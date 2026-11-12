"""Cross-code validation of Thermalio against independent external codes.

Solves the same manufactured problems with

* Thermalio (polygonal cell-centred FV; square and skewed meshes),
* FiPy 4.0.3 (NIST structured finite volumes),
* scikit-fem 12.0.2 (P1 and P2 continuous Galerkin),

all with backward Euler at identical time steps, and compares each code with
the exact solution and with each other.  A fourth panel checks Thermalio's
enthalpy Stefan solver against Neumann's classical similarity solution.

Outputs ``test_plots/benchmark/external_codes/{external_code_comparison.png,
external_code_timing.png, summary.json, convergence.csv}``.  The timing
figure compares time-to-solution, work-precision and per-step cost; every
code runs single-threaded SciPy sparse kernels on the same machine, so the
timings compare implementations, not hardware.

Run: ``pip install -r requirements-external.txt`` then
``MPLCONFIGDIR=/tmp/matplotlib python external_code_comparison.py``
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from scipy.optimize import brentq
from scipy.special import erf

from heat_solver.dmp import anisotropic_tensor
from heat_solver.enthalpy import EnthalpyStefanSolver
from heat_solver.external_codes import fipy_heat_solve, skfem_heat_solve
from heat_solver.meshes import generate_nonorthogonal_polygonal_mesh, generate_square_polygonal_mesh
from heat_solver.phase_change import ApparentHeatCapacityModel
from heat_solver.polygonal import PolygonalHeatSolver
from heat_solver.transport import ReactionDiffusionHeatSolver

OUT = Path("test_plots/benchmark/external_codes")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.grid": True, "grid.alpha": 0.3})

LEVELS = (8, 16, 32, 64)


# ---------------------------------------------------------------------------
# Manufactured problems
# ---------------------------------------------------------------------------


def isotropic_case():
    k = 0.1
    ex = lambda x, y, t: np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y)
    return {
        "name": "isotropic Dirichlet", "bbox": (-1.0, 1.0, -1.0, 1.0), "kappa": k, "reaction": 0.0,
        "exact": ex, "source": lambda x, y, t: (2 * k * np.pi**2 - 1.0) * ex(x, y, t),
        "dirichlet": ex, "t_end": 0.05, "dt": 2.5e-4,
    }


def anisotropic_case():
    K = anisotropic_tensor(ratio=5.0, angle=np.pi / 6) * 0.1
    ex = lambda x, y, t: np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y)

    def source(x, y, t):
        u = ex(x, y, t)
        uxy = np.pi**2 * np.exp(-t) * np.cos(np.pi * x) * np.cos(np.pi * y)
        return -u + np.pi**2 * (K[0, 0] + K[1, 1]) * u - (K[0, 1] + K[1, 0]) * uxy

    return {
        "name": "anisotropic (ratio 5, 30 deg)", "bbox": (-1.0, 1.0, -1.0, 1.0), "kappa": K, "reaction": 0.0,
        "exact": ex, "source": source, "dirichlet": ex, "t_end": 0.05, "dt": 2.5e-4,
    }


def pennes_robin_case():
    k, perf, h = 0.1, 2.0, 3.0
    ex = lambda x, y, t: np.exp(-t) * (2.0 + np.sin(np.pi * x) * np.cos(np.pi * y))

    def grad(x, y, t):
        return (np.exp(-t) * np.pi * np.cos(np.pi * x) * np.cos(np.pi * y),
                -np.exp(-t) * np.pi * np.sin(np.pi * x) * np.sin(np.pi * y))

    def source(x, y, t):
        u = ex(x, y, t)
        lap = -2 * np.pi**2 * np.exp(-t) * np.sin(np.pi * x) * np.cos(np.pi * y)
        return -u - k * lap + perf * u

    def robin_value(x, y, t, nx, ny):
        gx, gy = grad(x, y, t)
        return k * (gx * nx + gy * ny) + h * ex(x, y, t)

    return {
        "name": "Pennes + Robin", "bbox": (0.0, 1.0, 0.0, 1.0), "kappa": k, "reaction": perf, "h": h,
        "exact": ex, "source": source, "robin_value": robin_value, "t_end": 0.05, "dt": 2.5e-4,
    }


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def _rel_l2_cells(u, ue, areas):
    return float(np.sqrt(np.sum(areas * (u - ue) ** 2) / np.sum(areas * ue**2)))


def run_thermalio(case, n, mesh="square", flux="reconstructed"):
    if mesh == "square":
        v, p, c = generate_square_polygonal_mesh(nx=n, ny=n, bbox=case["bbox"])
    else:
        v, p, c = generate_nonorthogonal_polygonal_mesh(nx=n, ny=n, bbox=case["bbox"], skew=0.3)
    t0 = time.perf_counter()
    if "robin_value" in case:
        h, rv = case["h"], case["robin_value"]
        solver = ReactionDiffusionHeatSolver(
            v, p, case["kappa"], case["dt"], reaction_rate=case["reaction"], bc_type="robin",
            bc_func=lambda x, y, t, nx, ny: (h * np.ones_like(x), rv(x, y, t, nx, ny)),
            source_func=case["source"],
        )
    else:
        solver = PolygonalHeatSolver(
            v, p, case["kappa"], case["dt"], bc_type="dirichlet", bc_func=case["dirichlet"],
            source_func=case["source"], flux_discretization=flux, nonorthogonal_correction=True,
        )
    x, y = solver.cell_centers[:, 0], solver.cell_centers[:, 1]
    if flux == "quadratic":  # unknowns are cell averages: consistent initial data and error metric
        _, u = solver.solve(solver.cell_average(case["exact"], 0.0), 0.0, case["t_end"])
        wall = time.perf_counter() - t0
        ue = solver.cell_average(case["exact"], case["t_end"])
    else:
        _, u = solver.solve(case["exact"](x, y, 0.0), 0.0, case["t_end"])
        wall = time.perf_counter() - t0
        ue = case["exact"](x, y, case["t_end"])
    return {"centers": solver.cell_centers, "u": u, "err": _rel_l2_cells(u, ue, solver.cell_areas),
            "max_err": float(np.max(np.abs(u - ue))), "time": wall, "dofs": solver.M}


def run_fipy(case, n):
    t0 = time.perf_counter()
    centers, u = fipy_heat_solve(n, case["bbox"], case["kappa"], case["dt"], case["t_end"],
                                 lambda x, y: case["exact"](x, y, 0.0), case["source"], case["dirichlet"],
                                 reaction=case["reaction"])
    wall = time.perf_counter() - t0
    x0, x1, y0, y1 = case["bbox"]
    areas = np.full(len(u), (x1 - x0) * (y1 - y0) / len(u))
    ue = case["exact"](centers[:, 0], centers[:, 1], case["t_end"])
    return {"centers": centers, "u": u, "err": _rel_l2_cells(u, ue, areas),
            "max_err": float(np.max(np.abs(u - ue))), "time": wall, "dofs": len(u)}


def run_skfem(case, n, order):
    t0 = time.perf_counter()
    kw = dict(reaction=case["reaction"], order=order)
    if "robin_value" in case:
        kw["robin"] = (case["h"], case["robin_value"])
    else:
        kw["dirichlet"] = case["dirichlet"]
    res = skfem_heat_solve(n, case["bbox"], case["kappa"], case["dt"], case["t_end"],
                           lambda x, y: case["exact"](x, y, 0.0), case["source"], **kw)
    wall = time.perf_counter() - t0
    return {"result": res, "err": res.l2_error(case["exact"], case["t_end"]), "time": wall, "dofs": res.ndofs}


def orders(hs, errs):
    hs, errs = np.asarray(hs, float), np.asarray(errs, float)
    return (np.log(errs[:-1] / errs[1:]) / np.log(hs[:-1] / hs[1:])).tolist()


def study_case(case, include_fipy=True, include_skewed=False):
    rows = []
    span = case["bbox"][1] - case["bbox"][0]
    for n in LEVELS:
        h = span / n
        th = run_thermalio(case, n)
        row = {"case": case["name"], "n": n, "h": h, "thermalio": th["err"], "thermalio_time": th["time"],
               "thermalio_dofs": th["dofs"]}
        if include_skewed:
            sk = run_thermalio(case, n, mesh="skewed")
            row.update({"thermalio_skewed": sk["err"], "thermalio_skewed_time": sk["time"]})
            sq = run_thermalio(case, n, mesh="skewed", flux="quadratic")
            row.update({"thermalio_skewed_quadratic": sq["err"], "thermalio_skewed_quadratic_time": sq["time"]})
        if include_fipy:
            fp_ = run_fipy(case, n)
            row.update({"fipy": fp_["err"], "fipy_time": fp_["time"],
                        "diff_thermalio_fipy": _rel_l2_cells(th["u"], fp_["u"], np.ones(len(fp_["u"])))
                        if np.allclose(th["centers"], fp_["centers"]) else float("nan")})
        for order in (1, 2):
            fe = run_skfem(case, n, order)
            row.update({f"skfem_p{order}": fe["err"], f"skfem_p{order}_time": fe["time"],
                        f"skfem_p{order}_dofs": fe["dofs"]})
            at_centers = fe["result"].evaluate(th["centers"])
            row[f"diff_thermalio_skfem_p{order}"] = float(
                np.sqrt(np.mean((th["u"] - at_centers) ** 2) / np.mean(at_centers**2)))
        rows.append(row)
        print(case["name"], n, {k: (f"{v:.2e}" if isinstance(v, float) else v) for k, v in row.items()
                                if k in ("thermalio", "fipy", "skfem_p1", "skfem_p2")})
    return rows


# ---------------------------------------------------------------------------
# Neumann (one-phase) Stefan similarity solution
# ---------------------------------------------------------------------------


def neumann_stefan(nx=160, t_end=0.5):
    """Melting of a solid initially at T_m = 0 by a hot wall T_w = 1.

    Exact front ``s(t) = 2 lam sqrt(alpha t)`` with
    ``lam exp(lam^2) erf(lam) = St / sqrt(pi)``, ``St = c (T_w - T_m) / L``,
    and liquid temperature ``T = T_w (1 - erf(x / 2 sqrt(alpha t)) / erf(lam))``.
    """
    alpha, c, L, Tw = 0.05, 1.0, 2.0, 1.0
    St = c * Tw / L
    lam = brentq(lambda z: z * np.exp(z * z) * erf(z) - St / np.sqrt(np.pi), 1e-6, 5.0)
    length = 1.0
    dx = length / nx
    x_wall = 0.5 * dx  # Dirichlet band centres sit at x = dx/2

    def exact_T(x, t):
        xi = np.maximum(np.asarray(x, float) - x_wall, 0.0)
        tt = max(t, 1e-12)
        s = 2 * lam * np.sqrt(alpha * tt)
        return np.where(xi < s, Tw * (1 - erf(xi / (2 * np.sqrt(alpha * tt))) / erf(lam)), 0.0)

    v, p, _ = generate_square_polygonal_mesh(nx=nx, ny=5, bbox=(0.0, length, 0.0, 5 * dx))
    width = 1e-3
    model = ApparentHeatCapacityModel(0.0, width, latent_heat=L, specific_heat=c)
    solver = EnthalpyStefanSolver(v, p, alpha, 2e-3, model, bc_type="dirichlet",
                                  bc_func=lambda x, y, t: exact_T(x, t))
    times = np.linspace(0.02, t_end, 25)
    u = np.zeros(solver.M)
    t = 0.0
    front_num, front_ex, profiles = [], [], {}
    row = np.abs(solver.cell_centers[:, 1] - 2.5 * dx) < 1e-9
    order = np.argsort(solver.cell_centers[row, 0])
    xs = solver.cell_centers[row, 0][order]
    for tk in times:
        t, u = solver.solve(u, t, tk)
        H = model.enthalpy(u[row][order])
        melted = np.clip(H / L, 0.0, 1.0) * dx  # enthalpy-weighted melted length per cell
        front_num.append(float(np.sum(melted) - 0.5 * dx))  # band cell is held liquid
        front_ex.append(float(2 * lam * np.sqrt(alpha * tk)))
        profiles[float(tk)] = u[row][order]
    T_err = float(np.max(np.abs(profiles[float(times[-1])] - exact_T(xs, times[-1]))))
    return {"lambda": lam, "stefan": St, "times": times.tolist(), "front_numeric": front_num,
            "front_exact": front_ex, "x": xs.tolist(), "T_final": profiles[float(times[-1])].tolist(),
            "T_exact_final": exact_T(xs, times[-1]).tolist(),
            "max_front_error": float(np.max(np.abs(np.array(front_num) - np.array(front_ex)))),
            "max_T_error_final": T_err, "dx": dx}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

TIMING_STYLES = {
    "thermalio": ("Thermalio FV", "o-", "tab:blue"),
    "thermalio_skewed": ("Thermalio FV skewed (reconstructed)", "o--", "tab:cyan"),
    "thermalio_skewed_quadratic": ("Thermalio FV skewed (quadratic)", "D-", "tab:purple"),
    "fipy": ("FiPy FV", "s-", "tab:orange"),
    "skfem_p1": ("scikit-fem P1", "^-", "tab:green"),
    "skfem_p2": ("scikit-fem P2", "v-", "tab:red"),
}


def _dofs(row, code):
    if code.startswith("skfem"):
        return row[f"{code}_dofs"]
    return row["thermalio_dofs"]


def time_to_accuracy(times, errs, target):
    """Log-log interpolation of wall time at which ``err = target`` (None if not bracketed)."""
    t, e = np.log(np.asarray(times, float)), np.log(np.asarray(errs, float))
    for k in range(len(e) - 1):
        lo, hi = sorted((e[k], e[k + 1]))
        if lo <= np.log(target) <= hi and e[k] != e[k + 1]:
            w = (np.log(target) - e[k]) / (e[k + 1] - e[k])
            return float(np.exp(t[k] + w * (t[k + 1] - t[k])))
    return None


def timing_figure(results, case_meta):
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.6))
    rows = results["isotropic"]
    a = ax[0, 0]
    for code, (lab, mk, col) in TIMING_STYLES.items():
        if code in rows[0] and f"{code}_time" in rows[0]:
            a.loglog([_dofs(r, code) for r in rows], [r[f"{code}_time"] for r in rows], mk, color=col, label=lab)
    d = np.array([64, 20000])
    a.loglog(d, 2e-4 * d, "k:", lw=0.8, label=r"$\propto N$ (linear scaling)")
    a.set_xlabel("degrees of freedom $N$"); a.set_ylabel("wall time [s] (setup + %d steps)" % case_meta["isotropic"])
    a.set_title("(a) time to solution, isotropic case"); a.legend(fontsize=6.5)
    a = ax[0, 1]
    for code, (lab, mk, col) in TIMING_STYLES.items():
        if code in rows[0] and f"{code}_time" in rows[0]:
            per = [1e6 * r[f"{code}_time"] / case_meta["isotropic"] / _dofs(r, code) for r in rows]
            a.loglog([_dofs(r, code) for r in rows], per, mk, color=col, label=lab)
    a.set_xlabel("degrees of freedom $N$"); a.set_ylabel(r"cost per step per DOF [$\mu$s]")
    a.set_title("(b) per-DOF cost: fixed overheads dominate small problems"); a.legend(fontsize=6.5)
    a = ax[0, 2]
    targets = (1e-3, 1e-4)
    codes = [c for c in TIMING_STYLES if any(c in results[k][0] and f"{c}_time" in results[k][0] for k in results)]
    keys = list(results)
    width = 0.8 / len(codes)
    for ci, code in enumerate(codes):
        vals = []
        for key in keys:
            rr = results[key]
            if code in rr[0] and f"{code}_time" in rr[0]:
                vals.append(time_to_accuracy([r[f"{code}_time"] for r in rr], [r[code] for r in rr], targets[0]))
            else:
                vals.append(np.nan)
        xs = np.arange(len(keys)) + ci * width
        lab, _, col = TIMING_STYLES[code]
        a.bar(xs, [v if v is not None else np.nan for v in vals], width, color=col, label=lab)
        for x_, v in zip(xs, vals):
            if v is None:
                a.text(x_, 0.02, "not\nreached", ha="center", va="bottom", fontsize=5, rotation=90,
                       transform=a.get_xaxis_transform())
    a.set_yscale("log"); a.set_xticks(np.arange(len(keys)) + 0.4 - width / 2); a.set_xticklabels(keys, fontsize=8)
    a.set_ylabel("wall time [s]"); a.set_title(f"(c) time to reach {targets[0]:.0e} relative error")
    a.legend(fontsize=6)
    for a, key, title in zip(ax[1], keys, ("(d) work-precision: isotropic", "(e) work-precision: anisotropic",
                                            "(f) work-precision: Pennes + Robin")):
        rr = results[key]
        for code, (lab, mk, col) in TIMING_STYLES.items():
            if code in rr[0] and f"{code}_time" in rr[0]:
                a.loglog([r[f"{code}_time"] for r in rr], [r[code] for r in rr], mk, color=col, label=lab)
        for tg in targets:
            a.axhline(tg, color="0.6", lw=0.6, ls="--")
        a.set_xlabel("wall time [s]"); a.set_ylabel("relative $L^2$ error"); a.set_title(title)
        a.legend(fontsize=6.5)
    fig.suptitle("Accuracy vs cost: Thermalio, FiPy and scikit-fem on identical problems "
                 "(single core, backward Euler, same $\\Delta t$)")
    fig.tight_layout()
    fig.savefig(OUT / "external_code_timing.png", dpi=160)
    plt.close(fig)


def main():
    t0 = time.perf_counter()
    results = {
        "isotropic": study_case(isotropic_case()),
        "anisotropic": study_case(anisotropic_case(), include_skewed=True),
        "pennes_robin": study_case(pennes_robin_case(), include_fipy=False),
    }
    timing_figure(results, {k: int(round(c["t_end"] / c["dt"])) for k, c in
                            (("isotropic", isotropic_case()), ("anisotropic", anisotropic_case()),
                             ("pennes_robin", pennes_robin_case()))})
    stefan = neumann_stefan()
    print("stefan", stefan["max_front_error"], stefan["max_T_error_final"], "elapsed", time.perf_counter() - t0)

    styles = {
        "thermalio": ("Thermalio FV (square)", "o-", "tab:blue"),
        "thermalio_skewed": ("Thermalio FV reconstructed (skewed mesh)", "o--", "tab:cyan"),
        "thermalio_skewed_quadratic": ("Thermalio FV quadratic (skewed mesh)", "D-", "tab:purple"),
        "fipy": ("FiPy FV", "s-", "tab:orange"),
        "skfem_p1": ("scikit-fem P1", "^-", "tab:green"),
        "skfem_p2": ("scikit-fem P2", "v-", "tab:red"),
    }
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    for a, (key, title) in zip(ax[0], (("isotropic", "(a) isotropic, Dirichlet"),
                                       ("anisotropic", "(b) anisotropic tensor, Dirichlet"),
                                       ("pennes_robin", "(c) Pennes perfusion + Robin"))):
        rows = results[key]
        hs = [r["h"] for r in rows]
        for code, (lab, mk, col) in styles.items():
            if code in rows[0]:
                errs = [r[code] for r in rows]
                a.loglog(hs, errs, mk, color=col, label=f"{lab} ({orders(hs, errs)[-1]:.2f})")
        hs_a = np.array(hs)
        ref = rows[-1]["thermalio"]
        a.loglog(hs_a, ref * (hs_a / hs_a[-1]) ** 2, "k:", lw=0.8, label="slope 2")
        a.set_xlabel("$h$"); a.set_ylabel(r"relative $L^2$ error vs exact"); a.set_title(title)
        a.legend(fontsize=6.5)
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[0, 2].annotate("P2 reaches the backward-Euler\ntime-error floor", xy=(0.016, 6e-6), xytext=(0.02, 4e-5),
                      fontsize=7, arrowprops={"arrowstyle": "->", "lw": 0.7})
    a = ax[1, 0]
    for key, mk in (("isotropic", "o"), ("anisotropic", "s"), ("pennes_robin", "^")):
        rows = results[key]
        hs = [r["h"] for r in rows]
        if "diff_thermalio_fipy" in rows[0]:
            a.loglog(hs, [r["diff_thermalio_fipy"] for r in rows], mk + "-", color="tab:orange",
                     label=f"Thermalio vs FiPy, {key}")
        a.loglog(hs, [r["diff_thermalio_skfem_p2"] for r in rows], mk + "--", color="tab:red",
                 label=f"Thermalio vs skfem P2, {key}")
    a.set_xlabel("$h$"); a.set_ylabel("relative cross-code difference")
    a.set_title("(d) code-to-code agreement"); a.legend(fontsize=6.5)
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a = ax[1, 1]
    rows = results["anisotropic"]
    for code, (lab, mk, col) in styles.items():
        if code in rows[0] and f"{code}_time" in rows[0]:
            a.loglog([r[f"{code}_time"] for r in rows], [r[code] for r in rows], mk, color=col, label=lab)
    a.set_xlabel("wall time [s]"); a.set_ylabel("relative $L^2$ error")
    a.set_title("(e) work-precision, anisotropic case"); a.legend(fontsize=6.5)
    a = ax[1, 2]
    a.plot(np.sqrt(stefan["times"]), stefan["front_exact"], "k-", lw=2, label="Neumann exact front")
    a.plot(np.sqrt(stefan["times"]), stefan["front_numeric"], "ro", ms=4, label="Thermalio enthalpy solver")
    a.set_xlabel(r"$\sqrt{t}$"); a.set_ylabel("melt-front position $s(t)$")
    a.set_title(f"(f) Neumann Stefan (St={stefan['stefan']:.2f}), max front error {stefan['max_front_error']:.1e}")
    b = a.inset_axes([0.55, 0.12, 0.42, 0.38])
    b.plot(stefan["x"], stefan["T_exact_final"], "k-", lw=1.5)
    b.plot(stefan["x"], stefan["T_final"], "r--", lw=1)
    b.set_xlim(0, 0.6); b.set_title("$T(x)$ at $t_f$", fontsize=7); b.tick_params(labelsize=6)
    a.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "external_code_comparison.png", dpi=170)
    plt.close(fig)

    summary = {k: [{kk: vv for kk, vv in r.items()} for r in v] for k, v in results.items()}
    summary["orders"] = {
        key: {code: orders([r["h"] for r in rows], [r[code] for r in rows])
              for code in styles if code in rows[0]}
        for key, rows in results.items()
    }
    summary["neumann_stefan"] = {k: v for k, v in stefan.items() if k not in ("x", "T_final", "T_exact_final")}
    summary["versions"] = {}
    import fipy
    import skfem
    summary["versions"] = {"fipy": fipy.__version__, "scikit-fem": skfem.__version__}
    summary["time_to_accuracy"] = {
        key: {code: {f"{tg:.0e}": time_to_accuracy([r[f"{code}_time"] for r in rows], [r[code] for r in rows], tg)
                     for tg in (1e-3, 1e-4)}
              for code in TIMING_STYLES if code in rows[0] and f"{code}_time" in rows[0]}
        for key, rows in results.items()
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(OUT / "convergence.csv", "w", newline="") as fh:
        keys = sorted({k for rows in results.values() for r in rows for k in r})
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for rows in results.values():
            for r in rows:
                w.writerow(r)
    print(json.dumps(summary["orders"], indent=1))
    print("total", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
