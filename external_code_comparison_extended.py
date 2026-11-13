"""Extended cross-code comparison: Thermalio against seven independent codes.

Adds three codes to the FiPy / scikit-fem study (``external_code_comparison.py``):

* **NGSolve** (compiled C++ finite elements; P1, P2, P3; 1 thread and 4 threads),
* **PorePy** (cell-centred finite volumes, TPFA and MPFA-O, on the same square
  and skewed meshes as Thermalio),
* **py-pde** (Numba-compiled finite differences, backward Euler),

and runs everything on the same three manufactured problems (isotropic
Dirichlet, anisotropic Dirichlet, Pennes + Robin), with backward Euler and the
same time step.  Errors are relative L2 errors against the exact solution
(cell-centred codes: at cell centres; finite elements: integrated).

Timing.  Every run is timed twice, once for a single step and once for the full
run, which splits it into a one-off part (mesh, assembly, factorization,
just-in-time compilation) and the per-step cost.  py-pde reports its Numba
compilation directly.  A scaling study (n up to 256, isotropic) measures the
per-step cost against problem size and, for NGSolve, against thread count.

Outputs ``test_plots/benchmark/external_codes_extended/{cross_code_accuracy.png,
cross_code_cost.png, summary.json, convergence.csv}``.

Run: ``pip install -r requirements-external.txt`` (PorePy also needs the system
library ``libGLU``), then
``MPLCONFIGDIR=/tmp/matplotlib python external_code_comparison_extended.py``
(``--replot`` redraws the figures from ``summary.json``; ``--resume`` reuses the
accuracy runs saved in ``results_partial.json`` and redoes only the scaling study).
"""

from __future__ import annotations

import csv
import gc
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import sympy as sp

import external_code_comparison as base
from heat_solver.dmp import anisotropic_tensor
from heat_solver.external_codes import (
    SymbolicProblem,
    ngsolve_heat_solve,
    porepy_heat_solve,
    pypde_heat_solve,
)
from heat_solver.meshes import generate_nonorthogonal_polygonal_mesh

OUT = Path("test_plots/benchmark/external_codes_extended")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.grid": True, "grid.alpha": 0.3})
warnings.filterwarnings("ignore")

LEVELS = (8, 16, 32, 64)
SCALING_LEVELS = (32, 64, 128, 256)
SCALING_STEPS = 40
X, Y, T = sp.symbols("x y t")


def cases():
    """The three problems of ``external_code_comparison.py``, with a SymPy twin for NGSolve / py-pde."""
    sine = sp.exp(-T) * sp.sin(sp.pi * X) * sp.sin(sp.pi * Y)
    iso, ani, pen = base.isotropic_case(), base.anisotropic_case(), base.pennes_robin_case()
    iso["sym"] = SymbolicProblem(sine, iso["bbox"], iso["kappa"])
    ani["sym"] = SymbolicProblem(sine, ani["bbox"], ani["kappa"])
    pen["sym"] = SymbolicProblem(sp.exp(-T) * (2 + sp.sin(sp.pi * X) * sp.cos(sp.pi * Y)), pen["bbox"], pen["kappa"],
                                 reaction=pen["reaction"], robin_h=pen["h"])
    return {"isotropic": iso, "anisotropic": ani, "pennes_robin": pen}


# Every series: label, marker, colour, family.
STYLES = {
    "thermalio": ("Thermalio FV (TPFA + corr.)", "o-", "tab:blue", "fv"),
    "thermalio_face": ("Thermalio FV, face Dirichlet", "x--", "#6baed6", "fv"),
    "fipy": ("FiPy FV", "s-", "tab:orange", "fv"),
    "porepy_tpfa": ("PorePy TPFA", "P-", "tab:olive", "fv"),
    "porepy_mpfa": ("PorePy MPFA-O", "P--", "tab:brown", "fv"),
    "pypde": ("py-pde FD", "*-", "tab:pink", "fd"),
    "skfem_p1": ("scikit-fem P1", "^-", "#98df8a", "fe"),
    "skfem_p2": ("scikit-fem P2", "v-", "tab:green", "fe"),
    "ngsolve_p1": ("NGSolve P1", "^:", "#ff9896", "fe"),
    "ngsolve_p2": ("NGSolve P2", "v:", "tab:red", "fe"),
    "ngsolve_p3": ("NGSolve P3", "d:", "#8c1c13", "fe"),
    "ngsolve_p2_4t": ("NGSolve P2, 4 threads", "v-", "tab:purple", "fe"),
}
SKEW_STYLES = {
    "thermalio_skewed": ("Thermalio TPFA + correction", "o-", "tab:blue"),
    "thermalio_skewed_quadratic": ("Thermalio quadratic flux", "D-", "tab:cyan"),
    "porepy_tpfa_skewed": ("PorePy TPFA", "P-", "tab:olive"),
    "porepy_mpfa_skewed": ("PorePy MPFA-O", "P--", "tab:brown"),
}


def _rel(u, ue, w):
    return float(np.sqrt(np.sum(w * (u - ue) ** 2) / np.sum(w * ue**2)))


# ---------------------------------------------------------------------------
# Runners: each returns {"err", "time", "setup", "per_step", "dofs", "res"}
# ---------------------------------------------------------------------------


def _timed(case, fn, err_of, dofs_of, repeats=1):
    """Time ``fn(case)`` as a user sees it (cold) and split it into one-off cost and per-step cost.

    The reported ``time`` is a cold full run (nothing cached: geometry, assembly,
    factorization, compilation).  The per-step cost is the difference of two warm
    runs, the full run and a one-step run, so caches (e.g. Thermalio's geometry
    cache) cannot leak into it; with ``repeats > 1`` each warm time is the
    minimum over repeats.  ``setup`` is the cold time minus the stepping.
    """
    nsteps = int(round(case["t_end"] / case["dt"]))

    def clock(c):
        gc.collect()  # no garbage-collection pause from earlier runs inside the timed region
        gc.disable()
        try:
            t0 = time.perf_counter()
            out = fn(c)
            return out, time.perf_counter() - t0
        finally:
            gc.enable()

    res, cold = clock(case)

    def best(c):
        return min(clock(c)[1] for _ in range(repeats))

    warm_one = best(dict(case, t_end=case["dt"]))
    warm_full = best(case)
    per_step = max(warm_full - warm_one, 0.0) / max(nsteps - 1, 1)
    return {"err": err_of(res), "time": cold, "per_step": per_step, "setup": max(cold - nsteps * per_step, 0.0),
            "dofs": dofs_of(res), "res": res}


def run_thermalio(case, n, repeats=1, **kw):
    return _timed(case, lambda c: base.run_thermalio(c, n, **kw), lambda r: r["err"], lambda r: r["dofs"], repeats)


def run_fipy(case, n, repeats=1):
    return _timed(case, lambda c: base.run_fipy(c, n), lambda r: r["err"], lambda r: r["dofs"], repeats)


def run_skfem(case, n, order, repeats=1):
    return _timed(case, lambda c: base.run_skfem(c, n, order), lambda r: r["err"], lambda r: r["dofs"], repeats)


def run_ngsolve(case, n, order, threads=1, repeats=1):
    return _timed(case, lambda c: ngsolve_heat_solve(n, c["sym"], c["dt"], c["t_end"], order=order, threads=threads),
                  lambda r: r.l2_error(), lambda r: r.ndofs, repeats)


def run_porepy(case, n, method, skew=0.0, repeats=1):
    nodes = None
    if skew:
        nodes = generate_nonorthogonal_polygonal_mesh(nx=n, ny=n, bbox=case["bbox"], skew=skew)[0]
    ex = case["exact"]

    def fn(c):
        centers, u, vol = porepy_heat_solve(n, c["bbox"], c["kappa"], c["dt"], c["t_end"], lambda x, y: ex(x, y, 0.0),
                                            c["source"], c["dirichlet"], reaction=c["reaction"], method=method,
                                            nodes=nodes)
        return {"centers": centers, "u": u, "err": _rel(u, ex(centers[:, 0], centers[:, 1], c["t_end"]), vol),
                "dofs": len(u)}

    return _timed(case, fn, lambda r: r["err"], lambda r: r["dofs"], repeats)


def run_pypde(case, n):
    centers, u, times = pypde_heat_solve(n, case["sym"], case["dt"], case["t_end"])
    ue = case["exact"](centers[:, 0], centers[:, 1], case["t_end"])
    nsteps = int(round(case["t_end"] / case["dt"]))
    return {"err": _rel(u, ue, np.ones_like(u)), "time": times["total"], "per_step": times["stepping"] / nsteps,
            "setup": times["compile"], "dofs": len(u), "res": {"centers": centers, "u": u}}


def _cross(a, b):
    """Relative difference of two cell-centred results on the same cells (``nan`` if the cells differ)."""
    if a["centers"].shape != b["centers"].shape or not np.allclose(a["centers"], b["centers"]):
        return float("nan")
    return float(np.sqrt(np.mean((a["u"] - b["u"]) ** 2) / np.mean(b["u"] ** 2)))


def study_case(key, case):
    rows = []
    dirichlet = "dirichlet" in case
    span = case["bbox"][1] - case["bbox"][0]
    for n in LEVELS:
        row = {"case": key, "n": n, "h": span / n}
        runs = {"thermalio": run_thermalio(case, n)}
        if dirichlet:
            runs["thermalio_face"] = run_thermalio(case, n, dirichlet_mode="face")
            runs["fipy"] = run_fipy(case, n)
            runs["porepy_tpfa"] = run_porepy(case, n, "tpfa")
            runs["porepy_mpfa"] = run_porepy(case, n, "mpfa")
        runs["pypde"] = run_pypde(case, n)
        for order in (1, 2):
            runs[f"skfem_p{order}"] = run_skfem(case, n, order)
        for order in (1, 2, 3):
            runs[f"ngsolve_p{order}"] = run_ngsolve(case, n, order)
        runs["ngsolve_p2_4t"] = run_ngsolve(case, n, 2, threads=4)
        if key == "anisotropic":
            runs["thermalio_skewed"] = run_thermalio(case, n, mesh="skewed")
            runs["thermalio_skewed_quadratic"] = run_thermalio(case, n, mesh="skewed", flux="quadratic")
            runs["porepy_tpfa_skewed"] = run_porepy(case, n, "tpfa", skew=0.3)
            runs["porepy_mpfa_skewed"] = run_porepy(case, n, "mpfa", skew=0.3)
        for code, r in runs.items():
            row.update({code: r["err"], f"{code}_time": r["time"], f"{code}_setup": r["setup"],
                        f"{code}_per_step": r["per_step"], f"{code}_dofs": r["dofs"]})
        # Code-to-code agreement on identical cells (same 5-point stencil with face Dirichlet data).
        ref = runs["thermalio_face"]["res"] if dirichlet else runs["thermalio"]["res"]
        for code in ("fipy", "porepy_tpfa", "pypde"):
            if dirichlet and code in runs:
                row[f"diff_thermalio_face_{code}"] = _cross(ref, runs[code]["res"])
        th = runs["thermalio"]["res"]
        if not dirichlet:
            row["diff_thermalio_pypde_robin"] = _cross(th, runs["pypde"]["res"])
        p3 = runs["ngsolve_p3"]["res"].evaluate(th["centers"])
        row["diff_thermalio_ngsolve_p3"] = float(np.sqrt(np.mean((th["u"] - p3) ** 2) / np.mean(p3**2)))
        sk = runs["skfem_p2"]["res"]["result"].evaluate(th["centers"])
        ng2 = runs["ngsolve_p2"]["res"].evaluate(th["centers"])
        row["diff_skfem_p2_ngsolve_p2"] = float(np.sqrt(np.mean((sk - ng2) ** 2) / np.mean(ng2**2)))
        rows.append(row)
        print(key, n, {c: f"{row[c]:.2e}/{row[c + '_time']:.2f}s" for c in runs}, flush=True)
    return rows


def scaling_study(case):
    """Per-step cost vs problem size (isotropic case, ``SCALING_STEPS`` steps)."""
    c = dict(case, t_end=SCALING_STEPS * case["dt"])
    out = []
    for n in SCALING_LEVELS:
        row = {"n": n, "cells": n * n}
        runs = {
            "thermalio": run_thermalio(c, n, repeats=3),
            "thermalio_face": run_thermalio(c, n, repeats=3, dirichlet_mode="face"),
            "porepy_tpfa": run_porepy(c, n, "tpfa", repeats=3),
            "skfem_p1": run_skfem(c, n, 1, repeats=2),
            "ngsolve_p1": run_ngsolve(c, n, 1, repeats=3),
            "ngsolve_p1_4t": run_ngsolve(c, n, 1, threads=4, repeats=3),
            "ngsolve_p2": run_ngsolve(c, n, 2, repeats=3),
            "ngsolve_p2_4t": run_ngsolve(c, n, 2, threads=4, repeats=3),
        }
        if n <= 128:
            runs["fipy"] = run_fipy(c, n)
        try:
            runs["pypde"] = run_pypde(c, n)
        except Exception as exc:  # py-pde's implicit Euler is a fixed-point iteration: it needs dt alpha / h^2 small
            row["pypde_failure"] = f"{type(exc).__name__}: {exc}"
            print("scaling", n, "py-pde failed:", row["pypde_failure"], flush=True)
        for code, r in runs.items():
            row.update({f"{code}_per_step": r["per_step"], f"{code}_setup": r["setup"], f"{code}_dofs": r["dofs"],
                        f"{code}_err": r["err"]})
        out.append(row)
        print("scaling", n, {k: f"{row[k + '_per_step'] * 1e3:.2f}ms" for k in runs}, flush=True)
    return out


def orders(hs, errs):
    hs, errs = np.asarray(hs, float), np.asarray(errs, float)
    return (np.log(errs[:-1] / errs[1:]) / np.log(hs[:-1] / hs[1:])).tolist()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

TITLES = {"isotropic": "isotropic, Dirichlet", "anisotropic": "anisotropic tensor (ratio 5, 30°), Dirichlet",
          "pennes_robin": "Pennes perfusion + Robin"}
SCALING_STYLES = {
    "thermalio": ("Thermalio FV", "o-", "tab:blue"),
    "thermalio_face": ("Thermalio FV, face Dirichlet", "x--", "#6baed6"),
    "fipy": ("FiPy FV", "s-", "tab:orange"),
    "porepy_tpfa": ("PorePy TPFA", "P-", "tab:olive"),
    "pypde": ("py-pde FD (stepping only)", "*-", "tab:pink"),
    "skfem_p1": ("scikit-fem P1", "^-", "#98df8a"),
    "ngsolve_p1": ("NGSolve P1, 1 thread", "^:", "#ff9896"),
    "ngsolve_p1_4t": ("NGSolve P1, 4 threads", "^-", "tab:purple"),
    "ngsolve_p2": ("NGSolve P2, 1 thread", "v:", "tab:red"),
    "ngsolve_p2_4t": ("NGSolve P2, 4 threads", "v-", "tab:purple"),
}


def _series(rows, code):
    return code in rows[0] and not all(np.isnan(r.get(code, np.nan)) for r in rows)


def accuracy_figure(results):
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 9.2))
    for a, key in zip(ax[0], results):
        rows = results[key]
        hs = [r["h"] for r in rows]
        for code, (lab, mk, col, _) in STYLES.items():
            if _series(rows, code):
                errs = [r[code] for r in rows]
                a.loglog(hs, errs, mk, color=col, ms=4, label=f"{lab} ({orders(hs, errs)[-1]:.2f})")
        h = np.array(hs)
        a.loglog(h, rows[-1]["thermalio"] * (h / h[-1]) ** 2, "k:", lw=0.8, label="slope 2")
        floor = rows[-1]["ngsolve_p3"]  # P3 at the finest mesh: essentially the time error alone
        a.axhline(floor, color="0.5", lw=0.7, ls="-.")
        a.text(hs[0], floor * 1.15, "backward-Euler time-error floor", fontsize=6.5, color="0.35", ha="right")
        a.set_xlabel("$h$"); a.set_ylabel("relative $L^2$ error vs exact")
        a.set_title(f"({'abc'[list(results).index(key)]}) {TITLES[key]}")
        a.legend(fontsize=5.8, ncol=1, loc="lower right")
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    # (d) identical skewed meshes: Thermalio vs PorePy
    a = ax[1, 0]
    rows = results["anisotropic"]
    hs = [r["h"] for r in rows]
    for code, (lab, mk, col) in SKEW_STYLES.items():
        errs = [r[code] for r in rows]
        a.loglog(hs, errs, mk, color=col, label=f"{lab} ({orders(hs, errs)[-1]:.2f})")
    a.set_xlabel("$h$"); a.set_ylabel("relative $L^2$ error")
    a.set_title("(d) anisotropic case on identical skewed meshes (skew 0.3):\nfinite-volume flux choices")
    a.legend(fontsize=7)
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    # (e) code-to-code agreement
    a = ax[1, 1]
    pairs = {"diff_thermalio_face_fipy": ("Thermalio face vs FiPy", "s", "tab:orange"),
             "diff_thermalio_face_porepy_tpfa": ("Thermalio face vs PorePy TPFA", "P", "tab:olive"),
             "diff_thermalio_face_pypde": ("Thermalio face vs py-pde", "*", "tab:pink"),
             "diff_thermalio_pypde_robin": ("Thermalio vs py-pde (Robin)", "*", "tab:gray"),
             "diff_skfem_p2_ngsolve_p2": ("scikit-fem P2 vs NGSolve P2", "v", "tab:green"),
             "diff_thermalio_ngsolve_p3": ("Thermalio vs NGSolve P3", "d", "tab:red")}
    for key, ls in (("isotropic", "-"), ("anisotropic", "--"), ("pennes_robin", "-.")):
        rows = results[key]
        hs = [r["h"] for r in rows]
        for col_key, (lab, mk, col) in pairs.items():
            if col_key in rows[0]:
                a.loglog(hs, [max(r[col_key], 1e-16) for r in rows], mk + ls, color=col, ms=4, label=f"{lab}, {key}")
    a.axhspan(1e-16, 1e-11, color="0.85", alpha=0.6, lw=0)
    a.text(0.03, 0.04, "round-off: same discretization", transform=a.transAxes, fontsize=7)
    a.set_xlabel("$h$"); a.set_ylabel("relative difference between codes")
    a.set_title("(e) code-to-code agreement"); a.legend(fontsize=5.6, ncol=1)
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    # (f) accuracy per unknown
    a = ax[1, 2]
    rows = results["isotropic"]
    for code, (lab, mk, col, _) in STYLES.items():
        if _series(rows, code) and code != "ngsolve_p2_4t":
            a.loglog([r[f"{code}_dofs"] for r in rows], [r[code] for r in rows], mk, color=col, ms=4, label=lab)
    a.set_xlabel("unknowns $N$"); a.set_ylabel("relative $L^2$ error")
    a.set_title("(f) accuracy per unknown, isotropic case\n(backward-Euler time error ~1e-5 caps high order)")
    a.legend(fontsize=5.8)
    fig.suptitle("Thermalio against FiPy, PorePy, py-pde, scikit-fem and NGSolve on identical problems "
                 "(backward Euler, same $\\Delta t$; observed order of the finest pair in brackets)")
    fig.tight_layout()
    fig.savefig(OUT / "cross_code_accuracy.png", dpi=160)
    plt.close(fig)


def cost_figure(results, scaling):
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 9.2))
    for a, key in zip(ax[0], results):
        rows = results[key]
        for code, (lab, mk, col, _) in STYLES.items():
            if _series(rows, code):
                a.loglog([r[f"{code}_time"] for r in rows], [r[code] for r in rows], mk, color=col, ms=4, label=lab)
        for tg in (1e-3, 1e-4):
            a.axhline(tg, color="0.6", lw=0.6, ls="--")
        a.set_xlabel("wall time [s], one fresh solve (setup + compile + all steps)")
        a.set_ylabel("relative $L^2$ error")
        a.set_title(f"({'abc'[list(results).index(key)]}) work-precision: {TITLES[key]}")
        a.legend(fontsize=5.8)
    # (d) one-off vs stepping at n = 64, isotropic
    a = ax[1, 0]
    row = results["isotropic"][-1]
    codes = [c for c in STYLES if _series(results["isotropic"], c)]
    setup = np.array([row[f"{c}_setup"] for c in codes])
    stepping = np.array([row[f"{c}_time"] for c in codes]) - setup
    y = np.arange(len(codes))
    a.barh(y, setup, color="0.55", label="one-off: mesh, assembly, factorization, JIT compile")
    a.barh(y, stepping, left=setup, color="tab:blue", label="200 time steps")
    a.set_yticks(y); a.set_yticklabels([STYLES[c][0] for c in codes], fontsize=7); a.invert_yaxis()
    a.set_xscale("log"); a.set_xlabel("wall time [s]")
    a.set_title(f"(d) where the time goes, isotropic n = {row['n']}")
    a.legend(fontsize=6.5, loc="upper right")
    # (e) per-step cost vs size
    a = ax[1, 1]
    for code, (lab, mk, col) in SCALING_STYLES.items():
        pts = [(r[f"{code}_dofs"], r[f"{code}_per_step"]) for r in scaling
               if r.get(f"{code}_per_step", 0.0) > 0.0]
        if pts:
            a.loglog(*zip(*pts), mk, color=col, ms=4, label=lab)
    d = np.array([1e3, 3e5])
    a.loglog(d, 2e-7 * d, "k:", lw=0.8, label=r"$\propto N$")
    a.set_xlabel("unknowns $N$"); a.set_ylabel("cost per time step [s]")
    a.set_title("(e) per-step cost vs size (isotropic, one-off cost excluded)")
    a.legend(fontsize=5.8)
    # (f) time to 1e-4
    a = ax[1, 2]
    tg = 1e-4
    keys = list(results)
    codes = [c for c in STYLES if any(_series(results[k], c) for k in keys)]
    width = 0.84 / len(codes)
    for ci, code in enumerate(codes):
        for ki, key in enumerate(keys):
            rr = results[key]
            if not _series(rr, code):
                continue
            v, status = base.time_to_accuracy([r[f"{code}_time"] for r in rr], [r[code] for r in rr], tg,
                                              with_status=True)
            x_ = ki + ci * width
            lab, _, col, _ = STYLES[code]
            if v is None:
                a.text(x_, 0.02, "not reached", ha="center", va="bottom", fontsize=4.8, rotation=90,
                       transform=a.get_xaxis_transform())
                continue
            a.bar(x_, v, width, color=col, label=lab, edgecolor="white", lw=0.3,
                  hatch="///" if status == "coarsest" else None, alpha=0.55 if status == "coarsest" else 1.0)
    a.set_yscale("log"); a.set_xticks(np.arange(len(keys)) + 0.42 - width / 2); a.set_xticklabels(keys, fontsize=8)
    a.set_ylabel("wall time [s]")
    a.set_title(f"(f) time to reach {tg:.0e} relative error (one fresh solve)\n"
                "hatched: already below on the coarsest mesh (upper bound)", fontsize=9)
    handles, labels = a.get_legend_handles_labels()
    seen = {}
    for h_, l_ in zip(handles, labels):
        seen.setdefault(l_, h_)
    lo, hi = a.get_ylim()
    a.set_ylim(lo, hi * 40)
    a.legend(seen.values(), seen.keys(), fontsize=5.6, ncol=3, loc="upper center")
    fig.suptitle("Cost on one core (NGSolve also on 4 threads): one-off setup and compilation vs time stepping")
    fig.tight_layout()
    fig.savefig(OUT / "cross_code_cost.png", dpi=160)
    plt.close(fig)


def _versions():
    import fipy
    import ngsolve
    import pde
    import porepy
    import skfem

    return {"fipy": fipy.__version__, "scikit-fem": skfem.__version__, "ngsolve": ngsolve.__version__,
            "porepy": porepy.__version__, "py-pde": pde.__version__}


def main():
    t0 = time.perf_counter()
    cs = cases()
    run_pypde(dict(cs["isotropic"], t_end=cs["isotropic"]["dt"]), 8)  # one-time Numba start-up, not charged
    partial = OUT / "results_partial.json"
    if "--resume" in sys.argv and partial.exists():
        results = json.loads(partial.read_text())
    else:
        results = {key: study_case(key, case) for key, case in cs.items()}
        partial.write_text(json.dumps(results))
    scaling = scaling_study(cs["isotropic"])
    summary = {"results": results, "scaling": scaling, "scaling_steps": SCALING_STEPS, "versions": _versions(),
               "orders": {key: {code: orders([r["h"] for r in rows], [r[code] for r in rows])
                                for code in list(STYLES) + list(SKEW_STYLES) if code in rows[0]}
                          for key, rows in results.items()},
               "time_to_accuracy": {key: {code: dict(zip(("time", "status"), base.time_to_accuracy(
                   [r[f"{code}_time"] for r in rows], [r[code] for r in rows], 1e-4, with_status=True)))
                   for code in STYLES if code in rows[0]} for key, rows in results.items()}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(OUT / "convergence.csv", "w", newline="") as fh:
        keys = sorted({k for rows in results.values() for r in rows for k in r})
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for rows in results.values():
            w.writerows(rows)
    accuracy_figure(results)
    cost_figure(results, scaling)
    print("total", time.perf_counter() - t0)


def replot():
    summary = json.loads((OUT / "summary.json").read_text())
    accuracy_figure(summary["results"])
    cost_figure(summary["results"], summary["scaling"])


if __name__ == "__main__":
    replot() if "--replot" in sys.argv else main()
