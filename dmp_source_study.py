"""Direction B PR5: bound-preserving limiting for sign-changing sources.

Writes ``test_plots/dmp/sign_changing_sources.png`` and ``.json``:

(a) manufactured solution that *grows beyond* its initial/boundary range under
    a sign-changing source, anisotropic tensor on a skewed mesh: accuracy of
    high-order, low-order, PR4 global limiter (auto bounds), and the new
    ``source_envelope`` / ``source_local`` limiters;
(b) steep block + heater + sink: rigorous source envelope [m^n, M^n] vs the
    extrema produced by each scheme;
(c) maximum envelope violation and error against a fine reference;
(d) where the high-order scheme leaves the envelope.

Run: ``MPLCONFIGDIR=/tmp/matplotlib python dmp_source_study.py``
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

from heat_solver.afc import AFCMonotoneSolver
from heat_solver.dmp import anisotropic_tensor
from heat_solver.meshes import generate_nonorthogonal_polygonal_mesh, generate_nonorthogonal_tiled_polygonal_mesh
from heat_solver.mms import manufactured_case
from heat_solver.polygonal import PolygonalHeatSolver

OUT = Path("test_plots/dmp")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.grid": True, "grid.alpha": 0.3})

SCHEMES = {
    "high": ("high-order (reconstructed)", "o-", "tab:blue"),
    "low": ("low-order M-matrix", "s-", "tab:gray"),
    "global": ("PR4 global (auto bounds)", "D-", "tab:purple"),
    "source_envelope": ("source envelope (new)", "^-", "tab:red"),
    "source_local": ("source local, strict (new)", "v-", "tab:orange"),
    "source_local_s": ("source local, relaxed (new)", "P-", "tab:green"),
}


def _solve(v, p, alpha, dt, bc, src, u0, t_end, scheme):
    if scheme in ("high", "low"):
        s = PolygonalHeatSolver(v, p, alpha, dt, bc_type="dirichlet", bc_func=bc, source_func=src,
                                flux_discretization="reconstructed", nonorthogonal_correction=True,
                                monotone=(scheme == "low"))
    else:
        kw = {"limiter": scheme.replace("_s", "") if scheme == "source_local_s" else scheme}
        if scheme == "source_local_s":
            kw.update(limiter="source_local", smoothness_factor=1.0)
        s = AFCMonotoneSolver(v, p, alpha, dt, bc_func=bc, source_func=src, **kw)
    _, u = s.solve(u0, 0.0, t_end)
    return s, u


def study_mms():
    alpha = anisotropic_tensor(ratio=5.0, angle=np.pi / 6)
    case = manufactured_case("(0.2+4*t)*sin(pi*x)*sin(pi*y)", alpha=alpha.tolist(), model="diffusion")
    T = 0.05
    out = {k: [] for k in SCHEMES}
    hs = []
    for n in (12, 24, 48, 96):
        v, p, c = generate_nonorthogonal_polygonal_mesh(nx=n, ny=n, bbox=(-1, 1, -1, 1), skew=0.3)
        hs.append(2.0 / n)
        for key in SCHEMES:
            s, u = _solve(v, p, alpha, T / 50, case["boundary"], case["source"],
                          case["solution"](c[:, 0], c[:, 1], 0.0), T, key)
            ue = case["solution"](s.cell_centers[:, 0], s.cell_centers[:, 1], T)
            A = s.cell_areas
            out[key].append(float(np.sqrt(np.sum(A * (u - ue) ** 2) / np.sum(A * ue**2))))
        print("mms", n, {k: f"{v[-1]:.2e}" for k, v in out.items()})
    return {"h": hs, "errors": out, "initial_range": [-0.2, 0.2], "final_max": 0.4}


def heater_sink(Q0):
    def src(x, y, t):
        x = np.asarray(x, float); y = np.asarray(y, float)
        heat = ((x - 0.55) ** 2 + (y + 0.55) ** 2 < 0.3**2).astype(float)
        sink = ((x + 0.55) ** 2 + (y - 0.55) ** 2 < 0.3**2).astype(float)
        return Q0 * (heat - sink)
    return src


def study_steep(Q0=1.0):
    alpha = anisotropic_tensor(ratio=20.0, angle=np.pi / 6)
    src = heater_sink(Q0)
    bc = lambda x, y, t: np.zeros_like(np.asarray(x, float))
    dt, N = 1.5e-3, 20
    T = dt * N
    block = lambda c: ((np.abs(c[:, 0]) <= 0.35) & (np.abs(c[:, 1]) <= 0.35)).astype(float)
    v, p, c = generate_nonorthogonal_tiled_polygonal_mesh(nx_tiles=10, ny_tiles=10, bbox=(-1, 1, -1, 1), skew=0.4)
    vf, pf, cf = generate_nonorthogonal_tiled_polygonal_mesh(nx_tiles=32, ny_tiles=32, bbox=(-1, 1, -1, 1), skew=0.4)
    ref_s, ref_u = _solve(vf, pf, alpha, dt / 4, bc, src, block(cf), T, "high")
    ref = griddata(ref_s.cell_centers, ref_u, c, method="linear")

    probe = PolygonalHeatSolver(v, p, alpha, dt, bc_type="dirichlet", bc_func=bc, source_func=src,
                                flux_discretization="reconstructed", nonorthogonal_correction=True)
    q_int = src(c[:, 0], c[:, 1], 0.0)[~probe.is_boundary]
    env_t, env_lo, env_hi = [0.0], [0.0], [1.0]
    for n in range(N):
        env_t.append((n + 1) * dt)
        env_hi.append(env_hi[-1] + dt * max(q_int.max(), 0.0))
        env_lo.append(env_lo[-1] - dt * max((-q_int).max(), 0.0))
    env_lo, env_hi = np.array(env_lo), np.array(env_hi)

    results = {}
    for key in SCHEMES:
        u = block(c)
        ext = [(0.0, float(u.min()), float(u.max()))]
        if key in ("high", "low"):
            s = PolygonalHeatSolver(v, p, alpha, dt, bc_type="dirichlet", bc_func=bc, source_func=src,
                                    flux_discretization="reconstructed", nonorthogonal_correction=True,
                                    monotone=(key == "low"))
            worst = np.zeros_like(u)
            for n in range(N):
                _, u = s.solve(u, n * dt, (n + 1) * dt)
                ext.append(((n + 1) * dt, float(u.min()), float(u.max())))
                worst = np.maximum(worst, np.maximum(u - env_hi[n + 1], env_lo[n + 1] - u))
            activity = None
        else:
            s, u = _solve(v, p, alpha, dt, bc, src, block(c), T, key)
            ext += s.extrema_history
            worst = None
            activity = float(np.mean(s.limiter_activity))
        mins = np.array([e[1] for e in ext]); maxs = np.array([e[2] for e in ext])
        viol = float(max(np.max(maxs - env_hi), np.max(env_lo - mins), 0.0))
        A = s.cell_areas
        err = float(np.sqrt(np.nansum(A * (u - ref) ** 2) / np.nansum(A * ref**2)))
        results[key] = {"min": mins.tolist(), "max": maxs.tolist(), "violation": viol, "error": err,
                        "final_min": float(u.min()), "final_max": float(u.max()), "limiter_activity": activity,
                        "u": u, "worst": worst}
        print("steep", key, f"viol {viol:.3e} err {err:.3e} activity {activity}")
    ref_ext = (float(np.nanmin(ref)), float(np.nanmax(ref)))
    return {"t": env_t, "env_lo": env_lo.tolist(), "env_hi": env_hi.tolist(), "results": results,
            "centers": c, "reference_extrema": ref_ext, "Q0": Q0}


def main():
    t0 = time.perf_counter()
    mms = study_mms()
    steep = study_steep()
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    a = ax[0, 0]
    for key, (lab, mk, col) in SCHEMES.items():
        e = mms["errors"][key]
        order = np.polyfit(np.log(mms["h"]), np.log(e), 1)[0]
        a.loglog(mms["h"], e, mk, color=col, label=f"{lab} (fitted order {order:.2f})")
    a.set_xlabel("$h$"); a.set_ylabel(r"relative $L^2$ error")
    a.set_title("(a) growing solution, sign-changing source\n(anisotropic, skewed mesh)")
    a.legend(fontsize=7)
    a = ax[0, 1]
    t = np.array(steep["t"])
    env_lo, env_hi = np.array(steep["env_lo"]), np.array(steep["env_hi"])
    for key in ("high", "low", "global", "source_envelope", "source_local", "source_local_s"):
        lab, mk, col = SCHEMES[key]
        r = steep["results"][key]
        excess = np.maximum(np.array(r["max"]) - env_hi, env_lo - np.array(r["min"]))
        a.plot(t, excess, mk[0] + "-", ms=3, color=col, label=lab)
    a.axhline(0.0, color="k", lw=1)
    a.axhspan(0.0, 0.08, color="tab:red", alpha=0.06)
    a.text(0.001, 0.07, "outside the rigorous envelope", fontsize=7, color="tab:red")
    a.set_xlabel("$t$"); a.set_ylabel(r"$\max(\max_i u_i - M^n,\ m^n - \min_i u_i)$")
    a.set_title(f"(b) excess over the envelope $[m^n, M^n]$, steep block + heater/sink ($Q_0={steep['Q0']:.0f}$)",
                fontsize=9)
    a.legend(fontsize=7, loc="center right")
    a = ax[1, 0]
    keys = list(SCHEMES)
    viol = [max(steep["results"][k]["violation"], 1e-17) for k in keys]
    errs = [steep["results"][k]["error"] for k in keys]
    xpos = np.arange(len(keys))
    a.bar(xpos - 0.2, viol, 0.4, color="tab:red", label="max envelope violation")
    b = a.twinx(); b.grid(False)
    b.bar(xpos + 0.2, errs, 0.4, color="tab:blue", alpha=0.6, label="error vs fine reference")
    a.set_yscale("log"); a.set_ylim(1e-17, 1)
    a.set_xticks(xpos); a.set_xticklabels([SCHEMES[k][0].replace(" (", "\n(") for k in keys], fontsize=6.5, rotation=25, ha="right")
    a.set_ylabel("envelope violation", color="tab:red"); b.set_ylabel(r"relative $L^2$ error", color="tab:blue")
    a.set_title("(c) bounds vs accuracy (violations below 1e-16 shown at floor)")
    a = ax[1, 1]
    c = steep["centers"]
    w = steep["results"]["high"]["worst"]
    im = a.tripcolor(c[:, 0], c[:, 1], np.maximum(w, 0.0), shading="gouraud", cmap="Reds")
    fig.colorbar(im, ax=a, label="max over time of envelope violation")
    for (cx, cy), col in (((0.55, -0.55), "orange"), ((-0.55, 0.55), "cyan")):
        a.add_patch(plt.Circle((cx, cy), 0.3, fill=False, color=col, lw=1.5))
    a.add_patch(plt.Rectangle((-0.35, -0.35), 0.7, 0.7, fill=False, color="k", lw=1, ls="--"))
    a.set_aspect("equal"); a.grid(False)
    a.set_title("(d) high-order scheme: where it leaves the envelope\n(orange heater, cyan sink, dashed initial block)")
    fig.tight_layout()
    fig.savefig(OUT / "sign_changing_sources.png", dpi=170)
    plt.close(fig)
    summary = {"mms": mms, "steep": {k: v for k, v in steep.items() if k not in ("centers", "results")}}
    summary["steep"]["results"] = {k: {kk: vv for kk, vv in r.items() if kk not in ("u", "worst")}
                                   for k, r in steep["results"].items()}
    (OUT / "sign_changing_sources.json").write_text(json.dumps(summary, indent=2))
    print("done", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
