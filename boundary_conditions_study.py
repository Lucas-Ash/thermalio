"""Boundary-condition capability matrix: every solver x every bc_type.

For each solver and each boundary type the matching standard test problem
(:mod:`heat_solver.bc_test_problems`) is solved on three meshes and the
observed order of the relative L2 error is recorded.  Heat-equation solvers
use the classical exact problems; the extended transport models use the
manufactured problems.  Writes ``test_plots/boundary_conditions/
{capability_matrix.png, summary.json}``.

Run: ``MPLCONFIGDIR=/tmp/matplotlib python boundary_conditions_study.py``
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from heat_solver.afc import AFCMonotoneSolver
from heat_solver.bc_test_problems import heat_problem, transport_problem
from heat_solver.boundary_conditions import BC_TYPES
from heat_solver.curvilinear import CurvilinearHeatSolver
from heat_solver.enthalpy import ApparentCapacityPicardSolver, EnthalpyStefanSolver, SmoothPhaseChangeModel
from heat_solver.mesh_library import generate
from heat_solver.polygonal import PolygonalHeatSolver
from heat_solver.transport import (
    AdvectionDiffusionHeatSolver,
    FractionalHeatSolver,
    HyperbolicHeatSolver,
    ReactionDiffusionHeatSolver,
)
from heat_solver.triangular import NonUniformHeatSolver

OUT = Path("test_plots/boundary_conditions")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})

T_END = 0.05
LEVELS = (8, 16, 32, 64)


def _rel_l2(u, ue, w):
    return float(np.sqrt(np.sum(w * (u - ue) ** 2) / np.sum(w * ue**2)))


def _dt(n, span):
    """dt ~ h^2 so the first-order time error does not mask the spatial order."""
    h = span / n
    return min(0.25 * h * h, T_END / 10)


def run_cell_centred(make, prob, n, skew=0.0, averages=False):
    b = prob["bbox"]
    gen = "nonorthogonal" if skew else "square"
    params = dict(nx=n, ny=n, bbox=b) | ({"skew": skew} if skew else {})
    mesh = generate(gen, **params)
    dt = _dt(n, b[1] - b[0])
    s = make(mesh, prob, dt)
    if averages:
        u0 = s.cell_average(prob["exact"], 0.0)
        ue = s.cell_average(prob["exact"], T_END)
    else:
        c = s.cell_centers
        u0 = prob["exact"](c[:, 0], c[:, 1], 0.0)
        ue = prob["exact"](c[:, 0], c[:, 1], T_END)
    kw = {"du0": prob["initial_rate"](*s.cell_centers.T)} if "initial_rate" in prob and isinstance(
        s, HyperbolicHeatSolver) else {}
    _, u = s.solve(u0, 0.0, T_END, **kw)
    return _rel_l2(u, ue, s.cell_areas)


def run_triangular(prob, n):
    b = prob["bbox"]
    mesh = generate("delaunay", nx=n + 1, ny=n + 1, jitter=0.15, bbox=b, seed=2)
    p, tri = mesh.triangulation()
    s = NonUniformHeatSolver(p, tri, prob["alpha"], _dt(n, b[1] - b[0]), bc_type=prob["bc_type"],
                             bc_func=prob["bc_func"], source_func=prob["source"])
    _, u = s.solve(prob["exact"](p[:, 0], p[:, 1], 0.0), 0.0, T_END)
    return _rel_l2(u, prob["exact"](p[:, 0], p[:, 1], T_END), s.cv_area)


def run_curvilinear(prob, n):
    b = prob["bbox"]
    X, Y = generate("curvilinear", nx=n + 1, ny=n + 1, bbox=b, warp=0.08).structured()
    s = CurvilinearHeatSolver(X, Y, alpha=prob["alpha"], dt=_dt(n, b[1] - b[0]), bc_type=prob["bc_type"],
                              bc_func=prob["bc_func"], source_func=prob["source"])
    _, u = s.solve(prob["exact"](X, Y, 0.0), 0.0, T_END)
    return _rel_l2(u, prob["exact"](X, Y, T_END), s.J)


def heat_solvers():
    lin = SmoothPhaseChangeModel(latent_heat=0.0)
    common = lambda p: dict(bc_type=p["bc_type"], bc_func=p["bc_func"], source_func=p["source"])
    return {
        "Polygonal TPFA (square)": lambda p, n: run_cell_centred(
            lambda m, p, dt: PolygonalHeatSolver(*m, p["alpha"], dt, **common(p)), p, n),
        "Polygonal TPFA+corr (skewed)": lambda p, n: run_cell_centred(
            lambda m, p, dt: PolygonalHeatSolver(*m, p["alpha"], dt, nonorthogonal_correction=True, **common(p)),
            p, n, skew=0.3),
        "Polygonal quadratic (skewed)": lambda p, n: run_cell_centred(
            lambda m, p, dt: PolygonalHeatSolver(*m, p["alpha"], dt, flux_discretization="quadratic", **common(p)),
            p, n, skew=0.3, averages=True),
        "AFC global limiter (square)": lambda p, n: run_cell_centred(
            lambda m, p, dt: AFCMonotoneSolver(*m, p["alpha"], dt, limiter="global", flux_discretization="tpfa",
                                               **common(p)), p, n),
        "Enthalpy, L=0 (square)": lambda p, n: run_cell_centred(
            lambda m, p, dt: EnthalpyStefanSolver(*m, p["alpha"], dt, lin, **common(p)), p, n),
        "Apparent-capacity Picard, L=0": lambda p, n: run_cell_centred(
            lambda m, p, dt: ApparentCapacityPicardSolver(*m, p["alpha"], dt, lin, **common(p)), p, n),
        "Triangular (Delaunay)": run_triangular,
        "Curvilinear grid": run_curvilinear,
    }


def transport_solvers():
    def mk(cls, extra):
        return lambda p, n: run_cell_centred(
            lambda m, p, dt: cls(*m, p["alpha"], dt, bc_type=p["bc_type"], bc_func=p["bc_func"],
                                 source_func=p["source"], **extra(p)), p, n)
    return {
        "Cattaneo (hyperbolic)": ("cattaneo", mk(HyperbolicHeatSolver, lambda p: {"relaxation_time": p["tau"]})),
        "Caputo fractional": ("fractional", mk(FractionalHeatSolver, lambda p: {"beta": p["beta"]})),
        "Advection-diffusion": ("advection", mk(AdvectionDiffusionHeatSolver,
                                                lambda p: {"velocity": p["velocity"], "scheme": "central"})),
        "Pennes reaction-diffusion": ("pennes", mk(ReactionDiffusionHeatSolver,
                                                   lambda p: {"reaction_rate": p["perfusion"]})),
    }


def main():
    t0 = time.perf_counter()
    results = {}
    rows = []
    for name, run in heat_solvers().items():
        for bc in BC_TYPES:
            prob = heat_problem(bc, manufactured_boundary="skewed" in name)
            rows.append((name, bc, prob, run))
    for name, (model, run) in transport_solvers().items():
        for bc in BC_TYPES:
            rows.append((name, bc, transport_problem(model, bc), run))
    for name, bc, prob, run in rows:
        try:
            errs = [run(prob, n) for n in LEVELS]
            orders = [float(np.log2(errs[k] / errs[k + 1])) for k in range(len(errs) - 1)]
            exact = max(errs) < 1e-11  # solution reproduced to round-off (e.g. quadratic in space)
            results.setdefault(name, {})[bc] = {"errors": errs, "orders": orders,
                                                "status": "exact" if exact else "ok"}
            print(f"{name:32s} {bc:10s} err {errs[-1]:.2e} orders {np.round(orders, 2)} "
                  f"[{time.perf_counter() - t0:.0f}s]")
        except Exception as exc:  # noqa: BLE001 - record and continue
            results.setdefault(name, {})[bc] = {"status": f"error: {exc}"}
            print(f"{name:32s} {bc:10s} ERROR {exc}")

    names = list(results)
    order = np.full((len(names), len(BC_TYPES)), np.nan)
    err = np.full_like(order, np.nan)
    for i, nm in enumerate(names):
        for j, bc in enumerate(BC_TYPES):
            r = results[nm][bc]
            if r["status"] == "ok":
                order[i, j] = r["orders"][-1]
                err[i, j] = r["errors"][-1]
            elif r["status"] == "exact":
                order[i, j] = np.inf
                err[i, j] = max(r["errors"][-1], 1e-16)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6.2))
    im = ax[0].imshow(np.where(np.isinf(order), 2.2, order), cmap="RdYlGn", vmin=0.0, vmax=2.2, aspect="auto")
    for (i, j), v in np.ndenumerate(order):
        label = "fail" if np.isnan(v) else ("exact" if np.isinf(v) else f"{v:.2f}")
        ax[0].text(j, i, label, ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax[0], label="observed order (finest pair)")
    ax[0].set_title("Observed order of the relative $L^2$ error")
    im = ax[1].imshow(np.log10(err), cmap="viridis_r", aspect="auto")
    for (i, j), v in np.ndenumerate(err):
        ax[1].text(j, i, "fail" if np.isnan(v) else f"{v:.1e}", ha="center", va="center", fontsize=7, color="w")
    fig.colorbar(im, ax=ax[1], label=r"$\log_{10}$ relative $L^2$ error, finest mesh")
    ax[1].set_title(f"Error on the finest mesh (n={LEVELS[-1]})")
    for a in ax:
        a.set_xticks(range(len(BC_TYPES)))
        a.set_xticklabels(BC_TYPES)
        a.set_yticks(range(len(names)))
        a.set_yticklabels(names if a is ax[0] else [""] * len(names))
    fig.suptitle("Every solver x every boundary type (standard test problems; transport models manufactured)")
    fig.tight_layout()
    fig.savefig(OUT / "capability_matrix.png", dpi=160)
    plt.close(fig)
    (OUT / "summary.json").write_text(json.dumps({"levels": LEVELS, "t_end": T_END, "results": results},
                                                 indent=2))
    print("done", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
