#!/usr/bin/env python3
"""Demonstrative graphs for the monotonicity / AFC work (direction B).

Produces ``test_plots/dmp/monotonicity_showcase.png``:

* top row -- the diffused field of a bounded indicator block on a skewed,
  strongly anisotropic mesh under three schemes (high-order reconstructed,
  linear M-matrix projection, and nonlinear AFC), with cells that violate the
  bound [0, 1] marked in red;
* bottom row -- the max bound excursion per scheme, the smooth-solution accuracy
  per scheme, and the accuracy<->monotonicity trade-off scatter showing AFC in
  the best corner (bound-preserving AND accurate).

Usage:
    MPLCONFIGDIR=/tmp/matplotlib python dmp_afc_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heat_solver.afc import AFCMonotoneSolver
from heat_solver.dmp import anisotropic_tensor, bound_excursion
from heat_solver.geometry import polygon_area_and_centroid
from heat_solver.meshes import (
    generate_nonorthogonal_polygonal_mesh,
    generate_nonorthogonal_tiled_polygonal_mesh,
)
from heat_solver.polygonal import PolygonalHeatSolver

OUTPUT_DIR = ROOT / "test_plots" / "dmp"

SCHEME_LABELS = {
    "reconstructed": "High-order (reconstructed)",
    "monotone": "Linear monotone (M-matrix)",
    "afc": "Nonlinear AFC (this work)",
}


def _block_ic(centers, blk=(-0.35, 0.35, -0.35, 0.35)):
    x, y = centers[:, 0], centers[:, 1]
    return ((x >= blk[0]) & (x <= blk[1]) & (y >= blk[2]) & (y <= blk[3])).astype(float)


def _solve_block(kind, vertices, polygons, centers, alpha, dt=1.5e-3, t_end=9e-3):
    u0 = _block_ic(centers)
    zero = lambda x, y, t: np.zeros_like(np.asarray(x, dtype=float))
    if kind == "afc":
        s = AFCMonotoneSolver(vertices, polygons, alpha, dt, bc_func=zero,
                              source_func=zero, flux_discretization="reconstructed")
        _, u = s.solve(u0, 0.0, t_end)
    else:
        kw = {
            "reconstructed": {"flux_discretization": "reconstructed", "nonorthogonal_correction": True},
            "monotone": {"flux_discretization": "reconstructed", "nonorthogonal_correction": True, "monotone": True},
        }[kind]
        s = PolygonalHeatSolver(vertices, polygons, alpha, dt, bc_type="dirichlet",
                                bc_func=zero, source_func=zero, **kw)
        _, u = s.solve(u0, 0.0, t_end)
    return u


def _smooth_error(kind, alpha, n=24, skew=0.3, dt=2e-5, t_end=0.004):
    from heat_solver.mms import manufactured_case

    case = manufactured_case("exp(-t)*sin(pi*x)*sin(pi*y)", alpha=alpha.tolist(), model="diffusion")
    v, p, c = generate_nonorthogonal_polygonal_mesh(nx=n, ny=n, bbox=(-1, 1, -1, 1), skew=skew)
    areas = np.array([polygon_area_and_centroid(v[poly])[0] for poly in p])
    u0 = case["solution"](c[:, 0], c[:, 1], 0.0)
    ue = case["solution"](c[:, 0], c[:, 1], t_end)
    if kind == "afc":
        s = AFCMonotoneSolver(v, p, alpha, dt, bc_func=case["boundary"],
                              source_func=case["source"], flux_discretization="reconstructed")
        _, u = s.solve(u0, 0.0, t_end)
    else:
        kw = {
            "reconstructed": {"flux_discretization": "reconstructed", "nonorthogonal_correction": True},
            "monotone": {"flux_discretization": "reconstructed", "nonorthogonal_correction": True, "monotone": True},
        }[kind]
        s = PolygonalHeatSolver(v, p, alpha, dt, bc_type="dirichlet", bc_func=case["boundary"],
                                source_func=case["source"], **kw)
        _, u = s.solve(u0, 0.0, t_end)
    return float(np.sqrt(np.sum(areas * (u - ue) ** 2) / np.sum(areas * ue**2)))


def _field_panel(ax, vertices, polygons, centers, u, title):
    polys = [vertices[poly] for poly in polygons]
    coll = PolyCollection(polys, array=u, cmap="viridis", edgecolors="none")
    coll.set_clim(-0.08, 1.08)
    ax.add_collection(coll)
    out = (u < -1e-9) | (u > 1.0 + 1e-9)
    if np.any(out):
        ax.scatter(centers[out, 0], centers[out, 1], s=6, c="red", label="DMP violation")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.8)
    be = bound_excursion(u, 0.0, 1.0)
    excursion = max(be["overshoot"], be["undershoot"])
    ax.set_title(f"{title}\nmax excursion = {excursion:.2e}", fontsize=9)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return coll


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    alpha = anisotropic_tensor(ratio=20.0, angle=np.pi / 6)
    v, p, c = generate_nonorthogonal_tiled_polygonal_mesh(nx_tiles=10, ny_tiles=10, bbox=(-1, 1, -1, 1), skew=0.4)

    fields, excursions = {}, {}
    for kind in ("reconstructed", "monotone", "afc"):
        u = _solve_block(kind, v, p, c, alpha)
        fields[kind] = u
        be = bound_excursion(u, 0.0, 1.0)
        excursions[kind] = max(be["overshoot"], be["undershoot"])

    # Smooth-solution accuracy (milder anisotropy where high-order is well-behaved).
    alpha_mild = anisotropic_tensor(ratio=5.0, angle=np.pi / 6)
    errors = {kind: _smooth_error(kind, alpha_mild) for kind in ("reconstructed", "monotone", "afc")}

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    coll = None
    for ax, kind in zip(axes[0], ("reconstructed", "monotone", "afc")):
        coll = _field_panel(ax, v, p, c, fields[kind], SCHEME_LABELS[kind])
    fig.colorbar(coll, ax=axes[0].tolist(), shrink=0.8, label="temperature (physical bounds [0, 1])")

    order = ["reconstructed", "monotone", "afc"]
    colors = ["#d62728", "#1f77b4", "#2ca02c"]
    labels = ["high-order", "linear monotone", "AFC (this work)"]

    ax = axes[1, 0]
    ax.bar(labels, [excursions[k] for k in order], color=colors)
    ax.set_ylabel("max bound excursion")
    ax.set_title("Monotonicity: DMP violation\n(anisotropic block, skewed mesh)", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)

    ax = axes[1, 1]
    ax.bar(labels, [errors[k] for k in order], color=colors)
    ax.set_ylabel("relative L2 error")
    ax.set_title("Accuracy: smooth anisotropic\nsolution (skewed mesh)", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)

    ax = axes[1, 2]
    for k, col, lab in zip(order, colors, labels):
        ax.scatter(max(excursions[k], 1e-6), errors[k], s=90, c=col, label=lab, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("max bound excursion (log)  ->  worse")
    ax.set_ylabel("relative L2 error  ->  worse")
    ax.set_title("Accuracy <-> monotonicity trade-off\n(best = bottom-left)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.annotate("AFC: bounded AND accurate", xy=(1e-6, errors["afc"]),
                xytext=(3e-6, errors["afc"] * 1.8), fontsize=7,
                arrowprops=dict(arrowstyle="->", color="#2ca02c"))

    fig.suptitle("Bound-preserving high-resolution diffusion: linear monotone vs nonlinear AFC", fontsize=13)
    out_path = OUTPUT_DIR / "monotonicity_showcase.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print("Block-test max bound excursion:")
    for k in order:
        print(f"  {SCHEME_LABELS[k]:32s} {excursions[k]:.3e}")
    print("Smooth-solution relative L2 error:")
    for k in order:
        print(f"  {SCHEME_LABELS[k]:32s} {errors[k]:.3e}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
