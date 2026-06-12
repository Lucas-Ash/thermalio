#!/usr/bin/env python3
"""Use-case examples for the extended thermal-transport models.

This script demonstrates the three transport models added on top of the
classical Fourier polygonal solver and verifies each against a manufactured
solution:

  1. ``HyperbolicHeatSolver``       -- non-Fourier Cattaneo thermal waves.
  2. ``AdvectionDiffusionHeatSolver`` -- convective transport (upwind vs central).
  3. ``FractionalHeatSolver``       -- Caputo time-fractional subdiffusion.

Usage:
    MPLCONFIGDIR=/tmp/matplotlib python transport_demo.py

Writes ``test_plots/transport_models.png`` and prints convergence tables.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heat_solver.cases import (
    advection_diffusion_case,
    cattaneo_wave_case,
    fractional_subdiffusion_case,
)
from heat_solver.geometry import polygon_area_and_centroid
from heat_solver.meshes import generate_square_polygonal_mesh
from heat_solver.transport import (
    AdvectionDiffusionHeatSolver,
    FractionalHeatSolver,
    HyperbolicHeatSolver,
)


def _mesh(n, bbox):
    vertices, polygons, centers = generate_square_polygonal_mesh(nx=n, ny=n, bbox=bbox)
    areas = np.array([polygon_area_and_centroid(vertices[poly])[0] for poly in polygons])
    return vertices, polygons, centers, areas


def _rel_l2(u, u_exact, areas):
    return float(np.sqrt(np.sum(areas * (u - u_exact) ** 2)) / np.sqrt(np.sum(areas * u_exact**2)))


def hyperbolic_example():
    print("\n[1] Non-Fourier Cattaneo thermal wave:  tau u_tt + u_t - alpha lap(u) = Q")
    case = cattaneo_wave_case(alpha=0.1, tau=0.2)
    bbox = case["bbox"]
    t_end = 0.3
    print(f"    finite wave speed c = sqrt(alpha/tau) = {np.sqrt(case['alpha']/case['relaxation_time']):.4f}")
    print(f"    {'cells/side':>10} {'dt':>9} {'rel L2':>11}")
    for n in (16, 32, 64):
        vertices, polygons, centers, areas = _mesh(n, bbox)
        dt = t_end / (4 * n)
        solver = HyperbolicHeatSolver(
            vertices, polygons, case["alpha"], dt, case["relaxation_time"],
            bc_func=case["boundary"], source_func=case["source"],
        )
        u0 = case["solution"](centers[:, 0], centers[:, 1], 0.0)
        du0 = case["initial_rate"](centers[:, 0], centers[:, 1])
        t, u = solver.solve(u0, 0.0, t_end, du0=du0)
        u_exact = case["solution"](centers[:, 0], centers[:, 1], t)
        print(f"    {n:>10} {dt:>9.4f} {_rel_l2(u, u_exact, areas):>11.3e}")
    return centers, u, u_exact, n


def advection_example():
    print("\n[2] Convective transport:  u_t + v.grad(u) - alpha lap(u) = Q,  v = (0.8, 0.4)")
    case = advection_diffusion_case(alpha=0.05, velocity=(0.8, 0.4))
    bbox = case["bbox"]
    t_end = 0.3
    results = {}
    for scheme in ("upwind", "central"):
        print(f"    scheme = {scheme}")
        print(f"    {'cells/side':>10} {'cell Pe':>9} {'rel L2':>11}")
        for n in (16, 32, 64):
            vertices, polygons, centers, areas = _mesh(n, bbox)
            dt = t_end / 200
            solver = AdvectionDiffusionHeatSolver(
                vertices, polygons, case["alpha"], dt, case["velocity"],
                scheme=scheme, bc_func=case["boundary"], source_func=case["source"],
            )
            u0 = case["solution"](centers[:, 0], centers[:, 1], 0.0)
            t, u = solver.solve(u0, 0.0, t_end)
            u_exact = case["solution"](centers[:, 0], centers[:, 1], t)
            pe = solver.peclet_number(2.0 / n)
            print(f"    {n:>10} {pe:>9.3f} {_rel_l2(u, u_exact, areas):>11.3e}")
            results[scheme] = (centers, u, u_exact)
    return results


def fractional_example():
    print("\n[3] Anomalous subdiffusion (Caputo):  D_t^beta u - alpha lap(u) = Q")
    bbox = (0.0, 1.0, 0.0, 1.0)
    t_end = 0.5
    curves = {}
    for beta in (0.4, 0.6, 0.8):
        case = fractional_subdiffusion_case(alpha=0.1, beta=beta)
        print(f"    beta = {beta} (expected temporal order ~ {2 - beta:.1f})")
        print(f"    {'steps':>6} {'rel L2':>11} {'order':>7}")
        prev = None
        for nt in (15, 30, 60):
            n = 96
            vertices, polygons, centers, areas = _mesh(n, bbox)
            dt = t_end / nt
            solver = FractionalHeatSolver(
                vertices, polygons, case["alpha"], dt, beta,
                bc_func=case["boundary"], source_func=case["source"],
            )
            u0 = case["solution"](centers[:, 0], centers[:, 1], 0.0)
            t, u = solver.solve(u0, 0.0, t_end)
            u_exact = case["solution"](centers[:, 0], centers[:, 1], t)
            err = _rel_l2(u, u_exact, areas)
            order = "" if prev is None else f"{np.log2(prev / err):.2f}"
            print(f"    {nt:>6} {err:>11.3e} {order:>7}")
            prev = err
        curves[beta] = (centers, u)
    return curves


def _tri_field(ax, centers, values, title):
    tcf = ax.tripcolor(centers[:, 0], centers[:, 1], values, shading="gouraud")
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return tcf


def main():
    centers_h, u_h, ue_h, n_h = hyperbolic_example()
    adv = advection_example()
    frac = fractional_example()

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    _tri_field(axes[0, 0], centers_h, u_h, "Cattaneo wave: numerical")
    _tri_field(axes[0, 1], centers_h, ue_h, "Cattaneo wave: exact")
    _tri_field(axes[0, 2], centers_h, np.abs(u_h - ue_h), "Cattaneo wave: |error|")

    c_up, u_up, ue_up = adv["upwind"]
    c_ce, u_ce, ue_ce = adv["central"]
    _tri_field(axes[1, 0], c_up, u_up, "Advection-diffusion (upwind)")
    _tri_field(axes[1, 1], c_ce, u_ce, "Advection-diffusion (central)")
    cen_f, u_f = frac[0.6]
    _tri_field(axes[1, 2], cen_f, u_f, "Subdiffusion beta=0.6")

    fig.suptitle("Extended thermal-transport models", fontsize=14)
    out_dir = ROOT / "test_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "transport_models.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
