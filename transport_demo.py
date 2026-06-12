#!/usr/bin/env python3
"""Use-case examples for the extended thermal-transport models.

This script demonstrates the three transport models added on top of the
classical Fourier polygonal solver and verifies each against a manufactured
solution:

  1. ``HyperbolicHeatSolver``       -- non-Fourier Cattaneo thermal waves.
  2. ``AdvectionDiffusionHeatSolver`` -- convective transport (upwind vs central).
  3. ``FractionalHeatSolver``       -- Caputo time-fractional subdiffusion.
  4. Boundary-driven Cattaneo waves: a prescribed heat-flux pulse
     (``bc_type='flux'``, e.g. pulsed-laser heating) launches a thermal wave
     with finite speed sqrt(alpha/tau), compared against the parabolic
     Fourier response on the same mesh.  Robin/Neumann data are also
     supported by all three transport solvers.

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
from heat_solver.polygonal import PolygonalHeatSolver
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


def flux_pulse_example():
    """Boundary heat-flux pulse: finite-speed Cattaneo wave vs Fourier diffusion."""
    print("\n[4] Flux-driven Cattaneo pulse (bc_type='flux') vs Fourier on the same strip")
    alpha, tau = 0.05, 1.0
    c = np.sqrt(alpha / tau)
    q0, t_pulse, t_end = 1.0, 0.5, 4.0
    bbox = (0.0, 2.0, 0.0, 0.25)
    vertices, polygons, centers = generate_square_polygonal_mesh(nx=160, ny=20, bbox=bbox)
    dt = 0.01
    print(f"    wave speed c = sqrt(alpha/tau) = {c:.4f};  expected front at x = c*t = {c * t_end:.3f}")

    def pulse_flux(x, y, t, nx, ny):
        active = q0 if t <= t_pulse else 0.0
        return np.where(np.isclose(x, 0.0), active, 0.0)

    def pulse_neumann(x, y, t, nx, ny):
        return pulse_flux(x, y, t, nx, ny) / alpha

    hyperbolic = HyperbolicHeatSolver(
        vertices, polygons, alpha, dt, tau, bc_type="flux", bc_func=pulse_flux,
    )
    snapshots = {}
    for t_snap in (1.0, 2.0, t_end):
        _, u = hyperbolic.solve(np.zeros(hyperbolic.M), 0.0, t_snap)
        snapshots[t_snap] = u
    u_hyp = snapshots[t_end]

    fourier = PolygonalHeatSolver(
        vertices, polygons, alpha, dt, bc_type="neumann", bc_func=pulse_neumann,
    )
    _, u_fourier = fourier.solve(np.zeros(hyperbolic.M), 0.0, t_end)

    areas = hyperbolic.cell_areas
    injected = q0 * (bbox[3] - bbox[2]) * t_pulse
    print(f"    energy check (hyperbolic): sum(area*u) = {np.sum(areas * u_hyp):.6f}, injected = {injected:.6f}")
    print(f"    energy check (Fourier):    sum(area*u) = {np.sum(areas * u_fourier):.6f}")

    # Extract the centerline row of cells for profile plots.
    y_mid = 0.5 * (bbox[2] + bbox[3])
    h_y = (bbox[3] - bbox[2]) / 20
    row = np.abs(centers[:, 1] - y_mid) < 0.5 * h_y
    order = np.argsort(centers[row, 0])
    x_line = centers[row, 0][order]
    profile = {t_snap: u[row][order] for t_snap, u in snapshots.items()}
    profile_fourier = u_fourier[row][order]

    ahead = x_line > c * t_end + 0.3
    print(f"    max u ahead of front (hyperbolic): {np.max(np.abs(u_hyp[row][order][ahead])):.2e}")
    print(f"    max u ahead of front (Fourier):    {np.max(profile_fourier[ahead]):.2e}  (parabolic: no front)")
    return {
        "x": x_line,
        "profiles": profile,
        "fourier": profile_fourier,
        "c": c,
        "t_end": t_end,
        "centers": centers,
        "u_field": u_hyp,
    }


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
    pulse = flux_pulse_example()

    fig, axes = plt.subplots(3, 3, figsize=(13, 12))

    _tri_field(axes[0, 0], centers_h, u_h, "Cattaneo wave: numerical")
    _tri_field(axes[0, 1], centers_h, ue_h, "Cattaneo wave: exact")
    _tri_field(axes[0, 2], centers_h, np.abs(u_h - ue_h), "Cattaneo wave: |error|")

    c_up, u_up, ue_up = adv["upwind"]
    c_ce, u_ce, ue_ce = adv["central"]
    _tri_field(axes[1, 0], c_up, u_up, "Advection-diffusion (upwind)")
    _tri_field(axes[1, 1], c_ce, u_ce, "Advection-diffusion (central)")
    cen_f, u_f = frac[0.6]
    _tri_field(axes[1, 2], cen_f, u_f, "Subdiffusion beta=0.6")

    ax = axes[2, 0]
    for t_snap, u_line in sorted(pulse["profiles"].items()):
        ax.plot(pulse["x"], u_line, label=f"t = {t_snap:g}")
        ax.axvline(pulse["c"] * t_snap, color="gray", lw=0.8, ls=":")
    ax.set_title("Flux-pulse Cattaneo wave: centerline u(x, t)\n(dotted: wavefront x = ct)", fontsize=10)
    ax.set_xlabel("x")
    ax.legend(fontsize=8)

    ax = axes[2, 1]
    ax.plot(pulse["x"], pulse["profiles"][pulse["t_end"]], label="Cattaneo (flux BC)")
    ax.plot(pulse["x"], pulse["fourier"], "--", label="Fourier (Neumann BC)")
    ax.axvline(pulse["c"] * pulse["t_end"], color="gray", lw=0.8, ls=":")
    ax.set_title(f"Finite vs infinite propagation speed (t = {pulse['t_end']:g})", fontsize=10)
    ax.set_xlabel("x")
    ax.legend(fontsize=8)

    _tri_field(axes[2, 2], pulse["centers"], pulse["u_field"], "Flux-pulse Cattaneo field (strip)")
    axes[2, 2].set_aspect("auto")

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
