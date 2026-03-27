#!/usr/bin/env python3
"""
Compare backward Euler vs fully nonlinear Crank–Nicolson for phase-change runs.

Writes: test_plots/phase_change_time_scheme_comparison.png
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

from heat_solver.drivers import run_square_polygonal_test
from heat_solver.geometry import polygon_area_and_centroid


def _cell_areas(vertices, polygons):
    return np.array([polygon_area_and_centroid(vertices[poly])[0] for poly in polygons], dtype=float)


def _rel_l2_diff(u, u_ref, weights):
    diff = u - u_ref
    return np.sqrt(np.sum(weights * diff**2)) / (np.sqrt(np.sum(weights * u_ref**2)) + 1e-16)


def main():
    case = "stefan_apparent_capacity"
    alpha = 0.08
    t_init = 0.0
    t_end = 0.02
    bbox = (-1.0, 1.0, -1.0, 1.0)
    nx = ny = 12
    dts = np.array([0.01, 0.005, 0.0025, 0.00125], dtype=float)
    dt_ref = 3.125e-4
    phase_opts = {"max_iters": 100, "tol": 1e-10, "relaxation": 0.7}

    common = dict(
        case=case,
        alpha=alpha,
        t_init=t_init,
        t_end=t_end,
        nx=nx,
        ny=ny,
        bbox=bbox,
        nonorthogonal_correction=True,
        phase_change_options=phase_opts,
    )

    print("Computing reference solution (phase-change CN, very small dt)...")
    verts, polys, _centers, u_ref, _, _, _ = run_square_polygonal_test(
        dt=float(dt_ref),
        time_scheme="crank_nicolson",
        **common,
    )
    weights = _cell_areas(verts, polys)

    err_be = []
    err_cn = []
    for dt in dts:
        *_, u_be, _, _, _ = run_square_polygonal_test(
            dt=float(dt),
            time_scheme="backward_euler",
            **common,
        )
        *_, u_cn, _, _, _ = run_square_polygonal_test(
            dt=float(dt),
            time_scheme="crank_nicolson",
            **common,
        )
        err_be.append(_rel_l2_diff(u_be, u_ref, weights))
        err_cn.append(_rel_l2_diff(u_cn, u_ref, weights))

    err_be = np.asarray(err_be)
    err_cn = np.asarray(err_cn)
    slope_be = np.polyfit(np.log(dts), np.log(err_be), 1)[0]
    slope_cn = np.polyfit(np.log(dts), np.log(err_cn), 1)[0]

    print(f"Fitted BE slope: {slope_be:.3f}")
    print(f"Fitted CN slope: {slope_cn:.3f}")
    print(f"{'dt':>12} {'BE temporal':>14} {'CN temporal':>14}")
    for i, dt in enumerate(dts):
        print(f"{dt:12.6f} {err_be[i]:14.6e} {err_cn[i]:14.6e}")

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 5.0))
    ax.loglog(dts, err_be, "o-", lw=1.5, label=f"Backward Euler (slope~{slope_be:.2f})")
    ax.loglog(dts, err_cn, "s-", lw=1.5, label=f"Crank–Nicolson nonlinear (slope~{slope_cn:.2f})")
    ax.loglog(dts, err_be[-1] * (dts / dts[-1]) ** 1.0, "k:", alpha=0.45, label="slope 1")
    ax.loglog(dts, err_cn[-1] * (dts / dts[-1]) ** 2.0, "k--", alpha=0.45, label="slope 2")
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"$\|u(\Delta t)-u_\mathrm{ref}\|_{L^2}/\|u_\mathrm{ref}\|_{L^2}$")
    ax.set_title("Stefan phase-change temporal scaling\n(square polygonal mesh)")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(loc="best")

    out_dir = ROOT / "test_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase_change_time_scheme_comparison.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
