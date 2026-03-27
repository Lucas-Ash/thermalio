#!/usr/bin/env python3
"""
Compare backward Euler vs Crank–Nicolson on the polygonal FV heat solver.

The semi-discrete problem is ``M u' + A u = M Q`` (plus boundary terms for Neumann/Robin).
Crank–Nicolson is second-order accurate in **time** for that ODE; backward Euler is first-order.

Important: A plot of **total** error vs the **analytical PDE** solution mixes spatial truncation
with temporal error. On a modest mesh, spatial error (~1e-4) often **dominates**, so the CN curve
looks flat while BE still decreases — that is not a bug in CN, it is the wrong diagnostic.

This script produces **two** panels:
  1) Total relative L2 error vs the analytical solution (spatial + temporal).
  2) **Temporal** error only: relative L2 norm of ``u(dt) - u_ref`` on the same mesh, where
     ``u_ref`` is a Crank–Nicolson solution with a much smaller ``dt_ref`` (converged in time
     for the semi-discrete IVP). On panel 2, slope-1 (BE) and slope-2 (CN) references are meaningful.

Usage:
    MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python time_scheme_demo.py

Writes ``test_plots/time_scheme_comparison.png``.
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
    num = np.sqrt(np.sum(weights * diff**2))
    den = np.sqrt(np.sum(weights * u_ref**2)) + 1e-16
    return num / den


def main():
    alpha = 0.1
    t_init = 0.0
    t_end = 0.05
    case = "source_driven_sine"
    bbox = (0.0, 1.0, 0.0, 1.0)
    nx = ny = 20

    dts = np.array([0.025, 0.0125, 0.00625, 0.003125, 0.0015625])
    # Fine CN solution for the same semi-discrete IVP (manageable step count).
    dt_ref = 1e-5
    assert dt_ref < dts.min(), "dt_ref must be smaller than every dt in dts"

    common = dict(
        case=case,
        alpha=alpha,
        t_init=t_init,
        t_end=t_end,
        nx=nx,
        ny=ny,
        bbox=bbox,
        nonorthogonal_correction=True,
    )

    print("Computing reference solution (Crank–Nicolson, very small dt)...")
    verts, polys, _centers, u_ref, _, _, _ = run_square_polygonal_test(
        dt=float(dt_ref),
        time_scheme="crank_nicolson",
        **common,
    )
    weights = _cell_areas(verts, polys)

    err_be_total = []
    err_cn_total = []
    err_be_temp = []
    err_cn_temp = []

    for dt in dts:
        *_, u_be, _, _, res_be = run_square_polygonal_test(
            dt=float(dt), time_scheme="backward_euler", **common
        )
        *_, u_cn, _, _, res_cn = run_square_polygonal_test(
            dt=float(dt), time_scheme="crank_nicolson", **common
        )
        err_be_total.append(res_be["L2_rel"])
        err_cn_total.append(res_cn["L2_rel"])
        err_be_temp.append(_rel_l2_diff(u_be, u_ref, weights))
        err_cn_temp.append(_rel_l2_diff(u_cn, u_ref, weights))

    err_be_total = np.asarray(err_be_total)
    err_cn_total = np.asarray(err_cn_total)
    err_be_temp = np.asarray(err_be_temp)
    err_cn_temp = np.asarray(err_cn_temp)

    print(f"\nCase: {case}, mesh: {nx}x{ny}, t in [{t_init}, {t_end}], alpha={alpha}")
    print(f"Reference: CN with dt_ref={dt_ref:g} ({int(np.ceil((t_end - t_init) / dt_ref))} steps)\n")
    print(f"{'dt':>12}  {'L2_rel BE':>14}  {'L2_rel CN':>14}  |  {'temp BE':>14}  {'temp CN':>14}")
    for i, dt in enumerate(dts):
        print(
            f"{dt:12.6f}  {err_be_total[i]:14.6e}  {err_cn_total[i]:14.6e}  |  "
            f"{err_be_temp[i]:14.6e}  {err_cn_temp[i]:14.6e}"
        )

    # Log-log slope estimates (temporal panel, skip last point if noisy)
    log_dt = np.log(dts)
    for name, y in [("BE temporal", err_be_temp), ("CN temporal", err_cn_temp)]:
        slope, intercept = np.polyfit(log_dt, np.log(y), 1)
        print(f"  Fitted log-log slope ({name}): {slope:.3f} (expect ~1 for BE, ~2 for CN)")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    ax0.loglog(dts, err_be_total, "o-", label="Backward Euler", lw=1.5)
    ax0.loglog(dts, err_cn_total, "s-", label="Crank–Nicolson", lw=1.5)
    ax0.set_xlabel(r"$\Delta t$")
    ax0.set_ylabel(r"relative $L_2$ vs analytical (total error)")
    ax0.set_title("Total error (spatial + temporal)\nCN can plateau when spatial error dominates")
    ax0.legend(loc="best")
    ax0.grid(True, which="both", alpha=0.35)

    ax1.loglog(dts, err_be_temp, "o-", label="Backward Euler", lw=1.5)
    ax1.loglog(dts, err_cn_temp, "s-", label="Crank–Nicolson", lw=1.5)
    ax1.loglog(dts, err_be_temp[-1] * (dts / dts[-1]) ** 1.0, "k:", alpha=0.45, label="slope 1")
    ax1.loglog(dts, err_cn_temp[-1] * (dts / dts[-1]) ** 2.0, "k--", alpha=0.45, label="slope 2")
    ax1.set_xlabel(r"$\Delta t$")
    ax1.set_ylabel(r"$\|u(\Delta t) - u_{\mathrm{ref}}\|_{L^2} / \|u_{\mathrm{ref}}\|_{L^2}$")
    ax1.set_title(
        f"Temporal error only (ref = CN, $\\Delta t$={dt_ref:g})\nSlopes should match BE≈1, CN≈2"
    )
    ax1.legend(loc="best")
    ax1.grid(True, which="both", alpha=0.35)

    fig.suptitle("Polygonal heat solver: time integration comparison", fontsize=13, y=1.02)
    out_dir = ROOT / "test_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "time_scheme_comparison.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
