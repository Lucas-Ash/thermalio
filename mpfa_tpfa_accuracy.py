#!/usr/bin/env python3
"""
Plot relative L2 error vs analytical solution for TPFA vs MPFA on the same skewed polygonal meshes.

Uses ``sine_mode`` (u = 0 on ∂Ω) so MPFA’s homogeneous-vertex condensation is consistent with
Dirichlet cell values. Several mesh resolutions (nx = ny) isolate spatial accuracy; keep ``dt``
small so time error does not dominate.

Run:
    MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python mpfa_tpfa_accuracy.py

Output: ``test_plots/mpfa_tpfa_accuracy.png``
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

from heat_solver.drivers import run_nonorthogonal_polygonal_test


def main():
    nx_list = np.array([6, 8, 10, 12, 14, 16, 20])
    common = dict(
        case="sine_mode",
        alpha=0.1,
        dt=2e-3,
        t_init=0.0,
        t_end=0.02,
        bbox=(-1.0, 1.0, -1.0, 1.0),
        skew=0.35,
        nonorthogonal_correction=True,
    )

    err_tp = []
    err_mp = []
    for nx in nx_list:
        *_, res_tp = run_nonorthogonal_polygonal_test(flux_scheme="tpfa", nx=int(nx), ny=int(nx), **common)
        *_, res_mp = run_nonorthogonal_polygonal_test(flux_scheme="mpfa", nx=int(nx), ny=int(nx), **common)
        err_tp.append(res_tp["L2_rel"])
        err_mp.append(res_mp["L2_rel"])

    err_tp = np.asarray(err_tp)
    err_mp = np.asarray(err_mp)
    ratio_mp_over_tp = err_mp / (err_tp + 1e-20)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.8))

    ax0.semilogy(nx_list, err_tp, "o-", lw=2, ms=7, label="TPFA (two-point + correction)")
    ax0.semilogy(nx_list, err_mp, "s-", lw=2, ms=7, label="MPFA-O (subcell FEM / Schur)")
    ax0.set_xlabel(r"Mesh resolution $n_x = n_y$ (skewed polygonal mesh)")
    ax0.set_ylabel(r"Relative $L_2$ error vs analytical (sine mode)")
    ax0.set_title("Accuracy: lower curve is more accurate")
    ax0.legend(loc="best")
    ax0.grid(True, which="both", alpha=0.35)

    ax1.axhline(1.0, color="k", ls="--", lw=1, alpha=0.5, label="equal error")
    ax1.plot(nx_list, ratio_mp_over_tp, "D-", color="C2", lw=2, ms=6, label=r"$E_{\mathrm{MPFA}} / E_{\mathrm{TPFA}}$")
    ax1.set_xlabel(r"$n_x = n_y$")
    ax1.set_ylabel("Error ratio")
    ax1.set_title("MPFA error relative to TPFA\n(< 1 means MPFA more accurate)")
    ax1.legend(loc="best")
    ax1.grid(True, which="both", alpha=0.35)
    ax1.set_ylim(bottom=0)

    fig.suptitle(
        f"Sine mode, skew={common['skew']}, $t\\in[{common['t_init']}, {common['t_end']}]$, "
        rf"$\Delta t={common['dt']}$",
        fontsize=12,
        y=1.02,
    )
    out = ROOT / "test_plots" / "mpfa_tpfa_accuracy.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Relative L2 error vs analytical (sine mode, skewed mesh):")
    print(f"{'nx':>6}  {'TPFA L2_rel':>14}  {'MPFA L2_rel':>14}  {'MPFA/TPFA':>12}")
    for i, nx in enumerate(nx_list):
        print(f"{int(nx):6d}  {err_tp[i]:14.6e}  {err_mp[i]:14.6e}  {ratio_mp_over_tp[i]:12.4f}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
