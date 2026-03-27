#!/usr/bin/env python3
"""
Demonstrate MPFA-O (subcell FEM / condensation) vs TPFA two-point flux on a skewed polygonal mesh.

Uses ``sine_mode`` (u = 0 on the boundary) so the MPFA implementation (homogeneous vertex values)
is consistent with Dirichlet cell values.

Run:
    MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python mpfa_tpfa_demo.py

Output: ``test_plots/mpfa_vs_tpfa.png``
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
from heat_solver.plotting import create_polygonal_figure


def main():
    common = dict(
        case="sine_mode",
        alpha=0.1,
        dt=5e-3,
        t_init=0.0,
        t_end=0.03,
        nx=16,
        ny=16,
        bbox=(-1.0, 1.0, -1.0, 1.0),
        skew=0.35,
        nonorthogonal_correction=True,
    )

    v, p, c, u_tp, ue, d_tp, r_tp = run_nonorthogonal_polygonal_test(flux_scheme="tpfa", **common)
    _, _, _, u_mp, _, d_mp, r_mp = run_nonorthogonal_polygonal_test(flux_scheme="mpfa", **common)

    diff = u_tp - u_mp
    errmax = float(np.max(np.abs(diff)))
    print("TPFA L2_rel vs exact:", r_tp["L2_rel"])
    print("MPFA L2_rel vs exact:", r_mp["L2_rel"])
    print("max |u_tpfa - u_mpfa|:", errmax)

    fig, axs = plt.subplots(2, 2, figsize=(11, 9))
    from heat_solver.plotting import visualize_polygonal_mesh

    visualize_polygonal_mesh(v, p, None, ax=axs[0, 0])
    axs[0, 0].set_title("Skewed polygonal mesh (nonorthogonal)")

    vmin = min(u_tp.min(), u_mp.min(), ue.min())
    vmax = max(u_tp.max(), u_mp.max(), ue.max())
    visualize_polygonal_mesh(v, p, u_tp, ax=axs[0, 1])
    axs[0, 1].set_title("TPFA (two-point + correction)")
    axs[0, 1].collections[-1].set_clim(vmin, vmax)

    visualize_polygonal_mesh(v, p, u_mp, ax=axs[1, 0])
    axs[1, 0].set_title("MPFA-O (subcell FEM / Schur)")
    axs[1, 0].collections[-1].set_clim(vmin, vmax)

    visualize_polygonal_mesh(v, p, diff, ax=axs[1, 1], cmap="coolwarm")
    axs[1, 1].set_title(f"TPFA − MPFA (max |Δ| = {errmax:.2e})")

    for ax in axs.flat:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle("Polygonal heat solver: TPFA vs MPFA-O (sine mode, same mesh & time step)", fontsize=13)
    out = ROOT / "test_plots" / "mpfa_vs_tpfa.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
