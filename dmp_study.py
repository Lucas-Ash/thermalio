#!/usr/bin/env python3
"""Discrete-maximum-principle (DMP) study for the polygonal FV schemes (direction B).

Two parts:

1. **DMP / overshoot sweep**: diffuse a bounded indicator block (values in
   [0, 1]) with zero Dirichlet data across meshes x anisotropy ratios x flux
   schemes, reporting whether the assembled operator is an M-matrix and the
   solution over/undershoot beyond [0, 1].  Demonstrates that the base two-point
   flux is unconditionally monotone, the non-orthogonal correction and
   reconstructed flux violate the DMP under skew/anisotropy, and the monotone
   M-matrix projection restores it.

2. **Accuracy vs monotonicity trade-off**: on a smooth manufactured anisotropic
   solution on a skewed mesh, report the observed order of accuracy for each
   scheme alongside its overshoot.  Shows the reconstructed flux is most accurate
   but non-monotone, while the monotone projection removes overshoot at some
   accuracy cost (and is a no-op where the base scheme is already monotone).

Usage:
    MPLCONFIGDIR=/tmp/matplotlib python dmp_study.py
"""

from __future__ import annotations

import csv as _csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heat_solver.dmp import DMP_SCHEMES, anisotropic_tensor, run_dmp_case, run_dmp_study
from heat_solver.geometry import polygon_area_and_centroid
from heat_solver.meshes import (
    generate_nonorthogonal_tiled_polygonal_mesh,
    generate_square_polygonal_mesh,
)
from heat_solver.polygonal import PolygonalHeatSolver
from heat_solver.verification import observed_order

OUTPUT_DIR = ROOT / "test_plots" / "dmp"


def _mesh_builders():
    return {
        "square": lambda: generate_square_polygonal_mesh(nx=24, ny=24, bbox=(-1, 1, -1, 1)),
        "skewed_tiled": lambda: generate_nonorthogonal_tiled_polygonal_mesh(
            nx_tiles=8, ny_tiles=8, bbox=(-1, 1, -1, 1), skew=0.4
        ),
    }


def _anisotropy_panel():
    return {
        "isotropic": anisotropic_tensor(ratio=1.0),
        "aniso_x5": anisotropic_tensor(ratio=5.0, angle=np.pi / 6),
        "aniso_x20": anisotropic_tensor(ratio=20.0, angle=np.pi / 6),
    }


def dmp_sweep():
    records = run_dmp_study(_mesh_builders(), _anisotropy_panel())
    print("=== DMP / overshoot sweep (bounded block in [0, 1]) ===")
    header = f"{'mesh':13s} {'anisotropy':11s} {'scheme':24s} {'M-matrix':9s} {'overshoot':>10s} {'undershoot':>11s}"
    print(header)
    for r in records:
        print(
            f"{r['mesh']:13s} {r['anisotropy']:11s} {r['scheme']:24s} "
            f"{str(r['is_m_matrix']):9s} {r['overshoot']:10.3e} {r['undershoot']:11.3e}"
        )
    return records


def _solve_mms_skewed(case, alpha, tiles, dt, t_end, scheme_kwargs):
    v, p, c = generate_nonorthogonal_tiled_polygonal_mesh(
        nx_tiles=tiles, ny_tiles=tiles, bbox=(-1, 1, -1, 1), skew=0.4
    )
    areas = np.array([polygon_area_and_centroid(v[poly])[0] for poly in p])
    solver = PolygonalHeatSolver(
        v, p, alpha, dt, bc_type="dirichlet",
        bc_func=case["boundary"], source_func=case["source"], **scheme_kwargs,
    )
    u0 = case["solution"](c[:, 0], c[:, 1], 0.0)
    _, u = solver.solve(u0, 0.0, t_end)
    ue = case["solution"](c[:, 0], c[:, 1], t_end)
    err = float(np.sqrt(np.sum(areas * (u - ue) ** 2) / np.sum(areas * ue**2)))
    return err


def tradeoff_study():
    try:
        from heat_solver.mms import manufactured_case
    except ImportError:
        print("\n(SymPy not installed; skipping accuracy/monotonicity trade-off study.)")
        return []

    alpha = anisotropic_tensor(ratio=20.0, angle=np.pi / 6)
    case = manufactured_case(
        "exp(-t)*sin(pi*x)*sin(pi*y)", alpha=alpha.tolist(), model="diffusion",
    )
    tiles_coarse, tiles_fine = 8, 16
    h_coarse, h_fine = 2.0 / (3 * tiles_coarse), 2.0 / (3 * tiles_fine)
    dt, t_end = 1e-4, 0.01

    print("\n=== Accuracy vs monotonicity (smooth MMS, skewed mesh, anisotropy x20) ===")
    print(f"{'scheme':24s} {'err(coarse)':>12s} {'err(fine)':>12s} {'order':>7s} {'overshoot':>10s}")
    rows = []
    block_v, block_p, block_c = generate_nonorthogonal_tiled_polygonal_mesh(
        nx_tiles=tiles_fine, ny_tiles=tiles_fine, bbox=(-1, 1, -1, 1), skew=0.4
    )
    for label, kwargs in DMP_SCHEMES.items():
        e_c = _solve_mms_skewed(case, alpha, tiles_coarse, dt, t_end, kwargs)
        e_f = _solve_mms_skewed(case, alpha, tiles_fine, dt, t_end, kwargs)
        order = observed_order(e_c, e_f, h_coarse / h_fine)
        block = run_dmp_case(block_v, block_p, block_c, alpha, kwargs, dt=1.5e-3, t_end=9e-3)
        overshoot = max(block["bounds"]["overshoot"], block["bounds"]["undershoot"])
        ostr = "n/a" if order is None else f"{order:.2f}"
        print(f"{label:24s} {e_c:12.3e} {e_f:12.3e} {ostr:>7s} {overshoot:10.3e}")
        rows.append({"scheme": label, "err_coarse": e_c, "err_fine": e_f,
                     "order": order, "overshoot": overshoot})
    print(
        "  note: in this severe regime (ratio x20 + skew) base TPFA and the linear\n"
        "  M-matrix projection are ~O(1) inconsistent (the classic two-point-flux\n"
        "  failure on K-non-orthogonal meshes), while the consistent schemes\n"
        "  overshoot -- motivating a *nonlinear* monotone scheme (NTPFA) for both\n"
        "  consistency and the DMP (direction B PR2)."
    )
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sweep = dmp_sweep()
    tradeoff = tradeoff_study()

    (OUTPUT_DIR / "dmp_report.json").write_text(
        json.dumps({"sweep": sweep, "tradeoff": tradeoff}, indent=2, default=str), encoding="utf-8"
    )
    with (OUTPUT_DIR / "dmp_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=list(sweep[0].keys()))
        writer.writeheader()
        writer.writerows(sweep)
    print(f"\nwrote {OUTPUT_DIR / 'dmp_report.json'} and {OUTPUT_DIR / 'dmp_sweep.csv'}")


if __name__ == "__main__":
    main()
