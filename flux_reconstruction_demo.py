#!/usr/bin/env python3
"""
Print relative L2 errors for ``sine_mode`` on a skewed quadrilateral mesh: classical TPFA
(``nonorthogonal_correction=False``) vs ``flux_discretization='reconstructed'``.

Run from repo root::

    MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python flux_reconstruction_demo.py
"""

from heat_solver.drivers import run_polygonal_mesh_test
from heat_solver.meshes import generate_nonorthogonal_polygonal_mesh


def main():
    vertices, polygons, _ = generate_nonorthogonal_polygonal_mesh(nx=32, ny=32, skew=0.35)
    common = dict(
        alpha=0.1,
        dt=1e-3,
        t_init=0.0,
        t_end=0.02,
        case="sine_mode",
        nonorthogonal_correction=False,
    )
    _, _, _, _, _, _, r_tpfa = run_polygonal_mesh_test(
        vertices, polygons, flux_discretization="tpfa", **common
    )
    _, _, _, _, _, _, r_rec = run_polygonal_mesh_test(
        vertices, polygons, flux_discretization="reconstructed", **common
    )
    print("skewed nonorthogonal mesh 32x32 cells, sine_mode, classical TPFA (no NOC)")
    print(f"  TPFA L2_rel:           {r_tpfa['L2_rel']:.6e}")
    print(f"  reconstructed L2_rel:  {r_rec['L2_rel']:.6e}")
    print(f"  ratio (rec / tpfa):    {r_rec['L2_rel'] / r_tpfa['L2_rel']:.4f}")


if __name__ == "__main__":
    main()
