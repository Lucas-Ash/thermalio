"""
Regression: flux_discretization='reconstructed' vs classical TPFA on a skewed polygonal mesh.

Classical TPFA is compared with ``nonorthogonal_correction=False`` so the baseline is the
two-point transmissibility only (no gradient-based correction).
"""

import numpy as np
import pytest

from heat_solver.drivers import run_polygonal_mesh_test
from heat_solver.meshes import generate_nonorthogonal_polygonal_mesh
from heat_solver.polygonal import PolygonalHeatSolver


def test_reconstructed_beats_tpfa_on_skewed_sine_mode():
    vertices, polygons, _ = generate_nonorthogonal_polygonal_mesh(nx=24, ny=24, skew=0.35)
    kwargs = dict(
        alpha=0.1,
        dt=1e-3,
        t_init=0.0,
        t_end=0.02,
        case="sine_mode",
        nonorthogonal_correction=False,
    )
    _, _, _, _, _, _, res_tpfa = run_polygonal_mesh_test(
        vertices,
        polygons,
        flux_discretization="tpfa",
        **kwargs,
    )
    _, _, _, _, _, _, res_rec = run_polygonal_mesh_test(
        vertices,
        polygons,
        flux_discretization="reconstructed",
        **kwargs,
    )
    assert res_rec["L2_rel"] < res_tpfa["L2_rel"]
    assert res_tpfa["L2_rel"] > 1e-4


def test_mpfa_with_reconstructed_disallowed():
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    polygons = [[0, 1, 2, 3]]
    with pytest.raises(ValueError, match="reconstructed"):
        PolygonalHeatSolver(
            vertices,
            polygons,
            alpha=0.1,
            dt=0.01,
            flux_scheme="mpfa",
            flux_discretization="reconstructed",
        )


def test_invalid_flux_discretization():
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    polygons = [[0, 1, 2, 3]]
    with pytest.raises(ValueError, match="flux_discretization"):
        PolygonalHeatSolver(
            vertices,
            polygons,
            alpha=0.1,
            dt=0.01,
            flux_discretization="bogus",
        )
