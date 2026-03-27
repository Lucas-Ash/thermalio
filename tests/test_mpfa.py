import numpy as np
import pytest
from scipy.sparse.linalg import norm as sp_norm

from heat_solver.meshes import generate_nonorthogonal_polygonal_mesh, generate_square_polygonal_mesh
from heat_solver.polygonal import PolygonalHeatSolver


def _fro(a):
    return float(sp_norm(a, "fro"))


def test_mpfa_diffusion_matrix_symmetric():
    v, p, _ = generate_square_polygonal_mesh(nx=6, ny=6, bbox=(0.0, 1.0, 0.0, 1.0))
    s = PolygonalHeatSolver(v, p, alpha=0.2, dt=0.01, bc_type="dirichlet", flux_scheme="mpfa")
    s._assemble_system()
    d = (s.A - s.A.T).toarray()
    assert np.linalg.norm(d) < 1e-10


def test_mpfa_skew_mesh_differs_from_tpfa():
    """On a non-orthogonal mesh, MPFA (FEM) and TPFA assemble different stencils."""
    v, p, _ = generate_nonorthogonal_polygonal_mesh(nx=10, ny=10, bbox=(0.0, 1.0, 0.0, 1.0), skew=0.35)
    kwargs = dict(vertices=v, polygons=p, alpha=0.2, dt=0.01, bc_type="dirichlet", nonorthogonal_correction=True)
    tp = PolygonalHeatSolver(**kwargs, flux_scheme="tpfa")
    mp = PolygonalHeatSolver(**kwargs, flux_scheme="mpfa")
    tp._assemble_system()
    mp._assemble_system()
    rel = _fro(tp.A - mp.A) / (_fro(tp.A) + 1e-16)
    assert rel > 0.02


def test_mpfa_neumann_and_robin_use_same_diffusion_as_tpfa():
    """Neumann/Robin use TPFA diffusion fallback; solutions match ``flux_scheme='tpfa'``."""
    from heat_solver import get_analytical_case, run_square_polygonal_test

    for case in ("steady_linear_neumann", "steady_linear_robin"):
        info = get_analytical_case(case, alpha=0.1, t_end=0.1)
        kw = dict(
            case=case,
            alpha=0.1,
            dt=0.02,
            t_init=0.0,
            t_end=0.1,
            nx=10,
            ny=10,
            bbox=info["bbox"],
            nonorthogonal_correction=True,
        )
        *_, u_tp, _, _, _ = run_square_polygonal_test(flux_scheme="tpfa", **kw)
        *_, u_mp, _, _, _ = run_square_polygonal_test(flux_scheme="mpfa", **kw)
        assert np.allclose(u_tp, u_mp, rtol=0, atol=1e-12)


def test_mpfa_invalid_flux_scheme_raises():
    v, p, _ = generate_square_polygonal_mesh(nx=2, ny=2, bbox=(0.0, 1.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="flux_scheme"):
        PolygonalHeatSolver(v, p, alpha=0.2, dt=0.01, flux_scheme="mimetic")


def test_mpfa_sine_mode_solve_matches_driver():
    from heat_solver.drivers import run_nonorthogonal_polygonal_test

    kwargs = dict(
        case="sine_mode",
        alpha=0.1,
        dt=2e-3,
        t_init=0.0,
        t_end=0.02,
        nx=8,
        ny=8,
        bbox=(-1.0, 1.0, -1.0, 1.0),
        skew=0.35,
        nonorthogonal_correction=True,
        flux_scheme="mpfa",
    )
    *_, _u, _ue, _d, res = run_nonorthogonal_polygonal_test(**kwargs)
    assert np.isfinite(res["L2_rel"])
    assert res["L2_rel"] < 0.5
