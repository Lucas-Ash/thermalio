import numpy as np
import pytest

from heat_solver.afc import AFCMonotoneSolver
from heat_solver.dmp import anisotropic_tensor, bound_excursion, run_dmp_case
from heat_solver.geometry import polygon_area_and_centroid
from heat_solver.meshes import (
    generate_nonorthogonal_polygonal_mesh,
    generate_nonorthogonal_tiled_polygonal_mesh,
)
from heat_solver.polygonal import PolygonalHeatSolver


def _block_ic(centers, blk=(-0.35, 0.35, -0.35, 0.35)):
    x, y = centers[:, 0], centers[:, 1]
    return ((x >= blk[0]) & (x <= blk[1]) & (y >= blk[2]) & (y <= blk[3])).astype(float)


def _rel_l2(u, ue, areas):
    return float(np.sqrt(np.sum(areas * (u - ue) ** 2) / np.sum(areas * ue**2)))


def test_afc_is_bound_preserving_where_highorder_overshoots():
    # Steep front + strong anisotropy on a skewed mesh: the reconstructed flux
    # overshoots; AFC must stay within [0, 1] exactly.
    alpha = anisotropic_tensor(ratio=20.0, angle=np.pi / 6)
    v, p, c = generate_nonorthogonal_tiled_polygonal_mesh(nx_tiles=8, ny_tiles=8, bbox=(-1, 1, -1, 1), skew=0.4)

    recon = run_dmp_case(v, p, c, alpha, {
        "flux_scheme": "tpfa", "flux_discretization": "reconstructed", "nonorthogonal_correction": True,
    }, dt=1.5e-3, t_end=9e-3)
    assert recon["bounds"]["overshoot"] > 1e-3  # confirm the regime really violates the DMP

    solver = AFCMonotoneSolver(
        v, p, alpha, 1.5e-3, bc_func=lambda x, y, t: np.zeros_like(x),
        source_func=lambda x, y, t: np.zeros_like(x), flux_discretization="reconstructed",
    )
    _, u = solver.solve(_block_ic(c), 0.0, 9e-3)
    be = bound_excursion(u, 0.0, 1.0)
    assert be["overshoot"] == 0.0
    assert be["undershoot"] == 0.0


def test_afc_more_accurate_than_linear_monotone():
    # On a smooth anisotropic solution on a skewed mesh, AFC recovers much of the
    # high-order accuracy lost by the linear M-matrix projection, while both are
    # bound-preserving.
    pytest.importorskip("sympy")
    from heat_solver.mms import manufactured_case

    alpha = anisotropic_tensor(ratio=5.0, angle=np.pi / 6)
    case = manufactured_case("exp(-t)*sin(pi*x)*sin(pi*y)", alpha=alpha.tolist(), model="diffusion")
    v, p, c = generate_nonorthogonal_polygonal_mesh(nx=24, ny=24, bbox=(-1, 1, -1, 1), skew=0.3)
    areas = np.array([polygon_area_and_centroid(v[poly])[0] for poly in p])
    u0 = case["solution"](c[:, 0], c[:, 1], 0.0)
    ue = case["solution"](c[:, 0], c[:, 1], 0.004)

    def run_poly(monotone):
        s = PolygonalHeatSolver(
            v, p, alpha, 2e-5, bc_type="dirichlet", bc_func=case["boundary"],
            source_func=case["source"], flux_discretization="reconstructed",
            nonorthogonal_correction=True, monotone=monotone,
        )
        _, u = s.solve(u0, 0.0, 0.004)
        return _rel_l2(u, ue, areas)

    err_recon = run_poly(False)
    err_monotone = run_poly(True)
    s = AFCMonotoneSolver(v, p, alpha, 2e-5, bc_func=case["boundary"],
                          source_func=case["source"], flux_discretization="reconstructed")
    _, u_afc = s.solve(u0, 0.0, 0.004)
    err_afc = _rel_l2(u_afc, ue, areas)

    assert err_afc < 0.7 * err_monotone          # clearly better than linear monotone
    assert err_afc < 3.0 * err_recon             # within a small factor of high-order


def test_afc_recovers_high_order_when_no_limiting_needed():
    # Smooth solution on a square isotropic mesh has no spurious extrema, so AFC
    # should track the high-order reconstructed result closely.
    pytest.importorskip("sympy")
    from heat_solver.mms import manufactured_case
    from heat_solver.meshes import generate_square_polygonal_mesh

    case = manufactured_case("exp(-t)*sin(pi*x)*sin(pi*y)", alpha=0.1, model="diffusion")
    v, p, c = generate_square_polygonal_mesh(nx=24, ny=24, bbox=(-1, 1, -1, 1))
    areas = np.array([polygon_area_and_centroid(v[poly])[0] for poly in p])
    u0 = case["solution"](c[:, 0], c[:, 1], 0.0)
    ue = case["solution"](c[:, 0], c[:, 1], 0.01)

    s_recon = PolygonalHeatSolver(v, p, 0.1, 2e-5, bc_type="dirichlet", bc_func=case["boundary"],
                                  source_func=case["source"], flux_discretization="reconstructed")
    _, u_recon = s_recon.solve(u0, 0.0, 0.01)
    s_afc = AFCMonotoneSolver(v, p, 0.1, 2e-5, bc_func=case["boundary"],
                              source_func=case["source"], flux_discretization="reconstructed")
    _, u_afc = s_afc.solve(u0, 0.0, 0.01)

    assert _rel_l2(u_afc, ue, areas) < 1.5e-3
    # Both should be close on this benign case.
    assert _rel_l2(u_afc, u_recon, areas) < 5e-2
