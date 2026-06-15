import numpy as np
from scipy.sparse import csr_matrix

from heat_solver.dmp import (
    anisotropic_tensor,
    bound_excursion,
    m_matrix_metrics,
    make_monotone,
    run_dmp_case,
)
from heat_solver.geometry import polygon_area_and_centroid
from heat_solver.meshes import (
    generate_nonorthogonal_tiled_polygonal_mesh,
    generate_square_polygonal_mesh,
)


def test_m_matrix_metrics_detects_positive_offdiagonals():
    # Diffusion-like: zero row sums, one positive off-diagonal pair.
    A = csr_matrix(np.array([
        [2.0, -1.0, -1.0],
        [-1.0, 1.5, -0.5],
        [-1.0, 0.5, 0.5],  # the +0.5 is a positive off-diagonal (DMP violation)
    ]))
    m = m_matrix_metrics(A)
    assert m["num_positive_offdiag"] == 1
    assert np.isclose(m["max_positive_offdiag"], 0.5)
    assert m["is_m_matrix"] is False


def test_m_matrix_metrics_clean_m_matrix():
    A = csr_matrix(np.array([[2.0, -1.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, -1.0, 2.0]]))
    assert m_matrix_metrics(A)["is_m_matrix"] is True


def test_bound_excursion():
    be = bound_excursion(np.array([-0.05, 0.5, 1.2]), lo=0.0, hi=1.0)
    assert np.isclose(be["undershoot"], 0.05)
    assert np.isclose(be["overshoot"], 0.2)


def test_make_monotone_properties():
    # Non-M-matrix with zero row sums; projection must yield an M-matrix while
    # preserving zero row sums and (for symmetric input) symmetry.
    A = csr_matrix(np.array([
        [1.0, -1.5, 0.5],
        [-1.5, 1.0, 0.5],
        [0.5, 0.5, -1.0],
    ]))
    assert np.allclose(A.sum(axis=1), 0.0)
    Am = make_monotone(A)
    assert m_matrix_metrics(Am)["is_m_matrix"] is True
    assert np.allclose(Am.sum(axis=1), 0.0)            # conservation preserved
    assert abs((Am - Am.T)).max() < 1e-14              # symmetry preserved


def test_make_monotone_noop_on_m_matrix():
    A = csr_matrix(np.array([[2.0, -1.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, -1.0, 2.0]]))
    Am = make_monotone(A)
    assert abs((Am - A)).max() < 1e-14


def test_base_tpfa_is_monotone_under_anisotropy():
    # Base two-point flux is an M-matrix for any SPD diffusivity -> no overshoot.
    v, p, c = generate_nonorthogonal_tiled_polygonal_mesh(nx_tiles=6, ny_tiles=6, bbox=(-1, 1, -1, 1), skew=0.4)
    alpha = anisotropic_tensor(ratio=20.0, angle=np.pi / 6)
    res = run_dmp_case(v, p, c, alpha, {
        "flux_scheme": "tpfa", "flux_discretization": "tpfa", "nonorthogonal_correction": False,
    }, dt=1.5e-3, t_end=9e-3)
    assert res["m_matrix"]["is_m_matrix"] is True
    assert res["bounds"]["overshoot"] == 0.0
    assert res["bounds"]["undershoot"] == 0.0


def test_reconstructed_violates_dmp_and_monotone_restores_it():
    v, p, c = generate_nonorthogonal_tiled_polygonal_mesh(nx_tiles=6, ny_tiles=6, bbox=(-1, 1, -1, 1), skew=0.4)
    alpha = anisotropic_tensor(ratio=20.0, angle=np.pi / 6)

    violating = run_dmp_case(v, p, c, alpha, {
        "flux_scheme": "tpfa", "flux_discretization": "reconstructed", "nonorthogonal_correction": True,
    }, dt=1.5e-3, t_end=9e-3)
    assert violating["m_matrix"]["is_m_matrix"] is False
    assert violating["bounds"]["overshoot"] > 1e-4 or violating["bounds"]["undershoot"] > 1e-4

    monotone = run_dmp_case(v, p, c, alpha, {
        "flux_scheme": "tpfa", "flux_discretization": "reconstructed",
        "nonorthogonal_correction": True, "monotone": True,
    }, dt=1.5e-3, t_end=9e-3)
    assert monotone["m_matrix"]["is_m_matrix"] is True
    assert monotone["bounds"]["overshoot"] == 0.0
    assert monotone["bounds"]["undershoot"] == 0.0


def test_monotone_is_noop_on_orthogonal_isotropic():
    # On a square isotropic mesh the base operator is already an M-matrix, so the
    # monotone projection changes nothing -- accuracy is untouched there.
    from heat_solver.polygonal import PolygonalHeatSolver

    v, p, c = generate_square_polygonal_mesh(nx=16, ny=16, bbox=(-1, 1, -1, 1))
    kwargs = dict(bc_type="dirichlet", bc_func=lambda x, y, t: np.zeros_like(x),
                  source_func=lambda x, y, t: np.zeros_like(x))
    plain = PolygonalHeatSolver(v, p, 0.1, 1e-3, nonorthogonal_correction=False, **kwargs)
    mono = PolygonalHeatSolver(v, p, 0.1, 1e-3, nonorthogonal_correction=False, monotone=True, **kwargs)
    plain._assemble_system()
    mono._assemble_system()
    assert abs((plain.A - mono.A)).max() < 1e-14
