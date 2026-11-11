import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from heat_solver.afc import AFCMonotoneSolver
from heat_solver.dmp import anisotropic_tensor, bound_excursion
from heat_solver.meshes import (
    generate_hexagonal_polygonal_mesh,
    generate_nonorthogonal_polygonal_mesh,
    generate_nonorthogonal_tiled_polygonal_mesh,
    generate_square_polygonal_mesh,
)
from heat_solver.polygonal import PolygonalHeatSolver
from heat_solver.quadratic_flux import CellQuadrature, QuadraticReconstruction, polygon_second_moments

K = anisotropic_tensor(ratio=5.0, angle=np.pi / 6) * 0.1


def _perturbed(n, seed=0):
    v, p, _ = generate_square_polygonal_mesh(nx=n, ny=n, bbox=(-1, 1, -1, 1))
    v = v.copy()
    inner = (np.abs(v[:, 0]) < 1 - 1e-9) & (np.abs(v[:, 1]) < 1 - 1e-9)
    v[inner] += 0.2 * (2.0 / n) * np.random.default_rng(seed).uniform(-1, 1, (inner.sum(), 2))
    return v, p


MESHES = {
    "skewed": lambda: generate_nonorthogonal_polygonal_mesh(nx=10, ny=10, bbox=(-1, 1, -1, 1), skew=0.3)[:2],
    "tiled": lambda: generate_nonorthogonal_tiled_polygonal_mesh(nx_tiles=4, ny_tiles=4, bbox=(-1, 1, -1, 1), skew=0.3)[:2],
    "perturbed": lambda: _perturbed(10),
    "hex": lambda: generate_hexagonal_polygonal_mesh(nx=10, ny=10, spacing=0.2, bbox=(-1, 1, -1, 1))[:2],
}


def _quad_avg(c, M):
    # cell average of q(x, y) = x^2 + 3xy - 2y^2
    return c[0] ** 2 + 3 * c[0] * c[1] - 2 * c[1] ** 2 + M[0, 0] + 3 * M[0, 1] - 2 * M[1, 1]


def test_second_moments_match_quadrature():
    v, p = _perturbed(6)
    q = CellQuadrature(v, p)
    for i, poly in enumerate(p[:10]):
        area, c, M = polygon_second_moments(v[poly])
        assert area == pytest.approx(q.areas[i], rel=1e-12)
        mxx = q.average(lambda x, y: (x - c[0]) ** 2)[i]
        mxy = q.average(lambda x, y: (x - c[0]) * (y - c[1]))[i]
        assert M[0, 0] == pytest.approx(mxx, abs=1e-14)
        assert M[0, 1] == pytest.approx(mxy, abs=1e-14)


@pytest.mark.parametrize("mesh", list(MESHES))
def test_reconstruction_is_exact_for_quadratics(mesh):
    v, p = MESHES[mesh]()
    rec = QuadraticReconstruction(v, p)
    u = np.array([_quad_avg(rec.centers[i], rec.moments[i]) for i in range(len(p))])
    for i in range(len(p)):
        z = rec.coeffs[i] @ u[rec.indices(i)]
        x, y = rec.centers[i]
        if rec.degree[i] == 2:
            np.testing.assert_allclose(z, [2 * x + 3 * y, 3 * x - 4 * y, 2, 3, -4], atol=1e-10)


@pytest.mark.parametrize("mesh", list(MESHES))
def test_operator_reproduces_steady_quadratic_where_tpfa_does_not(mesh):
    v, p = MESHES[mesh]()
    q_src = -(2 * K[0, 0] + 3 * (K[0, 1] + K[1, 0]) - 4 * K[1, 1])
    errs = {}
    for disc in ("quadratic", "tpfa"):
        s = PolygonalHeatSolver(v, p, K, 1.0, flux_discretization=disc, nonorthogonal_correction=True)
        s._assemble_system()
        avg = np.array([_quad_avg(*polygon_second_moments(v[poly])[1:]) for poly in p])
        A, b = s.A.tolil(), s.cell_areas * q_src
        for i in np.where(s.is_boundary)[0]:
            A.rows[i], A.data[i], b[i] = [i], [1.0], avg[i]
        errs[disc] = np.max(np.abs(spsolve(A.tocsr(), b) - avg))
        # conservative: every column of A sums to zero (face fluxes cancel pairwise)
        assert np.max(np.abs(np.asarray(s.A.sum(axis=0)))) < 1e-12
    assert errs["quadratic"] < 1e-11
    if mesh != "skewed":  # on the affine shear the corrected TPFA is also exact
        assert errs["tpfa"] > 1e-4


def _transient_error(v, p, disc):
    def exact(x, y, t):
        return np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y)

    def source(x, y, t):
        u = exact(x, y, t)
        uxy = np.pi**2 * np.exp(-t) * np.cos(np.pi * x) * np.cos(np.pi * y)
        return -u + np.pi**2 * (K[0, 0] + K[1, 1]) * u - (K[0, 1] + K[1, 0]) * uxy

    s = PolygonalHeatSolver(v, p, K, 2e-4, bc_type="dirichlet", bc_func=exact, source_func=source,
                            time_scheme="crank_nicolson", flux_discretization=disc,
                            nonorthogonal_correction=True, cell_average_data=True)
    u = s.solve(s.cell_average(exact, 0.0), 0.0, 0.01)[1]
    ue = s.cell_average(exact, 0.01)
    return np.sqrt(np.sum(s.cell_areas * (u - ue) ** 2) / np.sum(s.cell_areas * ue**2))


def test_quadratic_flux_restores_second_order_on_skewed_mesh():
    mk = lambda n: generate_nonorthogonal_polygonal_mesh(nx=n, ny=n, bbox=(-1, 1, -1, 1), skew=0.3)[:2]
    e_q = [_transient_error(*mk(n), "quadratic") for n in (16, 32)]
    e_r = [_transient_error(*mk(n), "reconstructed") for n in (16, 32)]
    assert np.log2(e_q[0] / e_q[1]) > 1.75
    assert np.log2(e_r[0] / e_r[1]) < 1.3      # the legacy averaged-gradient flux loses order
    assert e_q[1] < 0.25 * e_r[1]


def test_quadratic_flux_converges_on_hexagonal_mesh():
    mk = lambda n: generate_hexagonal_polygonal_mesh(nx=n, ny=n, spacing=2.0 / n, bbox=(-1, 1, -1, 1))[:2]
    e = [_transient_error(*mk(n), "quadratic") for n in (16, 32)]
    assert np.log2(e[0] / e[1]) > 1.6


def test_default_cell_average_data_and_option_validation():
    v, p = MESHES["skewed"]()
    s = PolygonalHeatSolver(v, p, K, 0.01, flux_discretization="quadratic")
    assert s.cell_average_data
    assert not PolygonalHeatSolver(v, p, K, 0.01).cell_average_data
    with pytest.raises(ValueError):
        PolygonalHeatSolver(v, p, K, 0.01, flux_scheme="mpfa", flux_discretization="quadratic")


def test_afc_accepts_quadratic_target_and_stays_bounded():
    alpha = anisotropic_tensor(ratio=20.0, angle=np.pi / 6)
    v, p, c = generate_nonorthogonal_polygonal_mesh(nx=16, ny=16, bbox=(-1, 1, -1, 1), skew=0.3)
    u0 = ((np.abs(c[:, 0]) <= 0.35) & (np.abs(c[:, 1]) <= 0.35)).astype(float)
    zero = lambda x, y, t: np.zeros_like(np.asarray(x, float))
    s = AFCMonotoneSolver(v, p, alpha, 1.5e-3, bc_func=zero, source_func=zero,
                          flux_discretization="quadratic", limiter="global")
    _, u = s.solve(u0, 0.0, 9e-3)
    be = bound_excursion(u, 0.0, 1.0)
    assert be["overshoot"] == 0.0 and be["undershoot"] == 0.0
