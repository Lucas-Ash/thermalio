"""Tests that pin down the documented limitations of the quadratic flux.

Each test either *checks* a property that is not proved (coercivity, energy
stability), *documents* a known limitation so a change is noticed (no
monotonicity, wider stencil), or verifies the support added for one
(temperature-dependent conductivity with precomputed geometry).
"""

import numpy as np
import pytest
from scipy.linalg import eigvals, eigvalsh

from heat_solver import quadratic_flux as qf
from heat_solver.afc import AFCMonotoneSolver
from heat_solver.cases import get_analytical_case
from heat_solver.dmp import anisotropic_tensor, bound_excursion
from heat_solver.meshes import (
    generate_hexagonal_polygonal_mesh,
    generate_nonorthogonal_polygonal_mesh,
    generate_nonorthogonal_tiled_polygonal_mesh,
    generate_square_polygonal_mesh,
)
from heat_solver.nonlinear_cases import anisotropic_nonlinear_case
from heat_solver.polygonal import PolygonalHeatSolver


def _perturbed(n, seed=0):
    v, p, _ = generate_square_polygonal_mesh(nx=n, ny=n, bbox=(-1, 1, -1, 1))
    v = v.copy()
    inner = (np.abs(v[:, 0]) < 1 - 1e-9) & (np.abs(v[:, 1]) < 1 - 1e-9)
    v[inner] += 0.2 * (2.0 / n) * np.random.default_rng(seed).uniform(-1, 1, (inner.sum(), 2))
    return v, p


MESHES = {
    "skewed": lambda n=12: generate_nonorthogonal_polygonal_mesh(nx=n, ny=n, bbox=(-1, 1, -1, 1), skew=0.3)[:2],
    "hex": lambda n=12: generate_hexagonal_polygonal_mesh(nx=n, ny=n, spacing=2.0 / n, bbox=(-1, 1, -1, 1))[:2],
    "perturbed": lambda n=12: _perturbed(n),
    "tiled": lambda n=12: generate_nonorthogonal_tiled_polygonal_mesh(
        nx_tiles=n // 3, ny_tiles=n // 3, bbox=(-1, 1, -1, 1), skew=0.3)[:2],
}


def _interior_operator(v, p, K):
    s = PolygonalHeatSolver(v, p, K, 1.0, flux_discretization="quadratic")
    s._assemble_system()
    interior = ~s.is_boundary
    A = s.A.toarray()[np.ix_(interior, interior)]
    return s, A, s.cell_areas[interior]


# --- Limitation 1: coercivity is not proved -> check it numerically ---------------

@pytest.mark.parametrize("mesh", list(MESHES))
@pytest.mark.parametrize("ratio", [5.0, 1000.0])
def test_operator_is_coercive_and_time_stable(mesh, ratio):
    K = anisotropic_tensor(ratio=ratio, angle=np.pi / 6) * 0.1
    _, A, D = _interior_operator(*MESHES[mesh](), K)
    scale = np.sqrt(np.outer(D, D))
    # Coercivity: u^T A u > 0 for all u != 0 (symmetric part positive definite).
    assert eigvalsh(0.5 * (A + A.T) / scale).min() > 1e-3
    # Time stability: spectrum of D^{-1} A in the open right half plane, so backward
    # Euler and Crank--Nicolson are unconditionally stable on this mesh.
    assert eigvals(A / D[:, None]).real.min() > 1e-3


def test_backward_euler_energy_decays_without_sources():
    K = anisotropic_tensor(ratio=50.0, angle=np.pi / 6) * 0.1
    v, p = MESHES["perturbed"](16)
    zero = lambda x, y, t: np.zeros_like(np.asarray(x, float))
    s = PolygonalHeatSolver(v, p, K, 5e-3, bc_type="dirichlet", bc_func=zero, flux_discretization="quadratic")
    u = np.random.default_rng(1).uniform(-1, 1, s.M)
    u[s.is_boundary] = 0.0
    energies = []
    for k in range(20):
        u = s.solve(u, k * 5e-3, (k + 1) * 5e-3)[1]
        energies.append(np.sum(s.cell_areas * u**2))
    assert np.all(np.diff(energies) <= 1e-14)


# --- Limitation 2: not monotone (documented), AFC restores bounds -----------------

def test_quadratic_flux_is_not_monotone_but_afc_bounds_it():
    alpha = anisotropic_tensor(ratio=20.0, angle=np.pi / 6)
    v, p, c = generate_nonorthogonal_polygonal_mesh(nx=16, ny=16, bbox=(-1, 1, -1, 1), skew=0.3)
    s = PolygonalHeatSolver(v, p, alpha, 1.5e-3, flux_discretization="quadratic",
                            bc_type="dirichlet", bc_func=lambda x, y, t: np.zeros_like(x))
    s._assemble_system()
    A = s.A.tocoo()
    off = A.row != A.col
    assert np.mean(A.data[off] > 1e-14 * np.abs(A.data).max()) > 0.2  # not an M-matrix
    block = ((np.abs(c[:, 0]) <= 0.35) & (np.abs(c[:, 1]) <= 0.35)).astype(float)
    _, u = s.solve(block, 0.0, 9e-3)
    excursion = bound_excursion(u, 0.0, 1.0)
    assert max(excursion["overshoot"], excursion["undershoot"]) > 1e-3  # overshoots a steep front
    zero = lambda x, y, t: np.zeros_like(np.asarray(x, float))
    afc = AFCMonotoneSolver(v, p, alpha, 1.5e-3, bc_func=zero, source_func=zero,
                            flux_discretization="quadratic", limiter="global")
    be = bound_excursion(afc.solve(block, 0.0, 9e-3)[1], 0.0, 1.0)
    assert be["overshoot"] == 0.0 and be["undershoot"] == 0.0


# --- Limitation 3: wider stencil (documented bound) --------------------------------

def test_stencil_width_is_bounded():
    v, p = MESHES["skewed"](16)
    nnz = {}
    for disc in ("quadratic", "tpfa"):
        s = PolygonalHeatSolver(v, p, 0.1, 1.0, flux_discretization=disc, nonorthogonal_correction=True)
        s._assemble_system()
        nnz[disc] = s.A.nnz / s.M
    assert 15.0 < nnz["quadratic"] <= 25.0
    assert nnz["quadratic"] > nnz["tpfa"]


# --- Limitation 4: assembly cost -> geometry precomputed once, re-assembly vectorized

def test_operator_is_linear_in_conductivity():
    v, p = MESHES["tiled"]()
    s = PolygonalHeatSolver(v, p, 0.1, 1.0)
    op = qf.QuadraticDiffusionOperator(v, p, s.edge_to_cells)
    K1 = anisotropic_tensor(5.0, 0.3) * 0.1
    K2 = np.random.default_rng(0).uniform(0.05, 0.2, (op.n_faces, 2, 2))
    lhs = op.assemble(2.0 * K1 + K2)
    rhs = 2.0 * op.assemble(K1) + op.assemble(K2)
    assert abs(lhs - rhs).max() < 1e-13 * abs(lhs).max()


def test_nonlinear_solve_reuses_precomputed_geometry(monkeypatch):
    case = anisotropic_nonlinear_case()
    v, p = MESHES["skewed"](10)
    built = {"n": 0}
    calls = {"n": 0}
    init, assemble = qf.QuadraticDiffusionOperator.__init__, qf.QuadraticDiffusionOperator.assemble

    def counting_init(self, *a, **k):
        built["n"] += 1
        init(self, *a, **k)

    def counting_assemble(self, K):
        calls["n"] += 1
        return assemble(self, K)

    monkeypatch.setattr(qf.QuadraticDiffusionOperator, "__init__", counting_init)
    monkeypatch.setattr(qf.QuadraticDiffusionOperator, "assemble", counting_assemble)
    s = PolygonalHeatSolver(v, p, case["alpha"], 1e-3, bc_type="dirichlet", bc_func=case["exact"],
                            source_func=case["source"], flux_discretization="quadratic")
    s.solve(s.cell_average(case["exact"], 0.0), 0.0, 5e-3)
    assert built["n"] == 1
    assert calls["n"] > 5  # re-assembled in every Picard iteration


# --- Limitation 5 (now supported): temperature-dependent conductivity --------------

@pytest.mark.parametrize("mesh", list(MESHES))
def test_face_values_are_exact_for_quadratics(mesh):
    v, p = MESHES[mesh]()
    s = PolygonalHeatSolver(v, p, 0.1, 1.0)
    op = qf.QuadraticDiffusionOperator(v, p, s.edge_to_cells)
    q = lambda x, y, *a: 1 + 2 * x - y + x * x + 3 * x * y - 2 * y * y
    u = s.cell_average(q)
    xf = op.face_midpoints
    np.testing.assert_allclose(op.face_values(u), q(xf[:, 0], xf[:, 1]), atol=1e-11)


def _nonlinear_error(v, p, disc):
    case = anisotropic_nonlinear_case()
    s = PolygonalHeatSolver(v, p, case["alpha"], 2e-4, bc_type="dirichlet", bc_func=case["exact"],
                            source_func=case["source"], time_scheme="crank_nicolson", flux_discretization=disc,
                            nonorthogonal_correction=True, cell_average_data=True,
                            nonlinear_options={"max_iters": 50, "tol": 1e-11})
    assert s.temperature_dependent_diffusivity
    u = s.solve(s.cell_average(case["exact"], 0.0), 0.0, 0.01)[1]
    ue = s.cell_average(case["exact"], 0.01)
    return np.sqrt(np.sum(s.cell_areas * (u - ue) ** 2) / np.sum(s.cell_areas * (ue - 1.0) ** 2))


@pytest.mark.parametrize("mesh", ["skewed", "hex"])
def test_temperature_dependent_tensor_conductivity_converges_second_order(mesh):
    e = [_nonlinear_error(*MESHES[mesh](n), "quadratic") for n in (16, 32)]
    assert np.log2(e[0] / e[1]) > 1.6


def test_temperature_dependent_quadratic_beats_tpfa_on_perturbed_mesh():
    v, p = MESHES["perturbed"](24)
    assert _nonlinear_error(v, p, "quadratic") < 0.3 * _nonlinear_error(v, p, "tpfa")


def test_legacy_scalar_temperature_dependent_case_runs_with_quadratic_flux():
    case = get_analytical_case("temperature_dependent_diffusivity", 0.1, 0.05)
    v, p, _ = generate_nonorthogonal_polygonal_mesh(nx=12, ny=12, bbox=case["bbox"], skew=0.3)
    s = PolygonalHeatSolver(v, p, case["alpha"], 5e-3, bc_type="dirichlet", bc_func=case["solution"],
                            source_func=case["source"], flux_discretization="quadratic",
                            time_scheme="crank_nicolson", nonlinear_options=case["nonlinear_options"])
    u = s.solve(s.cell_average(case["solution"], 0.0), 0.0, 0.05)[1]
    # The solution is linear in space, which the quadratic flux reproduces exactly:
    # only the (small) Crank--Nicolson time error remains.
    assert np.max(np.abs(u - s.cell_average(case["solution"], 0.05))) < 2e-7


@pytest.mark.parametrize("disc", ["quadratic", "tpfa"])
def test_crank_nicolson_is_second_order_with_temperature_dependent_conductivity(disc):
    # Regression: the explicit half of CN used A(u^{n+1}) instead of A(u^n), which
    # made nonlinear CN first order in time (error halved with dt).
    case = get_analytical_case("temperature_dependent_diffusivity", 0.1, 0.05)
    v, p, _ = generate_nonorthogonal_polygonal_mesh(nx=10, ny=10, bbox=case["bbox"], skew=0.3)
    errs = []
    for dt in (5e-3, 2.5e-3):
        s = PolygonalHeatSolver(v, p, case["alpha"], dt, bc_type="dirichlet", bc_func=case["solution"],
                                source_func=case["source"], flux_discretization=disc, time_scheme="crank_nicolson",
                                nonlinear_options=case["nonlinear_options"], cell_average_data=True)
        u = s.solve(s.cell_average(case["solution"], 0.0), 0.0, 0.05)[1]
        errs.append(np.max(np.abs(u - s.cell_average(case["solution"], 0.05))))
    assert np.log2(errs[0] / errs[1]) > 1.8
