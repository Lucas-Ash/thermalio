"""Factorization reuse for Robin / radiative boundaries and faster first-time setup (performance item 3).

* Linear boundary closures (Robin with time-independent beta, Neumann, flux) reuse
  their factorization and reproduce the refactorize-every-step results bit for bit.
* Radiative closures use frozen-Jacobian Newton and agree with full Newton to the
  nonlinear tolerance.
* Vectorized curvilinear boundary rows, batched gradient stencils and generator
  centroids are bit-identical to the legacy per-row / per-cell code.
"""

import numpy as np
import pytest
from scipy.sparse import csc_matrix
from scipy.sparse import random as sparse_random
from scipy.sparse.linalg import spsolve

from heat_solver.bc_test_problems import heat_problem, transport_problem
from heat_solver.curvilinear import CurvilinearHeatSolver
from heat_solver.factor_cache import FactorCache, spsolve_factor
from heat_solver.fv_assembly import replace_rows
from heat_solver.geometry import polygon_area_and_centroid
from heat_solver.mesh_library import generate
from heat_solver.meshes import (
    generate_hexagonal_polygonal_mesh,
    generate_mixed_polygonal_mesh,
    generate_nonorthogonal_tiled_polygonal_mesh,
    generate_perturbed_quad_mesh,
)
from heat_solver.polygonal import PolygonalHeatSolver
from heat_solver.transport import FractionalHeatSolver, ReactionDiffusionHeatSolver


def _system(n=200, seed=0):
    A = sparse_random(n, n, density=0.03, random_state=seed)
    return (A + A.T + 10.0 * csc_matrix(np.eye(n))).tocsr()


@pytest.mark.parametrize("fmt", ["csr", "csc"])
def test_spsolve_factor_matches_spsolve_bit_for_bit(fmt):
    K = _system().asformat(fmt)
    rng = np.random.default_rng(1)
    solve = spsolve_factor(K)
    for _ in range(3):
        b = rng.standard_normal(K.shape[0])
        assert np.array_equal(solve(b), spsolve(K, b))


def test_factor_cache_reuses_until_inputs_change():
    base = _system()
    B = csc_matrix(np.diag(np.linspace(1, 2, base.shape[0]))).tocsr()
    cache, builds = FactorCache(), []

    def build(Bm):
        return lambda: (builds.append(1), base + 0.1 * Bm)[1]

    f1 = cache.factor("k", (base,), B, build(B))
    assert cache.factor("k", (base,), B.copy(), build(B)) is f1  # same values -> reused
    assert len(builds) == 1
    B2 = B.copy()
    B2.data[0] += 1e-14
    cache.factor("k", (base,), B2, build(B2))  # any change in B -> refactorized
    other = base.copy()
    cache.factor("k", (other,), B2, build(B2))  # a different base object -> refactorized
    assert len(builds) == 3


def _poly_run(bc, scheme="backward_euler", n=16, **kw):
    p = heat_problem(bc, manufactured_boundary=True)
    m = generate("nonorthogonal", nx=n, ny=n, bbox=p["bbox"], skew=0.3)
    s = PolygonalHeatSolver(*m, p["alpha"], 2e-3, bc_type=p["bc_type"], bc_func=p["bc_func"],
                            source_func=p["source"], time_scheme=scheme, nonorthogonal_correction=True, **kw)
    c = s.cell_centers
    _, u = s.solve(p["exact"](c[:, 0], c[:, 1], 0.0), 0.0, 0.05)
    return u, s


@pytest.mark.parametrize("scheme", ["backward_euler", "bdf2", "crank_nicolson"])
def test_polygonal_robin_reuses_factorization_and_matches_fresh_solves(scheme):
    u, s = _poly_run("robin", scheme)
    stats = s.last_solve_stats
    assert stats["boundary_factorizations"] <= 6  # one per distinct (dt, scheme coefficient)
    assert stats["factorizations_reused"] >= 18
    # Reference: refactorize every step (no reuse) -> identical to the bit.
    p = heat_problem("robin", manufactured_boundary=True)
    m = generate("nonorthogonal", nx=16, ny=16, bbox=p["bbox"], skew=0.3)
    ref = PolygonalHeatSolver(*m, p["alpha"], 2e-3, bc_type="robin", bc_func=p["bc_func"], source_func=p["source"],
                              time_scheme=scheme, nonorthogonal_correction=True)
    ref._factor_cache.lookup = lambda *a, **k: None  # never reuse
    c = ref.cell_centers
    _, u_ref = ref.solve(p["exact"](c[:, 0], c[:, 1], 0.0), 0.0, 0.05)
    assert np.array_equal(u, u_ref)


def test_robin_with_time_dependent_beta_refactorizes():
    m = generate("square", nx=10, ny=10, bbox=(0, 1, 0, 1))
    bc = lambda x, y, t, nx, ny: ((1.0 + t) * np.ones_like(x), np.ones_like(x))
    s = PolygonalHeatSolver(*m, 0.1, 1e-2, bc_type="robin", bc_func=bc)
    s.solve(np.zeros(s.M), 0.0, 0.1)
    assert s.last_solve_stats["boundary_factorizations"] == 10


@pytest.mark.parametrize("scheme", ["backward_euler", "bdf2", "crank_nicolson"])
def test_polygonal_radiative_frozen_jacobian_matches_newton(scheme):
    u, s = _poly_run("radiative", scheme)
    u_newton, _ = _poly_run("radiative", scheme, nonlinear_options={"radiative_jacobian": "newton"})
    assert np.max(np.abs(u - u_newton)) <= 1e-9 * np.max(np.abs(u_newton))
    assert s.last_solve_stats["boundary_factorizations"] <= 4


@pytest.mark.parametrize("bc", ["robin", "radiative"])
@pytest.mark.parametrize("model", ["pennes", "fractional"])
def test_transport_boundary_solves_reuse_factorizations(model, bc):
    p = transport_problem(model, bc)
    m = generate("square", nx=12, ny=12, bbox=p["bbox"])
    make = {"pennes": lambda: ReactionDiffusionHeatSolver(*m, p["alpha"], 5e-3, p["perfusion"], bc_type=bc,
                                                          bc_func=p["bc_func"], source_func=p["source"]),
            "fractional": lambda: FractionalHeatSolver(*m, p["alpha"], 5e-3, p["beta"], bc_type=bc,
                                                       bc_func=p["bc_func"], source_func=p["source"])}[model]
    s = make()
    c = s.cell_centers
    u0 = p["exact"](c[:, 0], c[:, 1], 0.0)
    _, u = s.solve(u0, 0.0, 0.1)
    assert len(s._factor_cache._entries) == 1  # one factorization for the whole run
    assert np.sqrt(np.mean((u - p["exact"](c[:, 0], c[:, 1], 0.1)) ** 2)) < 1e-3


@pytest.mark.parametrize("bc", ["neumann", "flux", "robin", "radiative"])
def test_curvilinear_vectorized_rows_match_legacy_lil_rows(bc):
    p = heat_problem(bc)
    X, Y = generate("curvilinear", nx=13, ny=11, bbox=p["bbox"], warp=0.1).structured()
    s = CurvilinearHeatSolver(X, Y, alpha=p["alpha"], dt=1e-3, bc_type=bc, bc_func=p["bc_func"])
    K = (s.M_diag - 1e-3 * s.A).tocsr()
    T = 1.0 + 0.1 * np.random.default_rng(0).random(s.N)
    rhs_old, rhs_new = np.ones(s.N), np.ones(s.N)
    lil = K.copy().tolil()
    s._apply_flux_boundary_rows(lil, rhs_old, 0.01, temperature=T)
    old = lil.tocsr()
    new = s._with_boundary_rows(K, rhs_new, 0.01, temperature=T)
    assert np.array_equal(old.indptr, new.indptr) and np.array_equal(old.indices, new.indices)
    assert old.data.tobytes() == new.data.tobytes() and rhs_old.tobytes() == rhs_new.tobytes()


def test_curvilinear_radiative_frozen_jacobian_matches_newton():
    p = heat_problem("radiative")
    X, Y = generate("curvilinear", nx=17, ny=17, bbox=p["bbox"], warp=0.08).structured()
    out = []
    for mode in ("frozen", "newton"):
        s = CurvilinearHeatSolver(X, Y, alpha=p["alpha"], dt=2e-3, bc_type="radiative", bc_func=p["bc_func"],
                                  source_func=p["source"], nonlinear_options={"radiative_jacobian": mode})
        out.append(s.solve(p["exact"](X, Y, 0.0), 0.0, 0.05)[1])
    assert np.max(np.abs(out[0] - out[1])) <= 1e-9 * np.max(np.abs(out[1]))


def test_replace_rows_matches_lil_assignment():
    K = _system(50, seed=2)
    rows = np.array([0, 7, 30, 49])
    seg = [np.array([0, 3, 9]), np.array([7]), np.array([1, 2, 30, 44]), np.array([10, 49])]
    ptr = np.concatenate([[0], np.cumsum([len(c) for c in seg])])
    cols = np.concatenate(seg)
    vals = np.arange(1.0, cols.size + 1.0)
    lil = K.tolil()
    for r, c, k0, k1 in zip(rows, seg, ptr[:-1], ptr[1:]):
        lil.rows[r] = list(c)
        lil.data[r] = list(vals[k0:k1])
    ref, new = lil.tocsr(), replace_rows(K, rows, ptr, cols, vals)
    assert np.array_equal(ref.indptr, new.indptr) and np.array_equal(ref.indices, new.indices)
    assert np.array_equal(ref.data, new.data)


@pytest.mark.parametrize("make", [
    lambda: generate_hexagonal_polygonal_mesh(nx=9, ny=8, spacing=0.2, jitter=0.2, seed=3),
    lambda: generate_mixed_polygonal_mesh(nx_tiles=4, ny_tiles=3),
    lambda: generate_nonorthogonal_tiled_polygonal_mesh(nx_tiles=4, ny_tiles=3, skew=0.3),
    lambda: generate_perturbed_quad_mesh(nx=11, ny=9, amplitude=0.25, seed=4),
])
def test_generator_centroids_bit_identical_to_per_cell_builder(make):
    vertices, polygons, centers = make()[:3]
    legacy = np.array([polygon_area_and_centroid(np.asarray(vertices)[poly])[1] for poly in polygons])
    assert np.asarray(centers).tobytes() == legacy.tobytes()


@pytest.mark.parametrize("gen, kw", [("perturbed", dict(nx=12, ny=12, amplitude=0.3)),
                                     ("hexagonal", dict(nx=8, ny=8, jitter=0.2)),
                                     ("nonorthogonal_tiled", dict(nx_tiles=3, ny_tiles=3))])
def test_batched_gradient_stencils_bit_identical_to_loop(gen, kw):
    s = PolygonalHeatSolver(*generate(gen, **kw), 0.1, 1e-3, nonorthogonal_correction=True)
    new, old = s._build_gradient_reconstruction(), s._build_gradient_reconstruction_loop()
    for a, b in zip(new, old):
        assert list(a) == list(b)
        assert all(np.asarray(a[k]).tobytes() == np.asarray(b[k]).tobytes() for k in a)
