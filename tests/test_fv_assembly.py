"""Vectorized diffusion assembly, Dirichlet rows and operator caching (performance item 1).

The vectorized assembly must reproduce the legacy per-face loop: same sparsity
pattern and values to round-off (the legacy 2-vector BLAS dot products may use
fused multiply-adds), and bit for bit where the arithmetic is exact.
"""

import math

import numpy as np
import pytest
from scipy.sparse import identity as sparse_identity
from scipy.sparse import random as sparse_random

from heat_solver.fv_assembly import identity_rows
from heat_solver.mesh_library import generate
from heat_solver.polygonal import PolygonalHeatSolver, _apply_dirichlet_rows_csr
from heat_solver.triangular import NonUniformHeatSolver

MESHES = {
    "square": lambda: generate("square", nx=10, ny=10),
    "hexagonal": lambda: generate("hexagonal", nx=6, ny=6, jitter=0.1),
    "perturbed": lambda: generate("perturbed", nx=8, ny=8),
    "mixed": lambda: generate("mixed", nx_tiles=3, ny_tiles=3),
    "skewed": lambda: generate("nonorthogonal", nx=8, ny=8, skew=0.3),
    "tiled": lambda: generate("nonorthogonal_tiled", nx_tiles=3, ny_tiles=3),
}
ALPHAS = {
    "scalar": 0.1,
    "tensor": np.array([[0.2, 0.06], [0.06, 0.05]]),
    "field": lambda x, y: 0.1 + 0.05 * np.sin(3 * x) * np.cos(2 * y),
    "tensor_field": lambda x, y: np.stack(
        [np.stack([0.1 + 0.02 * x * x, 0.01 + 0 * x], -1), np.stack([0.01 + 0 * x, 0.05 + 0.01 * y * y], -1)], -2),
}


def _assert_same_operator(A_new, A_old):
    A_new, A_old = A_new.tocsr(), A_old.tocsr()
    scale = abs(A_old).max()
    assert abs(A_new - A_old).max() <= 1e-14 * scale
    # Same stored pattern, except entries that cancel to round-off in one of the two sums.
    diff = ((A_new != 0).astype(int) - (A_old != 0).astype(int)).tocoo()
    for r, c in zip(diff.row, diff.col):
        assert max(abs(A_new[r, c]), abs(A_old[r, c])) <= 1e-14 * scale


@pytest.mark.parametrize("alpha", list(ALPHAS))
@pytest.mark.parametrize("mesh", list(MESHES))
def test_vectorized_assembly_matches_legacy_loop(mesh, alpha):
    m = MESHES[mesh]()
    for corr in (False, True):
        s = PolygonalHeatSolver(*m, ALPHAS[alpha], 1e-3, nonorthogonal_correction=corr)
        s._assemble_system()
        _assert_same_operator(s.A, s._assemble_tpfa_loop())
    s = PolygonalHeatSolver(*m, ALPHAS[alpha], 1e-3, flux_discretization="reconstructed")
    s._assemble_system()
    _assert_same_operator(s.A, s._assemble_diffusion_reconstructed_loop())


def test_vectorized_assembly_is_bit_identical_when_arithmetic_is_exact():
    m = generate("square", nx=16, ny=16)  # h = 1/8: every geometric quantity is exact
    for alpha in (0.25, np.array([[0.5, 0.125], [0.125, 0.25]])):
        s = PolygonalHeatSolver(*m, alpha, 1e-3, nonorthogonal_correction=True)
        s._assemble_system()
        old = s._assemble_tpfa_loop()
        assert (s.A != old).nnz == 0 and s.A.nnz == old.nnz


def test_temperature_dependent_tensor_assembly_matches_loop():
    m = MESHES["skewed"]()
    alpha = lambda x, y, T: np.stack([np.stack([0.1 + 0.05 * T, 0 * T], -1), np.stack([0 * T, 0.1 + 0.02 * T * T], -1)], -2)
    s = PolygonalHeatSolver(*m, alpha, 1e-3, nonorthogonal_correction=True)
    T = np.random.default_rng(0).uniform(0, 1, s.M)
    s._assemble_system(temperature=T)
    _assert_same_operator(s.A, s._assemble_tpfa_loop(temperature=T))


def test_scalar_only_alpha_callable_falls_back_to_pointwise_evaluation():
    m = MESHES["perturbed"]()
    alpha = lambda x, y: 0.1 + 0.05 * math.sin(float(x))  # math.sin rejects arrays
    s = PolygonalHeatSolver(*m, alpha, 1e-3)
    s._assemble_system()
    _assert_same_operator(s.A, s._assemble_tpfa_loop())


def test_identity_rows_matches_lil_construction():
    rng = np.random.default_rng(3)
    K = (sparse_random(60, 60, density=0.1, random_state=4) + sparse_identity(60)).tocsr()
    mask = rng.random(60) < 0.3
    lil = K.tolil()
    for i in np.flatnonzero(mask):
        lil.rows[i] = [i]
        lil.data[i] = [1.0]
    ref = lil.tocsr()
    new = identity_rows(K, mask)
    assert np.array_equal(new.indptr, ref.indptr) and np.array_equal(new.indices, ref.indices)
    assert np.array_equal(new.data, ref.data)


def test_inplace_dirichlet_rows_match_legacy_loop():
    K = (sparse_random(40, 40, density=0.15, random_state=5) + sparse_identity(40)).tocsr()
    rows = np.array([0, 3, 17, 39])
    ref = K.copy()
    for i in rows:
        start, end = ref.indptr[i], ref.indptr[i + 1]
        ref.data[start:end] = 0.0
        ref.data[start + np.searchsorted(ref.indices[start:end], i)] = 1.0
    new = K.copy()
    _apply_dirichlet_rows_csr(new, rows)
    assert np.array_equal(new.data, ref.data) and np.array_equal(new.indices, ref.indices)


def test_solve_reuses_operator_and_factorization_across_calls(monkeypatch):
    m = generate("square", nx=12, ny=12)
    ex = lambda x, y, t: np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y)
    s = PolygonalHeatSolver(*m, 0.1, 1e-3, bc_func=ex)
    c = s.cell_centers
    u0 = ex(c[:, 0], c[:, 1], 0.0)
    _, u_first = s.solve(u0, 0.0, 0.01)
    calls = []
    original = s._assemble_system
    monkeypatch.setattr(s, "_assemble_system", lambda *a, **k: (calls.append(1), original(*a, **k)))
    _, u_again = s.solve(u0, 0.0, 0.01)
    assert not calls  # operator and factorization reused
    assert np.array_equal(u_first, u_again)
    s.A = s.A.copy()  # any replacement of the operator invalidates the cache
    s.solve(u0, 0.0, 0.01)
    assert calls
    fresh = PolygonalHeatSolver(*m, 0.1, 1e-3, bc_func=ex)
    assert np.array_equal(fresh.solve(u0, 0.0, 0.01)[1], u_first)


def test_triangular_dirichlet_data_scalar_only_callback():
    p, tri = generate("delaunay", nx=8, ny=8).triangulation()
    vec = lambda x, y, t: 1.0 + 0.5 * x * y
    scal = lambda x, y, t: 1.0 + 0.5 * float(x) * float(y)  # float() rejects arrays
    u = [NonUniformHeatSolver(p, tri, 0.1, 1e-2, bc_func=f).solve(np.zeros(len(p)), 0.0, 0.05)[1] for f in (vec, scal)]
    assert np.array_equal(u[0], u[1])
