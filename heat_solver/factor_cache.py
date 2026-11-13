"""Reusing sparse factorizations when boundary conditions make the matrix look time-dependent.

With Robin or radiative boundaries the implicit matrix is ``base + coef * B``,
where ``B`` is the boundary closure.  The solvers used to refactorize it at
every step (and every nonlinear iteration), which dominated run time.

* :func:`spsolve_factor` factorizes ``K`` so that each solve reproduces
  ``scipy.sparse.linalg.spsolve(K, b)`` bit for bit (``spsolve`` on a CSR
  matrix factorizes the transpose and solves transposed; this does the same).
* :class:`FactorCache` keeps one factorization per key and reuses it while
  the operators it was built from are the same objects and ``B`` has the same
  values.  For Robin data with time-independent ``beta`` the matrix never
  changes, so results are identical to refactorizing every step.
* :func:`frozen_newton` solves the radiative boundary nonlinearity with a
  frozen Jacobian: the factorization is reused across iterations and steps
  and refreshed only when the iteration contracts slowly.  It converges to the
  same fixed point as Newton's method, to the nonlinear tolerance.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, issparse
from scipy.sparse.linalg import splu


def spsolve_factor(K):
    """Return ``solve`` with ``solve(b) == spsolve(K, b)`` bit for bit."""
    if not issparse(K) or K.format not in ("csr", "csc"):
        K = csc_matrix(K)
    K = K.copy()
    K.sum_duplicates()  # spsolve canonicalizes its input the same way
    K = K.astype(float)
    if K.format == "csc":
        lu = splu(K)
        return lambda b: lu.solve(np.asarray(b, dtype=float))
    # CSR data read as CSC is K^T: factorize it and solve the transposed system.
    lu = splu(csc_matrix((K.data, K.indices, K.indptr), shape=K.shape[::-1]))
    return lambda b: lu.solve(np.asarray(b, dtype=float), trans="T")


_ANY = object()  # lookup() sentinel: accept whatever B the stored factorization was built with


def _canonical(B):
    if B is None:
        return None
    B = csr_matrix(B, copy=True)
    B.sum_duplicates()
    return B


def _same_values(B1, B2):
    if B1 is None or B2 is None:
        return B1 is None and B2 is None
    return (B1.shape == B2.shape and np.array_equal(B1.indptr, B2.indptr)
            and np.array_equal(B1.indices, B2.indices) and np.array_equal(B1.data, B2.data))


class FactorCache:
    """Factorizations of matrices ``build()`` reused while their ingredients are unchanged.

    ``deps`` are objects compared by identity (the base operator, the mass
    matrix); ``B`` (sparse or ``None``) is compared by value.
    """

    def __init__(self):
        self._entries = {}

    def clear(self):
        self._entries.clear()

    def lookup(self, key, deps, B=None):
        entry = self._entries.get(key)
        if entry is None:
            return None
        old_deps, old_B, solve = entry
        if len(old_deps) != len(deps) or any(a is not b for a, b in zip(old_deps, deps)):
            return None
        if B is not _ANY and not _same_values(old_B, _canonical(B)):
            return None
        return solve

    def store(self, key, deps, B, build):
        solve = spsolve_factor(build())
        self._entries[key] = (tuple(deps), _canonical(B), solve)
        return solve

    def factor(self, key, deps, B, build):
        """Factorization of ``build()``; rebuilt only when ``deps`` or the values of ``B`` change."""
        solve = self.lookup(key, deps, B)
        return solve if solve is not None else self.store(key, deps, B, build)



def frozen_newton(u0, boundary, apply_base, rhs, coef, build_jacobian, cache, key, deps,
                  tol, max_iters, max_rate=0.5):
    """Solve ``base u + coef (B(u) u - b(u)) = rhs`` with a frozen Jacobian.

    ``boundary(v) -> (B, b)`` is the boundary closure linearized about ``v``
    (``B(v) v - b(v)`` is the exact nonlinear boundary term), ``apply_base(v)``
    returns ``base @ v`` and ``build_jacobian(B)`` the matrix
    ``base + coef * B`` to factorize.  The factorization stored under ``key``
    is reused across calls while ``deps`` are unchanged; it is refreshed at the
    current iterate when an update is not at least ``max_rate`` times smaller
    than the previous one.  Stops when the update is below
    ``tol * max(1, max|u|)``; with a contraction rate below ``max_rate <= 1/2``
    the remaining error is below that bound too.
    Returns ``(u, iterations, factorizations)``.
    """
    u = np.array(u0, dtype=float)
    solve = cache.lookup(key, deps, _ANY)
    prev = None
    n_factor = 0
    for it in range(1, max_iters + 1):
        B, b = boundary(u)
        if solve is None:
            solve = cache.store(key, deps, B, lambda: build_jacobian(B))
            n_factor += 1
        residual = apply_base(u) + coef * (B @ u - b) - rhs
        du = solve(residual)
        u = u - du
        err = float(np.max(np.abs(du)))
        if err <= tol * max(1.0, float(np.max(np.abs(u)))):
            return u, it, n_factor
        if prev is not None and err > max_rate * prev:
            solve = None  # slow contraction: refresh the Jacobian at the next iterate
            prev = None
        else:
            prev = err
    raise RuntimeError("Radiative boundary iteration did not converge; reduce dt or raise max_iters.")
