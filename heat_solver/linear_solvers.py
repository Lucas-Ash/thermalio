"""Linear-solver choice for the implicit steps: direct, Jacobi-preconditioned or multigrid Krylov.

``docs/performance_guide.md`` (section 3) measured which solver wins where:

* when the time step keeps the system mass-dominated (``dt ~ h^2``), CG with a
  Jacobi preconditioner converges in a mesh-independent number of iterations,
  needs no setup and beats a sparse LU above a few tens of thousands of
  unknowns (6x at 1M);
* with large steps (near-steady problems, Stefan problems) the Jacobi
  iteration count grows like ``1/h``; a sparse LU factorized once and reused
  over many steps is then fastest up to about a million unknowns, and
  algebraic multigrid (pyamg) beyond, or when the matrix changes every solve.

:func:`choose_linear_solver` encodes that with a calibrated cost model
(``linear_solver="auto"``);
:class:`IterativeSolver` runs the Krylov solves with a reused preconditioner
and, for Dirichlet systems, eliminates the identity rows' columns so the
matrix stays symmetric (CG needs that).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, bicgstab, cg, gmres

# Below this, a direct factorization is always used (nothing to gain, most robust).
DIRECT_MIN_UNKNOWNS = 20_000
# Above this, no direct factorization (memory: ~6 GB at 1M unknowns for 2D TPFA).
DIRECT_MAX_UNKNOWNS = 1_500_000
# Cost model for linear_solver="auto", in microseconds per unknown, calibrated on
# Thermalio's own Dirichlet systems (square meshes, 9k-1M cells; see
# docs/performance_guide.md).  Only the ratios matter for the choice.
COST = {
    "factor": 10.0,          # sparse LU at the reference size; grows like (N / ref)^0.5 per unknown
    "factor_ref": 262_144,
    "backsolve": 0.27,       # one pair of triangular solves
    "jacobi_iter": 0.02,     # one CG iteration with Jacobi
    "amg_setup": 2.3,        # smoothed-aggregation hierarchy
    "amg_iter": 0.18,        # one CG iteration with a V-cycle
}
KRYLOV_METHODS = ("cg", "bicgstab", "gmres")
PRECONDITIONERS = (None, "jacobi", "amg")


def amg_available():
    try:
        import pyamg  # noqa: F401
    except ImportError:
        return False
    return True


def is_symmetric(K, rtol=1e-12):
    K = csr_matrix(K)
    D = abs(K - K.T)
    return D.nnz == 0 or float(D.max()) <= rtol * float(abs(K).max())


def jacobi_condition_bound(K):
    """Gershgorin bound on the condition number of ``D^{-1} K`` (``inf`` if not diagonally dominant).

    With ``w = max_i sum_{j != i} |K_ij| / |K_ii|`` the eigenvalues of the
    Jacobi-scaled matrix lie in ``[1 - w, 1 + w]``.  For ``M + dt A`` with a
    diffusion operator ``A`` this is ``1 + 2 dt a_ii / m_i``: small when the mass
    term dominates, large for big steps.
    """
    K = csr_matrix(K)
    diag = np.abs(K.diagonal())
    off = np.asarray(abs(K).sum(axis=1)).ravel() - diag
    if np.any(diag == 0.0):
        return np.inf
    w = float(np.max(off / diag))
    return np.inf if w >= 1.0 else (1.0 + w) / (1.0 - w)


def estimated_costs(n, kappa, n_solves, matrix_changes=False):
    """Estimated seconds for ``n_solves`` solves with each strategy (see ``COST``)."""
    c = COST
    factor = 1e-6 * n * c["factor"] * (n / c["factor_ref"]) ** 0.5
    n_factor = n_solves if matrix_changes else 1
    jacobi_its = 2.5 * kappa**0.66 if np.isfinite(kappa) else np.inf
    amg_its = 1.5 + 1.5 * np.log10(max(kappa, 1.0)) if np.isfinite(kappa) else 10.0
    return {
        "direct": n_factor * factor + 1e-6 * n_solves * n * c["backsolve"],
        "jacobi": 1e-6 * n_solves * jacobi_its * n * c["jacobi_iter"],
        "amg": 1e-6 * n * c["amg_setup"] + 1e-6 * n_solves * amg_its * n * c["amg_iter"],
    }


def choose_linear_solver(K, options=None, n_solves=1, matrix_changes=False):
    """``(method, preconditioner, reason)`` for ``linear_solver="auto"``.

    Small systems use a direct factorization.  Otherwise the cheapest of
    direct (factorize once, or per solve when the matrix changes), Jacobi-CG
    and AMG-CG is picked from a calibrated cost model driven by the size, the
    Gershgorin condition bound of the Jacobi-scaled system and the number of
    solves.  Direct is ruled out above ``direct_max_unknowns`` (memory) and AMG
    without pyamg.  CG is used for symmetric systems, BiCGSTAB otherwise.
    """
    opts = options or {}
    n = K.shape[0]
    if n <= int(opts.get("direct_min_unknowns", DIRECT_MIN_UNKNOWNS)):
        return "direct", None, f"{n} unknowns <= direct_min_unknowns"
    kappa = jacobi_condition_bound(K)
    costs = estimated_costs(n, kappa, n_solves, matrix_changes)
    if n > int(opts.get("direct_max_unknowns", DIRECT_MAX_UNKNOWNS)):
        costs.pop("direct")
    if not amg_available():
        costs.pop("amg")
    choice = min(costs, key=costs.get)
    summary = ", ".join(f"{k} ~{v:.2g}s" for k, v in sorted(costs.items(), key=lambda kv: kv[1]))
    reason = f"N={n}, Jacobi condition bound {kappa:.3g}, {n_solves} solves: {summary}"
    if choice == "direct":
        return "direct", None, reason
    return ("cg" if is_symmetric(K) else "bicgstab"), choice, reason


def build_preconditioner(K, kind, symmetric=True):
    if kind is None:
        return None
    if kind == "jacobi":
        inv = 1.0 / K.diagonal()
        return LinearOperator(K.shape, matvec=lambda x: inv * x, dtype=float)
    if kind == "amg":
        import pyamg

        ml = pyamg.smoothed_aggregation_solver(csr_matrix(K), symmetry="hermitian" if symmetric else "nonsymmetric")
        return ml.aspreconditioner(cycle="V")
    raise ValueError(f"unknown preconditioner {kind!r}; use None, 'jacobi' or 'amg'.")


def eliminate_identity_rows(K, mask):
    """Split ``K`` (identity rows on ``mask``) into ``K_elim`` and the moved columns ``K_IB``.

    ``K x = b`` and ``K_elim x = b - K_IB (b * mask)`` have the same solution;
    ``K_elim`` is symmetric when the interior block is.
    """
    K = csr_matrix(K)
    row_of = np.repeat(np.arange(K.shape[0]), np.diff(K.indptr))
    move = mask[K.indices] & ~mask[row_of]
    # separate index arrays: eliminate_zeros() compacts them in place
    K_ib = csr_matrix((np.where(move, K.data, 0.0), K.indices.copy(), K.indptr.copy()), shape=K.shape)
    K_el = csr_matrix((np.where(move, 0.0, K.data), K.indices.copy(), K.indptr.copy()), shape=K.shape)
    K_ib.eliminate_zeros()
    K_el.eliminate_zeros()
    return K_el, K_ib


class IterativeSolver:
    """Preconditioned Krylov solves with preconditioners reused per key.

    A preconditioner built for one matrix is kept for later solves under the
    same key (later steps, nonlinear iterations, ``solve()`` calls); the Krylov
    method always solves the current matrix to the requested tolerance, so a
    stale preconditioner only costs iterations.  It is rebuilt when the
    iteration count exceeds twice the count right after it was built.
    """

    def __init__(self, method, preconditioner, options=None):
        if method not in KRYLOV_METHODS:
            raise ValueError(f"unknown Krylov method {method!r}.")
        if preconditioner not in PRECONDITIONERS:
            raise ValueError(f"unknown preconditioner {preconditioner!r}; use None, 'jacobi' or 'amg'.")
        self.method = method
        self.preconditioner = preconditioner
        opts = options or {}
        self.rtol = float(opts.get("rtol", 1e-12))
        self.atol = float(opts.get("atol", 0.0))
        self.maxiter = None if opts.get("maxiter") is None else int(opts["maxiter"])
        self._entries = {}
        self.stats = {"solves": 0, "iterations": 0, "preconditioner_builds": 0}

    def solve(self, K, b, x0=None, key=None, dirichlet_mask=None):
        entry = self._entries.get(key)
        if entry is None or entry["K"] is not K:
            if dirichlet_mask is not None and np.any(dirichlet_mask):
                K_sys, K_ib = eliminate_identity_rows(K, dirichlet_mask)
            else:
                K_sys, K_ib = csr_matrix(K), None
            M, base = (None, None) if entry is None or entry["stale"] else (entry["M"], entry["base_iters"])
            entry = {"K": K, "K_sys": K_sys, "K_ib": K_ib, "M": M, "base_iters": base, "stale": False}
            self._entries[key] = entry
        if entry["M"] is None and self.preconditioner is not None:
            entry["M"] = build_preconditioner(entry["K_sys"], self.preconditioner, symmetric=self.method == "cg")
            entry["base_iters"] = None
            self.stats["preconditioner_builds"] += 1
        rhs = b if entry["K_ib"] is None else b - entry["K_ib"] @ (b * dirichlet_mask)
        x, iters = self._krylov(entry["K_sys"], rhs, x0, entry["M"])
        if entry["base_iters"] is None:
            entry["base_iters"] = iters
        elif iters > 2 * entry["base_iters"] + 5:
            entry["stale"] = True  # rebuild the preconditioner for the next solve
        self.stats["solves"] += 1
        self.stats["iterations"] += iters
        return x

    def _krylov(self, K, b, x0, M):
        count = [0]

        def callback(*_):
            count[0] += 1

        kw = dict(x0=x0, rtol=self.rtol, atol=self.atol, maxiter=self.maxiter, M=M, callback=callback)
        if self.method == "cg":
            x, info = cg(K, b, **kw)
        elif self.method == "bicgstab":
            x, info = bicgstab(K, b, **kw)
        else:
            x, info = gmres(K, b, callback_type="pr_norm", **kw)
        if info != 0:
            raise RuntimeError(
                f"Iterative linear solver {self.method!r} (preconditioner {self.preconditioner!r}) did not "
                f"converge (SciPy info {info}). Try preconditioner='amg', a looser rtol, or linear_solver='direct'.")
        return x, count[0]
