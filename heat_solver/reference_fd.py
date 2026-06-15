"""An independent finite-difference reference solver for cross-code validation.

This is a deliberately separate, vanilla implementation of the constant-
coefficient Dirichlet heat equation

    u_t - alpha * laplacian(u) = Q(x, y, t)

on a uniform Cartesian grid, using a 5-point Laplacian and backward-Euler time
stepping.  It shares **no code** with the finite-volume machinery in
``heat_solver`` (no polygonal assembly, no flux schemes), so agreement between
this solver and the FV solvers is a genuine cross-implementation ("N-version
against an independent code") check -- the in-repo, dependency-free stand-in for
validating against an external package.

Scope: scalar constant ``alpha``, Dirichlet boundary data, rectangular domain.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import factorized


def solve_fd_reference(bbox, n, alpha, dt, t_end, solution, source, bc_func=None, t_init=0.0):
    """Solve ``u_t - alpha lap u = Q`` on an ``(n+1) x (n+1)`` uniform grid.

    Parameters
    ----------
    bbox : (xmin, xmax, ymin, ymax)
    n : int
        Number of cells per side (grid has ``n+1`` nodes per side).
    alpha : float
        Constant scalar diffusivity.
    solution : callable ``(x, y, t) -> u``
        Used for the initial condition and (if ``bc_func`` is None) Dirichlet data.
    source : callable ``(x, y, t) -> Q``.
    bc_func : optional callable ``(x, y, t) -> u`` for Dirichlet data.

    Returns
    -------
    (X, Y, u) : meshgrids and the final-time solution on the full node grid.
    """
    xmin, xmax, ymin, ymax = bbox
    bc_func = bc_func if bc_func is not None else solution
    xs = np.linspace(xmin, xmax, n + 1)
    ys = np.linspace(ymin, ymax, n + 1)
    hx = (xmax - xmin) / n
    hy = (ymax - ymin) / n
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    nint = n - 1  # interior nodes per side
    ninterior = nint * nint

    def idx(i, j):  # interior index for node (i, j), 1 <= i, j <= n-1
        return (j - 1) * nint + (i - 1)

    # Assemble the interior Laplacian operator L (so that -alpha L approximates
    # -alpha laplacian) as a sparse matrix.
    rows, cols, data = [], [], []
    cx = alpha / hx**2
    cy = alpha / hy**2
    for j in range(1, n):
        for i in range(1, n):
            k = idx(i, j)
            rows.append(k); cols.append(k); data.append(2.0 * cx + 2.0 * cy)
            for (ii, jj, c) in ((i - 1, j, cx), (i + 1, j, cx), (i, j - 1, cy), (i, j + 1, cy)):
                if 1 <= ii <= n - 1 and 1 <= jj <= n - 1:
                    rows.append(k); cols.append(idx(ii, jj)); data.append(-c)
    A = csr_matrix((data, (rows, cols)), shape=(ninterior, ninterior))

    lhs = (eye(ninterior, format="csr") / dt + A).tocsc()
    solve = factorized(lhs)

    # Initial condition.
    u = solution(X, Y, t_init).astype(float).copy()

    nsteps = max(int(round((t_end - t_init) / dt)), 1)
    dt_eff = (t_end - t_init) / nsteps
    if abs(dt_eff - dt) > 1e-12:
        # Re-factor for the adjusted uniform step.
        lhs = (eye(ninterior, format="csr") / dt_eff + A).tocsc()
        solve = factorized(lhs)

    interior_x = X[1:n, 1:n]
    interior_y = Y[1:n, 1:n]
    t = t_init
    for _ in range(nsteps):
        t_next = t + dt_eff
        # Apply Dirichlet boundary values on the full grid for this step.
        bc = bc_func(X, Y, t_next)
        u[0, :] = bc[0, :]
        u[-1, :] = bc[-1, :]
        u[:, 0] = bc[:, 0]
        u[:, -1] = bc[:, -1]

        rhs = (u[1:n, 1:n] / dt_eff + source(interior_x, interior_y, t_next)).ravel()
        # Move known boundary contributions to the RHS (5-point neighbours of the
        # interior ring that fall on the boundary).
        rhs_grid = np.zeros((nint, nint))
        rhs_grid += rhs.reshape(nint, nint)
        # left/right boundaries (i=1 uses x=0 column; i=n-1 uses x=n column)
        rhs_grid[:, 0] += (alpha / hx**2) * u[1:n, 0]
        rhs_grid[:, -1] += (alpha / hx**2) * u[1:n, n]
        rhs_grid[0, :] += (alpha / hy**2) * u[0, 1:n]
        rhs_grid[-1, :] += (alpha / hy**2) * u[n, 1:n]

        u_int = solve(rhs_grid.ravel())
        u[1:n, 1:n] = u_int.reshape(nint, nint)
        t = t_next

    # Final boundary refresh for consistency.
    bc = bc_func(X, Y, t)
    u[0, :] = bc[0, :]
    u[-1, :] = bc[-1, :]
    u[:, 0] = bc[:, 0]
    u[:, -1] = bc[:, -1]
    return X, Y, u


def relative_l2(u, u_ref):
    """Unweighted relative L2 difference of two equal-shaped fields."""
    u = np.asarray(u, dtype=float)
    u_ref = np.asarray(u_ref, dtype=float)
    return float(np.sqrt(np.sum((u - u_ref) ** 2)) / (np.sqrt(np.sum(u_ref**2)) + 1e-300))
