"""Algebraic flux correction (FCT) for high-resolution *and* monotone diffusion.

Direction B, PR2.  The linear M-matrix projection in :mod:`heat_solver.dmp`
restores the discrete maximum principle but is only first order (and reverts to
the inconsistent two-point behaviour on K-non-orthogonal meshes).  This module
adds a *nonlinear* scheme that is bound-preserving **and** recovers the accuracy
of the high-order (reconstructed / nonorthogonal-corrected) flux in smooth
regions, following Kuzmin's algebraic flux correction (a Flux-Corrected Transport
generalisation):

* the high-order operator ``A_H`` (consistent but not an M-matrix) and the
  low-order operator ``A_L = make_monotone(A_H)`` (an M-matrix, bound-preserving)
  differ by a symmetric artificial diffusion ``D = A_L - A_H``;
* ``D`` is decomposed into antisymmetric edge fluxes ``f_ij = d_ij (u_i - u_j)``
  whose sum recovers ``A_H`` from ``A_L``;
* a node-based **Zalesak limiter** scales each edge flux by ``alpha_ij in [0, 1]``
  so that adding it back never pushes a node outside the local bounds of the
  previous time level -- full correction (``alpha = 1 -> A_H``) where the
  solution is smooth, no correction (``alpha = 0 -> A_L``) near steep fronts.

The limited anti-diffusion is treated by a fixed-point (Picard) iteration on the
right-hand side, so the implicit operator stays the monotone M-matrix
``M + dt A_L``.  Dirichlet boundary data only (first increment).

References: Kuzmin (2009), "Explicit and implicit FEM-FCT algorithms with flux
linearization"; Zalesak (1979).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import factorized

from .dmp import make_monotone
from .polygonal import PolygonalHeatSolver, _apply_dirichlet_rows_csr


class AFCMonotoneSolver:
    """Bound-preserving high-resolution diffusion via algebraic flux correction.

    Parameters mirror the diffusion subset of :class:`PolygonalHeatSolver`
    (Dirichlet only).  ``flux_discretization`` selects the high-order target
    operator (``"reconstructed"`` by default, or ``"tpfa"`` with the
    nonorthogonal correction).
    """

    def __init__(
        self,
        vertices,
        polygons,
        alpha,
        dt,
        bc_func=None,
        source_func=None,
        flux_discretization="reconstructed",
        nonorthogonal_correction=True,
        smoothness_factor=0.0,
        max_iters=50,
        tol=1e-10,
    ):
        self._base = PolygonalHeatSolver(
            vertices, polygons, alpha, dt, bc_type="dirichlet",
            bc_func=bc_func, source_func=source_func,
            flux_discretization=flux_discretization,
            nonorthogonal_correction=nonorthogonal_correction,
        )
        self.dt = float(dt)
        self.M = self._base.M
        self.cell_centers = self._base.cell_centers
        self.cell_areas = self._base.cell_areas
        self.is_boundary = self._base.is_boundary
        self._boundary_idx = np.where(self.is_boundary)[0]
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        # Venkatakrishnan-style smoothness relaxation: a mesh-vanishing tolerance
        # eps ~ factor * U_ref * h^1.5 lets small (smooth) anti-diffusive fluxes
        # pass unlimited so smooth extrema are not clipped, while O(1) front jumps
        # are still strictly limited.  factor=0 recovers the strict PR2 limiter.
        self.smoothness_factor = float(smoothness_factor)
        self._h = np.sqrt(self.cell_areas)
        self._eps = np.zeros(self.M)

        self._base._assemble_system()
        self.A_H = self._base.A.tocsr()
        self.A_L = make_monotone(self.A_H).tocsr()
        self._build_edges()

    def _build_edges(self):
        """Symmetric artificial-diffusion edge coefficients d_ij = max(0, a_ij, a_ji)."""
        coo = self.A_H.tocoo()
        offdiag = {}
        for i, j, v in zip(coo.row, coo.col, coo.data):
            if i != j and v != 0.0:
                offdiag[(int(i), int(j))] = float(v)
        edges_i, edges_j, d = [], [], []
        for (i, j) in {(min(a, b), max(a, b)) for (a, b) in offdiag}:
            dij = max(0.0, offdiag.get((i, j), 0.0), offdiag.get((j, i), 0.0))
            if dij > 0.0:
                edges_i.append(i)
                edges_j.append(j)
                d.append(dij)
        self._edge_i = np.asarray(edges_i, dtype=np.intp)
        self._edge_j = np.asarray(edges_j, dtype=np.intp)
        self._edge_d = np.asarray(d, dtype=float)
        # Neighbour adjacency (for local bounds), boundary-inclusive.
        self._neighbors = [[] for _ in range(self.M)]
        for i, j in zip(self._edge_i, self._edge_j):
            self._neighbors[i].append(j)
            self._neighbors[j].append(i)

    def _local_bounds(self, u_ref):
        """Per-node [umin, umax] over the node and its edge-neighbours."""
        umin = u_ref.copy()
        umax = u_ref.copy()
        for i in range(self.M):
            nb = self._neighbors[i]
            if nb:
                vals = u_ref[nb]
                umin[i] = min(umin[i], vals.min())
                umax[i] = max(umax[i], vals.max())
        return umin, umax

    def _zalesak_alpha(self, u_iter, umin, umax):
        """Edge limiter coefficients alpha_ij in [0, 1] (Zalesak)."""
        ei, ej, dd = self._edge_i, self._edge_j, self._edge_d
        f = dd * (u_iter[ei] - u_iter[ej])  # antisymmetric raw anti-diffusive flux

        # Prelimiting: drop fluxes that are locally diffusive (would smear extrema).
        prelimit = f * (u_iter[ej] - u_iter[ei]) > 0.0
        f = np.where(prelimit, 0.0, f)

        P_plus = np.zeros(self.M)
        P_minus = np.zeros(self.M)
        np.add.at(P_plus, ei, np.maximum(0.0, f))
        np.add.at(P_plus, ej, np.maximum(0.0, -f))
        np.add.at(P_minus, ei, np.minimum(0.0, f))
        np.add.at(P_minus, ej, np.minimum(0.0, -f))

        scale = self.cell_areas / self.dt
        # Relax the bounds by the (mesh-vanishing) smoothness tolerance so that
        # smooth extrema are not clipped; eps == 0 gives the strict limiter.
        Q_plus = scale * np.maximum(umax - u_iter, self._eps)
        Q_minus = scale * np.minimum(umin - u_iter, -self._eps)

        with np.errstate(divide="ignore", invalid="ignore"):
            R_plus = np.where(P_plus > 0.0, np.minimum(1.0, Q_plus / P_plus), 1.0)
            R_minus = np.where(P_minus < 0.0, np.minimum(1.0, Q_minus / P_minus), 1.0)
        R_plus = np.clip(R_plus, 0.0, 1.0)
        R_minus = np.clip(R_minus, 0.0, 1.0)

        alpha = np.where(
            f > 0.0,
            np.minimum(R_plus[ei], R_minus[ej]),
            np.minimum(R_minus[ei], R_plus[ej]),
        )
        alpha = np.where(f == 0.0, 0.0, alpha)
        return alpha * f  # limited edge fluxes

    def _scatter_edges(self, edge_flux):
        F = np.zeros(self.M)
        np.add.at(F, self._edge_i, edge_flux)
        np.add.at(F, self._edge_j, -edge_flux)
        return F

    def solve(self, u0, t0, t_end):
        u = np.array(u0, dtype=float)
        nsteps = max(int(round((t_end - t0) / self.dt)), 1)
        dt = (t_end - t0) / nsteps

        # Smoothness tolerance from the solution scale (range of the initial data).
        if self.smoothness_factor > 0.0:
            u_ref = max(float(u.max() - u.min()), abs(float(u.max())), 1e-300)
            self._eps = self.smoothness_factor * u_ref * self._h**1.5
        else:
            self._eps = np.zeros(self.M)

        sys_L = (diags(self.cell_areas, format="csr") + dt * self.A_L).tocsr()
        _apply_dirichlet_rows_csr(sys_L, self._boundary_idx)
        factor = factorized(sys_L.tocsc())

        cx, cy = self.cell_centers[:, 0], self.cell_centers[:, 1]
        t = float(t0)
        for _ in range(nsteps):
            t_next = t + dt
            src = np.broadcast_to(np.asarray(self._base.source_func(cx, cy, t_next), dtype=float), (self.M,))
            bc = np.broadcast_to(np.asarray(self._base.bc_func(cx, cy, t_next), dtype=float), (self.M,))
            rhs_base = self.cell_areas * u + dt * (self.cell_areas * src)
            rhs_base[self._boundary_idx] = bc[self._boundary_idx]

            # Bounds from the previous time level's local neighbourhood.
            umin, umax = self._local_bounds(u)
            # Include the Dirichlet data so boundary-adjacent nodes are not over-limited.
            umin[self._boundary_idx] = np.minimum(umin[self._boundary_idx], bc[self._boundary_idx])
            umax[self._boundary_idx] = np.maximum(umax[self._boundary_idx], bc[self._boundary_idx])

            u_iter = factor(rhs_base)  # low-order predictor
            for _ in range(self.max_iters):
                limited = self._zalesak_alpha(u_iter, umin, umax)
                rhs = rhs_base + dt * self._scatter_edges(limited)
                rhs[self._boundary_idx] = bc[self._boundary_idx]
                u_new = factor(rhs)
                err = np.max(np.abs(u_new - u_iter))
                u_iter = u_new
                if err <= self.tol * max(1.0, np.max(np.abs(u_new))):
                    break
            u = u_iter
            t = t_next
        return t, u
