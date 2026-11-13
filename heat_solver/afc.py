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
``M + dt A_L`` (plus the boundary closure).  All boundary types are supported.

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
    _LIMITERS = ("local", "global", "source_envelope", "source_local")
    """Bound-preserving high-resolution diffusion via algebraic flux correction.

    Parameters mirror the diffusion subset of :class:`PolygonalHeatSolver`.
    ``bc_type`` accepts every boundary type of
    :mod:`heat_solver.boundary_conditions`; the (diagonal, non-negative)
    boundary closure is added to both operators, so the low-order operator
    stays an M-matrix.  ``flux_discretization`` selects the high-order target
    operator (``"reconstructed"`` by default, or ``"tpfa"`` with the
    nonorthogonal correction).

    ``limiter``:

    - ``"local"`` (default): Zalesak limiter against local neighbour bounds.
      High resolution, but clips smooth extrema unless ``smoothness_factor > 0``
      (a Venkatakrishnan-style, essentially-non-oscillatory relaxation).
    - ``"global"``: enforce the physical maximum-principle bounds ``[m, M]`` for
      every node (Zhang--Shu style).  Strictly bound-preserving even at coarse
      resolution AND it does not clip smooth interior extrema (which lie inside
      ``[m, M]``), so it recovers high-order accuracy on smooth solutions.  The
      bounds are ``bounds=(m, M)`` if given, else auto-detected from the initial
      data and the Dirichlet boundary samples.  This is rigorous when the
      continuous solution obeys a maximum principle (e.g. source-free or
      sign-definite forcing).  With a sign-changing source the true solution
      can leave the auto-detected ``[m, M]`` (a heat sink drives it below the
      initial/boundary minimum), and this limiter would clip physics.
    - ``"source_envelope"``: rigorous *time-dependent* global bounds for
      problems with sign-changing sources (Direction B, PR5).  Starting from
      ``[m^0, M^0]`` (initial + boundary data, or ``bounds``) they evolve as::

          M^{n+1} = max(M^n + dt * max_i q_i^+, max g^{n+1})
          m^{n+1} = min(m^n - dt * max_i q_i^-, min g^{n+1})

      which is the discrete comparison principle of the low-order M-matrix
      scheme (spatially constant super/sub-solutions).  The continuous
      solution obeys the analogous bound, so the limiter never clips the
      physical solution; with ``q = 0`` it reduces to ``"global"``.
    - ``"source_local"``: local bounds taken from the *source-including*
      low-order solution ``u^L`` at the new time level, ``[min, max]`` of
      ``u^L`` over each node's stencil.  Source-driven extrema appear in
      ``u^L`` and are therefore admissible, while the high-order correction
      cannot create extrema beyond what the monotone scheme produces.  Combine
      with ``smoothness_factor > 0`` to avoid clipping smooth extrema.
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
        limiter="local",
        bounds=None,
        max_iters=50,
        tol=1e-10,
        bc_type="dirichlet",
        dirichlet_mode="band",
    ):
        from .boundary_conditions import normalize_bc_type

        self.bc_type = normalize_bc_type(bc_type)
        self._base = PolygonalHeatSolver(
            vertices, polygons, alpha, dt, bc_type=self.bc_type,
            bc_func=bc_func, source_func=source_func,
            flux_discretization=flux_discretization,
            nonorthogonal_correction=nonorthogonal_correction,
            dirichlet_mode=dirichlet_mode,
        )
        self.dirichlet_mode = self._base.dirichlet_mode
        # band: boundary cells held at the data; face: data on the boundary faces
        # through the (diagonal, non-negative) closure alpha_n (g - u_c) / d.
        self._band_dirichlet = self._base._band_dirichlet
        self._face_dirichlet = self.bc_type == "dirichlet" and not self._band_dirichlet
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
        # Limiter bounds: "local" uses neighbour min/max (high resolution but
        # clips smooth extrema unless smoothness_factor > 0); "global" enforces
        # the physical maximum-principle bounds [m, M] for every node, which is
        # strictly bound-preserving AND does not clip smooth interior extrema.
        self.limiter = str(limiter).lower().strip()
        if self.limiter not in self._LIMITERS:
            raise ValueError(f"limiter must be one of {self._LIMITERS}.")
        self._user_bounds = None if bounds is None else (float(bounds[0]), float(bounds[1]))

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

    def _ambient_range(self, t):
        """Ambient temperatures of Robin (``value / beta``), radiative (``T_inf``) or face-Dirichlet (``g``) boundaries.

        By the comparison principle these bound the solution together with the
        initial data, so they enter the global and envelope bounds.
        """
        if self.bc_type not in ("robin", "radiative") and not self._face_dirichlet:
            return None
        from .boundary_conditions import parse_radiative, parse_robin

        faces = self._base.boundary_faces
        mid, nrm = faces["midpoints"], faces["normals"]
        if mid.size == 0:
            return None
        data = self._base._evaluate_bc(mid[:, 0], mid[:, 1], t, normals=nrm)
        if self._face_dirichlet:  # face data play the role of the ambient temperature
            amb = np.broadcast_to(np.asarray(data, dtype=float), (mid.shape[0],))
        elif self.bc_type == "robin":
            beta, value = parse_robin(data, mid.shape[0])
            amb = value[beta > 0] / beta[beta > 0]
        else:
            amb = parse_radiative(data, mid.shape[0])[1]
        return (float(np.min(amb)), float(np.max(amb))) if amb.size else None

    def _boundary(self, t, temperature=None):
        """``(B, b)`` closure for all but band Dirichlet boundaries (``B`` diagonal, non-negative)."""
        if self._band_dirichlet:
            return None, np.zeros(self.M)
        if self._face_dirichlet:  # constant matrix, built once; only the data g(t) change
            fc = self._face_data()
            return fc["B"], self._base._face_dirichlet_rhs(t, fc)
        # Diagonal closure (no tangential correction) keeps A_L + B an M-matrix.
        B, b = self._base._assemble_boundary_system(
            t, temperature=temperature if self.bc_type == "radiative" else None, tangential=False)
        return B.tocsr(), b

    def _face_data(self):
        """Face-Dirichlet closure without tangential correction (diagonal ``B``), cached."""
        if getattr(self, "_face_closure", None) is None:
            base, faces = self._base, self._base.boundary_faces
            B, _ = base._assemble_boundary_system(0.0, tangential=False)
            _, w, _, _, _ = base._boundary_face_relations(0.0, tangential=False)
            self._face_closure = {"B": B.tocsr(), "w": w, "cells": faces["cells"], "lengths": faces["lengths"],
                                  "midpoints": faces["midpoints"], "normals": faces["normals"]}
        return self._face_closure

    def _global_bounds(self, u0, t0, t_end):
        """Physical maximum-principle bounds [m, M] for the global limiter.

        Uses ``bounds`` if supplied, else the extrema of the initial data and the
        Dirichlet boundary (or Robin/radiative ambient) data across the time
        interval.  With non-zero Neumann/flux inflow there is no a-priori
        maximum principle: use ``limiter="source_envelope"``.
        """
        if self._user_bounds is not None:
            return self._user_bounds
        cx, cy = self.cell_centers[:, 0], self.cell_centers[:, 1]
        lo = float(np.min(u0))
        hi = float(np.max(u0))
        for t in (t0, 0.5 * (t0 + t_end), t_end):
            if self._band_dirichlet:
                bc = np.broadcast_to(np.asarray(self._base.bc_func(cx, cy, t), dtype=float), (self.M,))
                lo = min(lo, float(bc[self._boundary_idx].min()) if self._boundary_idx.size else lo)
                hi = max(hi, float(bc[self._boundary_idx].max()) if self._boundary_idx.size else hi)
            else:
                amb = self._ambient_range(t)
                if amb is not None:
                    lo, hi = min(lo, amb[0]), max(hi, amb[1])
        return lo, hi

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
        dirichlet = self._band_dirichlet  # face Dirichlet goes through the boundary closure
        bidx = self._boundary_idx

        # Smoothness tolerance from the solution scale (range of the initial data).
        if self.smoothness_factor > 0.0:
            u_ref = max(float(u.max() - u.min()), abs(float(u.max())), 1e-300)
            self._eps = self.smoothness_factor * u_ref * self._h**1.5
        else:
            self._eps = np.zeros(self.M)

        mass = diags(self.cell_areas, format="csr")
        constant_lhs = self.bc_type in ("dirichlet", "neumann", "flux")
        if constant_lhs:
            sys_L = (mass + dt * self.A_L).tocsr()
            if self._face_dirichlet:  # constant closure matrix
                sys_L = (sys_L + dt * self._boundary(t0)[0]).tocsr()
            if dirichlet:
                _apply_dirichlet_rows_csr(sys_L, bidx)
            factor = factorized(sys_L.tocsc())

        def low_order_factor(B):
            # The boundary closure B is diagonal and non-negative, so A_L + B is
            # still an M-matrix: the low-order scheme stays bound-preserving.
            return factor if constant_lhs else factorized((mass + dt * (self.A_L + B)).tocsc())

        cx, cy = self.cell_centers[:, 0], self.cell_centers[:, 1]
        global_m, global_M = self._global_bounds(u, t0, t_end) if self.limiter == "global" else (None, None)
        if self.limiter == "source_envelope":
            if self._user_bounds is not None:
                env_lo, env_hi = self._user_bounds
            else:
                vals = [u]
                if dirichlet:
                    bc0 = np.broadcast_to(np.asarray(self._base.bc_func(cx, cy, t0), dtype=float), (self.M,))
                    vals.append(bc0[bidx])
                env_lo = float(min(v.min() for v in vals if v.size))
                env_hi = float(max(v.max() for v in vals if v.size))
        interior = np.ones(self.M, dtype=bool)
        if dirichlet:
            interior[bidx] = False
        self.bound_history = []
        self.limiter_activity = []
        self.extrema_history = []

        t = float(t0)
        for _ in range(nsteps):
            t_next = t + dt
            src = np.broadcast_to(np.asarray(self._base.source_func(cx, cy, t_next), dtype=float), (self.M,))
            B, b = self._boundary(t_next, temperature=u)
            rhs_core = self.cell_areas * u + dt * (self.cell_areas * src)
            if dirichlet:
                bc = np.broadcast_to(np.asarray(self._base.bc_func(cx, cy, t_next), dtype=float), (self.M,))
                rhs_base = rhs_core.copy()
                rhs_base[bidx] = bc[bidx]
            else:
                rhs_base = rhs_core + dt * b
            solve_L = low_order_factor(B)
            u_low = solve_L(rhs_base)  # low-order (M-matrix) solution, source and boundary included
            if self.bc_type == "radiative":  # converge the nonlinear boundary for the predictor
                for _ in range(self.max_iters):
                    B, b = self._boundary(t_next, temperature=u_low)
                    solve_L = low_order_factor(B)
                    rhs_base = rhs_core + dt * b
                    u_prev, u_low = u_low, solve_L(rhs_base)
                    if np.max(np.abs(u_low - u_prev)) <= self.tol * max(1.0, np.max(np.abs(u_low))):
                        break

            if self.limiter == "global":
                umin = np.full(self.M, global_m)
                umax = np.full(self.M, global_M)
            elif self.limiter == "source_envelope":
                # Discrete comparison principle: boundary inflow (Neumann/flux) acts as
                # a source b_i/|Omega_i| in boundary cells; Robin/radiative ambient
                # temperatures and Dirichlet data bound the envelope from outside.
                q_eff = src + (b / self.cell_areas if self.bc_type in ("neumann", "flux") else 0.0)
                q_eff = q_eff[interior] if np.any(interior) else q_eff
                env_hi = env_hi + dt * max(float(np.max(q_eff)), 0.0)
                env_lo = env_lo - dt * max(float(np.max(-q_eff)), 0.0)
                if dirichlet and bidx.size:
                    env_hi = max(env_hi, float(bc[bidx].max()))
                    env_lo = min(env_lo, float(bc[bidx].min()))
                amb = self._ambient_range(t_next)
                if amb is not None:
                    env_lo, env_hi = min(env_lo, amb[0]), max(env_hi, amb[1])
                umin = np.full(self.M, env_lo)
                umax = np.full(self.M, env_hi)
            elif self.limiter == "source_local":
                umin, umax = self._local_bounds(u_low)
            else:
                # Bounds from the previous time level's local neighbourhood.
                umin, umax = self._local_bounds(u)
                if dirichlet:
                    umin[bidx] = np.minimum(umin[bidx], bc[bidx])
                    umax[bidx] = np.maximum(umax[bidx], bc[bidx])
                elif self._face_dirichlet:
                    # A boundary cell's neighbourhood includes its boundary faces' data.
                    g = self._base._face_dirichlet_values(t_next, self._face_data())
                    cells = self._base.boundary_faces["cells"]
                    np.minimum.at(umin, cells, g)
                    np.maximum.at(umax, cells, g)

            self.bound_history.append((t_next, float(np.min(umin)), float(np.max(umax))))
            u_iter = u_low  # low-order predictor
            for _ in range(self.max_iters):
                limited = self._zalesak_alpha(u_iter, umin, umax)
                if self.bc_type == "radiative":
                    B, b = self._boundary(t_next, temperature=u_iter)
                    solve_L = low_order_factor(B)
                    rhs_base = rhs_core + dt * b
                rhs = rhs_base + dt * self._scatter_edges(limited)
                if dirichlet:
                    rhs[bidx] = bc[bidx]
                u_new = solve_L(rhs)
                err = np.max(np.abs(u_new - u_iter))
                u_iter = u_new
                if err <= self.tol * max(1.0, np.max(np.abs(u_new))):
                    break
            full = dt * self._scatter_edges(self._edge_d * (u_iter[self._edge_i] - u_iter[self._edge_j]))
            used = dt * self._scatter_edges(limited)
            denom = float(np.sum(np.abs(full)))
            self.limiter_activity.append(0.0 if denom == 0.0 else 1.0 - float(np.sum(np.abs(used))) / denom)
            u = u_iter
            t = t_next
            self.extrema_history.append((t, float(u.min()), float(u.max())))
        return t, u
