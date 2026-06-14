"""Extended thermal-transport models built on the polygonal finite-volume core.

This module adds three transport models that go beyond the classical parabolic
Fourier heat equation handled by :class:`heat_solver.polygonal.PolygonalHeatSolver`.
Each model is implemented by *composing* a base ``PolygonalHeatSolver`` so that the
mesh geometry, the diffusion stiffness matrix ``A`` and the lumped mass matrix
``M_diag`` (cell areas) are reused unchanged:

``HyperbolicHeatSolver``
    Non-Fourier (Maxwell--Cattaneo--Vernotte) heat conduction with a finite
    thermal relaxation time ``tau``::

        tau * u_tt + u_t - div(alpha grad u) = Q

    The relaxation time gives heat a finite propagation speed
    ``c = sqrt(alpha / tau)`` (thermal waves / "second sound"), which is the
    distinguishing feature of micro/nanoscale and ultrafast-heating transport.
    Time integration is a second-order implicit (unconditionally stable)
    three-level scheme.

``AdvectionDiffusionHeatSolver``
    Convective heat transport with a prescribed (incompressible) velocity
    field ``v``::

        u_t + div(v u) - div(alpha grad u) = Q

    The convective face flux uses either a first-order ``upwind`` scheme
    (monotone, robust at high Peclet number) or a second-order ``central``
    scheme.  This is the standard model for forced convection and conjugate
    heat transfer.

``FractionalHeatSolver``
    Time-fractional (anomalous) subdiffusion with a Caputo derivative of order
    ``0 < beta < 1``::

        D_t^beta u - div(alpha grad u) = Q

    discretized with the classical L1 scheme.  Sub-diffusive transport
    (``beta < 1``) models heat conduction in fractal/disordered media and
    materials with long-memory (non-local-in-time) thermal response.

``ReactionDiffusionHeatSolver``
    Reaction-diffusion / Pennes bioheat equation with a linear reaction term::

        u_t - div(alpha grad u) + k(x, y) u = Q

    The ``k u`` term models tissue perfusion cooling (Pennes bioheat),
    volumetric Newton cooling, or first-order chemical heat consumption.

All three solvers support ``bc_type`` in ``{'dirichlet', 'neumann', 'robin',
'flux'}``.  ``'flux'`` prescribes the *inward* boundary heat flux per unit
length (``q_in = alpha * du/dn``) directly, which is the natural way to drive
the Cattaneo model with a boundary heat pulse (e.g. pulsed-laser heating).
See ``transport_demo.py`` for runnable examples and
``tests/test_transport.py`` for convergence checks.
"""

from __future__ import annotations

from math import gamma

import numpy as np
from scipy.sparse import csr_matrix, diags, lil_matrix
from scipy.sparse.linalg import factorized, spsolve

from .polygonal import PolygonalHeatSolver


def _as_velocity_callable(velocity):
    """Normalize ``velocity`` into ``f(x, y, t) -> (vx, vy)`` returning arrays."""
    if callable(velocity):
        return velocity
    vec = np.asarray(velocity, dtype=float)
    if vec.shape != (2,):
        raise ValueError("Constant velocity must be a length-2 (vx, vy) vector.")

    def constant_velocity(x, y, t):
        ones = np.ones_like(np.asarray(x, dtype=float))
        return vec[0] * ones, vec[1] * ones

    return constant_velocity


class _TransportBase:
    """Shared plumbing: a composed base solver plus boundary-condition helpers.

    Supported ``bc_type`` values:

    - ``'dirichlet'``: ``bc_func(x, y, t) -> u`` on the boundary.
    - ``'neumann'``:   ``bc_func(x, y, t[, nx, ny]) -> du/dn`` (outward normal
      derivative), same convention as :class:`PolygonalHeatSolver`.
    - ``'flux'``:      ``bc_func(x, y, t[, nx, ny]) -> q_in``, the prescribed
      *inward* heat flux per unit boundary length, ``q_in = alpha * du/dn``.
      Unlike ``'neumann'`` this prescribes the physical flux directly without
      referencing the conductivity -- the natural form for boundary heat
      pulses (laser/contact heating) in the Cattaneo model.
    - ``'robin'``:     ``bc_func(x, y, t[, nx, ny]) -> (beta, value)`` for
      ``alpha * du/dn + beta * u = value``.
    """

    _BC_TYPES = ("dirichlet", "neumann", "robin", "flux")

    def __init__(
        self,
        vertices,
        polygons,
        alpha,
        dt,
        bc_type="dirichlet",
        bc_func=None,
        source_func=None,
        **base_kwargs,
    ):
        self.bc_type = str(bc_type).lower().strip()
        if self.bc_type not in self._BC_TYPES:
            raise ValueError(
                f"{type(self).__name__} supports bc_type in {self._BC_TYPES}; got {bc_type!r}."
            )
        # 'flux' reuses the base solver's Neumann face bookkeeping (normals,
        # midpoints, callback signature detection); the flux value itself is
        # integrated directly in _boundary_system without an alpha factor.
        base_bc = "neumann" if self.bc_type == "flux" else self.bc_type
        self._base = PolygonalHeatSolver(
            vertices,
            polygons,
            alpha,
            dt,
            bc_type=base_bc,
            bc_func=bc_func,
            source_func=source_func,
            **base_kwargs,
        )
        self.dt = float(dt)
        self.M = self._base.M
        self.cell_centers = self._base.cell_centers
        self.cell_areas = self._base.cell_areas
        self.is_boundary = self._base.is_boundary
        self._boundary_idx = np.where(self.is_boundary)[0]
        # Assemble the (constant) diffusion operator and lumped mass once.
        self._base._assemble_system()
        self.A = self._base.A.tocsr()
        self.area_diag = diags(self.cell_areas, format="csr")
        # Only Robin data can change the LHS matrix between steps; Dirichlet rows
        # are replaced and Neumann/flux data only feed the right-hand side.
        self._bc_lhs_constant = self.bc_type != "robin"

    @staticmethod
    def _uniform_schedule(t0, t_end, dt):
        span = float(t_end) - float(t0)
        if span <= 0:
            return 0, 0.0
        nsteps = max(int(round(span / float(dt))), 1)
        return nsteps, span / nsteps

    def _source(self, t):
        cx = self.cell_centers[:, 0]
        cy = self.cell_centers[:, 1]
        return np.broadcast_to(
            np.asarray(self._base.source_func(cx, cy, t), dtype=float), (self.M,)
        ).astype(float)

    def _bc_values(self, t):
        cx = self.cell_centers[:, 0]
        cy = self.cell_centers[:, 1]
        return np.broadcast_to(
            np.asarray(self._base.bc_func(cx, cy, t), dtype=float), (self.M,)
        ).astype(float)

    def _apply_dirichlet(self, lhs_csr):
        return self._base._apply_dirichlet_identity_rows(lhs_csr)

    def _boundary_system(self, t):
        """Boundary closure ``(B, b)`` so that ``A_total = A + B`` and ``rhs += b``.

        ``B`` is ``None`` when the boundary adds no matrix coupling (Dirichlet
        rows are replaced instead; Neumann/flux data only contribute to ``b``).
        """
        if self.bc_type == "dirichlet":
            return None, np.zeros(self.M)
        if self.bc_type == "flux":
            faces = self._base.boundary_faces
            cells = faces["cells"]
            if cells.size == 0:
                return None, np.zeros(self.M)
            midpoints = faces["midpoints"]
            flux_in = np.broadcast_to(
                np.asarray(
                    self._base._evaluate_bc(
                        midpoints[:, 0], midpoints[:, 1], t, normals=faces["normals"]
                    ),
                    dtype=float,
                ),
                (cells.size,),
            )
            b = np.bincount(cells, weights=faces["lengths"] * flux_in, minlength=self.M)
            return None, b
        if self.bc_type == "neumann":
            _, b = self._base._assemble_boundary_system(t)
            return None, b
        return self._base._assemble_boundary_system(t)  # robin


class HyperbolicHeatSolver(_TransportBase):
    """Cattaneo--Vernotte (hyperbolic, non-Fourier) heat conduction.

    Solves ``tau u_tt + u_t - div(alpha grad u) = Q`` with a second-order,
    unconditionally stable three-level implicit scheme on the semi-discrete
    system ``tau M u'' + M u' + A u = M Q``::

        u'' ~ (u^{n+1} - 2 u^n + u^{n-1}) / dt^2
        u'  ~ (u^{n+1} - u^{n-1}) / (2 dt)

    Parameters
    ----------
    relaxation_time : float
        Thermal relaxation time ``tau > 0``.  The thermal wave speed is
        ``sqrt(alpha / tau)``.
    """

    def __init__(self, vertices, polygons, alpha, dt, relaxation_time, **kwargs):
        super().__init__(vertices, polygons, alpha, dt, **kwargs)
        self.tau = float(relaxation_time)
        if self.tau <= 0.0:
            raise ValueError("relaxation_time (tau) must be positive for the hyperbolic model.")

    def wave_speed(self):
        """Finite propagation speed of thermal disturbances, ``sqrt(alpha/tau)``.

        Only meaningful for scalar isotropic ``alpha``; returns ``None`` otherwise.
        """
        alpha = self._base.alpha
        if callable(alpha):
            return None
        a = np.asarray(alpha, dtype=float)
        if a.ndim == 0:
            return float(np.sqrt(a / self.tau))
        return None

    def solve(self, u0, t0, t_end, du0=None):
        """Integrate from ``t0`` to ``t_end``.

        ``du0`` is the initial time derivative ``du/dt`` at ``t0`` (defaults to
        zero).  It is used to build the second-order startup level ``u^{-1}``.
        """
        u_n = np.array(u0, dtype=float)
        v0 = np.zeros(self.M) if du0 is None else np.broadcast_to(
            np.asarray(du0, dtype=float), (self.M,)
        ).astype(float)

        nsteps, dt = self._uniform_schedule(t0, t_end, self.dt)
        if nsteps == 0:
            return float(t0), u_n

        # Second-order startup level u^{-1} from a Taylor expansion using the
        # initial acceleration implied by the PDE (including boundary closure):
        #   tau a0 = -v0 - ((A + B) u0 - b)/area + Q0
        #   u^{-1} = u0 - dt v0 + dt^2/2 a0
        q0 = self._source(t0)
        B0, b0 = self._boundary_system(t0)
        A0 = self.A if B0 is None else (self.A + B0).tocsr()
        a0 = (q0 + (b0 - A0 @ u_n) / self.cell_areas - v0) / self.tau
        u_nm1 = u_n - dt * v0 + 0.5 * dt * dt * a0

        c2 = self.tau / (dt * dt)
        c1 = 1.0 / (2.0 * dt)
        base_lhs = ((c2 + c1) * self.area_diag + self.A).tocsr()
        if self.bc_type == "dirichlet":
            factor = factorized(self._apply_dirichlet(base_lhs).tocsc())
        elif self._bc_lhs_constant:
            factor = factorized(base_lhs.tocsc())
        else:
            factor = None

        t = float(t0)
        for _ in range(nsteps):
            t_next = t + dt
            q_next = self._source(t_next)
            rhs = self.cell_areas * q_next + self.cell_areas * (
                2.0 * c2 * u_n + (c1 - c2) * u_nm1
            )
            if self.bc_type == "dirichlet":
                bc = self._bc_values(t_next)
                rhs[self._boundary_idx] = bc[self._boundary_idx]
                u_np1 = factor(rhs)
            else:
                B, b = self._boundary_system(t_next)
                rhs = rhs + b
                if factor is not None:
                    u_np1 = factor(rhs)
                else:
                    u_np1 = spsolve((base_lhs + B).tocsr(), rhs)
            u_nm1, u_n = u_n, u_np1
            t = t_next
        return t, u_n


class AdvectionDiffusionHeatSolver(_TransportBase):
    """Convective heat transport ``u_t + div(v u) - div(alpha grad u) = Q``.

    A constant-in-time velocity field is assumed so the convection matrix is
    assembled once.  ``scheme='upwind'`` (default) is first-order and monotone;
    ``scheme='central'`` is second-order but may oscillate at high Peclet number.
    Time integration is backward Euler (default) or Crank--Nicolson.
    """

    def __init__(
        self,
        vertices,
        polygons,
        alpha,
        dt,
        velocity,
        scheme="upwind",
        time_scheme="backward_euler",
        **kwargs,
    ):
        super().__init__(vertices, polygons, alpha, dt, **kwargs)
        self.velocity = _as_velocity_callable(velocity)
        self.scheme = str(scheme).lower().strip()
        if self.scheme not in {"upwind", "central"}:
            raise ValueError("scheme must be 'upwind' or 'central'.")
        self.time_scheme = str(time_scheme).lower().strip()
        if self.time_scheme not in {"backward_euler", "crank_nicolson"}:
            raise ValueError("time_scheme must be 'backward_euler' or 'crank_nicolson'.")
        self.C = self._assemble_convection().tocsr()
        if self.bc_type != "dirichlet":
            # Boundary rows stay active for Neumann/Robin/flux data, so div(v u)
            # must keep its boundary-face contribution.
            self.C = (self.C + self._convection_boundary_closure()).tocsr()

    def _assemble_convection(self):
        """Cell-centered FV convection matrix C with (C u)_i ~ int_cell_i div(v u)."""
        C = lil_matrix((self.M, self.M))
        centers = self.cell_centers
        verts = self._base.vertices
        for edge, cells in self._base.edge_to_cells.items():
            if len(cells) != 2:
                continue
            i, j = cells
            if j < i:
                i, j = j, i
            v0, v1 = verts[edge[0]], verts[edge[1]]
            edge_vec = v1 - v0
            edge_len = np.linalg.norm(edge_vec)
            if edge_len == 0:
                continue
            tangent = edge_vec / edge_len
            normal = np.array([-tangent[1], tangent[0]])
            # Orient the normal to point from cell i to cell j.
            if np.dot(centers[j] - centers[i], normal) < 0:
                normal = -normal
            midpoint = 0.5 * (v0 + v1)
            vx, vy = self.velocity(np.array([midpoint[0]]), np.array([midpoint[1]]), 0.0)
            v_face = np.array([float(np.ravel(vx)[0]), float(np.ravel(vy)[0])])
            flow = float(np.dot(v_face, normal)) * edge_len  # signed volumetric flux, i -> j

            if self.scheme == "central":
                # u_face = 0.5 (u_i + u_j)
                C[i, i] += 0.5 * flow
                C[i, j] += 0.5 * flow
                C[j, i] -= 0.5 * flow
                C[j, j] -= 0.5 * flow
            else:  # upwind
                if flow >= 0.0:  # outflow from i into j -> use u_i
                    C[i, i] += flow
                    C[j, i] -= flow
                else:  # inflow to i from j -> use u_j
                    C[i, j] += flow
                    C[j, j] -= flow
        return C

    def _convection_boundary_closure(self):
        """First-order boundary convective flux using the cell value (``u_face ~ u_i``).

        Exact for impermeable boundaries (``v . n = 0``); first-order accurate
        otherwise.  Outflow-dominant boundaries (``v . n >= 0``) are recommended
        for robustness at high Peclet number.
        """
        faces = self._base.boundary_faces
        cells = faces["cells"]
        if cells.size == 0:
            return diags(np.zeros(self.M), format="csr")
        midpoints = faces["midpoints"]
        vx, vy = self.velocity(midpoints[:, 0], midpoints[:, 1], 0.0)
        flow = (
            np.ravel(vx) * faces["normals"][:, 0] + np.ravel(vy) * faces["normals"][:, 1]
        ) * faces["lengths"]
        return diags(np.bincount(cells, weights=flow, minlength=self.M), format="csr")

    def peclet_number(self, length_scale):
        """Mesh/representative Peclet number ``|v| * L / alpha`` (scalar alpha)."""
        alpha = self._base.alpha
        if callable(alpha) or np.asarray(alpha, dtype=float).ndim != 0:
            return None
        cx = self.cell_centers[:, 0]
        cy = self.cell_centers[:, 1]
        vx, vy = self.velocity(cx, cy, 0.0)
        speed = float(np.max(np.hypot(np.ravel(vx), np.ravel(vy))))
        return speed * float(length_scale) / float(np.asarray(alpha))

    def solve(self, u0, t0, t_end):
        u = np.array(u0, dtype=float)
        nsteps, dt = self._uniform_schedule(t0, t_end, self.dt)
        if nsteps == 0:
            return float(t0), u

        K = (self.A + self.C).tocsr()
        theta = 1.0 if self.time_scheme == "backward_euler" else 0.5
        base_lhs = (self.area_diag + theta * dt * K).tocsr()
        if self.bc_type == "dirichlet":
            factor = factorized(self._apply_dirichlet(base_lhs).tocsc())
        elif self._bc_lhs_constant:
            factor = factorized(base_lhs.tocsc())
        else:
            factor = None

        t = float(t0)
        for _ in range(nsteps):
            t_next = t + dt
            q_next = self._source(t_next)
            if self.time_scheme == "backward_euler":
                rhs = self.cell_areas * u + dt * (self.cell_areas * q_next)
            else:
                q_prev = self._source(t)
                rhs = (self.area_diag - 0.5 * dt * K) @ u + 0.5 * dt * (
                    self.cell_areas * (q_next + q_prev)
                )
            if self.bc_type == "dirichlet":
                bc = self._bc_values(t_next)
                rhs[self._boundary_idx] = bc[self._boundary_idx]
                u = factor(rhs)
            else:
                B_next, b_next = self._boundary_system(t_next)
                if self.time_scheme == "backward_euler":
                    rhs = rhs + dt * b_next
                else:
                    B_prev, b_prev = self._boundary_system(t)
                    rhs = rhs + 0.5 * dt * (b_prev + b_next)
                    if B_prev is not None:
                        rhs = rhs - 0.5 * dt * (B_prev @ u)
                if factor is not None:
                    u = factor(rhs)
                else:
                    u = spsolve((base_lhs + theta * dt * B_next).tocsr(), rhs)
            t = t_next
        return t, u


class FractionalHeatSolver(_TransportBase):
    """Time-fractional subdiffusion ``D_t^beta u - div(alpha grad u) = Q``.

    Uses the L1 discretization of the Caputo derivative of order ``0 < beta < 1``::

        D_t^beta u(t_{n+1}) ~ sigma * sum_{k=0}^{n} b_k (u^{n+1-k} - u^{n-k})

    with ``sigma = dt^{-beta} / Gamma(2 - beta)`` and
    ``b_k = (k+1)^{1-beta} - k^{1-beta}``.  The scheme is globally accurate of
    order ``2 - beta`` in time.  ``beta -> 1`` recovers classical Fourier
    diffusion (backward Euler).
    """

    def __init__(self, vertices, polygons, alpha, dt, beta, **kwargs):
        super().__init__(vertices, polygons, alpha, dt, **kwargs)
        self.beta = float(beta)
        if not (0.0 < self.beta < 1.0):
            raise ValueError("Fractional order beta must satisfy 0 < beta < 1.")

    def _l1_weights(self, n):
        k = np.arange(n + 1, dtype=float)
        return (k + 1.0) ** (1.0 - self.beta) - k ** (1.0 - self.beta)

    def solve(self, u0, t0, t_end):
        u0 = np.array(u0, dtype=float)
        nsteps, dt = self._uniform_schedule(t0, t_end, self.dt)
        if nsteps == 0:
            return float(t0), u0

        sigma = dt ** (-self.beta) / gamma(2.0 - self.beta)
        base_lhs = (sigma * self.area_diag + self.A).tocsr()
        if self.bc_type == "dirichlet":
            factor = factorized(self._apply_dirichlet(base_lhs).tocsc())
        elif self._bc_lhs_constant:
            factor = factorized(base_lhs.tocsc())
        else:
            factor = None

        history = [u0.copy()]  # history[m] == u^m
        weights = self._l1_weights(nsteps)  # b_0 .. b_nsteps

        t = float(t0)
        for n in range(nsteps):
            t_next = t + dt
            q_next = self._source(t_next)
            # Telescoping L1 history sum for k = 1 .. n (the k=0 term carries u^{n+1}).
            hist_sum = np.zeros(self.M)
            for k in range(1, n + 1):
                hist_sum += weights[k] * (history[n + 1 - k] - history[n - k])
            g = history[n] - hist_sum  # = u^n - sum_{k>=1} b_k (u^{n+1-k} - u^{n-k})
            rhs = self.cell_areas * q_next + sigma * (self.cell_areas * g)
            if self.bc_type == "dirichlet":
                bc = self._bc_values(t_next)
                rhs[self._boundary_idx] = bc[self._boundary_idx]
                u_np1 = factor(rhs)
            else:
                B, b = self._boundary_system(t_next)
                rhs = rhs + b
                if factor is not None:
                    u_np1 = factor(rhs)
                else:
                    u_np1 = spsolve((base_lhs + B).tocsr(), rhs)
            history.append(u_np1)
            t = t_next
        return t, history[-1]


class ReactionDiffusionHeatSolver(_TransportBase):
    """Reaction-diffusion / Pennes bioheat solver ``u_t - div(alpha grad u) + k u = Q``.

    The reaction rate ``k`` (perfusion coefficient) may be a non-negative scalar
    or a callable ``k(x, y)``.  It contributes a diagonal block ``R = diag(area*k)``
    so the semi-discrete system is ``M u' + (A + R) u = M Q``.  Time integration
    is backward Euler (default) or Crank--Nicolson.

    For constant scalar ``alpha`` and ``k``, the source-free mode
    ``sin(pi x) sin(pi y)`` decays at rate ``2 pi^2 alpha + k`` -- faster than
    pure diffusion -- which is the physical signature of perfusion cooling.
    """

    def __init__(
        self,
        vertices,
        polygons,
        alpha,
        dt,
        reaction_rate,
        time_scheme="backward_euler",
        **kwargs,
    ):
        super().__init__(vertices, polygons, alpha, dt, **kwargs)
        self.time_scheme = str(time_scheme).lower().strip()
        if self.time_scheme not in {"backward_euler", "crank_nicolson"}:
            raise ValueError("time_scheme must be 'backward_euler' or 'crank_nicolson'.")
        cx = self.cell_centers[:, 0]
        cy = self.cell_centers[:, 1]
        if callable(reaction_rate):
            k = np.asarray(reaction_rate(cx, cy), dtype=float)
        else:
            k = np.asarray(reaction_rate, dtype=float)
        k = np.broadcast_to(k, (self.M,)).astype(float)
        if np.any(k < 0.0):
            raise ValueError("reaction_rate (perfusion coefficient) must be non-negative.")
        self.reaction_rate = k
        self.R = diags(self.cell_areas * k, format="csr")

    def solve(self, u0, t0, t_end):
        u = np.array(u0, dtype=float)
        nsteps, dt = self._uniform_schedule(t0, t_end, self.dt)
        if nsteps == 0:
            return float(t0), u

        K = (self.A + self.R).tocsr()
        theta = 1.0 if self.time_scheme == "backward_euler" else 0.5
        base_lhs = (self.area_diag + theta * dt * K).tocsr()
        if self.bc_type == "dirichlet":
            factor = factorized(self._apply_dirichlet(base_lhs).tocsc())
        elif self._bc_lhs_constant:
            factor = factorized(base_lhs.tocsc())
        else:
            factor = None

        t = float(t0)
        for _ in range(nsteps):
            t_next = t + dt
            q_next = self._source(t_next)
            if self.time_scheme == "backward_euler":
                rhs = self.cell_areas * u + dt * (self.cell_areas * q_next)
            else:
                q_prev = self._source(t)
                rhs = (self.area_diag - 0.5 * dt * K) @ u + 0.5 * dt * (
                    self.cell_areas * (q_next + q_prev)
                )
            if self.bc_type == "dirichlet":
                bc = self._bc_values(t_next)
                rhs[self._boundary_idx] = bc[self._boundary_idx]
                u = factor(rhs)
            else:
                B_next, b_next = self._boundary_system(t_next)
                if self.time_scheme == "backward_euler":
                    rhs = rhs + dt * b_next
                else:
                    B_prev, b_prev = self._boundary_system(t)
                    rhs = rhs + 0.5 * dt * (b_prev + b_next)
                    if B_prev is not None:
                        rhs = rhs - 0.5 * dt * (B_prev @ u)
                if factor is not None:
                    u = factor(rhs)
                else:
                    u = spsolve((base_lhs + theta * dt * B_next).tocsr(), rhs)
            t = t_next
        return t, u
