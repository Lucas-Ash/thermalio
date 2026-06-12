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

All three solvers currently support Dirichlet boundary data, which keeps the
manufactured-solution verification unambiguous; the bulk operators are the
research-relevant part.  See ``transport_demo.py`` for runnable examples and
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
    """Shared plumbing: a composed base solver plus Dirichlet helpers."""

    def __init__(self, vertices, polygons, alpha, dt, bc_func=None, source_func=None, **base_kwargs):
        base_kwargs.setdefault("bc_type", "dirichlet")
        if str(base_kwargs["bc_type"]).lower() != "dirichlet":
            raise ValueError(
                f"{type(self).__name__} currently supports bc_type='dirichlet' only."
            )
        self._base = PolygonalHeatSolver(
            vertices,
            polygons,
            alpha,
            dt,
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
        # initial acceleration implied by the PDE:
        #   tau a0 = -v0 - (A u0)/area + Q0   ->   u^{-1} = u0 - dt v0 + dt^2/2 a0
        q0 = self._source(t0)
        a0 = (-v0 - (self.A @ u_n) / self.cell_areas + q0) / self.tau
        u_nm1 = u_n - dt * v0 + 0.5 * dt * dt * a0

        c2 = self.tau / (dt * dt)
        c1 = 1.0 / (2.0 * dt)
        lhs = (c2 + c1) * self.area_diag + self.A
        lhs = self._apply_dirichlet(lhs.tocsr())
        factor = factorized(lhs.tocsc())

        t = float(t0)
        for _ in range(nsteps):
            t_next = t + dt
            q_next = self._source(t_next)
            rhs = self.cell_areas * q_next + self.cell_areas * (
                2.0 * c2 * u_n + (c1 - c2) * u_nm1
            )
            bc = self._bc_values(t_next)
            rhs[self._boundary_idx] = bc[self._boundary_idx]
            u_np1 = factor(rhs)
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
        if self.time_scheme == "backward_euler":
            lhs = self._apply_dirichlet((self.area_diag + dt * K).tocsr())
            factor = factorized(lhs.tocsc())
        else:
            lhs = self._apply_dirichlet((self.area_diag + 0.5 * dt * K).tocsr())
            factor = factorized(lhs.tocsc())

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
            bc = self._bc_values(t_next)
            rhs[self._boundary_idx] = bc[self._boundary_idx]
            u = factor(rhs)
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
        lhs = self._apply_dirichlet((sigma * self.area_diag + self.A).tocsr())
        factor = factorized(lhs.tocsc())

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
            bc = self._bc_values(t_next)
            rhs[self._boundary_idx] = bc[self._boundary_idx]
            u_np1 = factor(rhs)
            history.append(u_np1)
            t = t_next
        return t, history[-1]
