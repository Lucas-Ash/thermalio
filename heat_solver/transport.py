"""Extended thermal-transport models built on the polygonal finite-volume core.

This module adds transport models that go beyond the classical parabolic
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
    Time integration is an implicit (unconditionally stable) three-level
    scheme: the time differences are centered at ``t_n`` but diffusion and
    source are evaluated at ``t_{n+1}``, so it is first-order accurate in time
    (measured order ~1.0; see direction_CD_enthalpy_adjoint_report.tex).

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

``HyperbolicStefanSolver`` and ``FractionalStefanSolver``
    Apparent-heat-capacity phase-change variants of the Cattaneo and Caputo
    solvers, respectively.  These couple ``ApparentHeatCapacityModel`` into the
    time term, yielding ``tau u_tt + c(u) u_t - div(alpha grad u) = Q`` and
    ``c(u) D_t^beta u - div(alpha grad u) = Q``.

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

from .factor_cache import FactorCache, frozen_newton
from .phase_change import ApparentHeatCapacityModel
from .polygonal import PolygonalHeatSolver, _AndersonAccelerator


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

    ``dirichlet_mode`` (forwarded to the base solver) chooses how Dirichlet data
    are imposed: ``'band'`` (default) holds the boundary cells at the data,
    ``'face'`` imposes them on the boundary faces (every cell an unknown), as
    FiPy does; see ``docs/boundary_conditions.md``.
    """

    _BC_TYPES = ("dirichlet", "neumann", "robin", "flux", "radiative")

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
        base_bc = self.bc_type
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
        self.phase_change_model = self._base.phase_change_model
        if (
            self.phase_change_model is not None
            and not isinstance(self.phase_change_model, ApparentHeatCapacityModel)
        ):
            raise TypeError("phase_change_model must be an ApparentHeatCapacityModel instance or None.")
        self.phase_change_options = self._base.phase_change_options
        # dirichlet_mode="band" (default) holds the boundary cells at the data
        # with identity rows; "face" imposes it on the boundary faces through the
        # closure alpha_n (g - v) / d (Robin with beta -> infinity), as FiPy does.
        self.dirichlet_mode = self._base.dirichlet_mode
        self._band_dirichlet = self._base._band_dirichlet
        self._face_dirichlet = self.bc_type == "dirichlet" and not self._band_dirichlet
        self._face_closure = None
        # Robin data, the (linearized, temperature-dependent) radiative closure and
        # the face-Dirichlet closure enter the LHS; band Dirichlet rows are replaced
        # and Neumann/flux data only feed the right-hand side.
        self._bc_lhs_constant = self.bc_type not in ("robin", "radiative") and not self._face_dirichlet
        # ...but Robin data with time-independent beta leave it unchanged, and the radiative
        # Jacobian changes slowly: reuse factorizations (heat_solver/factor_cache.py).
        self._factor_cache = FactorCache()

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

    def _effective_heat_capacity(self, temperature):
        if self.phase_change_model is None:
            return np.ones_like(np.asarray(temperature, dtype=float))
        temperature = np.asarray(temperature, dtype=float)
        return np.broadcast_to(
            np.asarray(self.phase_change_model.effective_heat_capacity(temperature), dtype=float),
            temperature.shape,
        )

    def _picard_options(self):
        opts = self.phase_change_options
        return (
            int(opts.get("max_iters", 30)),
            float(opts.get("tol", 1e-9)),
            float(opts.get("relaxation", 1.0)),
            int(opts.get("anderson_depth", 0)),
        )

    def _raise_on_nonconvergence(self):
        """Whether a non-converged Picard step raises (default) or is recorded.

        Set ``phase_change_options['raise_on_nonconvergence'] = False`` to let
        studies build convergence/failure maps without aborting the sweep.
        """
        return bool(self.phase_change_options.get("raise_on_nonconvergence", True))

    def _init_solve_report(self):
        """Reset per-solve nonlinear convergence metadata (capability: reporting)."""
        max_iters, tol, relaxation, anderson_depth = self._picard_options()
        self.solve_report = {
            "converged": True,
            "n_steps": 0,
            "failed_steps": 0,
            "failed_step_indices": [],
            "iterations": [],
            "max_iterations": 0,
            "max_allowed_iterations": int(max_iters),
            "mean_iterations": 0.0,
            "residuals": [],
            "final_residual": 0.0,
            "max_residual": 0.0,
            "tolerance": float(tol),
            "relaxation": float(relaxation),
            "anderson_depth": int(anderson_depth),
            "min_capacity": 1.0,
            "max_capacity": 1.0,
        }

    def _record_step(self, iters, converged, capacity_max=1.0, capacity_min=None, residual=None):
        report = getattr(self, "solve_report", None)
        if report is None:
            self._init_solve_report()
            report = self.solve_report
        step_index = report["n_steps"] + 1
        if capacity_min is None:
            capacity_min = capacity_max
        residual_value = float("nan") if residual is None else float(residual)
        report["n_steps"] += 1
        report["iterations"].append(int(iters))
        report["max_iterations"] = max(report["max_iterations"], int(iters))
        report["mean_iterations"] = float(np.mean(report["iterations"]))
        report["residuals"].append(residual_value)
        report["final_residual"] = residual_value
        if np.isfinite(residual_value):
            report["max_residual"] = max(report["max_residual"], residual_value)
        report["min_capacity"] = min(report["min_capacity"], float(capacity_min))
        report["max_capacity"] = max(report["max_capacity"], float(capacity_max))
        if not converged:
            report["converged"] = False
            report["failed_steps"] += 1
            report["failed_step_indices"].append(int(step_index))


    @staticmethod
    def _relaxed_picard_update(accel, u_iter, u_raw, relaxation):
        if accel is not None:
            u_next = accel.step(u_iter, u_raw)
            if np.all(np.isfinite(u_next)):
                return u_next
            accel.reset()
        if relaxation != 1.0:
            return relaxation * u_raw + (1.0 - relaxation) * u_iter
        return u_raw

    def _boundary_system(self, t, temperature=None):
        """Boundary closure ``(B, b)`` so that ``A_total = A + B`` and ``rhs += b``.

        ``B`` is ``None`` when the boundary adds no matrix coupling (band
        Dirichlet rows are replaced instead; Neumann/flux data only contribute to ``b``).
        Radiative closures are Newton-linearized about ``temperature``.
        """
        if self._band_dirichlet:
            return None, np.zeros(self.M)
        if self._face_dirichlet:
            # Constant matrix part (built once); only the data g(t) change.
            if self._face_closure is None:
                self._face_closure = self._base._face_dirichlet_closure()
            fc = self._face_closure
            return fc["B"], self._base._face_dirichlet_rhs(t, fc)
        if self.bc_type in ("neumann", "flux"):
            _, b = self._base._assemble_boundary_system(t)
            return None, b
        if self.bc_type == "radiative":
            if temperature is None:
                raise ValueError("radiative boundaries need the current temperature iterate.")
            return self._base._assemble_boundary_system(t, temperature=temperature)
        return self._base._assemble_boundary_system(t)  # robin

    def _boundary_solve(self, lhs_base, rhs_base, t, coef, u_guess, factor=None, max_iters=60, tol=1e-12):
        """Solve ``(lhs_base + coef B) u = rhs_base + coef b`` for every closure (all but band Dirichlet).

        Linear boundary types need one solve (``factor`` reuses a factorization
        when the LHS is constant).  For Robin and face-Dirichlet data the factorization of
        ``lhs_base + coef B`` is reused while ``B`` is unchanged (same values,
        so the result equals a fresh solve bit for bit).  Radiative boundaries
        are nonlinear: frozen-Jacobian Newton on the boundary nonlinearity until
        the update is below ``tol`` (:func:`heat_solver.factor_cache.frozen_newton`).
        """
        if self.bc_type != "radiative":
            B, b = self._boundary_system(t)
            rhs = rhs_base + coef * b
            if factor is not None:
                return factor(rhs)
            if B is None:
                return spsolve(lhs_base.tocsr(), rhs)
            solve = self._factor_cache.factor(("robin", float(coef)), (lhs_base,), B,
                                              lambda: (lhs_base + coef * B).tocsr())
            return solve(rhs)
        u, _, _ = frozen_newton(
            u0=u_guess,
            boundary=lambda v: self._boundary_system(t, temperature=v),
            apply_base=lambda v: lhs_base @ v,
            rhs=rhs_base, coef=coef,
            build_jacobian=lambda B: (lhs_base + coef * B).tocsr(),
            cache=self._factor_cache, key=("radiative", round(float(coef), 12)), deps=(lhs_base,),
            tol=tol, max_iters=max_iters)
        return u


class HyperbolicHeatSolver(_TransportBase):
    """Cattaneo--Vernotte (hyperbolic, non-Fourier) heat conduction.

    Solves ``tau u_tt + u_t - div(alpha grad u) = Q`` with an unconditionally
    stable three-level implicit scheme (first order in time, because ``A u``
    and ``Q`` are taken at ``t_{n+1}``) on the semi-discrete
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
        if self.phase_change_model is not None:
            return self._solve_phase_change(u0, t0, t_end, du0=du0)

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
        B0, b0 = self._boundary_system(t0, temperature=u_n)
        A0 = self.A if B0 is None else (self.A + B0).tocsr()
        a0 = (q0 + (b0 - A0 @ u_n) / self.cell_areas - v0) / self.tau
        u_nm1 = u_n - dt * v0 + 0.5 * dt * dt * a0

        c2 = self.tau / (dt * dt)
        c1 = 1.0 / (2.0 * dt)
        base_lhs = ((c2 + c1) * self.area_diag + self.A).tocsr()
        if self._band_dirichlet:
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
            if self._band_dirichlet:
                bc = self._bc_values(t_next)
                rhs[self._boundary_idx] = bc[self._boundary_idx]
                u_np1 = factor(rhs)
            else:
                u_np1 = self._boundary_solve(base_lhs, rhs, t_next, 1.0, u_n, factor)
            u_nm1, u_n = u_n, u_np1
            t = t_next
        return t, u_n

    def _solve_phase_change(self, u0, t0, t_end, du0=None):
        """Picard solve for ``tau u_tt + c(u) u_t - div(alpha grad u) = Q``."""
        u_n = np.array(u0, dtype=float)
        v0 = np.zeros(self.M) if du0 is None else np.broadcast_to(
            np.asarray(du0, dtype=float), (self.M,)
        ).astype(float)

        nsteps, dt = self._uniform_schedule(t0, t_end, self.dt)
        if nsteps == 0:
            return float(t0), u_n

        q0 = self._source(t0)
        B0, b0 = self._boundary_system(t0, temperature=u_n)
        A0 = self.A if B0 is None else (self.A + B0).tocsr()
        cp0 = self._effective_heat_capacity(u_n)
        a0 = (q0 + (b0 - A0 @ u_n) / self.cell_areas - cp0 * v0) / self.tau
        u_nm1 = u_n - dt * v0 + 0.5 * dt * dt * a0

        c2 = self.tau / (dt * dt)
        c1 = 1.0 / (2.0 * dt)
        max_iters, tol, relaxation, anderson_depth = self._picard_options()
        self._init_solve_report()
        t = float(t0)
        bidx = self._boundary_idx
        # Optional per-step history (set ``solver.record_history = True``) used by
        # heat_solver.boundary_flux to reconstruct consistent Dirichlet fluxes.
        record = bool(getattr(self, "record_history", False))
        if record:
            self.history = {"t": [float(t0)], "u": [u_n.copy()], "u_start": u_nm1.copy(),
                            "capacity": [], "dt": dt, "tau": self.tau}

        for _ in range(nsteps):
            t_next = t + dt
            q_next = self._source(t_next)
            rhs_base = self.cell_areas * q_next + self.cell_areas * (2.0 * c2 * u_n)
            if self._band_dirichlet:
                bc = self._bc_values(t_next)
                rhs_base[bidx] = bc[bidx]
            u_iter = u_n.copy()
            accel = _AndersonAccelerator(anderson_depth) if anderson_depth > 0 else None

            converged = False
            iters_done = max_iters
            cp_eff = self._effective_heat_capacity(u_iter)
            final_residual = float("inf")
            for _iter in range(max_iters):
                cp_eff = self._effective_heat_capacity(u_iter)
                lhs = (diags(self.cell_areas * (c2 + c1 * cp_eff), format="csr") + self.A).tocsr()
                rhs = rhs_base + self.cell_areas * (c1 * cp_eff - c2) * u_nm1
                if self._band_dirichlet:
                    lhs = self._apply_dirichlet(lhs)
                    rhs[bidx] = bc[bidx]
                else:
                    B, b = self._boundary_system(t_next, temperature=u_iter)
                    rhs = rhs + b
                    if B is not None:
                        lhs = (lhs + B).tocsr()
                u_raw = spsolve(lhs, rhs)
                u_next = self._relaxed_picard_update(accel, u_iter, u_raw, relaxation)
                err = np.max(np.abs(u_next - u_iter))
                scale = max(1.0, np.max(np.abs(u_next)))
                final_residual = float(err / scale)
                u_iter = u_next
                if err <= tol * scale:
                    converged = True
                    iters_done = _iter + 1
                    break

            self._record_step(
                iters_done, converged,
                capacity_max=float(np.max(cp_eff)),
                capacity_min=float(np.min(cp_eff)),
                residual=final_residual,
            )
            if not converged and self._raise_on_nonconvergence():
                raise RuntimeError(
                    "Hyperbolic Stefan solve did not converge. "
                    "Try smaller dt or larger phase_change_options['max_iters']."
                )
            if record:
                self.history["t"].append(float(t_next))
                self.history["u"].append(u_iter.copy())
                self.history["capacity"].append(np.asarray(cp_eff, dtype=float).copy())

            u_nm1, u_n = u_n, u_iter
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
        faces = self._base.boundary_faces
        if faces["cells"].size:
            vx, vy = self.velocity(faces["midpoints"][:, 0], faces["midpoints"][:, 1], 0.0)
            self._boundary_flow = (np.ravel(vx) * faces["normals"][:, 0]
                                   + np.ravel(vy) * faces["normals"][:, 1]) * faces["lengths"]
        else:
            self._boundary_flow = np.zeros(0)
        # Faces whose convective flux uses the BC-consistent face value u_f:
        # every face for the central scheme, inflow faces only for upwind (an
        # upwind outflow face keeps the donor-cell value so the scheme stays monotone).
        self._face_value_mask = (np.ones(self._boundary_flow.size, dtype=bool) if self.scheme == "central"
                                 else self._boundary_flow < 0.0)
        if self.bc_type in ("neumann", "flux"):
            # Boundary rows stay active, so div(v u) keeps its boundary-face
            # contribution flow * u_f with u_f = u_c + d du/dn: the u_c part is
            # constant (here), the data part enters the RHS in _boundary_system.
            self.C = (self.C + self._convection_boundary_closure()).tocsr()
        elif (self.bc_type in ("robin", "radiative") or self._face_dirichlet) and self._boundary_flow.size:
            donor = np.where(self._face_value_mask, 0.0, self._boundary_flow)
            cells = faces["cells"]
            self.C = (self.C + diags(np.bincount(cells, weights=donor, minlength=self.M))).tocsr()

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

    def _boundary_system(self, t, temperature=None):
        """Diffusive closure plus the convective boundary flux ``flow_f u_f``.

        The face value comes from the boundary condition itself
        (``u_f = lam (T u)_f + mu``, see ``PolygonalHeatSolver._boundary_face_relations``)
        instead of the cell value, which makes the convective boundary flux
        second-order consistent for every boundary type.  With ``scheme="upwind"``
        only inflow faces use it; outflow faces keep the donor-cell value.
        """
        B, b = super()._boundary_system(t, temperature=temperature)
        if self._band_dirichlet or self._boundary_flow.size == 0:
            return B, b
        from scipy.sparse import coo_matrix

        cells = self._base.boundary_faces["cells"]
        if self._face_dirichlet:  # u_f = g: the convective flux of every face-value face is data
            flow = np.where(self._face_value_mask, self._boundary_flow, 0.0)
            g = self._base._face_dirichlet_values(t, self._face_closure)
            return B, b - np.bincount(cells, weights=flow * g, minlength=self.M)
        T, _, _, lam, mu = self._base._boundary_face_relations(t, temperature=temperature)
        flow = np.where(self._face_value_mask, self._boundary_flow, 0.0)
        b = b - np.bincount(cells, weights=flow * mu, minlength=self.M)
        if self.bc_type in ("robin", "radiative"):
            E = coo_matrix((flow * lam, (cells, np.arange(cells.size))),
                           shape=(self.M, cells.size)).tocsr()
            Cb = (E @ T).tocsr()
            B = Cb if B is None else (B + Cb).tocsr()
        return B, b

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
        if self._band_dirichlet:
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
            if self._band_dirichlet:
                bc = self._bc_values(t_next)
                rhs[self._boundary_idx] = bc[self._boundary_idx]
                u = factor(rhs)
            else:
                if self.time_scheme == "crank_nicolson":
                    # Explicit half at the old level (radiation linearized about u^n).
                    B_prev, b_prev = self._boundary_system(t, temperature=u)
                    rhs = rhs + 0.5 * dt * b_prev
                    if B_prev is not None:
                        rhs = rhs - 0.5 * dt * (B_prev @ u)
                u = self._boundary_solve(base_lhs, rhs, t_next, theta * dt, u, factor)
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

    def __init__(self, vertices, polygons, alpha, dt, beta, memory_window=None, **kwargs):
        super().__init__(vertices, polygons, alpha, dt, **kwargs)
        self.beta = float(beta)
        if not (0.0 < self.beta < 1.0):
            raise ValueError("Fractional order beta must satisfy 0 < beta < 1.")
        # Short-memory window: keep only the most recent ``memory_window`` L1
        # history lags (None -> full history).  This bounds the per-step cost and
        # storage of the Caputo sum at the price of dropping the oldest memory.
        if memory_window is not None:
            memory_window = int(memory_window)
            if memory_window < 1:
                raise ValueError("memory_window must be a positive integer or None.")
        self.memory_window = memory_window

    def _l1_weights(self, n):
        k = np.arange(n + 1, dtype=float)
        return (k + 1.0) ** (1.0 - self.beta) - k ** (1.0 - self.beta)

    def _history_lag_range(self, n):
        """L1 history lags k = 1..kmax retained at step n (short-memory aware)."""
        if self.memory_window is None:
            return range(1, n + 1)
        return range(1, min(n, self.memory_window) + 1)

    def solve(self, u0, t0, t_end):
        if self.phase_change_model is not None:
            return self._solve_phase_change(u0, t0, t_end)

        u0 = np.array(u0, dtype=float)
        nsteps, dt = self._uniform_schedule(t0, t_end, self.dt)
        if nsteps == 0:
            return float(t0), u0

        sigma = dt ** (-self.beta) / gamma(2.0 - self.beta)
        base_lhs = (sigma * self.area_diag + self.A).tocsr()
        if self._band_dirichlet:
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
            # Telescoping L1 history sum (the k=0 term carries u^{n+1}); a short
            # memory window keeps only the most recent lags.
            hist_sum = np.zeros(self.M)
            for k in self._history_lag_range(n):
                hist_sum += weights[k] * (history[n + 1 - k] - history[n - k])
            g = history[n] - hist_sum  # = u^n - sum_{k>=1} b_k (u^{n+1-k} - u^{n-k})
            rhs = self.cell_areas * q_next + sigma * (self.cell_areas * g)
            if self._band_dirichlet:
                bc = self._bc_values(t_next)
                rhs[self._boundary_idx] = bc[self._boundary_idx]
                u_np1 = factor(rhs)
            else:
                u_np1 = self._boundary_solve(base_lhs, rhs, t_next, 1.0, history[n], factor)
            history.append(u_np1)
            t = t_next
        return t, history[-1]

    def _solve_phase_change(self, u0, t0, t_end):
        """Picard solve for ``c(u) D_t^beta u - div(alpha grad u) = Q``."""
        u0 = np.array(u0, dtype=float)
        nsteps, dt = self._uniform_schedule(t0, t_end, self.dt)
        if nsteps == 0:
            return float(t0), u0

        sigma = dt ** (-self.beta) / gamma(2.0 - self.beta)
        history = [u0.copy()]
        weights = self._l1_weights(nsteps)
        max_iters, tol, relaxation, anderson_depth = self._picard_options()
        self._init_solve_report()
        t = float(t0)
        bidx = self._boundary_idx

        for n in range(nsteps):
            t_next = t + dt
            q_next = self._source(t_next)
            hist_sum = np.zeros(self.M)
            for k in self._history_lag_range(n):
                hist_sum += weights[k] * (history[n + 1 - k] - history[n - k])
            g = history[n] - hist_sum
            rhs_base = self.cell_areas * q_next
            if self._band_dirichlet:
                bc = self._bc_values(t_next)
                rhs_base[bidx] = bc[bidx]
            u_iter = history[n].copy()
            accel = _AndersonAccelerator(anderson_depth) if anderson_depth > 0 else None

            converged = False
            iters_done = max_iters
            cp_eff = self._effective_heat_capacity(u_iter)
            final_residual = float("inf")
            for _iter in range(max_iters):
                cp_eff = self._effective_heat_capacity(u_iter)
                lhs = (diags(sigma * self.cell_areas * cp_eff, format="csr") + self.A).tocsr()
                rhs = rhs_base + sigma * self.cell_areas * cp_eff * g
                if self._band_dirichlet:
                    lhs = self._apply_dirichlet(lhs)
                    rhs[bidx] = bc[bidx]
                else:
                    B, b = self._boundary_system(t_next, temperature=u_iter)
                    rhs = rhs + b
                    if B is not None:
                        lhs = (lhs + B).tocsr()
                u_raw = spsolve(lhs, rhs)
                u_next = self._relaxed_picard_update(accel, u_iter, u_raw, relaxation)
                err = np.max(np.abs(u_next - u_iter))
                scale = max(1.0, np.max(np.abs(u_next)))
                final_residual = float(err / scale)
                u_iter = u_next
                if err <= tol * scale:
                    converged = True
                    iters_done = _iter + 1
                    break

            self._record_step(
                iters_done, converged,
                capacity_max=float(np.max(cp_eff)),
                capacity_min=float(np.min(cp_eff)),
                residual=final_residual,
            )
            if not converged and self._raise_on_nonconvergence():
                raise RuntimeError(
                    "Fractional Stefan solve did not converge. "
                    "Try smaller dt or larger phase_change_options['max_iters']."
                )

            history.append(u_iter)
            t = t_next
        return t, history[-1]


class HyperbolicStefanSolver(HyperbolicHeatSolver):
    """Cattaneo apparent-heat-capacity phase-change solver.

    This is a named convenience wrapper around :class:`HyperbolicHeatSolver`
    with ``phase_change_model`` required, solving
    ``tau u_tt + c(u) u_t - div(alpha grad u) = Q``.
    """

    def __init__(self, vertices, polygons, alpha, dt, relaxation_time, phase_change_model, **kwargs):
        if phase_change_model is None:
            raise ValueError("HyperbolicStefanSolver requires phase_change_model.")
        super().__init__(
            vertices,
            polygons,
            alpha,
            dt,
            relaxation_time,
            phase_change_model=phase_change_model,
            **kwargs,
        )


class FractionalStefanSolver(FractionalHeatSolver):
    """Caputo apparent-heat-capacity phase-change solver.

    This is a named convenience wrapper around :class:`FractionalHeatSolver`
    with ``phase_change_model`` required, solving
    ``c(u) D_t^beta u - div(alpha grad u) = Q``.
    """

    def __init__(self, vertices, polygons, alpha, dt, beta, phase_change_model, **kwargs):
        if phase_change_model is None:
            raise ValueError("FractionalStefanSolver requires phase_change_model.")
        super().__init__(
            vertices,
            polygons,
            alpha,
            dt,
            beta,
            phase_change_model=phase_change_model,
            **kwargs,
        )


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
        if self._band_dirichlet:
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
            if self._band_dirichlet:
                bc = self._bc_values(t_next)
                rhs[self._boundary_idx] = bc[self._boundary_idx]
                u = factor(rhs)
            else:
                if self.time_scheme == "crank_nicolson":
                    # Explicit half at the old level (radiation linearized about u^n).
                    B_prev, b_prev = self._boundary_system(t, temperature=u)
                    rhs = rhs + 0.5 * dt * b_prev
                    if B_prev is not None:
                        rhs = rhs - 0.5 * dt * (B_prev @ u)
                u = self._boundary_solve(base_lhs, rhs, t_next, theta * dt, u, factor)
            t = t_next
        return t, u
