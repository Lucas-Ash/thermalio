"""Discrete adjoints beyond scalar Pennes perfusion.

Every routine here differentiates the *discrete* time-stepping scheme exactly
("optimize-then-discretize" is avoided), so the returned gradients agree with
finite differences of the very same forward solver to round-off / nonlinear
solver tolerance.  The common pattern is the Lagrangian

    L(u, lambda, theta) = J(u) - sum_n lambda_n . R^n(u^n, u^{n-1}, ...; theta)

where ``R^n = 0`` is the residual of time step ``n``.  Stationarity in the
states gives a backward-in-time adjoint recurrence for ``lambda_n`` and the
gradient is ``dJ/dtheta = dJ/dtheta|_explicit - sum_n lambda_n . dR^n/dtheta``
(plus initial-data terms).  One forward and one adjoint sweep give the full
gradient regardless of the number of parameters.

Implemented adjoints
--------------------
:class:`LinearHeatAdjoint`
    Backward-Euler ``u_t - div(kappa grad u) + k u = Q`` with a conductivity
    *field* ``kappa = sum_j c_j phi_j``, perfusion field ``k = sum_j p_j psi_j``,
    source amplitudes ``Q = Q_0 + g(t) sum_j s_j chi_j`` and a Robin
    heat-transfer coefficient ``h`` (``kappa du/dn + h (u - u_amb) = 0``).
:func:`enthalpy_stefan_adjoint`
    The *nonlinear* enthalpy-primary Stefan solver
    (:class:`heat_solver.enthalpy.EnthalpyStefanSolver`) including the
    Cattaneo relaxation term on variable (adaptive) steps; parameters
    ``latent_heat``, ``alpha`` and ``tau``.  With ``latent_heat = 0`` this is
    the adjoint of the hyperbolic (Cattaneo) heat equation.
:func:`fractional_heat_adjoint`
    The Caputo L1 scheme of :class:`heat_solver.transport.FractionalHeatSolver`
    with respect to the fractional order ``beta`` and diffusivity ``alpha``.
:func:`taylor_test`
    Gradient verification: first-order vs second-order Taylor remainders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import gamma as gamma_fn

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu
from scipy.special import digamma

from .enthalpy import (
    EnthalpyStefanSolver,
    three_level_coefficients,
    temperature_sensitivity_to_latent_heat,
)
from .polygonal import PolygonalHeatSolver


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


@dataclass
class AdjointGradient:
    """Objective, gradient and the states produced by one forward/adjoint pass."""

    objective: float
    gradient: np.ndarray
    parameter_names: tuple
    predicted: np.ndarray
    residual: np.ndarray
    observation_steps: np.ndarray
    forward_states: np.ndarray | None = None
    adjoint_states: np.ndarray | None = None
    extras: dict = field(default_factory=dict)

    def as_dict(self):
        return dict(zip(self.parameter_names, (float(g) for g in self.gradient)))


class _Observations:
    """Sparse/multi-time least-squares misfit ``J = 1/2 sum w (P u - d)^2``."""

    def __init__(self, times, observed, n_cells, sensor_indices=None, weights=None):
        self.times = np.asarray(times, dtype=float).reshape(-1)
        if self.times.size == 0:
            raise ValueError("observation_times must be non-empty.")
        if np.any(np.diff(self.times) <= 0.0):
            raise ValueError("observation_times must be strictly increasing.")
        self.sensors = (
            np.arange(n_cells) if sensor_indices is None
            else np.asarray(sensor_indices, dtype=int).reshape(-1)
        )
        if self.sensors.size == 0 or np.any((self.sensors < 0) | (self.sensors >= n_cells)):
            raise ValueError("sensor_indices must select valid cells.")
        obs = np.asarray(observed, dtype=float)
        if obs.ndim == 1:
            obs = obs.reshape(self.times.size, self.sensors.size)
        if obs.shape != (self.times.size, self.sensors.size):
            raise ValueError("observed must have shape (n_times, n_sensors).")
        self.observed = obs
        if weights is None:
            w = np.ones_like(obs)
        else:
            w = np.asarray(weights, dtype=float)
            if w.ndim == 1 and w.size == self.sensors.size:
                w = np.tile(w, (self.times.size, 1))
            w = w.reshape(obs.shape)
        self.weights = w
        self.n_cells = n_cells

    def match_steps(self, step_times):
        step_times = np.asarray(step_times, dtype=float)
        steps = []
        for t in self.times:
            i = int(np.argmin(np.abs(step_times - t)))
            if abs(step_times[i] - t) > 1e-9 * max(1.0, abs(t)):
                raise ValueError(f"observation time {t} is not on the time-step grid.")
            if i == 0:
                raise ValueError("observation times must be strictly after the initial time.")
            steps.append(i)
        self.steps = np.asarray(steps, dtype=int)
        self.lookup = {int(s): j for j, s in enumerate(self.steps)}
        return self.steps

    def evaluate(self, fields_at_steps):
        """``fields_at_steps[j]`` is the observed field at observation ``j``."""
        pred = np.vstack([np.asarray(f)[self.sensors] for f in fields_at_steps])
        resid = pred - self.observed
        return 0.5 * float(np.sum(self.weights * resid**2)), pred, resid

    def field_gradient(self, step, resid):
        """``dJ/d(field)`` at a time step (zero if not observed)."""
        g = np.zeros(self.n_cells)
        j = self.lookup.get(int(step))
        if j is not None:
            np.add.at(g, self.sensors, self.weights[j] * resid[j])
        return g


def taylor_test(objective, gradient, theta, direction=None, hs=None, seed=0):
    """Taylor-remainder gradient check.

    For ``J(theta + h d)`` the zeroth-order remainder ``|J(theta+hd) - J(theta)|``
    decays like ``O(h)`` while the first-order remainder
    ``|J(theta+hd) - J(theta) - h g.d|`` decays like ``O(h^2)`` iff ``g`` is the
    exact gradient.  Returns the remainders and their observed rates.
    """
    theta = np.asarray(theta, dtype=float)
    g = np.asarray(gradient, dtype=float)
    if direction is None:
        rng = np.random.default_rng(seed)
        direction = rng.standard_normal(theta.size)
    d = np.asarray(direction, dtype=float)
    if hs is None:
        hs = np.max(np.abs(theta)) * 0.1 * 0.5 ** np.arange(6) / max(np.max(np.abs(d)), 1e-300)
    hs = np.asarray(hs, dtype=float)
    j0 = float(objective(theta))
    r0, r1 = [], []
    for h in hs:
        jh = float(objective(theta + h * d))
        r0.append(abs(jh - j0))
        r1.append(abs(jh - j0 - h * float(g @ d)))
    r0, r1 = np.array(r0), np.array(r1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate0 = np.log(r0[:-1] / r0[1:]) / np.log(hs[:-1] / hs[1:])
        rate1 = np.log(r1[:-1] / r1[1:]) / np.log(hs[:-1] / hs[1:])
    return {"h": hs, "remainder0": r0, "remainder1": r1, "rate0": rate0, "rate1": rate1,
            "directional_derivative": float(g @ d)}


def _edge_structure(vertices, polygons, nonorthogonal_correction=True):
    """Geometry-only decomposition ``A(kappa) = sum_e kappa(x_e) G_e`` of the FV diffusion operator."""
    probe = PolygonalHeatSolver(
        vertices, polygons, lambda x, y, temp: np.ones_like(np.asarray(x, dtype=float)), 1.0,
        temperature_dependent_diffusivity=True,
        nonorthogonal_correction=nonorthogonal_correction,
    )
    va = probe._vec_assembly
    if va is None:
        raise RuntimeError("vectorized edge assembly is unavailable for this mesh.")
    return probe, va


# ---------------------------------------------------------------------------
# Linear parabolic adjoint: conductivity / perfusion / source fields, Robin h
# ---------------------------------------------------------------------------


class LinearHeatAdjoint:
    """Backward-Euler discrete adjoint for ``u_t - div(kappa grad u) + k u = Q``.

    Parameter vector ``theta = [c (n_kappa), p (n_perf), s (n_src), h (robin only)]``

    * ``kappa(x) = sum_j c_j phi_j(x)`` evaluated at face midpoints (the
      operator is linear in face conductivities, ``A = sum_e kappa_e G_e``);
    * ``k(x) = sum_j p_j psi_j(x)`` at cell centers;
    * ``Q(x, t) = Q_0(x, t) + g(t) sum_j s_j chi_j(x)``;
    * ``h``: Robin coefficient in ``kappa du/dn + h (u - u_amb) = 0``.

    Boundary types: ``'dirichlet'`` (``bc_func(x, y, t)`` on boundary cells,
    the convention of all Thermalio FV solvers), ``'robin'`` and ``'flux'``
    (prescribed inward flux ``bc_func(x, y, t)`` at boundary faces).
    """

    def __init__(
        self,
        vertices,
        polygons,
        dt,
        conductivity_basis,
        perfusion_basis=(),
        source_basis=(),
        source_time_profile=None,
        background_source=None,
        bc_type="dirichlet",
        bc_func=None,
        robin_ambient=0.0,
        nonorthogonal_correction=True,
    ):
        self.dt = float(dt)
        self.bc_type = str(bc_type).lower()
        if self.bc_type not in {"dirichlet", "robin", "flux"}:
            raise ValueError("bc_type must be 'dirichlet', 'robin' or 'flux'.")
        probe, va = _edge_structure(vertices, polygons, nonorthogonal_correction)
        self.M = probe.M
        self.cell_areas = probe.cell_areas
        self.cell_centers = probe.cell_centers
        self.is_boundary = probe.is_boundary
        self._bidx = np.where(probe.is_boundary)[0]
        self._interior = np.ones(self.M, dtype=bool)
        if self.bc_type == "dirichlet":
            self._interior[self._bidx] = False
        self._va = va
        self.boundary_faces = probe.boundary_faces
        cx, cy = self.cell_centers[:, 0], self.cell_centers[:, 1]
        mid = va["midpoints"]
        self.kappa_basis = list(conductivity_basis)
        self.n_kappa = len(self.kappa_basis)
        if self.n_kappa == 0:
            raise ValueError("conductivity_basis must contain at least one function.")
        # Basis evaluated at interior faces (n_edges x n_kappa) and boundary faces.
        self.Phi_edges = np.column_stack([np.broadcast_to(np.asarray(f(mid[:, 0], mid[:, 1]), dtype=float), (mid.shape[0],)) for f in self.kappa_basis])
        bmid = self.boundary_faces["midpoints"]
        self.Phi_bfaces = np.column_stack([np.broadcast_to(np.asarray(f(bmid[:, 0], bmid[:, 1]), dtype=float), (bmid.shape[0],)) for f in self.kappa_basis]) if bmid.size else np.zeros((0, self.n_kappa))
        self.Psi = np.column_stack([np.broadcast_to(np.asarray(f(cx, cy), dtype=float), (self.M,)) for f in perfusion_basis]) if len(perfusion_basis) else np.zeros((self.M, 0))
        self.Chi = np.column_stack([np.broadcast_to(np.asarray(f(cx, cy), dtype=float), (self.M,)) for f in source_basis]) if len(source_basis) else np.zeros((self.M, 0))
        self.n_perf = self.Psi.shape[1]
        self.n_src = self.Chi.shape[1]
        self.n_robin = 1 if self.bc_type == "robin" else 0
        self.source_time_profile = source_time_profile or (lambda t: 1.0)
        self.background_source = background_source
        self.bc_func = bc_func or (lambda x, y, t: np.zeros_like(np.asarray(x, dtype=float)))
        self.robin_ambient = float(robin_ambient)
        names = [f"kappa_{j}" for j in range(self.n_kappa)]
        names += [f"perfusion_{j}" for j in range(self.n_perf)]
        names += [f"source_{j}" for j in range(self.n_src)]
        names += ["robin_h"] * self.n_robin
        self.parameter_names = tuple(names)

    @property
    def n_parameters(self):
        return self.n_kappa + self.n_perf + self.n_src + self.n_robin

    def split(self, theta):
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if theta.size != self.n_parameters:
            raise ValueError(f"theta must have {self.n_parameters} entries.")
        i = 0
        c = theta[i:i + self.n_kappa]; i += self.n_kappa
        p = theta[i:i + self.n_perf]; i += self.n_perf
        s = theta[i:i + self.n_src]; i += self.n_src
        h = float(theta[i]) if self.n_robin else None
        return c, p, s, h

    # -- operators -------------------------------------------------------------
    def diffusion_matrix(self, edge_kappa):
        va = self._va
        data = edge_kappa[va["all_emap"]] * va["all_geom"]
        return coo_matrix((data, (va["all_rows"], va["all_cols"])), shape=(self.M, self.M)).tocsr()

    def _robin_terms(self, c, h):
        faces = self.boundary_faces
        alpha_n = self.Phi_bfaces @ c
        d = faces["distances"]
        denom = alpha_n + h * d
        eff = alpha_n * h / denom
        return faces["cells"], faces["lengths"], alpha_n, d, denom, eff

    def operator(self, theta):
        c, p, s, h = self.split(theta)
        K = self.diffusion_matrix(self.Phi_edges @ c)
        if self.n_perf:
            K = K + diags(self.cell_areas * (self.Psi @ p), format="csr")
        b_const = np.zeros(self.M)
        if self.bc_type == "robin":
            cells, lengths, _, _, _, eff = self._robin_terms(c, h)
            K = K + diags(np.bincount(cells, weights=lengths * eff, minlength=self.M), format="csr")
            b_const = np.bincount(cells, weights=lengths * eff * self.robin_ambient, minlength=self.M)
        return K.tocsr(), b_const

    def _load(self, s, t):
        cx, cy = self.cell_centers[:, 0], self.cell_centers[:, 1]
        q = np.zeros(self.M)
        if self.background_source is not None:
            q += np.broadcast_to(np.asarray(self.background_source(cx, cy, t), dtype=float), (self.M,))
        if self.n_src:
            q += float(self.source_time_profile(t)) * (self.Chi @ s)
        return self.cell_areas * q

    def _boundary_load(self, t):
        if self.bc_type != "flux":
            return np.zeros(self.M)
        faces = self.boundary_faces
        mid = faces["midpoints"]
        q_in = np.broadcast_to(np.asarray(self.bc_func(mid[:, 0], mid[:, 1], t), dtype=float), (mid.shape[0],))
        return np.bincount(faces["cells"], weights=faces["lengths"] * q_in, minlength=self.M)

    # -- forward ---------------------------------------------------------------
    def forward(self, theta, u0, t_end):
        nsteps = max(int(round(float(t_end) / self.dt)), 1)
        dt = float(t_end) / nsteps
        c, p, s, h = self.split(theta)
        K, b_const = self.operator(theta)
        lhs = (diags(self.cell_areas, format="csr") + dt * K).tolil()
        if self.bc_type == "dirichlet":
            for i in self._bidx:
                lhs.rows[i] = [i]
                lhs.data[i] = [1.0]
        lhs = lhs.tocsr()
        lu = splu(lhs.tocsc())
        u = np.broadcast_to(np.asarray(u0, dtype=float), (self.M,)).astype(float)
        states = np.empty((nsteps + 1, self.M))
        states[0] = u
        cx, cy = self.cell_centers[:, 0], self.cell_centers[:, 1]
        for n in range(1, nsteps + 1):
            t = n * dt
            rhs = self.cell_areas * u + dt * (self._load(s, t) + b_const + self._boundary_load(t))
            if self.bc_type == "dirichlet":
                g = np.broadcast_to(np.asarray(self.bc_func(cx, cy, t), dtype=float), (self.M,))
                rhs[self._bidx] = g[self._bidx]
            u = lu.solve(rhs)
            states[n] = u
        return states, dt, lhs, lu

    def objective(self, theta, u0, t_end, observation_times, observed, sensor_indices=None, weights=None):
        states, dt, _, _ = self.forward(theta, u0, t_end)
        obs = _Observations(observation_times, observed, self.M, sensor_indices, weights)
        steps = obs.match_steps(dt * np.arange(states.shape[0]))
        return obs.evaluate(states[steps])[0]

    # -- adjoint ---------------------------------------------------------------
    def gradient(self, theta, u0, t_end, observation_times, observed, sensor_indices=None, weights=None):
        theta = np.asarray(theta, dtype=float)
        states, dt, lhs, lu = self.forward(theta, u0, t_end)
        n = states.shape[0] - 1
        obs = _Observations(observation_times, observed, self.M, sensor_indices, weights)
        steps = obs.match_steps(dt * np.arange(n + 1))
        J, pred, resid = obs.evaluate(states[steps])
        lu_T = splu(lhs.T.tocsc())
        mask = self._interior.astype(float)
        c, p, s, h = self.split(theta)

        va = self._va
        edge_w = np.zeros(va["n_edges"])
        perf_w = np.zeros(self.M)
        src_g = np.zeros(self.n_src)
        robin_face_w = None
        if self.bc_type == "robin":
            cells, lengths, alpha_n, d, denom, eff = self._robin_terms(c, h)
            robin_face_w = np.zeros(cells.size)
        adjoints = np.zeros_like(states)
        lam_next = np.zeros(self.M)
        for m in range(n, 0, -1):
            rhs = obs.field_gradient(m, resid) + self.cell_areas * mask * lam_next
            lam = lu_T.solve(rhs)
            adjoints[m] = lam
            lam_i = lam * mask
            u = states[m]
            # -lambda . dR/dtheta, dR/dc_j = dt A_j u  -> per-edge accumulation
            edge_w -= dt * np.bincount(
                va["all_emap"], weights=va["all_geom"] * lam_i[va["all_rows"]] * u[va["all_cols"]],
                minlength=va["n_edges"],
            )
            perf_w -= dt * lam_i * self.cell_areas * u
            if self.n_src:
                src_g += dt * float(self.source_time_profile(m * dt)) * (self.Chi.T @ (self.cell_areas * lam_i))
            if robin_face_w is not None:
                robin_face_w -= dt * lam_i[cells] * lengths * (u[cells] - self.robin_ambient)
            lam_next = lam

        grad_c = self.Phi_edges.T @ edge_w
        parts = [grad_c]
        if self.n_perf:
            parts.append(self.Psi.T @ perf_w)
        if self.n_src:
            parts.append(src_g)
        if robin_face_w is not None:
            de_dalpha = h * h * d / denom**2
            de_dh = alpha_n**2 / denom**2
            parts[0] = parts[0] + self.Phi_bfaces.T @ (robin_face_w * de_dalpha)
            parts.append(np.array([float(np.sum(robin_face_w * de_dh))]))
        grad = np.concatenate(parts)
        return AdjointGradient(
            objective=J, gradient=grad, parameter_names=self.parameter_names,
            predicted=pred, residual=resid, observation_steps=steps,
            forward_states=states, adjoint_states=adjoints,
            extras={"edge_sensitivity": edge_w, "dt": dt},
        )


# ---------------------------------------------------------------------------
# Nonlinear enthalpy (Stefan / Cattaneo-Stefan) adjoint
# ---------------------------------------------------------------------------


_ENTHALPY_PARAMETERS = ("latent_heat", "alpha", "tau")


def _scalar_alpha(solver):
    alpha = solver._base.alpha
    if callable(alpha) or np.asarray(alpha).ndim != 0:
        raise ValueError("enthalpy_stefan_adjoint requires a scalar diffusivity alpha.")
    return float(alpha)


def enthalpy_stefan_adjoint(
    solver,
    u0,
    t_end,
    observation_times,
    observed,
    sensor_indices=None,
    weights=None,
    du0=None,
    parameters=("latent_heat", "alpha"),
    t0=0.0,
):
    """Discrete adjoint of :class:`EnthalpyStefanSolver` (Stefan / Cattaneo-Stefan).

    Runs the forward solve (honouring the solver's uniform or adaptive step
    sequence; ``observation_times`` are added as output times so they lie on
    the grid), then integrates the adjoint recurrence backwards over the
    *accepted* steps.  The step sequence is treated as fixed (the step-size
    controller is not differentiated), which is the standard convention for
    adaptive discrete adjoints.

    Observations are temperatures ``T(H)`` at sensor cells.  Supported
    parameters: ``latent_heat``, ``alpha`` (scalar diffusivity) and ``tau``
    (requires ``solver.tau > 0``).  Boundary types: ``dirichlet`` and ``flux``.
    """
    if not isinstance(solver, EnthalpyStefanSolver):
        raise TypeError("solver must be an EnthalpyStefanSolver.")
    if solver.bc_type not in {"dirichlet", "flux"}:
        raise ValueError("enthalpy_stefan_adjoint supports 'dirichlet' and 'flux' boundaries.")
    params = tuple(parameters)
    for name in params:
        if name not in _ENTHALPY_PARAMETERS:
            raise ValueError(f"unknown parameter {name!r}; choose from {_ENTHALPY_PARAMETERS}.")
    if "tau" in params and solver.tau <= 0.0:
        raise ValueError("'tau' sensitivity needs relaxation_time > 0.")
    alpha = _scalar_alpha(solver)
    A1 = solver.A / alpha
    model = solver.model
    tau = solver.tau
    area = solver.cell_areas
    M = solver.M
    mask = solver._interior.astype(float)
    bidx = solver._boundary_idx
    dirichlet = solver.bc_type == "dirichlet"

    obs = _Observations(observation_times, observed, M, sensor_indices, weights)
    temp0 = np.broadcast_to(np.asarray(u0, dtype=float), (M,)).astype(float)
    solver.solve(temp0, t0, t_end, du0=du0, output_times=list(obs.times), store_history=True)
    hist = solver.history
    times = np.asarray(hist["t"])
    H = np.asarray(hist["H"])
    ks = np.asarray(hist["k"])
    kps = np.asarray(hist["k_prev"])
    N = H.shape[0] - 1
    steps = obs.match_steps(times)
    T = np.array([solver.temperature(h) for h in H])
    J, pred, resid = obs.evaluate(T[steps])
    cap = np.array([np.asarray(model.effective_heat_capacity(tt), dtype=float) for tt in T])

    coeffs = [None] + [three_level_coefficients(ks[n - 1], kps[n - 1], tau) for n in range(1, N + 1)]
    K_ops = [None]
    for n in range(1, N + 1):
        K, _ = solver._operator(times[n])
        K_ops.append(K)

    # Startup level H^{-1} (tau > 0) and its parameter derivatives.
    k0 = ks[0]
    if tau > 0.0:
        h_m1 = solver._initial_previous_level(H[0], temp0, t0, k0, du0)
        v_temp = np.zeros(M) if du0 is None else np.broadcast_to(np.asarray(du0, dtype=float), (M,)).astype(float)
        v0 = np.asarray(model.effective_heat_capacity(temp0), dtype=float) * v_temp
        K0, b0 = solver._operator(t0)
        a0 = (solver._source(t0) + (b0 - K0 @ temp0) / area - v0) / tau
    else:
        h_m1 = H[0].copy()

    def H_level(n):
        return h_m1 if n == -1 else H[n]

    # ---- adjoint sweep ----
    lam = np.zeros((N + 3, M))  # lam[n] for n = 1..N; lam[N+1] = lam[N+2] = 0
    for m in range(N, 0, -1):
        g = obs.field_gradient(m, resid) / cap[m]  # dJ/dH^m = T'(H) dJ/dT
        if m + 1 <= N:
            g += area * mask * coeffs[m + 1][1] * lam[m + 1]
        if m + 2 <= N:
            g += area * mask * coeffs[m + 2][2] * lam[m + 2]
        Jm = solver.jacobian(H[m], coeffs[m][0], K_ops[m])
        lam[m] = splu(Jm.T.tocsc()).solve(g)

    # ---- gradient assembly ----
    grads = {}
    phi0 = np.asarray(model.liquid_fraction(temp0), dtype=float)
    d_init_terms = area * mask * coeffs[1][1] * lam[1]
    if N >= 2:
        d_init_terms = d_init_terms + area * mask * coeffs[2][2] * lam[2]
    s1_term = area * mask * coeffs[1][2] * lam[1]

    if "latent_heat" in params:
        total = 0.0
        for m in range(1, N + 1):
            dTdL = temperature_sensitivity_to_latent_heat(model, T[m])
            total += float(obs.field_gradient(m, resid) @ dTdL)  # explicit observation term
            dR = K_ops[m] @ dTdL
            if dirichlet:
                bc = solver._bc_values(times[m])
                dR = dR * mask
                dR[bidx] = -np.asarray(model.liquid_fraction(bc), dtype=float)[bidx]
            total -= float(lam[m] @ dR)
        total += float(d_init_terms @ phi0)
        if tau > 0.0:
            # v0 = c_app(T0) dT0/dt and dc_app/dL|_T = dphi/dT.
            dphi0 = _dphi_dT(model, temp0)
            dv0 = dphi0 * v_temp
            dh_m1 = phi0 - k0 * dv0 + 0.5 * k0 * k0 * (-dv0 / tau)
            total += float(s1_term @ dh_m1)
        grads["latent_heat"] = total

    if "alpha" in params:
        total = 0.0
        for m in range(1, N + 1):
            total -= float(lam[m] @ (mask * (A1 @ T[m])))
        if tau > 0.0:
            dh_m1 = 0.5 * k0 * k0 * (-(A1 @ temp0) / area / tau)
            total += float(s1_term @ dh_m1)
        grads["alpha"] = total

    if "tau" in params:
        total = 0.0
        for m in range(1, N + 1):
            k, kp = ks[m - 1], kps[m - 1]
            span = k + kp
            d2 = 2.0 / span * ((H[m] - H_level(m - 1)) / k - (H_level(m - 1) - H_level(m - 2)) / kp)
            total -= float(lam[m] @ (area * mask * d2))
        dh_m1 = 0.5 * k0 * k0 * (-a0 / tau)
        total += float(s1_term @ dh_m1)
        grads["tau"] = total

    return AdjointGradient(
        objective=J,
        gradient=np.array([grads[p] for p in params]),
        parameter_names=params,
        predicted=pred,
        residual=resid,
        observation_steps=steps,
        forward_states=T,
        adjoint_states=lam[1:N + 1],
        extras={"times": times, "dts": ks, "enthalpy": H},
    )


def _dphi_dT(model, temperature):
    if hasattr(model, "d_liquid_fraction_d_temperature"):
        return np.asarray(model.d_liquid_fraction_d_temperature(temperature), dtype=float)
    raise AttributeError("phase model must provide d_liquid_fraction_d_temperature for tau-startup sensitivities.")


# ---------------------------------------------------------------------------
# Fractional (Caputo L1) adjoint
# ---------------------------------------------------------------------------


def _l1_coefficients(m, b):
    """Coefficients ``c_j`` (j = 0..m) of ``sum_k b_k (u^{m-k} - u^{m-1-k})``."""
    c = np.zeros(m + 1)
    j = np.arange(m + 1)
    c[1:] += b[m - j[1:]]
    c[:-1] -= b[m - 1 - j[:-1]]
    return c


def fractional_heat_adjoint(
    vertices,
    polygons,
    alpha,
    dt,
    beta,
    u0,
    t_end,
    observation_times,
    observed,
    sensor_indices=None,
    weights=None,
    bc_type="dirichlet",
    bc_func=None,
    source_func=None,
):
    """Discrete adjoint of the Caputo L1 scheme w.r.t. ``beta`` and scalar ``alpha``.

    Forward: ``sigma area sum_j c^{(m)}_j u^j + K u^m = area Q^m + b^m`` with
    ``sigma = dt^{-beta}/Gamma(2-beta)`` and L1 weights
    ``b_k = (k+1)^{1-beta} - k^{1-beta}`` (identical to
    :class:`FractionalHeatSolver` with full memory).  The adjoint inherits the
    memory: ``lambda_m`` couples to all *later* adjoint states.
    """
    from .transport import FractionalHeatSolver

    beta = float(beta)
    alpha = float(alpha)
    ref = FractionalHeatSolver(vertices, polygons, alpha, dt, beta, bc_type=bc_type,
                               bc_func=bc_func, source_func=source_func)
    if ref.bc_type not in {"dirichlet", "flux"}:
        raise ValueError("fractional_heat_adjoint supports 'dirichlet' and 'flux' boundaries.")
    M = ref.M
    area = ref.cell_areas
    A1 = ref.A / alpha
    N, dt = ref._uniform_schedule(0.0, t_end, dt)
    mask = np.ones(M)
    bidx = ref._boundary_idx
    dirichlet = ref.bc_type == "dirichlet"
    if dirichlet:
        mask[bidx] = 0.0
    k = np.arange(N + 1, dtype=float)
    b = (k + 1.0) ** (1.0 - beta) - k ** (1.0 - beta)
    with np.errstate(divide="ignore", invalid="ignore"):
        logk = np.where(k > 0, np.log(np.maximum(k, 1.0)), 0.0)
    db = -((k + 1.0) ** (1.0 - beta)) * np.log(k + 1.0) + np.where(k > 0, k ** (1.0 - beta) * logk, 0.0)
    sigma = dt ** (-beta) / gamma_fn(2.0 - beta)
    dsigma = sigma * (-np.log(dt) + digamma(2.0 - beta))

    lhs = (sigma * diags(area, format="csr") + ref.A).tocsr()
    if dirichlet:
        lhs = ref._apply_dirichlet(lhs)
    lu = splu(lhs.tocsc())
    u = np.zeros((N + 1, M))
    u[0] = np.broadcast_to(np.asarray(u0, dtype=float), (M,))
    for m in range(1, N + 1):
        t = m * dt
        c = _l1_coefficients(m, b)
        hist = c[:-1] @ u[:m]
        _, bvec = ref._boundary_system(t)
        rhs = area * ref._source(t) + bvec - sigma * area * hist
        if dirichlet:
            g = ref._bc_values(t)
            rhs[bidx] = g[bidx]
        u[m] = lu.solve(rhs)

    obs = _Observations(observation_times, observed, M, sensor_indices, weights)
    steps = obs.match_steps(dt * np.arange(N + 1))
    J, pred, resid = obs.evaluate(u[steps])
    lu_T = splu(lhs.T.tocsc())
    lam = np.zeros((N + 1, M))
    coeff_cache = {m: _l1_coefficients(m, b) for m in range(1, N + 1)}
    for m in range(N, 0, -1):
        rhs = obs.field_gradient(m, resid)
        if m < N:
            later = np.array([coeff_cache[l][m] for l in range(m + 1, N + 1)])
            rhs = rhs - sigma * area * mask * (later @ lam[m + 1:])
        lam[m] = lu_T.solve(rhs)

    g_beta = 0.0
    g_alpha = 0.0
    for m in range(1, N + 1):
        c = coeff_cache[m]
        dc = _l1_coefficients(m, db)
        v = c @ u[:m + 1]
        dv = dc @ u[:m + 1]
        lm = lam[m] * mask
        g_beta -= float(lm @ (area * (dsigma * v + sigma * dv)))
        g_alpha -= float(lm @ (A1 @ u[m]))
    return AdjointGradient(
        objective=J, gradient=np.array([g_beta, g_alpha]), parameter_names=("beta", "alpha"),
        predicted=pred, residual=resid, observation_steps=steps,
        forward_states=u, adjoint_states=lam, extras={"dt": dt},
    )
