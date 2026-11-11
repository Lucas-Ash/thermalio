"""Enthalpy-primary phase-change solver with adaptive time stepping.

This module implements the *enthalpy formulation* of (optionally non-Fourier)
Stefan problems on Thermalio's polygonal finite-volume discretization::

    tau H_tt + H_t - div(alpha grad T(H)) = Q,        H = c T + L phi(T)

The primary unknown is the volumetric enthalpy ``H``; temperature is recovered
pointwise through the monotone inverse ``T = T(H)``.  Compared with the
apparent-heat-capacity (temperature-primary) formulation used by
:class:`heat_solver.transport.HyperbolicStefanSolver` this has two structural
advantages for stiff phase change:

* **Exact discrete energy conservation.**  The time difference acts on ``H``,
  so latent heat released/absorbed in a step is accounted for exactly, even
  when a cell jumps across the whole mushy interval in one step (the classical
  "latent-heat jumping" failure of apparent-capacity schemes).
* **Bounded nonlinearity.**  The Newton Jacobian is
  ``gamma * diag(area) + K diag(dT/dH)`` with ``0 < dT/dH = 1/c_app <= 1/c``,
  so a narrow mushy zone (huge ``c_app``) makes the nonlinearity *weaker*
  rather than producing an O(L / width) capacity spike in the operator.

Time integration uses a variable-step three-level scheme that reduces to
backward Euler for ``tau = 0`` and to the Cattaneo three-level scheme of
:class:`~heat_solver.transport.HyperbolicHeatSolver` for ``tau > 0`` on uniform
steps.  :class:`AdaptiveStepOptions` enables error-controlled step selection
(extrapolation-based local error estimate on ``H``) combined with
failure-driven step reduction when Newton does not converge.

Phase models need ``enthalpy(T)``, ``temperature_from_enthalpy(H)``,
``effective_heat_capacity(T)`` and ``liquid_fraction(T)`` plus ``latent_heat``
and ``specific_heat`` attributes; both
:class:`~heat_solver.phase_change.ApparentHeatCapacityModel` (piecewise-linear
mushy ramp) and :class:`SmoothPhaseChangeModel` (``tanh`` ramp) qualify.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu

from .transport import _TransportBase


# ---------------------------------------------------------------------------
# Phase models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SmoothPhaseChangeModel:
    """Smooth (``C^infinity``) enthalpy law ``H = c T + L phi(T)``.

    ``phi(T) = (1 + tanh((T - T_m) / delta)) / 2`` is the liquid fraction.  A
    smooth law is convenient for manufactured-solution verification and for
    discrete-adjoint gradient checks (the piecewise-linear ramp of
    :class:`ApparentHeatCapacityModel` has kinks where Newton and finite
    differences are only piecewise smooth).  ``latent_heat = 0`` gives a plain
    linear material ``H = c T``.
    """

    melt_temperature: float = 0.0
    half_width: float = 0.1
    latent_heat: float = 1.0
    specific_heat: float = 1.0

    def __post_init__(self):
        if self.half_width <= 0.0:
            raise ValueError("half_width must be positive.")
        if self.latent_heat < 0.0:
            raise ValueError("latent_heat must be non-negative.")
        if self.specific_heat <= 0.0:
            raise ValueError("specific_heat must be positive.")

    def _z(self, temperature):
        return (np.asarray(temperature, dtype=float) - self.melt_temperature) / self.half_width

    def liquid_fraction(self, temperature):
        return 0.5 * (1.0 + np.tanh(self._z(temperature)))

    def d_liquid_fraction_d_temperature(self, temperature):
        th = np.tanh(self._z(temperature))
        return 0.5 * (1.0 - th**2) / self.half_width

    def effective_heat_capacity(self, temperature):
        return self.specific_heat + self.latent_heat * self.d_liquid_fraction_d_temperature(temperature)

    def enthalpy(self, temperature):
        temperature = np.asarray(temperature, dtype=float)
        return self.specific_heat * temperature + self.latent_heat * self.liquid_fraction(temperature)

    def temperature_from_enthalpy(self, enthalpy, tol=1e-14, max_iters=200):
        """Invert ``H(T)`` with a bracketed, safeguarded Newton iteration.

        Because ``0 <= phi <= 1`` the root lies in ``[(H - L)/c, H/c]``.  The
        iteration starts from the inverse of the equivalent linear ramp over
        ``[T_m - delta, T_m + delta]`` and falls back to bisection whenever
        Newton leaves the bracket or fails to halve ``|H(T) - H|`` (``rtsafe``).
        """
        h = np.asarray(enthalpy, dtype=float)
        c = self.specific_heat
        L = self.latent_heat
        if L == 0.0:
            return h / c
        lo = (h - L) / c
        hi = h / c
        d = self.half_width
        tm = self.melt_temperature
        h_lo = c * (tm - d)
        h_hi = c * (tm + d) + L
        t = np.where(h <= h_lo, h / c, np.where(h >= h_hi, (h - L) / c,
                     (tm - d) + (h - h_lo) * (2.0 * d) / (h_hi - h_lo)))
        t = np.clip(t, lo, hi)
        scale = np.maximum(1.0, np.abs(h))
        f_prev = np.full_like(t, np.inf)
        for _ in range(max_iters):
            f = self.enthalpy(t) - h
            lo = np.where(f < 0.0, t, lo)
            hi = np.where(f > 0.0, t, hi)
            t_new = t - f / self.effective_heat_capacity(t)
            bisect = (
                (t_new <= lo) | (t_new >= hi) | ~np.isfinite(t_new)
                | (np.abs(f) > 0.5 * np.abs(f_prev))
            )
            t_new = np.where(bisect, 0.5 * (lo + hi), t_new)
            f_prev = np.where(bisect, np.inf, f)
            done = (np.abs(t_new - t) <= tol * scale) | (np.abs(f) <= 4e-16 * scale)
            t = np.where(np.abs(f) <= 4e-16 * scale, t, t_new)
            if np.all(done):
                break
        return t


def _model_with_latent_heat(model, latent_heat):
    return replace(model, latent_heat=float(latent_heat))


def temperature_sensitivity_to_latent_heat(model, temperature):
    """``dT/dL`` at fixed enthalpy: ``-phi(T) / c_app(T)`` (implicit differentiation of H = cT + L phi)."""
    return -np.asarray(model.liquid_fraction(temperature), dtype=float) / np.asarray(
        model.effective_heat_capacity(temperature), dtype=float
    )


# ---------------------------------------------------------------------------
# Adaptive step control
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdaptiveStepOptions:
    """Error-controlled + failure-driven time-step selection.

    The local truncation error of the backward-Euler enthalpy update is
    estimated from the gap between the implicit solution and a linear
    extrapolation of the two previous levels (see :func:`local_error_estimate`).
    A step is accepted when the scaled error norm ``err <= 1``; the next step is
    ``dt * clip(safety * err**(-1/2), shrink_min, growth_max)``.  If Newton
    fails, the step is retried with ``dt * failure_shrink``.
    """

    rtol: float = 1e-3
    atol: float = 1e-4
    dt_min: float = 1e-9
    dt_max: float = np.inf
    safety: float = 0.9
    growth_max: float = 2.0
    shrink_min: float = 0.2
    failure_shrink: float = 0.5
    max_rejections: int = 60
    norm: str = "max"

    def __post_init__(self):
        if self.rtol < 0 or self.atol < 0 or (self.rtol == 0 and self.atol == 0):
            raise ValueError("rtol/atol must be non-negative and not both zero.")
        if not (0.0 < self.shrink_min < 1.0 < self.growth_max):
            raise ValueError("require 0 < shrink_min < 1 < growth_max.")
        if not (0.0 < self.failure_shrink < 1.0):
            raise ValueError("failure_shrink must lie in (0, 1).")
        if self.norm not in {"max", "rms"}:
            raise ValueError("norm must be 'max' or 'rms'.")


def local_error_estimate(h_new, h_n, h_nm1, k, k_prev):
    """Extrapolation-based estimate of the backward-Euler local error.

    With ``H^p = H^n + (k/k_prev)(H^n - H^{n-1})`` (linear extrapolation) one
    has ``H^{n+1} - H^p = k (2k + k_prev)/2 * H'' + O(k^3)`` while the BE local
    error is ``k^2/2 * H''``; hence ``LTE ~ k/(2k + k_prev) * (H^{n+1} - H^p)``.
    """
    predictor = h_n + (k / k_prev) * (h_n - h_nm1)
    return (k / (2.0 * k + k_prev)) * (h_new - predictor)


def _scaled_error_norm(err, h_new, h_n, opts):
    scale = opts.atol + opts.rtol * np.maximum(np.abs(h_new), np.abs(h_n))
    ratio = np.abs(err) / scale
    if opts.norm == "max":
        return float(np.max(ratio))
    return float(np.sqrt(np.mean(ratio**2)))


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


class NewtonFailure(RuntimeError):
    """Raised internally when a Newton step does not converge."""


def three_level_coefficients(k, k_prev, tau):
    """Coefficients of ``tau D2 H + D1 H = gamma H^{n+1} - p H^n - s H^{n-1}``.

    ``D2`` is the variable-step second difference; ``D1`` is backward Euler for
    ``tau = 0`` and the variable-step centered difference for ``tau > 0``.
    """
    span = k + k_prev
    gamma = 2.0 * tau / (span * k)
    p = 2.0 * tau / span * (1.0 / k + 1.0 / k_prev)
    s = -2.0 * tau / (span * k_prev)
    if tau == 0.0:
        gamma += 1.0 / k
        p += 1.0 / k
    else:
        gamma += 1.0 / span
        s += 1.0 / span
    return gamma, p, s


class EnthalpyStefanSolver(_TransportBase):
    """Enthalpy-primary (non-Fourier) Stefan solver on polygonal meshes.

    Solves ``tau H_tt + H_t - div(alpha grad T(H)) = Q`` with Newton's method on
    the enthalpy at every step.  ``tau = 0`` gives the classical enthalpy method.

    Parameters
    ----------
    phase_model :
        Enthalpy law (``ApparentHeatCapacityModel`` or ``SmoothPhaseChangeModel``).
    relaxation_time : float
        Cattaneo relaxation time ``tau >= 0``.
    newton_options : dict
        ``max_iters`` (25), ``tol`` (1e-10, relative update), ``rtol_residual``
        (1e-10, scaled residual) and ``line_search`` (True).
    adaptive : AdaptiveStepOptions or None
        ``None`` runs uniform steps of size ``dt`` (Newton failure then raises
        or is recorded, see ``raise_on_nonconvergence``).  Otherwise ``dt`` is
        only the initial step.
    """

    def __init__(
        self,
        vertices,
        polygons,
        alpha,
        dt,
        phase_model,
        relaxation_time=0.0,
        newton_options=None,
        adaptive=None,
        raise_on_nonconvergence=True,
        **kwargs,
    ):
        if "phase_change_model" in kwargs:
            raise TypeError("Pass the enthalpy law as phase_model, not phase_change_model.")
        super().__init__(vertices, polygons, alpha, dt, **kwargs)
        self.model = phase_model
        self.tau = float(relaxation_time)
        if self.tau < 0.0:
            raise ValueError("relaxation_time must be non-negative.")
        opts = {"max_iters": 25, "tol": 1e-10, "rtol_residual": 1e-10, "line_search": True}
        if newton_options:
            opts.update(dict(newton_options))
        self.newton_options = opts
        if adaptive is True:
            adaptive = AdaptiveStepOptions()
        self.adaptive = adaptive
        self.raise_on_nonconvergence = bool(raise_on_nonconvergence)
        self._interior = np.ones(self.M, dtype=bool)
        if self.bc_type == "dirichlet":
            self._interior[self._boundary_idx] = False

    # -- helpers -----------------------------------------------------------
    def enthalpy(self, temperature):
        return np.asarray(self.model.enthalpy(temperature), dtype=float)

    def temperature(self, enthalpy):
        return np.asarray(self.model.temperature_from_enthalpy(enthalpy), dtype=float)

    def _operator(self, t):
        B, b = self._boundary_system(t)
        K = self.A if B is None else (self.A + B).tocsr()
        return K, b

    def _residual(self, h, gamma, rhs_hist, K, b, dir_h):
        """Nonlinear residual; Dirichlet rows read ``H_i - H(g_i)``."""
        temp = self.temperature(h)
        res = self.cell_areas * (gamma * h - rhs_hist) + K @ temp - b
        if dir_h is not None:
            idx = self._boundary_idx
            res[idx] = h[idx] - dir_h[idx]
        return res, temp

    def jacobian(self, h, gamma, K):
        """Newton Jacobian ``gamma diag(area) + K diag(dT/dH)`` with identity Dirichlet rows."""
        temp = self.temperature(h)
        dTdH = 1.0 / np.asarray(self.model.effective_heat_capacity(temp), dtype=float)
        J = (diags(gamma * self.cell_areas, format="csr") + K @ diags(dTdH, format="csr")).tocsr()
        if self.bc_type == "dirichlet":
            J = self._apply_dirichlet(J)
        return J

    def _newton(self, h_guess, gamma, rhs_hist, K, b, dir_h):
        max_iters = int(self.newton_options["max_iters"])
        tol = float(self.newton_options["tol"])
        rtol_res = float(self.newton_options["rtol_residual"])
        line_search = bool(self.newton_options["line_search"])
        h = h_guess.copy()
        res, _ = self._residual(h, gamma, rhs_hist, K, b, dir_h)
        res_scale = max(1.0, float(np.max(np.abs(self.cell_areas * gamma * h))))
        res_norm = float(np.linalg.norm(res))
        for it in range(1, max_iters + 1):
            J = self.jacobian(h, gamma, K)
            delta = splu(J.tocsc()).solve(-res)
            step = 1.0
            while True:
                h_try = h + step * delta
                res_try, _ = self._residual(h_try, gamma, rhs_hist, K, b, dir_h)
                norm_try = float(np.linalg.norm(res_try))
                if not line_search or norm_try <= (1.0 - 1e-4 * step) * res_norm or step < 1.0 / 64:
                    break
                step *= 0.5
            h, res, res_norm = h_try, res_try, norm_try
            upd = step * float(np.max(np.abs(delta)))
            if not np.all(np.isfinite(h)):
                raise NewtonFailure("non-finite Newton iterate")
            max_res = float(np.max(np.abs(res)))
            if max_res <= rtol_res * res_scale or upd <= tol * max(1.0, float(np.max(np.abs(h)))):
                return h, it, max_res / res_scale
        raise NewtonFailure(f"Newton did not converge in {max_iters} iterations")

    def _initial_previous_level(self, h0, t0_temp, t0, k0, du0):
        """Second-order startup level ``H^{-1}`` for ``tau > 0`` (Taylor expansion)."""
        if self.tau == 0.0:
            return h0.copy()
        v_temp = np.zeros(self.M) if du0 is None else np.broadcast_to(
            np.asarray(du0, dtype=float), (self.M,)
        ).astype(float)
        v0 = np.asarray(self.model.effective_heat_capacity(t0_temp), dtype=float) * v_temp
        K0, b0 = self._operator(t0)
        a0 = (self._source(t0) + (b0 - K0 @ t0_temp) / self.cell_areas - v0) / self.tau
        return h0 - k0 * v0 + 0.5 * k0 * k0 * a0

    # -- main driver ---------------------------------------------------------
    def solve(self, u0, t0, t_end, du0=None, output_times=None, store_history=False):
        """Integrate from ``t0`` to ``t_end``; returns ``(t, T)``.

        ``u0`` is the initial *temperature*; ``du0`` its time derivative (only
        used for ``tau > 0``).  ``output_times`` are hit exactly (also in
        adaptive mode) and their temperature fields are stored in
        ``self.outputs``.  ``store_history=True`` keeps every accepted level
        (needed by :class:`heat_solver.adjoint.EnthalpyStefanAdjoint`).
        """
        t0 = float(t0)
        t_end = float(t_end)
        temp0 = np.broadcast_to(np.asarray(u0, dtype=float), (self.M,)).astype(float)
        h_n = self.enthalpy(temp0)
        outputs = sorted(float(s) for s in (output_times or []) if t0 < float(s) <= t_end + 1e-14)
        targets = sorted(set(outputs + [t_end]))

        report = {
            "times": [t0], "dts": [], "newton_iterations": [], "residuals": [],
            "error_estimates": [], "accepted_steps": 0, "rejected_error": 0,
            "rejected_newton": 0, "failed_steps": 0, "converged": True,
            "total_newton_iterations": 0, "energy": [float(np.sum(self.cell_areas * h_n))],
            "energy_balance_residual": [], "energy_input": [0.0],
        }
        self.outputs = {}
        self.history = {"t": [t0], "H": [h_n.copy()], "k": [], "k_prev": []} if store_history else None
        self.solve_report = report
        if t_end <= t0:
            return t0, temp0

        adaptive = self.adaptive
        k = float(self.dt) if adaptive is None else min(float(self.dt), adaptive.dt_max)
        if adaptive is None:
            nsteps, k = self._uniform_schedule(t0, t_end, self.dt)
        h_nm1 = self._initial_previous_level(h_n, temp0, t0, k, du0)
        k_prev = k
        have_two_levels = False  # error estimate needs a genuine previous step
        t = t0
        target_idx = 0
        rejections = 0

        while t < t_end - 1e-13 * max(1.0, abs(t_end)):
            target = targets[target_idx]
            remaining = target - t
            if adaptive is not None and k < remaining < 2.0 * k:
                # Avoid a sliver step just before an output time: split the rest evenly.
                k_try = 0.5 * remaining
            else:
                k_try = min(k, remaining)
            hit_target = k_try >= remaining - 1e-13 * max(1.0, abs(target))
            if hit_target:
                k_try = remaining
            t_next = t + k_try
            gamma, p, s = three_level_coefficients(k_try, k_prev, self.tau)
            q_next = self._source(t_next)
            rhs_hist = q_next + p * h_n + s * h_nm1
            K, b = self._operator(t_next)
            dir_h = None
            if self.bc_type == "dirichlet":
                dir_h = self.enthalpy(self._bc_values(t_next))
            guess = h_n + (k_try / k_prev) * (h_n - h_nm1) if have_two_levels else h_n.copy()
            if dir_h is not None:
                guess[self._boundary_idx] = dir_h[self._boundary_idx]
            try:
                h_new, iters, resid = self._newton(guess, gamma, rhs_hist, K, b, dir_h)
            except NewtonFailure:
                report["total_newton_iterations"] += int(self.newton_options["max_iters"])
                if adaptive is None:
                    report["failed_steps"] += 1
                    report["converged"] = False
                    if self.raise_on_nonconvergence:
                        raise RuntimeError(
                            "Enthalpy Newton solve did not converge; reduce dt or enable adaptive stepping."
                        )
                    return t, self.temperature(h_n)
                report["rejected_newton"] += 1
                rejections += 1
                k = k_try * adaptive.failure_shrink
                if k < adaptive.dt_min or rejections > adaptive.max_rejections:
                    report["converged"] = False
                    report["failed_steps"] += 1
                    raise RuntimeError("Adaptive enthalpy solve: step size underflow after Newton failures.")
                continue
            report["total_newton_iterations"] += iters

            err_norm = None
            if adaptive is not None and have_two_levels and self.tau == 0.0:
                lte = local_error_estimate(h_new, h_n, h_nm1, k_try, k_prev)
                err_norm = _scaled_error_norm(lte, h_new, h_n, adaptive)
                if err_norm > 1.0 and k_try > adaptive.dt_min * 1.0001:
                    report["rejected_error"] += 1
                    rejections += 1
                    if rejections > adaptive.max_rejections:
                        raise RuntimeError("Adaptive enthalpy solve: too many rejected steps.")
                    k = max(adaptive.dt_min, k_try * max(adaptive.shrink_min, adaptive.safety * err_norm ** -0.5))
                    continue
            elif adaptive is not None and have_two_levels and self.tau > 0.0:
                # Curvature indicator for the three-level scheme (same estimator,
                # interpreted as a step-size indicator rather than an exact LTE).
                lte = local_error_estimate(h_new, h_n, h_nm1, k_try, k_prev)
                err_norm = _scaled_error_norm(lte, h_new, h_n, adaptive)
                if err_norm > 1.0 and k_try > adaptive.dt_min * 1.0001:
                    report["rejected_error"] += 1
                    rejections += 1
                    if rejections > adaptive.max_rejections:
                        raise RuntimeError("Adaptive enthalpy solve: too many rejected steps.")
                    k = max(adaptive.dt_min, k_try * max(adaptive.shrink_min, adaptive.safety * err_norm ** -0.5))
                    continue

            # ---- accept ----
            rejections = 0
            temp_new = self.temperature(h_new)
            interior = self._interior
            flux_term = (self.cell_areas * q_next - K @ temp_new + b)
            if self.tau == 0.0:
                balance = float(np.sum(self.cell_areas[interior] * (h_new - h_n)[interior])
                                - k_try * np.sum(flux_term[interior]))
            else:
                balance = float("nan")
            report["energy_balance_residual"].append(balance)
            report["energy_input"].append(report["energy_input"][-1] + k_try * float(np.sum(flux_term[interior])))
            if self.history is not None:
                self.history["t"].append(t_next)
                self.history["H"].append(h_new.copy())
                self.history["k"].append(k_try)
                self.history["k_prev"].append(k_prev)
            h_nm1, h_n = h_n, h_new
            k_prev = k_try
            t = t_next
            have_two_levels = True
            report["times"].append(t)
            report["dts"].append(k_try)
            report["newton_iterations"].append(iters)
            report["residuals"].append(resid)
            report["error_estimates"].append(np.nan if err_norm is None else err_norm)
            report["accepted_steps"] += 1
            report["energy"].append(float(np.sum(self.cell_areas * h_n)))
            if hit_target:
                if target in outputs:
                    self.outputs[target] = temp_new
                target_idx = min(target_idx + 1, len(targets) - 1)

            if adaptive is not None and err_norm is not None:
                factor = min(adaptive.growth_max,
                             max(adaptive.shrink_min, adaptive.safety * max(err_norm, 1e-12) ** -0.5))
                k_new = k_try * factor
                if hit_target and k_try < k:
                    # The step was shortened only to land on an output time.
                    k_new = max(k_new, k)
                k = float(np.clip(k_new, adaptive.dt_min, adaptive.dt_max))
            # uniform mode keeps k
        self.enthalpy_field = h_n
        return t, self.temperature(h_n)


class ApparentCapacityPicardSolver(_TransportBase):
    """Reference temperature-primary (apparent-capacity) backward-Euler solver.

    Solves ``c_app(T) T_t - div(alpha grad T) = Q`` with the classical Picard
    iteration ``c_app(T^k) (T^{k+1} - T^n) / dt + K T^{k+1} = Q`` (optionally
    relaxed / Anderson-accelerated) for *any* enthalpy law (the polygonal
    solver's built-in path only accepts the piecewise-linear ramp).  It is the
    baseline the enthalpy formulation is compared against: it never raises on
    non-convergence, and its report carries the discrete energy defect
    ``sum_i area_i [H(T^{n+1}) - H(T^n)] - dt * (fluxes + sources)``.
    """

    def __init__(self, vertices, polygons, alpha, dt, phase_model, picard_options=None, **kwargs):
        super().__init__(vertices, polygons, alpha, dt, **kwargs)
        self.model = phase_model
        opts = {"max_iters": 100, "tol": 1e-8, "relaxation": 1.0, "anderson_depth": 0}
        if picard_options:
            opts.update(dict(picard_options))
        self.picard_options = opts
        self._interior = np.ones(self.M, dtype=bool)
        if self.bc_type == "dirichlet":
            self._interior[self._boundary_idx] = False

    def solve(self, u0, t0, t_end):
        from .polygonal import _AndersonAccelerator

        u = np.broadcast_to(np.asarray(u0, dtype=float), (self.M,)).astype(float)
        nsteps, dt = self._uniform_schedule(t0, t_end, self.dt)
        max_iters = int(self.picard_options["max_iters"])
        tol = float(self.picard_options["tol"])
        relaxation = float(self.picard_options["relaxation"])
        depth = int(self.picard_options["anderson_depth"])
        report = {"iterations": [], "failed_steps": 0, "converged": True,
                  "energy_defect": [], "total_iterations": 0}
        self.solve_report = report
        t = float(t0)
        bidx = self._boundary_idx
        for _ in range(nsteps):
            t_next = t + dt
            q = self._source(t_next)
            B, b = self._boundary_system(t_next)
            K = self.A if B is None else (self.A + B).tocsr()
            bc = self._bc_values(t_next) if self.bc_type == "dirichlet" else None
            u_iter = u.copy()
            accel = _AndersonAccelerator(depth) if depth > 0 else None
            converged = False
            iters = max_iters
            for it in range(max_iters):
                cap = np.asarray(self.model.effective_heat_capacity(u_iter), dtype=float)
                lhs = (diags(self.cell_areas * cap / dt, format="csr") + K).tocsr()
                rhs = self.cell_areas * (cap * u / dt + q) + b
                if bc is not None:
                    lhs = self._apply_dirichlet(lhs)
                    rhs[bidx] = bc[bidx]
                u_raw = splu(lhs.tocsc()).solve(rhs)
                u_next = self._relaxed_picard_update(accel, u_iter, u_raw, relaxation)
                err = float(np.max(np.abs(u_next - u_iter)))
                u_iter = u_next
                if not np.all(np.isfinite(u_iter)):
                    break
                if err <= tol * max(1.0, float(np.max(np.abs(u_iter)))):
                    converged = True
                    iters = it + 1
                    break
            report["iterations"].append(iters)
            report["total_iterations"] += iters
            if not converged:
                report["failed_steps"] += 1
                report["converged"] = False
            flux = self.cell_areas * q - K @ u_iter + b
            h_new = np.asarray(self.model.enthalpy(u_iter), dtype=float)
            h_old = np.asarray(self.model.enthalpy(u), dtype=float)
            inter = self._interior
            report["energy_defect"].append(float(
                np.sum(self.cell_areas[inter] * (h_new - h_old)[inter]) - dt * np.sum(flux[inter])
            ))
            u = u_iter
            t = t_next
        return t, u
