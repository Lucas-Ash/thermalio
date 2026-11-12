"""Consistent boundary-flux reconstruction for Dirichlet energy audits.

Thermalio's cell-centred solvers impose Dirichlet data by replacing the
equations of the boundary-cell band ``B`` with ``u_i = g_i``.  The heat that
enters or leaves the domain through ``partial Omega`` is therefore not a
quantity the solver computes -- earlier Direction C audits *inferred* it from
the enthalpy change, which makes the energy audit tautological.

This module reconstructs it independently from the discrete operator, using
the classical *consistent (reaction) flux* idea (the finite-volume analogue of
the Lagrange-multiplier / reaction-force boundary flux of Hughes et al., 2000):
evaluate the **unconstrained** balance equation of every Dirichlet cell at the
computed solution.  Its residual

    r_i = |Omega_i| * (discrete time-derivative term)_i + (K u)_i - |Omega_i| q_i - b_i,
    i in B,

is exactly the boundary heat flow needed to hold ``u_i = g_i``.  Because the FV
operator is conservative (every face flux is added to one cell and subtracted
from its neighbour, so ``sum_i (A u)_i = 0``), summing all cell equations gives
the exact discrete identity

    sum_i |Omega_i| (time term)_i = sum_i |Omega_i| q_i + sum_i b_i + sum_{i in B} r_i,

so ``E_flux = sum_n dt * sum_B r^n`` closes the scheme's own balance to the
nonlinear-solver tolerance.  Comparing it with the *enthalpy* change then
measures what the scheme gets wrong (for apparent-capacity schemes: the
latent-heat energy defect), which is the stronger audit Direction C asked for.

Functions
---------
:func:`band_interface_flux`
    Flux from the Dirichlet band into the interior cells (interior-only audit).
:func:`hyperbolic_stefan_flux_audit`
    Full audit of an apparent-capacity Cattaneo-Stefan run
    (:class:`heat_solver.transport.HyperbolicStefanSolver` with
    ``record_history = True``).
:func:`enthalpy_solver_flux_audit`
    Full audit of an :class:`heat_solver.enthalpy.EnthalpyStefanSolver` run
    (``store_history=True``).
"""

from __future__ import annotations

import numpy as np


def _conservation_defect(K):
    """``max_j |sum_i K_ij|``: zero for a conservative (flux-form) operator."""
    return float(np.max(np.abs(np.asarray(K.sum(axis=0)).ravel())))


def band_interface_flux(K, u, boundary_idx):
    """Heat flow from the Dirichlet band into the interior, ``-sum_{i in I} (K u)_i``.

    This closes the energy balance of the *interior* cells only; it misses the
    energy stored in (or released by) the band cells themselves.
    """
    interior = np.ones(K.shape[0], dtype=bool)
    interior[np.asarray(boundary_idx, dtype=int)] = False
    return -float(np.sum((K @ u)[interior]))


def hyperbolic_stefan_flux_audit(solver, model):
    """Flux-based energy audit of an apparent-capacity Cattaneo--Stefan run.

    The solver's three-level step (uniform ``dt``) reads, on every cell,

        |Omega| [tau D2 u + c_n D1 u] + A u = |Omega| q   (+ r on Dirichlet cells)

    with ``D2 u = (u^{n+1} - 2u^n + u^{n-1})/dt^2``,
    ``D1 u = (u^{n+1} - u^{n-1})/(2 dt)`` and ``c_n`` the apparent capacity of
    the last Picard iterate.  Summing over cells and steps gives

        relaxation_storage + capacity_work = source_energy + boundary_flux_energy
                                         + nonlinear_residual_energy

    exactly (the last term is the summed interior residual of unconverged
    nonlinear steps and vanishes when every step converges), where the relaxation term telescopes to
    ``tau sum |Omega| [(u^N - u^{N-1}) - (u^0 - u^{-1})]/dt``.  The enthalpy
    audit residual is ``Delta E - (source + flux - relaxation)``
    ``= (Delta E - capacity_work) + nonlinear_residual_energy``: the
    apparent-capacity energy defect plus any unconverged-step residual.
    """
    hist = getattr(solver, "history", None)
    if hist is None:
        raise ValueError("set solver.record_history = True before solve().")
    if solver.bc_type != "dirichlet":
        raise ValueError("hyperbolic_stefan_flux_audit targets Dirichlet runs.")
    dt = float(hist["dt"])
    tau = float(hist["tau"])
    area = solver.cell_areas
    bidx = solver._boundary_idx
    A = solver.A
    us = [hist["u_start"]] + list(hist["u"])  # us[k] = u^{k-1}
    times = hist["t"]
    n_steps = len(hist["capacity"])
    interior = np.ones(solver.M, dtype=bool)
    interior[bidx] = False
    flux_per_step, source_per_step, capacity_work, band_flux_per_step = [], [], 0.0, []
    nonlinear_residual = 0.0
    capacity_work_interior = 0.0
    per_cell_flux = np.zeros(bidx.size)
    for n in range(n_steps):
        u_prev, u_cur, u_next = us[n], us[n + 1], us[n + 2]
        cap = hist["capacity"][n]
        d2 = (u_next - 2.0 * u_cur + u_prev) / dt**2
        d1 = (u_next - u_prev) / (2.0 * dt)
        q = solver._source(times[n + 1])
        time_term = area * (tau * d2 + cap * d1)
        residual = time_term + A @ u_next - area * q
        r_b = residual[bidx]
        nonlinear_residual += dt * float(np.sum(residual[interior]))
        flux_per_step.append(float(np.sum(r_b)))
        per_cell_flux += dt * r_b
        source_per_step.append(float(np.sum(area * q)))
        capacity_work += dt * float(np.sum(area * cap * d1))
        capacity_work_interior += dt * float(np.sum((area * cap * d1)[interior]))
        band_flux_per_step.append(band_interface_flux(A, u_next, bidx))
    flux_energy = dt * float(np.sum(flux_per_step))
    source_energy = dt * float(np.sum(source_per_step))
    u_end, u_endm1, u_0, u_m1 = us[-1], us[-2], us[1], us[0]
    relaxation = tau * float(np.sum(area * ((u_end - u_endm1) - (u_0 - u_m1)))) / dt
    h0 = float(np.sum(area * np.asarray(model.enthalpy(u_0), dtype=float)))
    h1 = float(np.sum(area * np.asarray(model.enthalpy(u_end), dtype=float)))
    delta_e = h1 - h0
    identity_residual = relaxation + capacity_work - source_energy - flux_energy - nonlinear_residual
    # Interior-only balance: the band cells are clamped to g and may jump through
    # the whole mushy interval in one step (initial data inconsistent with g),
    # a latent-heat change the apparent-capacity time term never sees.  The
    # interior audit isolates the scheme's own defect.
    hI0 = float(np.sum((area * np.asarray(model.enthalpy(us[1]), dtype=float))[interior]))
    hI1 = float(np.sum((area * np.asarray(model.enthalpy(us[-1]), dtype=float))[interior]))
    relax_I = tau * float(np.sum((area * ((us[-1] - us[-2]) - (us[1] - us[0])))[interior])) / dt
    src_I = dt * float(np.sum([np.sum((area * solver._source(times[n + 1]))[interior]) for n in range(n_steps)]))
    band_in = dt * float(np.sum(band_flux_per_step))
    interior_closure = (hI1 - hI0) - (src_I + band_in - relax_I)
    closure = delta_e - (source_energy + flux_energy - relaxation)
    report = getattr(solver, "solve_report", {}) or {}
    scale = max(abs(delta_e), abs(flux_energy), 1e-300)
    return {
        "enthalpy_change": delta_e,
        "boundary_flux_energy": flux_energy,
        "energy_extracted": -flux_energy,
        "source_energy": source_energy,
        "relaxation_storage": relaxation,
        "capacity_work": capacity_work,
        "scheme_identity_residual": identity_residual,
        "nonlinear_residual_energy": nonlinear_residual,
        "capacity_defect": delta_e - capacity_work,
        "failed_steps": int(report.get("failed_steps", 0)),
        "energy_closure_residual": closure,
        "relative_energy_closure_residual": closure / scale,
        "band_interface_energy": band_in,
        "interior_enthalpy_change": hI1 - hI0,
        "interior_capacity_defect": (hI1 - hI0) - capacity_work_interior,
        "interior_energy_closure_residual": interior_closure,
        "band_capacity_defect": (delta_e - capacity_work) - ((hI1 - hI0) - capacity_work_interior),
        "flux_history": np.cumsum(dt * np.asarray(flux_per_step)),
        "times": np.asarray(times[1:]),
        "per_boundary_cell_energy": per_cell_flux,
        "operator_conservation_defect": _conservation_defect(A),
    }


def enthalpy_solver_flux_audit(solver):
    """Flux-based audit of an :class:`EnthalpyStefanSolver` run with ``store_history=True``.

    For ``tau = 0`` each step satisfies
    ``|Omega|(H^{n+1}-H^n)/k + K T^{n+1} = |Omega| q + b + r`` on every cell, so
    ``Delta E = source + b + flux`` holds to the Newton tolerance; any residual
    is a solver (not formulation) error.  For natural boundary types the
    ``prescribed_flux_energy`` is the full boundary inflow ``sum(b - B T)``
    (the Robin / radiative closure ``B`` lives in ``K``).
    """
    from .enthalpy import three_level_coefficients

    hist = getattr(solver, "history", None)
    if hist is None:
        raise ValueError("solve with store_history=True first.")
    area = solver.cell_areas
    bidx = solver._boundary_idx
    H = hist["H"]
    times = hist["t"]
    h_m1 = H[0]
    flux, src, bnd, per_cell = [], [], [], np.zeros(bidx.size)
    for n in range(len(hist["k"])):
        k, kp = hist["k"][n], hist["k_prev"][n]
        gamma, p, s = three_level_coefficients(k, kp, solver.tau)
        h_prev2 = h_m1 if n == 0 else H[n - 1]
        if n == 0 and solver.tau > 0:
            raise ValueError("enthalpy_solver_flux_audit currently supports tau = 0.")
        q = solver._source(times[n + 1])
        temp = solver.temperature(H[n + 1])
        K, b = solver._operator(times[n + 1], temperature=temp)
        residual = area * (gamma * H[n + 1] - p * H[n] - s * h_prev2 - q) + K @ temp - b
        r_b = residual[bidx] if solver.bc_type == "dirichlet" else np.zeros(0)
        flux.append(k * float(np.sum(r_b)))
        if r_b.size:
            per_cell += k * r_b
        src.append(k * float(np.sum(area * q)))
        if solver.bc_type == "dirichlet":
            bnd.append(k * float(np.sum(b)))
        else:
            # Natural boundaries: the boundary inflow is sum(b - B T) (Robin /
            # radiative closures carry a temperature-dependent part in K); the
            # conservative interior part of K sums to zero.
            bnd.append(k * float(np.sum(b) - np.sum(K @ temp)))
    delta_e = float(np.sum(area * (H[-1] - H[0])))
    flux_energy = float(np.sum(flux))
    closure = delta_e - (float(np.sum(src)) + float(np.sum(bnd)) + flux_energy)
    return {
        "enthalpy_change": delta_e,
        "boundary_flux_energy": flux_energy,
        "energy_extracted": -flux_energy,
        "source_energy": float(np.sum(src)),
        "prescribed_flux_energy": float(np.sum(bnd)),
        "energy_closure_residual": closure,
        "relative_energy_closure_residual": closure / max(abs(delta_e), abs(flux_energy), 1e-300),
        "flux_history": np.cumsum(flux),
        "times": np.asarray(times[1:]),
        "per_boundary_cell_energy": per_cell,
    }
