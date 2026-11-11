# Thermalio research backlog

Researched directions for adapting Thermalio toward novel, impactful work. Each
note follows `TEMPLATE.md` and records the scope, design decisions, findings,
and open follow-ups for its direction. The notes themselves do not affect any
test, sweep, or regression run. Technical write-ups for C and D live in the repo
root (`direction_C_nonfourier_report.tex`, `direction_D_inverse_report.tex`,
`direction_CD_enthalpy_adjoint_report.tex` for the enthalpy/adaptive/adjoint follow-up,
`direction_ABC_bounds_flux_crosscode_report.tex` for source limiters, flux audits and
external-code validation).

| ID | Direction | Status | Note |
|----|-----------|--------|------|
| A | Open V&V benchmark suite for non-classical heat transport | **landed** | [direction_A_vv_benchmarks.md](direction_A_vv_benchmarks.md) |
| B | Monotone / bound-preserving polygonal & MPFA FV for non-Fourier / fractional / anisotropic models | **landed** | [direction_B_monotone_polygonal_fv.md](direction_B_monotone_polygonal_fv.md) |
| C | 2D non-Fourier / fractional phase change (hyperbolic & fractional Stefan) | **landed** (solvers, sweeps, diagnostics, applications, report) | [direction_C_nonfourier_phase_change.md](direction_C_nonfourier_phase_change.md) |
| D | Inverse-problem / parameter-identification testbed (bioheat, non-Fourier) | **landed** (PR1–PR7, report) | [direction_D_inverse_problems.md](direction_D_inverse_problems.md) |

## Why these

Thermalio is unusual as a research platform: it unifies a broad set of physics
(classical, anisotropic, temperature-dependent, phase change, radiation, Cattaneo
hyperbolic, advection, fractional subdiffusion, Pennes bioheat, functionally
graded) behind one verification-first finite-volume framework with
manufactured-solution convergence sweeps across multiple mesh paradigms. The
directions above each exploit that breadth + rigor against a gap confirmed in the
current literature. Direction A came first because it was the highest-leverage,
lowest-risk starting point and strengthens the foundation the others build on.

## Open follow-ups across directions

- **A:** transcendental (Mittag-Leffler) Caputo cases; registering MMS cases in
  the regression sweep; nonlinear cross-code comparison.  (External-code
  validation vs FiPy and scikit-fem landed 2026-11.)
- **B:** coercivity analysis and vectorized assembly of the quadratic flux;
  geometric NTPFA; sharper local source envelopes.  (Sign-changing-source
  limiters and the second-order quadratic flux landed 2026-11.)
- **C:** second-order (Newmark-averaged) Cattaneo time stepping, adaptivity for
  the apparent-capacity and fractional solvers, consistent Dirichlet initial
  data (removes the band latent-heat jump), sharp-interface tracking.
  (Enthalpy-primary formulation, adaptive stepping and boundary-flux audits
  landed 2026-11.)
- **D:** field parameters and Robin/radiative boundaries in the nonlinear
  adjoint; sensor-design metrics; calibrated PINN studies in a JAX environment.
  (Field, Robin, fractional and nonlinear Stefan adjoints landed 2026-11.)
