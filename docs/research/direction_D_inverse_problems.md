# Direction D: Inverse-problem / parameter-identification testbed

- **Status:** PR1-PR7 initial implementation
- **Owner:** —
- **Last updated:** 2026-06-15

## Motivation & V&V question
Thermalio has no adjoint/sensitivity machinery, but its *verified, fast* forward
solver plus manufactured ground truth is an ideal clean testbed for estimating
physical parameters — perfusion `k`, relaxation time `tau`, fractional order
`beta`, or functionally-graded-conductivity parameters — from (noisy) temperature
data, and for a **fair, reproducible PINN/ML-vs-trusted-FV comparison**. Clinically
impactful (ablation / cryosurgery planning).

V&V question: *given synthetic temperature observations from a verified forward
model, how identifiable are the parameters, how robust is recovery to noise, and
how do classical optimization and ML approaches compare against the same trusted
reference?*

## Scope
- **Implemented first increment:** a thin parameter-estimation layer using SciPy
  `least_squares` around arbitrary trusted forward maps, plus synthetic
  observation/noise utilities, identifiability scans, and Pennes perfusion
  recovery tests.
- **Implemented second increment:** vector-parameter estimation, Tikhonov-style
  prior regularization, and Cartesian identifiability grid scans, verified on
  joint diffusivity/perfusion recovery.
- **Implemented diagnostics increment:** sparse/multi-time observation operators,
  finite-difference residual Jacobians, least-squares adjoint products `J.T r`,
  and Gauss-Newton covariance diagnostics.
- **Implemented adjoint/UQ increment:** a discrete backward-Euler adjoint for
  scalar Pennes perfusion gradients, normal-approximation confidence intervals,
  and bootstrap/noise-ensemble summaries.
- **Implemented field-inversion/runner increment:** normalized Gaussian perfusion
  field bases, matrix/smoothness regularization, and a reproducible Pennes field
  inverse runner that writes JSON/CSV/PNG study artifacts.
- **Implemented PINN/ML baseline increment:** a dependency-light comparison
  runner that reports trusted FV inverse, RBF-ridge ML surrogate, and training-
  grid lookup baselines with a shared schema that external PINN implementations
  can plug into.
- **Implemented JAX PINN adapter increment:** optional `heat_solver.pinn_jax`
  backend for source-free Pennes inverse PINNs, with lazy JAX imports, trainable
  scalar perfusion, PDE/data/initial/boundary losses, and JSON/CSV/PNG report
  output.
- **Out of scope (later):** adjoints for nonlinear/transport/fractional solvers;
  full Bayesian inversion; PINN implementation and head-to-head benchmark;
  high-dimensional cellwise field recovery.

## Design sketch
- New: `heat_solver/inverse.py` — objective `J(theta) = ||u(theta) - u_obs||`,
  wrapping a driver run; optimizer driver; noise injection utilities.
- Reuse: any transport/reaction solver as the forward map; manufactured cases for
  ground-truth parameter values; `build_error_report` for residuals.

## Reuse map
- `heat_solver/transport.py` (ReactionDiffusionHeatSolver, Hyperbolic, Fractional);
  `heat_solver/cases.py` manufactured solutions as ground truth; `scipy.optimize`.

## Verification plan
- Recover known manufactured parameters from noise-free synthetic data to high
  accuracy; degrade noise and report recovery error vs noise level; show the
  objective is convex/identifiable near the truth for single parameters.

## Progress plots
- `test_plots/direction_D_inverse_problems/pennes_perfusion_inverse_study.png`
  shows the original PR1 scalar perfusion objective scan and noise response.
- `test_plots/direction_D_inverse_problems/alpha_perfusion_identifiability_grid.png`
  shows the PR2 joint diffusivity/perfusion cost surface with the recovered
  least-squares point.
- `test_plots/direction_D_inverse_problems/perfusion_noise_ensemble.png`
  shows scalar perfusion recovery over a small ensemble of noise realizations.
- `test_plots/direction_D_inverse_problems/alpha_perfusion_noise_recovery.png`
  shows joint diffusivity/perfusion recovery as observation noise increases.
- `test_plots/direction_D_inverse_problems/regularization_path_diagnostic.png`
  shows how the Tikhonov prior selects a stable solution in an underdetermined
  inverse problem.
- `test_plots/direction_D_inverse_problems/sparse_multitime_sensor_recovery.png`
  shows sparse sensor placement, noisy multi-time observations, and the recovered
  perfusion fit.
- `test_plots/direction_D_inverse_problems/sensitivity_adjoint_diagnostics.png`
  shows finite-difference observation sensitivities, the adjoint product `J.T r`,
  and a local covariance-derived parameter standard deviation.
- `test_plots/direction_D_inverse_problems/pennes_discrete_adjoint_gradient_check.png`
  compares the Pennes discrete-adjoint gradient against a central finite-
  difference objective derivative.
- `test_plots/direction_D_inverse_problems/perfusion_uncertainty_intervals.png`
  compares local covariance confidence intervals and bootstrap intervals for a
  sparse multi-time perfusion fit.
- `test_plots/direction_D_inverse_problems/pennes_field_inverse_study/`
  contains the first regularized field-inversion runner output:
  `pennes_field_inverse_summary.json`,
  `pennes_field_inverse_coefficients.csv`, and
  `pennes_field_inverse_fields.png`.
- `test_plots/direction_D_inverse_problems/pinn_ml_baseline_comparison/`
  contains the first baseline-comparison runner output:
  `pennes_ml_baseline_summary.json`,
  `pennes_ml_baseline_metrics.csv`, and
  `pennes_ml_baseline_comparison.png`.
- `test_plots/direction_D_inverse_problems/jax_pinn_baseline/`
  contains the optional JAX PINN baseline output when run in a JAX-enabled
  environment: `pennes_jax_pinn_summary.json`,
  `pennes_jax_pinn_history.csv`, and `pennes_jax_pinn_training.png`.

## Further development priorities
- **Broader PDE adjoint support:** the current discrete adjoint covers scalar
  reaction/perfusion in the backward-Euler Pennes model. Extend the same pattern
  to diffusivity, source amplitudes, Robin/radiative boundary parameters,
  Cattaneo/Fractional transport, and nonlinear phase-change solves.
- **Observation design:** sparse-sensor and multi-time observation operators are
  now available. Next, add sensor-design metrics such as D-optimality, time-window
  comparisons, and automated observability/rank diagnostics.
- **Uncertainty quantification:** local covariance intervals and bootstrap
  summaries are available. Next add profile-likelihood scans, model-mismatch
  studies, and likelihood-aware reports that distinguish observation noise from
  discretization error.
- **Regularized field inversion:** low-dimensional Gaussian perfusion-field
  inversion is available. Next steps are mesh-aware Laplacian/TV penalties,
  adaptive basis placement, bound constraints informed by physiology, and
  extension to conductivity/source fields.
- **Inverse runners and reports:** a Pennes field-inversion runner now writes
  JSON/CSV/PNG artifacts. Generalize this into a suite runner with repeatable
  scenario definitions, noise ensembles, timing metadata, and comparison tables.
- **Model comparison:** trusted FV, RBF-ridge surrogate, and grid-lookup baselines
  are available with identical synthetic observations and error metrics. A JAX
  PINN adapter is now available as an optional backend; next, run calibrated
  larger PINN studies in a JAX-enabled environment and feed those report rows
  into the comparison schema.

## Risks & open questions
- Ill-posedness / non-identifiability (e.g. tau vs alpha trade-offs); regularization
  needs. Finite-difference gradients are cost-heavy — motivates a later adjoint.

## PR breakdown
- PR1: single-parameter estimation + identifiability/noise study. Implemented in
  `heat_solver/inverse.py` with perfusion-recovery verification in
  `tests/test_inverse.py`.
- PR2: multi-parameter estimation and regularization. Implemented in
  `heat_solver/inverse.py` with joint diffusivity/perfusion recovery tests.
- PR3: sparse/multi-time observations and sensitivity diagnostics. Implemented in
  `heat_solver/inverse.py` with solver-backed sparse perfusion recovery tests.
- PR4: Pennes discrete adjoint and UQ diagnostics. Implemented in
  `heat_solver/inverse.py` with adjoint-vs-finite-difference and confidence/
  bootstrap tests.
- PR5: regularized field inversion and inverse-study runner. Implemented in
  `heat_solver/inverse.py` with low-dimensional Gaussian perfusion-field
  recovery and report-writing tests.
- PR6: PINN/ML comparison baselines. Implemented as a trusted-FV-vs-RBF-surrogate
  Pennes baseline runner with report-writing tests.
- PR7: optional JAX PINN adapter. Implemented in `heat_solver/pinn_jax.py` with
  lazy-import tests and backend execution tests that run when JAX is installed.
- PR7+: calibrated PINN studies, broader PDE adjoints, and high-dimensional field
  inversion.

## References
- Pennes bioheat inverse problems (perfusion estimation); PINN inverse-problem
  literature for benchmarking.
