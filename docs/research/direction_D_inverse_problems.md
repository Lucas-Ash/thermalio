# Direction D: Inverse-problem / parameter-identification testbed

- **Status:** PR1/PR2 plus Pennes adjoint and UQ diagnostics
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
- **Out of scope (later):** adjoints for nonlinear/transport/fractional solvers;
  full Bayesian inversion; PINN implementation and head-to-head benchmark; field
  (spatially varying) parameter recovery.

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
- **Regularized field inversion:** extend parameter vectors to low-dimensional
  spatial bases for perfusion/conductivity fields, with smoothness penalties and
  mesh-aware regularization.
- **Inverse runners and reports:** add a reproducible driver similar to the
  verification suite that writes JSON/CSV reports and plot bundles for each
  inverse study.
- **Model comparison:** use the trusted FV inverse results as baselines for later
  PINN/ML comparisons, with identical synthetic observations and error metrics.

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
- PR4+: broader PDE adjoints, field inversion, and ML comparison.

## References
- Pennes bioheat inverse problems (perfusion estimation); PINN inverse-problem
  literature for benchmarking.
