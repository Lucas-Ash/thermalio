# Direction D: Inverse-problem / parameter-identification testbed

- **Status:** PR1/PR2 initial implementation
- **Owner:** —
- **Last updated:** 2026-06-14

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
- **Out of scope (later):** adjoint/auto-diff gradients; full Bayesian inversion;
  PINN implementation and head-to-head benchmark; field (spatially varying)
  parameter recovery.

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

## Risks & open questions
- Ill-posedness / non-identifiability (e.g. tau vs alpha trade-offs); regularization
  needs. Finite-difference gradients are cost-heavy — motivates a later adjoint.

## PR breakdown
- PR1: single-parameter estimation + identifiability/noise study. Implemented in
  `heat_solver/inverse.py` with perfusion-recovery verification in
  `tests/test_inverse.py`.
- PR2: multi-parameter estimation and regularization. Implemented in
  `heat_solver/inverse.py` with joint diffusivity/perfusion recovery tests.
- PR2+: ML comparison.

## References
- Pennes bioheat inverse problems (perfusion estimation); PINN inverse-problem
  literature for benchmarking.
