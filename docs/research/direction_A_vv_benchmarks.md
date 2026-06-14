# Direction A: Open V&V benchmark suite for non-classical heat transport

- **Status:** in-progress
- **Owner:** —
- **Last updated:** 2026-06-14

## Motivation & V&V question
The emerging-model community (non-Fourier/Cattaneo, fractional subdiffusion,
Pennes bioheat, functionally graded, anisotropic) has **no shared, reproducible
manufactured-solution + convergence + cross-scheme benchmark**. Thermalio already
has the backbone (MMS cases, multi-level convergence sweep across 7 mesh types,
regression baselines). This direction adds the pieces that make such a benchmark
*citable and trustworthy*: auto-derived (not hand-coded) sources, rigorous
order-of-accuracy / Richardson reporting, and cross-scheme agreement checks.

V&V question: *for each model and discretization, what is the observed order of
accuracy, is it in the asymptotic range, and do independent schemes agree?*

## Scope
- **In scope (current increment):**
  1. SymPy MMS source auto-derivation (`heat_solver/mms.py`).
  2. Richardson extrapolation + observed-order reporting (`heat_solver/verification.py`,
     surfaced in `tests.py` convergence summaries).
  3. Cross-scheme "N-version" agreement harness (`heat_solver/nversion.py`).
- **Out of scope (deferred):** general (non-monomial-in-time) Caputo derivatives;
  interpolation-based pointwise cross-*mesh* comparison; cross-code validation
  (FEniCS/etc.); auto-registering MMS cases into `CASE_SETTINGS`/`iter_test_jobs`
  (touches regression-snapshotted job set — needs a baseline reseed).

## Design sketch
- New: `heat_solver/mms.py`, `heat_solver/verification.py`, `heat_solver/nversion.py`,
  `requirements-dev.txt`, `tests/test_{mms,verification,nversion}.py`.
- Modified: `tests.py` `_write_case_summary` (+ `_augment_with_gci`) — append-only
  new CSV columns `L2_p_obs`, `L2_p_obs_3grid`, `L2_extrap_err`, `asymptotic`.
- Key signatures: `mms.manufactured_case(u, *, alpha, model, bc_type, ...)`;
  `verification.triplet_report(hs, errors)` / `observed_order` / `richardson_*` /
  `gci`; `nversion.run_nversion(case, *, meshes, variants, tol, ...)`.

## Reuse map
- `heat_solver/drivers.py` runners (`run_square_polygonal_test`,
  `run_nonorthogonal_tiled_polygonal_test`) and `build_error_report` — unchanged.
- `heat_solver/cases.py` hand-coded sources are the ground truth the SymPy module
  reproduces.
- SymPy imported lazily; core solvers never depend on it.

## Verification plan
- `tests/test_mms.py`: auto-derived sources reproduce 6 hand-coded cases to
  ≤1e-13 (source_driven_sine, functionally_graded, cattaneo_wave,
  advection_diffusion, pennes forced, fractional_subdiffusion); Neumann/Robin
  boundary data match the steady-linear cases; a fresh MMS case converges at
  order >1.7 through the real solver.
- `tests/test_verification.py`: synthetic `e=C h^p` recovers p, extrapolated
  error →0, asymptotic ratio →1; oscillatory data flagged.
- `tests/test_nversion.py`: smooth Dirichlet case — square mesh agrees to ~5e-4
  across tpfa/reconstructed/mpfa; tiled mesh to ~9e-3; MPFA-on-tiled recorded as
  a solver-failure skip (a robustness finding).
- `qa_regression.py` runs clean twice (CSV-only additions don't touch baselines).

## Risks & open questions
- SymPy availability → lazy import + `importorskip` + `requirements-dev.txt`.
- Caputo handler is monomial-in-time only (documented).
- Observed order reflects the dominant (space+time) error of the sweep; report
  the asymptotic flag so users can judge validity.

## PR breakdown
- PR1: `verification.py` + reporting + docs scaffold (done).
- PR2: `mms.py` + tests + dev deps (done).
- PR3: `nversion.py` + tests (done).
- Follow-ups: non-monomial Caputo; pointwise cross-mesh comparison; cross-code
  validation; register MMS cases into the regression sweep (with reseed).

## References
- P.J. Roache, *Verification and Validation in Computational Science and
  Engineering* (GCI / observed order).
- ASME V&V 20-2009 (solution verification, Richardson extrapolation).
- Method of Manufactured Solutions literature (Salari & Knupp).
