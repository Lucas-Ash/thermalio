# Direction C: 2D non-Fourier / fractional phase change (hyperbolic & fractional Stefan)

- **Status:** landed (PR1/PR2 solvers, expansion steps 1-3, computing-capability steps 4-8, technical report `direction_C_nonfourier_report.tex`; enthalpy-primary formulation + adaptive stepping follow-up, report `direction_CD_enthalpy_adjoint_report.tex`)
- **Owner:** —
- **Last updated:** 2026-11-11

## Motivation & V&V question
Phase change and the extended transport models are currently **not composable** in
Thermalio (the `ApparentHeatCapacityModel` couples to the classical parabolic
solver, not to the Cattaneo or fractional time-stepping). Combining them yields
**hyperbolic Stefan** (`tau u_tt + c(T) u_t - div(alpha grad u) = Q`) and
**fractional Stefan** (`c(T) D_t^beta u - div(...) = Q`) problems in 2D on
unstructured meshes — largely unexplored (existing non-Fourier Stefan work is
mostly 1D or classical Fourier). Applications: laser ablation, ultrafast freezing
/ cryosurgery, additive-manufacturing melt pools.

V&V question: *can a conservative 2D FV scheme reproduce manufactured
traveling-interface solutions for non-Fourier / fractional phase change, and what
is its observed order?*

## Scope
- **Implemented first increment:** composed solvers coupling apparent heat
  capacity with the Cattaneo three-level scheme and the Caputo L1 scheme, plus
  manufactured moving-interface verification cases.
- **Expansion-study diagnostics:** three follow-on research directions now have
  reproducible plots under
  `test_plots/direction_C_nonfourier_phase_change/expansion_studies/`:
  relaxation-aware hyperbolic Stefan physics, fractional-memory effects, and
  latent-heat/mushy-zone stiffness.
- **Out of scope (later):** sharp-interface tracking (post-processed interface
  diagnostics exist, see step 4); alloy/multi-component solidification.
  Application studies have since landed (step 5).

## Design sketch
- New: extend `heat_solver/transport.py` (`_TransportBase` already reuses the
  diffusion matrix A and lumped mass) with an apparent-heat-capacity-aware
  hyperbolic solver; new manufactured case in `cases.py`.
- Reuse: `ApparentHeatCapacityModel` (effective capacity, enthalpy), the existing
  `HyperbolicHeatSolver` three-level time scheme, Anderson-accelerated Picard from
  `polygonal.py`.

## Reuse map
- `heat_solver/phase_change.py`; `heat_solver/transport.py` HyperbolicHeatSolver /
  FractionalHeatSolver; `cases.stefan_apparent_capacity_*`.

## Verification plan
- Manufactured traveling `tanh` interface with a relaxation term; observed order
  in space and time via direction A's reporting; latent-heat energy balance check.

## Risks & open questions
- Nonlinear (c(T)) + hyperbolic/fractional coupling convergence; mushy-zone
  stiffness; need relaxation/Anderson tuning.

## Expansion steps 1-3 — implemented study runners

The three expansion targets below now have solver-backed **parameter-sweep
runners** in `direction_c_studies.py`, writing JSON/CSV/PNG to
`test_plots/direction_C_nonfourier_phase_change/sweeps/`:

1. **`relaxation_sweep`** — sweeps relaxation time `tau` x time resolution,
   recording manufactured error, the finite wave speed `sqrt(alpha/tau)`, and
   nonlinear stability metadata (max Picard iterations, failed steps).
2. **`fractional_memory_sweep`** — beta-calibration (observed temporal order via
   fine-dt self-convergence matches `2-beta` to ~0.1: 1.61/1.47/1.31 vs
   1.60/1.40/1.20) plus a **memory-compression** study using the new
   `memory_window` short-memory option (error grows monotonically as the window
   shrinks: full 2.3e-3 -> window-2 8.8e-3).
3. **`mushy_stiffness_map`** — latent-heat x transition-half-width stiffness map
   (analytic capacity spike 8 -> 241) with strict (no-Anderson) Picard
   convergence metadata; also flags mushy zones the discrete mesh under-resolves.

Supporting solver/case enhancements (in `heat_solver/transport.py`,
`heat_solver/cases.py`):

- `FractionalHeatSolver(..., memory_window=K)` short-memory L1 truncation
  (capability 2: fractional memory compression).
- `solve_report` nonlinear-convergence metadata (iteration counts, failed steps,
  peak capacity) + `phase_change_options['raise_on_nonconvergence']=False` for
  building failure/stability maps (capability 6).
- `hyperbolic_stefan_apparent_capacity_case(..., latent_heat,
  transition_half_width, specific_heat, speed)` parameterized to keep
  latent-heat/mushy sweeps manufactured-exact.

Verification: `tests/test_direction_c.py` (memory-window accuracy/validation,
convergence metadata, non-raising failure recording, parameterized-case
manufactured exactness, runner smoke test).

## Three important expansion targets

1. **Relaxation-aware hyperbolic Stefan physics.**  Non-Fourier phase change is
   not just classical Stefan with an extra parameter: the relaxation term
   `tau T_tt` changes the operator balance near the moving interface and sets a
   finite thermal-wave speed `sqrt(alpha / tau)`.  This is important for
   pulsed-laser melting, ultrafast ablation, and cryogenic shock problems where
   the Fourier infinite-speed assumption is physically suspect.  A useful next
   increment would add parameter-sweep runners over `tau`, front speed, and flux
   pulse width, with stability/convergence metadata.

2. **Fractional-memory Stefan dynamics.**  The Caputo/L1 history weights retain
   a long memory of previous interface states, and the memory tail changes
   strongly with fractional order `beta`.  This is important for porous,
   disordered, or heterogeneous phase-change media where thermal response is
   slower than classical diffusion.  A useful next increment would add
   beta-calibration studies and memory-compression tests, because naive
   fractional history storage becomes expensive for long simulations.

3. **Latent-heat and mushy-zone stiffness.**  Apparent heat capacity turns latent
   heat into a narrow, high-amplitude capacity spike.  This couples directly to
   the hyperbolic/fractional time term, so omitting `c(T)` produces the wrong
   interface dynamics even when the diffusion/source terms are unchanged.  A
   useful next increment would add systematic sweeps over latent heat, transition
   width, Picard relaxation, and Anderson depth, with failure maps and
   enthalpy-based preconditioning.

## Implemented computing capabilities (steps 4-8)

- **Step 4 — sharp-interface diagnostics** (`heat_solver/interface_diagnostics.py`):
  post-processing for a cell-centered field + apparent-capacity model —
  liquid-fraction field and area-weighted liquid volume fraction,
  solid/mushy/liquid area fractions, sensible/latent enthalpy budget,
  melt-isotherm interface position(s) along a centerline, mushy-zone thickness,
  and front speed.  Verified against an analytic `tanh` front (interface
  position and mushy thickness match to within a cell width).
- **Step 5 — application-study runners** (`direction_c_applications.py`):
  reproducible 2D scenarios beyond manufactured solutions, each writing
  JSON/CSV/PNG to `test_plots/direction_C_nonfourier_phase_change/applications/`:
    - `pulsed_laser_melting` — a boundary heat-flux pulse melts a cold solid slab
      (hyperbolic/Cattaneo Stefan, finite wave speed); reports melt-front
      advance, peak temperature, liquid fraction, mushy thickness, injected
      energy, sensible/latent enthalpy, and an **energy-closure residual**
      (enthalpy rise vs injected boundary energy ~0.3% at full resolution).
    - `cryosurgery_freezing` — a cold cryoprobe Dirichlet boundary freezes a warm
      domain; reports freezing-front margin, frozen volume fraction, minimum
      temperature, and extracted enthalpy.
    - `moving_scan_melt_pool` — a moving volumetric Gaussian heat source creates
      an additive-manufacturing-like melt-pool track; reports source energy,
      peak temperature, melt-pool length/width, liquid fraction, mushy thickness,
      and latent/sensible enthalpy.
    - `dual_pulse_remelting` — two separated laser pulses reheat an already
      partially transformed slab; reports remelting amplitude, injected energy,
      peak liquid fraction, interface advance, and the latent-heat state retained
      between pulses.
    - `rapid_solidification_quench` — a hot liquid slab is quenched by cold
      boundaries; reports liquid-fraction collapse, solid-fraction growth,
      peak-temperature decay, and enthalpy removed from the domain.
    - `buried_hot_inclusion_relaxation` — a localized hot inclusion relaxes and
      refreezes inside a colder matrix; reports melted area, peak temperature,
      liquid-fraction decay, and extracted enthalpy.
  These runners record nonlinear-convergence metadata and tolerate the onset
  stiffness via the non-raising solve mode (motivating capabilities 1 & 3).
  Verified by `tests/test_direction_c_applications.py` (analytic diagnostics +
  runner physical-sanity / energy-closure checks).
- **Step 6 — nonlinear convergence reporting** (`heat_solver/transport.py`,
  `heat_solver/interface_diagnostics.py`): phase-change solvers now retain
  per-step Picard/Anderson iteration counts, normalized final update residuals,
  failed-step indices, tolerance/relaxation/Anderson settings, and effective
  heat-capacity minima/maxima.  `summarize_solve_report(...)` compresses the
  full report into CSV/JSON-friendly fields used by all Direction C application
  runners.
- **Step 7 — energy and enthalpy conservation audits**
  (`heat_solver/interface_diagnostics.py`, `direction_c_applications.py`):
  `enthalpy_audit(...)` standardizes the apparent-capacity balance
  `observed enthalpy change - (energy_in - energy_out)`.  All six application
  runners now write sensible, latent, total, and initial enthalpy; expected
  enthalpy change; injected/source/extracted energy; absolute closure residual;
  and relative closure residual.  Flux-driven laser/remelting studies use
  prescribed flux/source energy, while Dirichlet cooling/freezing studies report
  inferred extracted enthalpy as the boundary-energy audit.
- **Step 8 — parameter sweep infrastructure** (`direction_c_studies.py`):
  `parameter_sweep_suite` creates a sparse benchmark matrix over relaxation
  time `tau`, fractional order `beta`, latent heat, transition half-width,
  manufactured front speed, mesh resolution, and time resolution.  Each row
  records the model family, variant name, swept parameter, manufactured
  relative `L2` error, time step, wave speed where applicable, nonlinear
  convergence status, failed steps, iteration statistics, maximum residual, and
  maximum apparent capacity.  The suite writes
  `parameter_sweep_suite.csv/json/png` under
  `test_plots/direction_C_nonfourier_phase_change/sweeps/`, alongside a sweeps
  README describing all Direction C sweep artifacts.

## Boundary coverage (2026-11)

The Cattaneo, Cattaneo-Stefan, fractional, advection-diffusion and Pennes solvers,
plus `EnthalpyStefanSolver` and `ApparentCapacityPicardSolver`, accept all five
boundary types, radiation included. Each implicit step iterates the
Newton-linearized radiative closure. In the enthalpy Newton solver the closure
is part of the residual, and its exact derivative is in the Jacobian. The
advection boundary flux uses the BC-consistent face value. Manufactured
problems (`bc_test_problems.transport_problem`) converge at second order for
every model x type. `enthalpy_solver_flux_audit` re-assembles the closure at
each stored temperature, so radiative runs still close their energy balance.
See `docs/boundary_conditions.md`.

## Additional computing capabilities that would help

1. **Adaptive time stepping for stiff phase change.** *Landed* for the
   enthalpy solver (`AdaptiveStepOptions`): extrapolation-based local error
   estimate, elementary controller, Newton-failure step halving, exact output
   times and sliver avoidance.  On a laser-pulse + cooling problem it is
   10-120x more accurate than uniform steps at comparable Newton work.  Still
   open: adaptivity for the apparent-capacity and fractional solvers, a PI
   controller, and a true error estimator for `tau > 0`.

2. **Accurate memory compression for fractional Stefan.**  The Caputo L1
   scheme sums the full time history.  Short-memory windowing (`memory_window`)
   is available but trades accuracy for cost; sum-of-exponentials or adaptive
   history compression would keep accuracy at larger final times and finer
   resolutions.

3. **Enthalpy-primary nonlinear formulation.** *Landed* as
   `heat_solver/enthalpy.py::EnthalpyStefanSolver` (Fourier and Cattaneo,
   `tau H_tt + H_t - div(alpha grad T(H)) = Q`, Newton in `H` with line
   search).  Energy conservative to <=1.5e-10; never failed in a 35-case
   stiffness sweep where apparent-capacity Picard failed in 20.  Also found:
   the three-level Cattaneo scheme (shared with `HyperbolicHeatSolver`) is
   first order in time, not second; a Newmark-averaged variant is the next
   step.  Still open: fractional enthalpy formulation.

4. **Interface curvature and phase-fraction contours.**  Step 4 now extracts
   interface position, front speed, and mushy-zone thickness along a
   centerline.  Still missing are interface curvature and full 2D
   phase-fraction contours, which would sharpen comparisons against Stefan
   theory and experimental front-position data.

5. **Application-study runners.**  Direction C now has six reproducible 2D
   scenarios beyond manufactured solutions: pulsed-laser melting, cryosurgery
   freezing margins, additive-manufacturing-like scan tracks, dual-pulse
   remelting, rapid solidification quenching, and buried hot-inclusion
   relaxation.  The next useful increment would turn these seeds into a
   benchmark matrix with controlled sweeps over pulse spacing, scan speed,
   quench strength, inclusion size, `tau`, latent heat, and mushy-zone width.

6. **Localized nonlinear failure diagnostics.**  Step 6 now records run-level
   residual and iteration histories.  A useful refinement would identify the
   cells responsible for the largest nonlinear update, so failure maps can show
   whether convergence loss is tied to the moving front, a boundary layer, or a
   volumetric heat-source hot spot.

7. **Explicit boundary-flux reconstruction for Dirichlet audits.** *Landed*
   (`heat_solver/boundary_flux.py`): consistent reaction flux = residual of the
   unconstrained Dirichlet-cell equations, exact global balance for
   conservative operators.  The cryosurgery and quench runners now audit
   against this flux (previously the closure was zero by construction).
   Findings (`direction_c_flux_audit_study.py`): apparent-capacity closure
   errors of 26% / 32% (24-cell mesh), decomposed exactly into a Dirichlet-band
   latent-heat jump `-L * band area` (dt-independent), an interior capacity
   defect that shrinks with dt, and unconverged Picard residuals; the enthalpy
   solver closes to 1e-16.  Report: `direction_ABC_bounds_flux_crosscode_report.tex`.

8. **Higher-dimensional design-of-experiments sweeps.**  Step 8 now provides a
   sparse one-at-a-time benchmark matrix over the main Direction C axes.  A
   useful next increment would add Latin-hypercube or factorial designs with
   parallel execution, resumable jobs, and aggregate sensitivity plots so broad
   parameter studies can scale beyond the current lightweight matrix.

## Enthalpy + adaptive follow-up (2026-11)

- Code: `heat_solver/enthalpy.py` (`EnthalpyStefanSolver`, `SmoothPhaseChangeModel`,
  `AdaptiveStepOptions`, `ApparentCapacityPicardSolver` baseline).
- Studies: `direction_c_enthalpy_studies.py` ->
  `test_plots/direction_C_nonfourier_phase_change/enthalpy_adaptive/`
  (`enthalpy_mms_convergence.png`, `stiffness_robustness.png`,
  `adaptive_pulse_cooling.png`, `summary.json`).
- Tests: `tests/test_enthalpy.py`.
- Write-up: `direction_CD_enthalpy_adjoint_report.tex` (formulation, proofs,
  results).

## Expansion plots

- `test_plots/direction_C_nonfourier_phase_change/expansion_studies/hyperbolic_relaxation_balance.png`
  shows the manufactured hyperbolic Stefan operator balance near the interface,
  the finite-speed scaling `sqrt(alpha / tau)`, and solver-backed manufactured
  errors over stable relaxation-time values.
- `test_plots/direction_C_nonfourier_phase_change/expansion_studies/fractional_memory_convergence.png`
  shows L1 Caputo memory weights, cumulative history retention, expected L1
  temporal-order scaling, and solver-backed FractionalStefanSolver errors for
  several `beta` values.
- `test_plots/direction_C_nonfourier_phase_change/expansion_studies/mushy_zone_capacity_coupling.png`
  shows apparent-capacity/enthalpy curves for several mushy-zone widths and a
  solver-backed centerline comparison demonstrating that omitting apparent
  capacity breaks the manufactured hyperbolic Stefan balance.
- `test_plots/direction_C_nonfourier_phase_change/sweeps/parameter_sweep_suite.png`
  shows the step-8 benchmark matrix: manufactured errors, nonlinear iteration
  costs/failures, stiffness-vs-residual behavior, and coverage by swept
  parameter axis.

## PR breakdown
- PR1: hyperbolic Stefan solver + manufactured case + convergence test.
  Implemented as `HyperbolicStefanSolver` and
  `hyperbolic_stefan_apparent_capacity`.
- PR2: fractional Stefan solver + manufactured case + convergence test.
  Implemented as `FractionalStefanSolver` and
  `fractional_stefan_apparent_capacity`.
- PR2+ (done): expansion steps 1-3 study runners (`direction_c_studies.py`) --
  relaxation/stability sweep, fractional-memory calibration + compression, and
  latent-heat/mushy-zone stiffness map -- plus the `memory_window`,
  convergence-metadata, and parameterized-case enhancements that back them.
- PR2+ (done): computing-capability steps 4-8 -- interface diagnostics,
  application runners, nonlinear convergence reporting, energy/enthalpy audits,
  and the broad Direction C parameter-sweep suite.

## References
- Non-Fourier Stefan problem (1D) literature; DPL bioheat phase-change
  (cryosurgery) via RBF/BEM — this direction targets conservative 2D FV instead.
