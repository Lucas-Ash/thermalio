# Direction C: 2D non-Fourier / fractional phase change (hyperbolic & fractional Stefan)

- **Status:** PR1/PR2 initial implementation + expansion-study diagnostics
- **Owner:** —
- **Last updated:** 2026-06-14

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
- **Out of scope (later):** sharp-interface tracking; alloy/multi-component
  solidification; application studies.

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

## PR breakdown
- PR1: hyperbolic Stefan solver + manufactured case + convergence test.
  Implemented as `HyperbolicStefanSolver` and
  `hyperbolic_stefan_apparent_capacity`.
- PR2: fractional Stefan solver + manufactured case + convergence test.
  Implemented as `FractionalStefanSolver` and
  `fractional_stefan_apparent_capacity`.
- PR2+: relaxation/front-speed studies, fractional-memory compression and
  calibration, and latent-heat/mushy-zone nonlinear robustness studies.

## References
- Non-Fourier Stefan problem (1D) literature; DPL bioheat phase-change
  (cryosurgery) via RBF/BEM — this direction targets conservative 2D FV instead.
