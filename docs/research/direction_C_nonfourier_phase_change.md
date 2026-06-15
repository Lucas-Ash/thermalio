# Direction C: 2D non-Fourier / fractional phase change (hyperbolic & fractional Stefan)

- **Status:** PR1/PR2 initial implementation
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

## PR breakdown
- PR1: hyperbolic Stefan solver + manufactured case + convergence test.
  Implemented as `HyperbolicStefanSolver` and
  `hyperbolic_stefan_apparent_capacity`.
- PR2: fractional Stefan solver + manufactured case + convergence test.
  Implemented as `FractionalStefanSolver` and
  `fractional_stefan_apparent_capacity`.
- PR2+: application studies.

## References
- Non-Fourier Stefan problem (1D) literature; DPL bioheat phase-change
  (cryosurgery) via RBF/BEM — this direction targets conservative 2D FV instead.
