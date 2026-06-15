# Direction C Application Studies

This directory contains reproducible example scenarios for non-Fourier
phase-change studies.  Each scenario writes:

- `<scenario>.png` — summary plot with scenario-specific time histories and final
  temperature field.
- `<scenario>_diagnostics.png` — Step 6/7 dashboard with nonlinear convergence
  and energy/enthalpy audit histories.
- `<scenario>.csv` — sampled diagnostic time history, including convergence
  metadata and energy/enthalpy audit columns.
- `<scenario>.json` — run parameters and final diagnostic summary, including
  final convergence and relative closure-residual metrics.

Regenerate all artifacts from the repository root with:

```bash
MPLCONFIGDIR=/tmp/matplotlib /home/user/thermalio/.venv/bin/python direction_c_applications.py
```

For a faster smoke-test version:

```bash
MPLCONFIGDIR=/tmp/matplotlib /home/user/thermalio/.venv/bin/python direction_c_applications.py --quick
```

## Scenarios

- `pulsed_laser_melting`: boundary heat-flux pulse melting a cold slab, with
  melt-front, liquid-fraction, injected-energy, and energy-closure diagnostics.
- `cryosurgery_freezing`: cold probe freezing a warm domain, with freezing
  margin, frozen fraction, minimum temperature, and extracted enthalpy.
- `moving_scan_melt_pool`: moving volumetric heat source for an
  additive-manufacturing-like melt-pool track, with melt-pool length/width and
  source-energy diagnostics.
- `dual_pulse_remelting`: two separated boundary pulses showing remelting and
  latent-heat retention between pulses.
- `rapid_solidification_quench`: hot liquid slab quenched by cold boundaries,
  showing liquid-fraction collapse and solid-fraction growth.
- `buried_hot_inclusion_relaxation`: localized hot inclusion relaxing inside a
  colder matrix, showing refreezing, melted-area decay, and enthalpy removal.

## Shared Diagnostics

Every CSV includes the same Step 6/7 audit columns:

- Nonlinear convergence: `solve_converged`, `solve_steps`, `failed_steps`,
  `max_iterations`, `mean_iterations`, `final_residual`, `max_residual`,
  `min_capacity`, `max_capacity`, `tolerance`, `relaxation`, and
  `anderson_depth`.
- Energy/enthalpy balance: `energy_in`, `energy_out`,
  `initial_total_enthalpy`, `total_enthalpy`, `sensible_enthalpy`,
  `latent_enthalpy`, `enthalpy_change`, `expected_enthalpy_change`,
  `energy_closure_residual`, and `relative_energy_closure_residual`.
