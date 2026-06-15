# Direction C Parameter Sweeps

This directory contains solver-backed Direction C sweep datasets.  Each sweep
writes JSON, CSV, and PNG artifacts for reproducible follow-up analysis.

Regenerate all sweeps from the repository root with:

```bash
MPLCONFIGDIR=/tmp/matplotlib /home/user/thermalio/.venv/bin/python direction_c_studies.py
```

For a smaller smoke-test pass:

```bash
MPLCONFIGDIR=/tmp/matplotlib /home/user/thermalio/.venv/bin/python direction_c_studies.py --quick
```

## Sweep Artifacts

- `relaxation_sweep.*`: hyperbolic Stefan relaxation-time and time-resolution
  sweep, including manufactured error, finite wave speed, and nonlinear
  convergence metadata.
- `fractional_beta_calibration.*`: fractional Stefan beta calibration with
  observed temporal order compared against the L1 expectation `2 - beta`.
- `fractional_memory_window.*`: short-memory fractional Stefan study showing
  error versus retained Caputo history lags.
- `mushy_stiffness_map.*`: latent-heat and mushy-zone-width stiffness map with
  apparent-capacity spikes, sampled capacity, iteration counts, and failures.
- `parameter_sweep_suite.*`: Step 8 sparse benchmark matrix over relaxation
  time, fractional order, latent heat, transition half-width, front speed, mesh
  resolution, and time resolution.

## Step 8 Fields

`parameter_sweep_suite.csv` uses one row per run:

- Physics/numerics: `family`, `variant`, `swept_parameter`, `tau`, `beta`,
  `latent_heat`, `transition_half_width`, `front_speed`, `mesh_n`,
  `time_steps`, `dt`, and `wave_speed`.
- Verification: `rel_l2` against the manufactured exact solution.
- Nonlinear robustness: `solve_converged`, `failed_steps`, `max_iterations`,
  `mean_iterations`, `max_residual`, and `max_capacity`.
