# Report Images

This directory contains report figure helpers and caption snippets for the
technical LaTeX reports:

- `direction_D_inverse_report.tex`
- `direction_C_nonfourier_report.tex`

Each report figure is stored as:

- `*.png` for the rendered figure.
- `*.caption.tex` for the LaTeX caption and label used by the report.

Regenerate the Direction D figures from the repository root with:

```bash
MPLCONFIGDIR=/tmp/matplotlib /home/user/thermalio/.venv/bin/python report_images/generate_direction_d_report_images.py
```

The generator is deterministic and uses the Direction D inverse-problem utilities
plus the solver-backed artifacts already stored under
`test_plots/direction_D_inverse_problems/`.

Check the Direction C report figures and rewrite its caption snippets without
regenerating simulations with:

```bash
MPLCONFIGDIR=/tmp/matplotlib /home/user/thermalio/.venv/bin/python report_images/generate_direction_c_report_images.py
```

Regenerate Direction C application and sweep figures deliberately with:

```bash
MPLCONFIGDIR=/tmp/matplotlib /home/user/thermalio/.venv/bin/python report_images/generate_direction_c_report_images.py --regenerate
```

Current generated figure topics:

- Direction D inverse workflow.
- Scalar perfusion identifiability and noise robustness.
- Sensor/time information content.
- Multi-parameter diffusivity/perfusion identifiability.
- Profile likelihood versus local Gauss-Newton curvature.
- Sensitivity and uncertainty diagnostics.
- Regularization-path behavior.
- Gaussian field-basis coverage and overlap.
- Field-inversion coefficient recovery.
- FV/ML/JAX PINN baseline comparison.

Current Direction C report topics:

- Apparent-capacity Stefan phase-change formulation.
- Hyperbolic/Cattaneo Stefan solver and finite thermal-wave speed.
- Fractional/Caputo Stefan solver and memory compression.
- Sharp-interface diagnostics, nonlinear convergence metadata, and energy
  closure audits.
- Application studies for melting, freezing, remelting, quenching, and buried
  inclusions.
- Parameter-sweep infrastructure over the main Direction C physics/numerics.
