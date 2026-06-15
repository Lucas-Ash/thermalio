# Direction D Report Images

This directory contains the expanded figure set for
`direction_D_inverse_report.tex`.

Each report figure is stored as:

- `*.png` for the rendered figure.
- `*.caption.tex` for the LaTeX caption and label used by the report.

Regenerate the figures from the repository root with:

```bash
MPLCONFIGDIR=/tmp/matplotlib /home/user/thermalio/.venv/bin/python report_images/generate_direction_d_report_images.py
```

The generator is deterministic and uses the Direction D inverse-problem utilities
plus the solver-backed artifacts already stored under
`test_plots/direction_D_inverse_problems/`.
