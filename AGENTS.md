# AGENTS Guide

## Repository Purpose

This repository implements finite-volume heat solvers on several mesh types and includes analytical verification and QA regression tooling.

The main PDE handled here is:

`partial_t u - alpha * Delta u = Q(x, y, t)`

with support for:

- Dirichlet boundary conditions
- Neumann boundary conditions
- Robin boundary conditions
- Volumetric source terms

## Important Files

- `heat_solver/polygonal.py`
  Polygonal cell-centered finite-volume solver.
  This is the main solver used by the current verification workflow.
  Supports Dirichlet, Neumann, Robin, and `source_func`.

- `heat_solver/triangular.py`
  Vertex-centered solver on Delaunay meshes.
  Supports Dirichlet and `source_func`.
  It does not support Neumann or Robin. The drivers intentionally reject those cases for triangular runs.

- `heat_solver/cases.py`
  Manufactured / analytical solutions and case metadata.
  If adding a new analytical verification case, this is the first file to update.
  Cases may provide:
  - `solution`
  - `boundary`
  - `bc_type`
  - `source`
  - `bbox`

- `heat_solver/drivers.py`
  High-level mesh runners that connect cases, meshes, solvers, and error reporting.
  Use these rather than calling solvers directly for verification work.

- `heat_solver/meshes.py`
  Mesh generators.
  Includes:
  - hexagonal polygonal mesh
  - square polygonal mesh
  - mixed polygonal mesh
  - skewed non-orthogonal quadrilateral mesh
  - skewed non-orthogonal tiled polygonal mesh
  - nonuniform Delaunay mesh

- `heat_solver/plotting.py`
  Plot generation helpers used by the verification script.

- `tests.py`
  Expensive manufactured-solution verification sweep (not the unit tests).
  Runs every case in `CASE_SETTINGS` across all refinement levels, on the mesh
  types each case supports (`_mesh_jobs(...)`): curvilinear and square polygonal
  for most cases, the tiled non-orthogonal mesh for several, and the full
  seven-mesh set (`MESH_ORDER`) for `anisotropic_heat_kernel`.

- `qa_regression.py`
  Regression harness for numerical datasets.
  First run seeds baselines.
  Later runs compare fresh outputs to stored baselines.

- `tests/`
  Pytest unit-test suite (fast, ~2 min). This is what CI runs.
  `tests/test_polygonal_boundary_conditions.py` holds the polygonal Neumann / Robin checks.

- `heat_solver/transport.py`, `heat_solver/interface_diagnostics.py`
  Extended transport models (Cattaneo, advection-diffusion, fractional, Pennes)
  and the Direction C hyperbolic / fractional Stefan solvers plus phase-change
  diagnostics and enthalpy audits.

- `heat_solver/verification.py`, `mms.py`, `nversion.py`, `reference_fd.py`
  Direction A V&V tooling (observed order / GCI, SymPy MMS, cross-scheme checks,
  independent FD reference). `benchmark_suite.py` ties them together.

- `heat_solver/dmp.py`, `heat_solver/afc.py`
  Direction B discrete-maximum-principle diagnostics and bound-preserving AFC schemes.

- `heat_solver/enthalpy.py`, `heat_solver/adjoint.py`
  Enthalpy-primary Stefan solver with adaptive time stepping, and exact discrete
  adjoints (fields, Robin, fractional, nonlinear Stefan). Verify new adjoints
  with `adjoint.taylor_test` (second-order remainder rate ~2). Studies:
  `direction_c_enthalpy_studies.py`, `direction_d_adjoint_studies.py`; report:
  `direction_CD_enthalpy_adjoint_report.tex` (`pdflatex`, run twice).

- `heat_solver/quadratic_flux.py`
  `flux_discretization="quadratic"` for `PolygonalHeatSolver`: second order on
  non-orthogonal / anisotropic problems. Its unknowns are cell averages:
  source and Dirichlet data are cell-integrated automatically
  (`cell_average_data`), and initial data / error checks should use
  `solver.cell_average(f, t)`. The legacy `reconstructed` flux loses order on
  skewed meshes; prefer `quadratic` (or `tpfa` + `nonorthogonal_correction`).

- `heat_solver/boundary_flux.py`, `heat_solver/external_codes.py`
  Dirichlet reaction-flux energy audits (set `solver.record_history = True` on
  `HyperbolicStefanSolver`), and FiPy / scikit-fem adapters for cross-code
  validation (`pip install -r requirements-external.txt`; their tests skip if
  absent; CI runs them in the `external-validation` job). Studies:
  `dmp_source_study.py`, `direction_c_flux_audit_study.py`,
  `external_code_comparison.py`; report:
  `direction_ABC_bounds_flux_crosscode_report.tex`.

- `heat_solver/inverse.py`, `heat_solver/pinn_jax.py`
  Direction D inverse problems (least squares, Pennes adjoint, UQ, field inversion,
  ML/PINN baselines). `pinn_jax.py` needs optional JAX and its tests skip without it.

- `docs/research/`
  One design / status note per research direction (A-D). Update the matching note
  when you land work on a direction.

- `requirements.txt`, `requirements-dev.txt`, `requirements-external.txt`
  Exact-pinned runtime, test, and optional external-code dependencies.

- `.github/workflows/ci.yml`
  CI: installs the pinned deps on Python 3.13, byte-compiles, runs `pytest tests`;
  a second job installs `requirements-external.txt` and runs the cross-code tests.

## Environment Setup

```bash
cd /home/user/thermalio
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
```

The commands below assume this `.venv`. Dependencies are pinned exactly; if you
bump a pin, rerun the unit tests and `qa_regression.py` and commit the new pins
with any fixes they require.

## Current Verification Workflow

### Unit tests (run these first; CI runs the same command)

```bash
cd /home/user/thermalio
MPLCONFIGDIR=/tmp/matplotlib /home/user/thermalio/.venv/bin/python -m pytest tests -q
```

Use `python -m pytest` (not bare `pytest`) so the repo root is on `sys.path`.
Expect 2 skips in `tests/test_pinn_jax.py` when JAX is not installed.

### Main verification sweep

Run:

```bash
cd /home/user/thermalio
MPLCONFIGDIR=/tmp/matplotlib /home/user/thermalio/.venv/bin/python tests.py
```

This writes plots and convergence summaries under `test_plots/`.

### QA regression

Run:

```bash
cd /home/user/thermalio
MPLCONFIGDIR=/tmp/matplotlib /home/user/thermalio/.venv/bin/python qa_regression.py
```

Behavior:

- If a baseline dataset is missing, it is created under `qa_regression/baseline/`
- Fresh results are always written under `qa_regression/latest/`
- The latest JSON report is written to `qa_regression/latest/regression_report.json`
- Nonzero exit means regression failure

### Boundary-condition regression

Run:

```bash
cd /home/user/thermalio
MPLCONFIGDIR=/tmp/matplotlib /home/user/thermalio/.venv/bin/python -m pytest tests/test_polygonal_boundary_conditions.py -q
```

## Boundary and Source Conventions

### Dirichlet

`bc_func(x, y, t) -> value`

### Neumann

The polygonal solver interprets Neumann data as the outward normal derivative:

`du/dn`

If normals are accepted by the callback:

`bc_func(x, y, t, nx, ny) -> du/dn`

### Robin

The polygonal solver uses:

`alpha * du/dn + beta * u = value`

The callback may return:

- `(beta, value)`, or
- `{"beta": ..., "value": ...}`

If normals are accepted:

`bc_func(x, y, t, nx, ny) -> (beta, value)`

### Source term

`source_func(x, y, t) -> Q(x, y, t)`

This is attached through case metadata as `source`.

## Current Manufactured Cases

Examples already in `heat_solver/cases.py`:

- `sine_mode`
- `harmonic_polynomial`
- `source_driven_sine`
- `steady_linear_neumann`
- `steady_linear_robin`
- `linear_patch`
- `hot_block`
- `off_axis_wave`
- `nyquist_oscillations`
- `point_source`

`source_driven_sine` is the current analytical source-term example:

`u(x, y, t) = exp(-t) * sin(pi x) * sin(pi y)`

with

`Q(x, y, t) = (2 * alpha * pi^2 - 1) * u(x, y, t)`

## Practical Change Rules

- If you add a new analytical case, update `heat_solver/cases.py` and usually `tests.py`.
- If the case uses Neumann or Robin, do not route it through the Delaunay / triangular solver.
- If you change mesh/job definitions in `tests.py`, remember that `qa_regression.py` consumes `tests.iter_test_jobs()`.
- If you intentionally change numerical output, rerun `qa_regression.py` once to seed/update baselines, then rerun it again to confirm clean comparison mode.
- For simple syntax validation, use:

```bash
python -m compileall -q heat_solver tests tests.py qa_regression.py benchmark_suite.py
```
- `np.cross` on 2-vectors raises under the pinned NumPy 2.x. Use
  `heat_solver.geometry.cross2d` for 2D cross products.

## Notes For Future Agents

- The repo is a git repository; GitHub Actions CI runs on every push and pull request.
- `pytest` is part of `requirements-dev.txt`. The `tests/` suite is the fast gate;
  `tests.py` and `qa_regression.py` are the slower numerical sweeps.
- `qa_regression/` is gitignored, so baselines are local to each checkout and are
  not run in CI.
- `MPLCONFIGDIR=/tmp/matplotlib` is useful in this environment to avoid matplotlib cache permission warnings.

## Common Tasks

### Add a new analytical case

1. Add the exact solution, and if needed the boundary and source callbacks, in `heat_solver/cases.py`.
2. Register the case in `get_analytical_case(...)` in `heat_solver/cases.py`.
3. Add timing / parameter settings for the case in `tests.py` under `CASE_SETTINGS`.
4. If the case should be part of dataset regression, no extra QA wiring is needed as long as `tests.py` includes it.
5. Run:

```bash
MPLCONFIGDIR=/tmp/matplotlib /home/user/thermalio/.venv/bin/python tests.py
MPLCONFIGDIR=/tmp/matplotlib /home/user/thermalio/.venv/bin/python qa_regression.py
MPLCONFIGDIR=/tmp/matplotlib /home/user/thermalio/.venv/bin/python qa_regression.py
```

### Add a new mesh to the verification workflow

1. Add the mesh generator in `heat_solver/meshes.py`.
2. Add a matching runner in `heat_solver/drivers.py`.
3. Export it from `heat_solver/__init__.py` if it should be public.
4. Wire it into `tests.py`:
   - add a config helper
   - add a save/run helper
   - update `MESH_ORDER`
   - update `_mesh_jobs(...)`
   - update `_resolution_metric(...)`
5. If `qa_regression.py` should cover it, expose a corresponding runner in the `RUNNERS` map and make sure `tests.iter_test_jobs()` yields jobs for it.

### Update QA baselines after an intentional numerical change

1. Run:

```bash
MPLCONFIGDIR=/tmp/matplotlib /home/user/thermalio/.venv/bin/python qa_regression.py
```

2. If new jobs were added, this seeds missing baseline entries automatically.
3. If existing outputs changed intentionally, replace the relevant baseline directories under `qa_regression/baseline/` with the matching directories from `qa_regression/latest/`.
4. Rerun:

```bash
MPLCONFIGDIR=/tmp/matplotlib /home/user/thermalio/.venv/bin/python qa_regression.py
```

5. Only treat the baseline update as complete if the second run reports all `PASS`.
