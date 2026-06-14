# Thermalio

**Thermalio** is a Python toolkit for **2D heat (diffusion) problems** on unstructured and structured meshes. It implements several **cell- and vertex-centered finite-volume** discretizations and ships with a large **manufactured-solution verification** suite: analytical references, convergence across refinement levels, and optional **numerical regression** baselines.

The core model is the transient heat equation with volumetric sources:

$$
\frac{\partial u}{\partial t} - \nabla \cdot (\boldsymbol{\alpha} \nabla u) = Q(\mathbf{x}, t)
$$

with **Dirichlet**, **Neumann**, **Robin**, and **nonlinear radiative** boundary data where supported. Extensions include **anisotropic** diffusivity $\boldsymbol{\alpha}$, **temperature-dependent** $\alpha(u)$, and **phase change** via an **apparent heat capacity** formulation.

---

## Meshing & discretizations

The library couples the same physical cases to multiple mesh generators and solvers:

| Capability | Notes |
|------------|--------|
| **Polygonal finite volumes** | Cell-centered fluxes on general polygons (hex-dominant, squares, mixed tilings, skewed quads). Non-orthogonal corrections for distorted cells. |
| **Delaunay / triangular** | Vertex-centered solver on nonuniform Delaunay triangulations (primarily Dirichlet + sources). |
| **Curvilinear quads** | Mapped structured grids with mild warping for geometry fidelity tests. |

Representative **numerical vs. exact fields and pointwise error** on the *same* manufactured case (*anisotropic heat kernel*, finest level):

<p align="center">
  <img src="test_plots/anisotropic_heat_kernel/level_05_superfine/curvilinear.png" alt="Curvilinear mesh verification" width="32%" />
  <img src="test_plots/anisotropic_heat_kernel/level_05_superfine/polygonal.png" alt="Hex polygonal mesh verification" width="32%" />
  <img src="test_plots/anisotropic_heat_kernel/level_05_superfine/delaunay.png" alt="Delaunay mesh verification" width="32%" />
</p>
<p align="center">
  <img src="test_plots/anisotropic_heat_kernel/level_05_superfine/square_polygonal.png" alt="Square polygonal mesh verification" width="32%" />
  <img src="test_plots/anisotropic_heat_kernel/level_03_fine/nonorthogonal_tiled_polygonal.png" alt="Non-orthogonal tiled polygonal verification" width="32%" />
</p>

*Left-to-right, top: curvilinear, hexagonal polygonal, Delaunay; bottom: square polygonal, non-orthogonal tiled polygonal.*

---

## Physics & modeling highlights

**Linear diffusion & sources** — Fundamental solutions (Gaussian / heat kernel), eigenmodes, harmonics, and **prescribed sources** $Q$ matched to closed-form $u$.

<p align="center">
  <img src="test_plots/heat_kernel/level_05_superfine/curvilinear.png" alt="Heat kernel on curvilinear mesh" width="48%" />
  <img src="test_plots/point_source/level_05_superfine/curvilinear.png" alt="Point source / discrete kernel on curvilinear mesh" width="48%" />
</p>

**Stiff spatial features** — Discontinuous initial data (convolved block), **rotated plane waves**, and **high-frequency** modes probe resolution limits and orientation dependence.

<p align="center">
  <img src="test_plots/nyquist_oscillations/level_03_fine/nonorthogonal_tiled_polygonal.png" alt="Nyquist oscillations on tiled mesh" width="70%" />
</p>

**Anisotropic diffusion** — $\boldsymbol{\alpha}$ a full $2\times2$ tensor (elliptic kernels, rotated principal directions).

**Functionally graded materials** — **Spatially graded conductivity** $\alpha(x)=\alpha_0 e^{\gamma x}$ (e.g. thermal-barrier coatings), with a manufactured solution that closes the full $\nabla\cdot(\alpha\nabla u)=\alpha\nabla^2 u + \nabla\alpha\cdot\nabla u$ flux (case `functionally_graded`, second-order convergent).

**Nonlinear material response** — **Temperature-dependent diffusivity** $\alpha = \alpha(x,u)$ with iterative solves; solutions and sources are manufactured for strict consistency.

<p align="center">
  <img src="test_plots/temperature_dependent_diffusivity/level_05_superfine/square_polygonal.png" alt="Temperature-dependent diffusivity verification" width="48%" />
  <img src="test_plots/temperature_dependent_diffusivity/level_03_fine/nonorthogonal_tiled_polygonal.png" alt="Temperature-dependent diffusivity on tiled mesh" width="48%" />
</p>

**Phase change** — **Apparent heat capacity** model (latent heat smeared across a mushy interval) with manufactured **Stefan-type** traveling-interface temperature fields.

<p align="center">
  <img src="test_plots/stefan_apparent_capacity/level_05_superfine/square_polygonal.png" alt="Stefan apparent capacity verification" width="70%" />
</p>

**Radiation boundary conditions** — **Nonlinear radiative** boundaries (emissivity / Stefan–Boltzmann-type closure) with a manufactured solution that closes the forcing and boundary residuals.

<p align="center">
  <img src="test_plots/radiative_manufactured/level_05_superfine/square_polygonal.png" alt="Radiative BC manufactured solution" width="48%" />
  <img src="test_plots/radiative_manufactured/level_03_fine/nonorthogonal_tiled_polygonal.png" alt="Radiative BC on non-orthogonal tiled mesh" width="48%" />
</p>

**Extended transport models** — beyond the classical parabolic Fourier law, `heat_solver/transport.py` adds three research-oriented transport models (manufactured-solution verified, reusing the polygonal FV diffusion operator). All three support **Dirichlet**, **Neumann**, **Robin**, and **prescribed-flux** boundary data — `bc_type='flux'` prescribes the inward boundary heat flux $q_{\mathrm{in}}=\alpha\,\partial u/\partial n$ directly (the natural form for pulsed-laser/contact heating of the Cattaneo model), and is discretely energy-conservative:

- **Non-Fourier / Cattaneo–Vernotte thermal waves** — $\tau\,\partial_{tt} u + \partial_t u - \nabla\cdot(\alpha\nabla u) = Q$, giving heat a finite propagation speed $c=\sqrt{\alpha/\tau}$ ("second sound"), integrated with an unconditionally stable second-order three-level scheme (`HyperbolicHeatSolver`).
- **Advection–diffusion (convective transport)** — $\partial_t u + \nabla\cdot(\mathbf{v}\,u) - \nabla\cdot(\alpha\nabla u) = Q$, with a prescribed velocity field and a choice of first-order monotone **upwind** or second-order **central** face flux (Péclet-number aware) (`AdvectionDiffusionHeatSolver`).
- **Anomalous / time-fractional subdiffusion** — Caputo derivative $D_t^\beta u - \nabla\cdot(\alpha\nabla u) = Q$ for $0<\beta<1$ via the L1 scheme (order $2-\beta$ in time), modeling long-memory thermal response in disordered media (`FractionalHeatSolver`).
- **Reaction–diffusion / Pennes bioheat** — $\partial_t u - \nabla\cdot(\alpha\nabla u) + k\,u = Q$, where the linear reaction term $k\,u$ models tissue perfusion cooling (Pennes bioheat), volumetric Newton cooling, or first-order chemical heat consumption; the source-free mode decays at the faster rate $2\pi^2\alpha + k$ (`ReactionDiffusionHeatSolver`).

Runnable examples and convergence tables: `python transport_demo.py` (writes `test_plots/transport_models.png`), including a **flux-pulse Cattaneo wave** demo that contrasts the finite wavefront $x=ct$ against the instantaneous parabolic Fourier response; unit checks in `tests/test_transport.py`.

<p align="center">
  <img src="test_plots/transport_models.png" alt="Extended thermal-transport models" width="90%" />
</p>

---

## Repository layout & workflows

| Path | Role |
|------|------|
| `heat_solver/polygonal.py` | Main **polygonal** cell-centered solver (broadest BC and physics support). |
| `heat_solver/transport.py` | **Extended transport models**: hyperbolic (Cattaneo), advection–diffusion, time-fractional subdiffusion, and reaction–diffusion / Pennes bioheat solvers. |
| `heat_solver/triangular.py` | **Delaunay / triangular** vertex-centered solver. |
| `heat_solver/curvilinear.py` | **Curvilinear** mapped-grid solver. |
| `heat_solver/cases.py` | Manufactured solutions, sources, BC callbacks, case metadata. |
| `heat_solver/drivers.py` | High-level runners wiring meshes, solvers, and error metrics. |
| `heat_solver/meshes.py` | Mesh generators (hex, square, mixed, skewed, tiled, Delaunay). |
| `tests.py` | Full verification sweep: plots and `convergence_summary` under `test_plots/`. |
| `qa_regression.py` | Compare fresh numerical outputs to stored baselines. |

**Run the main verification sweep** (writes figures and CSV/TXT summaries under `test_plots/`):

```bash
cd /path/to/thermalio
MPLCONFIGDIR=/tmp/matplotlib python tests.py
```

**Boundary-condition spot checks** (polygonal Neumann / Robin): `test_polygonal_boundary_conditions.py`.

Dependencies are the usual scientific Python stack (**NumPy**, **SciPy**, **Matplotlib**). Use a virtual environment and install those packages if they are not already present.

---

## Math in this README

Equations use **GitHub-style** delimiters: `$…$` for inline math and `$$…$$` on their own lines for display math (this matches [GitHub’s math syntax](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions)). If the **Markdown preview** in VS Code or Cursor shows raw `$` text, turn on **Settings → Markdown: Math** (`markdown.math.enabled`).

---

## Summary

Thermalio is built for **reproducible numerical heat-equation studies**: multiple **mesh paradigms**, **linear and nonlinear** material laws, **phase change** and **radiation** at the boundary, and **automated verification** against analytical references—illustrated throughout by the regression plots in `test_plots/`.
