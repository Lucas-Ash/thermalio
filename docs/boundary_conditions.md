# Boundary conditions

Every Thermalio solver accepts the same five boundary types, with the same
callback conventions (`heat_solver/boundary_conditions.py`):

| `bc_type`   | `bc_func(x, y, t[, nx, ny])` returns         | condition                                   |
|-------------|----------------------------------------------|---------------------------------------------|
| `dirichlet` | `u`                                          | `u = g`                                     |
| `neumann`   | `du/dn` (outward normal derivative)          | `du/dn = g`                                 |
| `robin`     | `(beta, value)` or `{"beta", "value"}`       | `alpha du/dn + beta u = value`              |
| `flux`      | `q_in` (inward heat flux per unit length)    | `alpha du/dn = q_in`                        |
| `radiative` | `(eps, T_inf[, sigma])` or a dict            | `alpha du/dn = -eps sigma (u^4 - T_inf^4)`  |

`sigma` defaults to the SI Stefan–Boltzmann constant. Callbacks that take five
positional arguments receive the outward unit normal. Unknown types raise
`ValueError` in every solver.

## Capability matrix

| Solver                                                    | dirichlet | neumann | robin | flux | radiative |
|-----------------------------------------------------------|:---------:|:-------:|:-----:|:----:|:---------:|
| `PolygonalHeatSolver` (tpfa / +correction / reconstructed / quadratic / mpfa\*) | ✓ | ✓ | ✓ | ✓ | ✓ |
| `AFCMonotoneSolver` (global and envelope limiters)        | ✓ | ✓ | ✓ | ✓ | ✓ |
| `EnthalpyStefanSolver`, `ApparentCapacityPicardSolver`    | ✓ | ✓ | ✓ | ✓ | ✓ |
| `NonUniformHeatSolver` (vertex-centred P1, Delaunay)      | ✓ | ✓ | ✓ | ✓ | ✓ |
| `CurvilinearHeatSolver`                                   | ✓ | ✓ | ✓ | ✓ | ✓ |
| `HyperbolicHeatSolver` (Cattaneo), `HyperbolicStefanSolver` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `FractionalHeatSolver` (Caputo L1)                        | ✓ | ✓ | ✓ | ✓ | ✓ |
| `AdvectionDiffusionHeatSolver`                            | ✓ | ✓ | ✓ | ✓ | ✓ |
| `ReactionDiffusionHeatSolver` (Pennes)                    | ✓ | ✓ | ✓ | ✓ | ✓ |

\* `flux_scheme="mpfa"` keeps its condensed MPFA operator for Dirichlet data and
uses the corrected two-point closure for the other types.

Before this round the triangular solver and the AFC solver were Dirichlet-only,
the transport models lacked radiation, the curvilinear and polygonal solvers
lacked `flux`, and `drivers.run_test` rejected non-Dirichlet Delaunay cases.

## Dirichlet on faces or on a band of cells

`dirichlet_mode=...` on every cell-centred solver: `PolygonalHeatSolver`, the transport
models (`HyperbolicHeatSolver`, `FractionalHeatSolver`, `AdvectionDiffusionHeatSolver`,
`ReactionDiffusionHeatSolver`, `HyperbolicStefanSolver`, `FractionalStefanSolver`),
`EnthalpyStefanSolver`, `ApparentCapacityPicardSolver` and `AFCMonotoneSolver`:

- **`"band"`** (default). The boundary cells are held at the data at their centroids
  (`u_c = g(x_c)`), with identity rows in the matrix.
- **`"face"`**. Every cell is an unknown, and the data are imposed on the boundary
  faces through the face flux `α_n (g_f − v) / d`. Here `d` is the centroid-to-face
  distance and `v` the cell value, extrapolated tangentially on skewed cells. This is
  what FiPy does. It is the Robin closure below with β → ∞, so it shares that
  machinery: factorization reuse, iterative solvers and all time schemes.
  - On Cartesian meshes it reproduces FiPy to 4e-14 (`tests/test_face_dirichlet.py`).
  - It is second order on square, skewed, hexagonal and tiled meshes.

Both modes are second order. On Cartesian meshes they are about equally accurate:
8.33e-5 (band) against 8.34e-5 (face) on the isotropic cross-code problem at n = 64,
and within 1.5% of each other for every solver in the table below. First-order upwind
advection is the exception: its face-mode error is 14% larger. On skewed meshes (skew 0.3)
face mode has a larger error constant: 3x on the sine-mode problem, 2x on the Pennes
problem. The cause is the one-sided least-squares gradient in the boundary cells,
which the face closure uses for its tangential correction.

Face mode differs in three more ways:
- the boundary cells carry real errors, as in FiPy;
- the boundary heat flow is the closure flux itself, with no band reaction to
  reconstruct;
- in phase-change runs, boundary cells are not clamped to the data, so none of
  them jumps through the mushy interval in a single step.

### Face mode in the other solvers

The face closure `(B, b)` is constant in `B` and linear in the data, so every solver
handles it the way it handles Robin data (`heat_solver/transport.py`,
`heat_solver/enthalpy.py`, `heat_solver/afc.py`):

- **Transport models.** `B` is built once per solver. The factorization of
  `lhs + B` is reused across steps (`FactorCache`), and the Picard loops of the
  Stefan variants add `B` like any closure.
- **Advection–diffusion.** The convective boundary flux uses `u_f = g` on every
  face (central) or on inflow faces only (upwind; outflow faces keep the donor
  cell, so the scheme stays monotone).
- **Enthalpy and apparent-capacity solvers.** `B` and `b` enter `K` and the
  residual, and there are no identity rows. The per-step energy balance then
  covers the whole domain.
- **AFC.** The closure is used without the tangential correction (`B` diagonal,
  non-negative), so `A_L + B` stays an M-matrix. The face data join the global
  bounds, the source envelope and the local bounds of boundary cells. The limited
  solution then stays within the initial and face data.
- **Energy audits** (`heat_solver/boundary_flux.py`). Both audits use the closure
  flux `b − B u` as the boundary heat flow.

Measured with backward Euler on the standard problems (`tests/test_face_dirichlet_solvers.py`,
relative L² error at n = 32, square mesh):

| solver | band | face | face order (16→32) |
|---|---|---|---|
| Cattaneo | 5.83e-6 | 5.83e-6 | 2.03 |
| Caputo fractional | 1.01e-5 | 1.02e-5 | 2.05 |
| Advection–diffusion (central) | 1.05e-5 | 1.07e-5 | 1.99 |
| Pennes | 8.73e-6 | 8.74e-6 | 1.98 |
| Enthalpy / apparent capacity (L = 0) | 1.026e-4 | 1.030e-4 | 2.00 |
| AFC (global and local limiters) | 1.026e-4 | 1.030e-4 | 2.00 |

In the hyperbolic Stefan audit of a cold-wall probe, the band's latent-heat jump term
(`−L × band area`) disappears. The audit residual falls from 67% of the enthalpy
change in band mode to 3.8% in face mode. What remains is the apparent-capacity
defect, plus the residual of the Picard steps that did not converge.

The curvilinear and triangular solvers do not need the option. Their unknowns are
grid nodes and mesh vertices, and the boundary unknowns lie on the boundary, so
Dirichlet data are already imposed there.

## Discretization

Every non-Dirichlet condition prescribes the inward heat flux
`q_in = alpha du/dn`, linearized as `q_in ≈ c − a u`. This is exact for the linear
types and is a Newton tangent about the current iterate for radiation.

**Cell-centred (polygonal, AFC, enthalpy, transport).** The boundary face value
`u_f` is eliminated from the face balance. With `alpha_n` the normal diffusivity
and `d` the centroid-to-face distance:

- Robin: `a = alpha_n beta / (alpha_n + beta d)` and `c = alpha_n value / (alpha_n + beta d)`.
- Radiation: the scalar face balance `alpha_n (u_f − v)/d = −eps sigma (u_f^4 − T_inf^4)`
  is solved by Newton on every face, then linearized in `v`. At convergence the
  nonlinear face balance holds exactly.
- On non-orthogonal boundary cells, `v` is the cell value extrapolated to the
  face foot point, `v = u_c + s·∇u_c` (least-squares gradient). This tangential
  correction restores second order on skewed meshes. AFC uses `v = u_c` so the
  low-order operator stays an M-matrix.
- Each closure also returns the face relation `u_f = λ v + μ`. The
  advection–diffusion solver uses it for the convective boundary flux
  `(v·n) u_f`. With the central scheme it applies on every face; with upwind,
  only on inflow faces (outflow faces use the donor cell).

**Vertex-centred P1 (triangular).** The natural boundary term
`∫ q_in φ_i ds` is integrated by trapezoid lumping on every boundary edge, using
that edge's outward normal. Boundary control volumes use the lumped-P1 area
when the problem is not Dirichlet, which keeps natural boundaries second-order
consistent.

**Curvilinear grid.** Flux rows use the existing one-sided boundary stencils.
`flux` maps onto them with `q_in / alpha`.

**Radiation in time steppers.** Each implicit step iterates the Newton-linearized
boundary system to convergence:
- The polygonal, curvilinear and transport solvers use a frozen Jacobian: the
  residual is re-evaluated with the closure linearized about each iterate, while
  one factorization is reused across iterations and steps and refreshed only when
  an update shrinks by less than half (`heat_solver/factor_cache.py`). This gives
  the same fixed point as full Newton, to `nonlinear_options["tol"]`.
  `nonlinear_options={"radiative_jacobian": "newton"}` restores full Newton.
- Phase-change and temperature-dependent-conductivity runs keep the full
  Picard/Newton loop, which re-assembles the closure each iteration.
- Robin closures with time-independent `beta` reuse one factorization for the
  whole run, with results bit-identical to refactorizing every step.
- Crank–Nicolson evaluates the explicit half with the old-level closure.
- In the enthalpy Newton solver, the radiative term is part of the residual, and
  its exact derivative is in the Jacobian.

**Bounds (AFC).** For Robin and radiative data, the ambient temperatures
`value / beta` and `T_inf` join the global bounds and the local envelopes. For
Neumann and flux data, the boundary inflow `b_i / |Omega_i|` enters the envelope
as a source.

## Standard test problems

Defined in `heat_solver/bc_test_problems.py`:

| `bc_type`   | problem                                                                        | exact solution |
|-------------|--------------------------------------------------------------------------------|----------------|
| `dirichlet` | decaying sine mode on `[0,1]^2`                                                | `e^{-2π²αt} sin πx sin πy` |
| `neumann`   | insulated plate on `[0,1]^2`                                                   | `e^{-2π²αt} cos πx cos πy` |
| `robin`     | convective cooling of `[-1,1]^2` into ambient 0, `Bi = h/α = 6` (Carslaw & Jaeger) | `cos λx cos λy e^{-2αλ²t}`, `λ tan λ = Bi` |
| `flux`      | uniform heating `q` through all walls of `[-1,1]^2`                            | `q(x²+y²)/(2α) + 2qt` |
| `radiative` | radiating plate, `eps = 0.5`, `sigma = 1` (manufactured `T_inf`)               | `1 + 0.4 e^{-t} cos(πx/2) cos(πy/2)` |

`heat_problem(..., manufactured_boundary=True)` generates the data from the
exact gradient. This makes the same solutions valid on the sheared domain of the
non-orthogonal mesh generator. `transport_problem(model, bc_type)` gives
manufactured problems for the Cattaneo, fractional, advection and Pennes models:
`u = 1 + 0.3(1 + t + t²) cos(πx/2) cos(πy/2)`, whose Caputo derivative is exact.

## Results

`boundary_conditions_study.py` writes `test_plots/boundary_conditions/capability_matrix.png`
and `summary.json`. It runs meshes `n = 8, 16, 32, 64`, with `T = 0.05` and `dt ∝ h²`.
The table gives the finest-pair observed order of the relative L² error:

| solver                         | dirichlet | neumann | robin | flux  | radiative |
|--------------------------------|:---------:|:-------:|:-----:|:-----:|:---------:|
| Polygonal TPFA (square)        | 2.00 | 2.00 | 1.96 | exact | 1.99 |
| Polygonal TPFA+corr (skew 0.3) | 1.98 | 1.95 | 1.96 | 1.87  | 1.98 |
| Polygonal quadratic (skew 0.3) | 1.96 | 2.08 | 2.02 | exact | 1.97 |
| AFC global limiter             | 2.00 | 2.00 | 1.96 | exact | 2.00 |
| Enthalpy (L = 0)               | 2.00 | 2.00 | 1.96 | exact | 2.00 |
| Apparent-capacity Picard (L = 0) | 2.00 | 2.00 | 1.96 | exact | 2.00 |
| Triangular (Delaunay P1)       | 2.05 | 1.88 | 2.01 | 2.04  | 2.00 |
| Curvilinear grid (warp 0.08)   | 2.00 | 1.35 | 1.60 | 1.60  | 1.68 |
| Cattaneo                       | 2.02 | 2.01 | 2.01 | 2.01  | 2.01 |
| Caputo fractional              | 2.04 | 2.02 | 2.04 | 2.02  | 2.04 |
| Advection–diffusion (central)  | 1.99 | 2.00 | 2.00 | 2.00  | 2.00 |
| Pennes                         | 2.00 | 2.00 | 2.00 | 2.00  | 2.00 |

"exact" means the quadratic-in-space flux solution is reproduced to round-off
(~1e-14).

Physical checks in `tests/test_boundary_conditions.py`:
- Insulated plates conserve heat to 1e-10.
- Uniform wall heating injects exactly `q × perimeter × t`.
- Convective and radiative cooling stay between the data and the ambient
  temperature and relax towards ambient.
- The tangential correction vanishes on orthogonal meshes.

## Known limitations

- **Curvilinear grid.** Non-Dirichlet orders fall to 1.35–1.7 at `n = 64`, and the
  Neumann error constant is large (1e-2). Both come from the one-sided corner
  stencils of the structured solver, which were not changed in this round.
- **Adjoints.** `heat_solver/adjoint.py` still covers Dirichlet, Robin and flux
  for the linear adjoint, and Dirichlet and flux for the fractional / enthalpy
  adjoints. Radiative adjoints are not implemented.
- **Dirichlet on cell-centred solvers.** These impose it on the boundary-cell band by
  default. Every one of them also offers `dirichlet_mode="face"` (section above). It is
  not available with cell-averaged unknowns (`flux_discretization="quadratic"`), where it
  would be first order. On skewed meshes its error constant is 2–3x the band's.
