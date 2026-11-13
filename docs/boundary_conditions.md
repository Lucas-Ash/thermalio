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

`PolygonalHeatSolver(..., dirichlet_mode=...)`:

- **`"band"`** (default). The boundary cells are held at the data at their centroids
  (`u_c = g(x_c)`), with identity rows in the matrix.
- **`"face"`**. Every cell is an unknown, and the data are imposed on the boundary
  faces through the face flux `α_n (g_f − v) / d`. Here `d` is the centroid-to-face
  distance and `v` the cell value, extrapolated tangentially on skewed cells. This is
  what FiPy does. It is the Robin closure below with β → ∞, so it shares that
  machinery: factorization reuse, iterative solvers and all time schemes.
  - On Cartesian meshes it reproduces FiPy to 4e-14 (`tests/test_face_dirichlet.py`).
  - It is second order on square, skewed, hexagonal and tiled meshes.

Both modes are second order and, on the cross-code problems, about equally accurate:
8.33e-5 (band) against 8.34e-5 (face) on the isotropic problem at n = 64. Face mode
differs in two ways:
- the boundary cells carry real errors, as in FiPy;
- the boundary heat flow is the closure flux itself, with no band reaction to
  reconstruct.

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
  default. `PolygonalHeatSolver` also offers `dirichlet_mode="face"` (section below).
  It is not available with cell-averaged unknowns (`flux_discretization="quadratic"`),
  where it would be first order. The transport, enthalpy and AFC solvers use the band
  only.
