# Gap analysis: what keeps Thermalio from matching established codes

- **Last updated:** 2026-11-11
- **Compared against:** FiPy 4.0.3 (NIST finite volumes) and scikit-fem 12.0.2
  (finite elements), measured in this repo (`external_code_comparison.py`);
  FEniCS / deal.II (not pip-installable here) by documented capability only.

## Where Thermalio is already at or above par

- **Verification depth:** manufactured solutions, observed order/GCI, N-version
  and cross-code checks, Taylor-tested discrete adjoints, energy audits.  This is
  more systematic than the default tooling of the compared codes.
- **Accuracy on anisotropic and non-orthogonal problems:** with the quadratic
  flux it is second order on skewed, hexagonal, randomly perturbed and mixed
  polygonal meshes.  On the rotated-tensor benchmark it is 17x more accurate
  than FiPy's tensor diffusion at n = 64.
- **Non-classical physics in one framework:** Cattaneo, Caputo fractional,
  Pennes, non-Fourier Stefan (apparent-capacity and enthalpy), bound-preserving
  AFC with sign-changing sources, and exact discrete adjoints for all of them.

## Gaps, ranked by how much they block parity

| # | Gap | Evidence | Established codes |
|---|-----|----------|-------------------|
| 1 | **Mesh import/export (2D by design)** — *addressed 2026-11* | `mesh_io` + `mesh_library`: Gmsh/VTU/npz import and export, saved generator configurations with stored geometry, all solvers and drivers accept imported meshes; set-up of 65k cells 18 s -> <1 s. Remaining: no Gmsh 4 import without meshio, no curved (high-order) boundary geometry | FiPy reads Gmsh; scikit-fem reads meshio formats |
| 2 | **Setup cost dominated by Python loops** | 65,536 cells: mesh 2.8 s + solver init 8.3 s + assembly 3.9 s (≈15 s before the first solve) vs FiPy 4.2 s and scikit-fem P1 6.3 s *for the entire run*; quadratic-flux geometry precompute 6.3 s at 16k cells | Vectorized or compiled assembly |
| 3 | **No scalable linear solvers or parallelism** | `_solve_sparse_linear_system`: SciPy LU, or *unpreconditioned* CG/BiCGSTAB; no AMG/ILU, no PETSc, no MPI | FEniCS/deal.II: PETSc/Trilinos, AMG, MPI to 10^9 DoF; FiPy: optional PETSc/Trilinos |
| 4 | **Dirichlet imposed on a band of boundary cells** — *addressed 2026-11 for `PolygonalHeatSolver`* | `dirichlet_mode="face"` imposes the data on boundary faces as FiPy does (reproduces FiPy to 4e-14 on Cartesian meshes). Correction: the "4x larger error than FiPy" measured earlier came from the comparison running the legacy `reconstructed` flux, not from the band. With the default two-point flux, band and face are equally accurate (8.33e-5 vs 8.34e-5 against FiPy's 8.34e-5). Remaining: face mode for the transport / enthalpy / AFC solvers and for the quadratic flux (cell-averaged unknowns); the latent-heat band jump in the enthalpy energy audits | Face/nodal Dirichlet |
| 5 | **Inconsistent boundary-condition and feature coverage across solvers** — *addressed 2026-11 for the forward solvers* | One shared BC module: every forward solver (polygonal incl. quadratic/MPFA fallback, AFC, enthalpy, triangular, curvilinear, transport) accepts dirichlet/neumann/robin/flux/radiative, second order on standard problems (`docs/boundary_conditions.md`). Remaining: curvilinear non-Dirichlet orders 1.35-1.7 (corner stencils); adjoints still Dirichlet/Robin/flux (linear) and Dirichlet/flux (fractional, enthalpy); MPFA proper is still Dirichlet-only | One BC system shared by all operators |
| 6 | **No operator/term abstraction** | 12 solver classes with 10 hand-written `solve()` loops; each physics re-implements time stepping, BCs and nonlinear iteration | FiPy term algebra (`TransientTerm == DiffusionTerm + ...`); FEniCS UFL weak forms with automatic Jacobians |
| 7 | **Limited time integration and nonlinear solvers** | Backward Euler / Crank–Nicolson; the three-level Cattaneo scheme is first order; adaptivity only in the enthalpy solver; Newton only in the enthalpy solver, Picard(+Anderson) elsewhere.  Nonlinear CN was first order until this round (fixed) | PETSc TS (BDF, IMEX, adaptive RK) and SNES Newton through FEniCS/deal.II.  FiPy and scikit-fem are similar to Thermalio here |
| 8 | **No higher-order discretizations** | FV only, second order at best; at n = 64 scikit-fem P2 is 21x more accurate on the isotropic case (1.5e-5 vs 3.3e-4) | P2+ FEM, hp-adaptivity (deal.II) |
| 9 | **No spatial adaptivity (AMR)** | Uniform refinement only | deal.II/FEniCS adaptive refinement with error estimators |
| 10 | **Software packaging and I/O** | No `pyproject.toml` (used via `sys.path`); no releases; no API docs; no VTK/XDMF output; about 250 MB of generated plots tracked in git; dozens of top-level scripts | `pip install`, versioned releases, API docs, ParaView output |
| 11 | **Validation scope** | Cross-code comparison covers linear problems only; nonlinear/phase-change validated against the Neumann solution only; no experimental data | Long validation records and user communities |

## Known numerical limitations still open

- **Quadratic flux:** not monotone (about 40% positive off-diagonals; it overshoots steep fronts, and AFC with the quadratic target fixes this); about 21 nonzeros per row; coercivity is not proved.  Numerically, the symmetric part is positive definite on all four mesh families for anisotropy up to 1e4 (`tests/test_quadratic_flux_limitations.py`).
- **Rough meshes:** convergence on randomly perturbed / tiled meshes is pre-asymptotic until about 10^4 cells.
- **Apparent-capacity Stefan solvers:** energy defects of 26-32% on coarse grids (`direction_c_flux_audit_study.py`); the enthalpy solver removes them.

## Suggested roadmap (impact per effort)

1. **Packaging + mesh I/O** (`pyproject.toml`, meshio import, VTK/XDMF export).
   Low effort; makes the code usable on real geometries and in ParaView.
2. **Vectorize geometry and assembly** (NumPy/Numba for edge maps, gradient
   reconstruction and the quadratic-flux precompute) and add **AMG / ILU
   preconditioning** (pyamg).  This closes gaps 2-3 for single-node problems.
3. **Face-based Dirichlet** as an option for every FV operator.  It removes the
   band artefacts and the 4x error-constant penalty.
4. **Unify the solver core:** a single spatial operator (flux choice + BC
   registry + capacity model) and separate time integrators (BE, CN, BDF2,
   embedded adaptive) and nonlinear solvers (Picard, Anderson, Newton with
   assembled Jacobians).  This fixes gaps 5-7 structurally and halves the
   code.
5. **3D polyhedral FV** (the quadratic reconstruction generalizes directly).
   High effort; needed for parity with FEniCS/deal.II on real devices and tissue.
6. **Error estimators + h-adaptivity** on top of 4.
7. **Nonlinear cross-code and experimental validation** (e.g. FEniCS Stefan
   benchmarks where installable; published melting/solidification experiments).
