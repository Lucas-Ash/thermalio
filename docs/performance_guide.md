# Performance guide: accuracy, timing, acceleration and parallelism

This guide answers three questions with measurements from this repository:

1. How do Thermalio's accuracy and speed compare with FiPy and scikit-fem?
2. Where does Thermalio's run time actually go?
3. How much do acceleration techniques and parallelism matter, and in what order?

All timings were taken on one 4-core x86-64 cloud VM (no GPU), with Python 3.13,
NumPy 2.5, SciPy 1.18 and pyamg 5.4. Every code uses SciPy's single-threaded sparse
kernels unless a panel says otherwise. Absolute times will differ on other
machines; the ratios and scaling exponents are the transferable part.

Scripts and figures:

| script | figures (`test_plots/benchmark/...`) |
|---|---|
| `external_code_comparison.py` | `external_codes/external_code_comparison.png`, `external_codes/external_code_timing.png` |
| `performance_study.py` | `performance/where_time_goes.png`, `linear_solvers.png`, `parallelism.png`, `accuracy_as_acceleration.png` |

## The short answer

- **For Thermalio today, parallelism inside a single solve buys almost nothing.**
  Only about 20% of a run is linear algebra, so by Amdahl's law even unlimited
  cores give at most about 1.3x. The rest is serial Python work.
- **Serial acceleration comes first, and it is large.**
  - Vectorizing the operator assembly makes it 260–470x faster, with a bit-identical
    matrix.
  - Together with cached mesh geometry, that makes a 100-step run 3–8x faster.
  - Crank–Nicolson reaches a 1e-5 time error in 28 steps where backward Euler
    needs about 9,700.
  - Algebraic multigrid is 12x faster than plain CG for large time steps at one
    million unknowns.
- **Parallelism pays in two places.** Many independent solves (parameter sweeps,
  uncertainty quantification, inverse problems) run 3.3–3.5x faster on 4 cores with
  no code changes. Once the code is vectorized, threaded kernels give 3.1–3.5x on 4
  cores and the Amdahl limit rises from 1.3x to 6.3x.
- **Thermalio is 2D, so problems stay small.** Even h = 1/1000 is only one million
  cells, and that solves in 0.3 s per time step on one core. GPUs and distributed
  memory (MPI) matter for 3D codes and for 10^7+ unknowns. For Thermalio they are a
  late, optional step.

## 1. Thermalio vs FiPy vs scikit-fem

The same manufactured problems are solved by all codes:

- backward Euler with the same time step, 200 steps;
- meshes from n = 8 to n = 64 per side;
- relative L² error against the exact solution.

Finest mesh (n = 64), wall time including setup:

| problem | Thermalio FV | FiPy FV | scikit-fem P1 | scikit-fem P2 |
|---|---|---|---|---|
| isotropic Dirichlet | 3.3e-4 in 1.5 s | 8.3e-5 in 6.5 s | 2.0e-3 in 4.0 s | 1.5e-5 in 44 s |
| anisotropic tensor (ratio 5, 30°) | 3.1e-5 in 1.6 s | 5.3e-4 in 6.8 s | 1.9e-3 in 4.0 s | 1.5e-5 in 91 s |
| Pennes perfusion + Robin | 2.5e-5 in 2.3 s | — | 1.0e-4 in 3.7 s | 5.5e-6 in 35 s |

The scikit-fem P2 column has about 4x as many unknowns as the others. Thermalio's
quadratic flux on the skewed anisotropic mesh reaches 1.45e-5 in 6.0 s.

How to read `external_code_timing.png`:

- **Per-DOF cost (panel b).** Thermalio costs 2–3 µs per time step per unknown
  at every size. scikit-fem costs 5–13 µs and FiPy 8–230 µs.
  - FiPy rebuilds its matrix in Python every step. That is a fixed overhead of about
    15 ms per step, so it dominates small problems.
  - Our scikit-fem adapter re-assembles the load by quadrature and refactorizes the
    matrix every step.
- **Time to a target error (panel c).**
  - Isotropic case, 1e-3: Thermalio 0.5 s, scikit-fem P2 1.2 s, FiPy 4.0 s.
    scikit-fem P1 never gets there by n = 64.
  - Anisotropic case, 1e-3: Thermalio 0.03 s, FiPy 4.0 s. FiPy's anisotropic flux
    converges at order 0.9.
- **Work–precision (panels d–f).** Per digit of accuracy:
  - Thermalio has the lowest cost on the anisotropic problem, and on the Pennes
    problem down to about 1e-4.
  - On the isotropic Dirichlet problem, FiPy is 4x more accurate at the same mesh.
    FiPy imposes Dirichlet data on boundary faces; Thermalio fixes a band of cells,
    which shifts the boundary by h/2 (gap-analysis item 4).
  - scikit-fem P2 wins at very tight tolerances (1e-5 and below). That is the
    higher-order effect of section 4.

These timings compare implementations as driven by our adapters, not the best each
library can do. The scikit-fem adapter, for example, could reuse its factorization.

## 2. Measure first: where the time goes

`where_time_goes.png` splits a 100-step Thermalio solve (`PolygonalHeatSolver`,
Dirichlet, square mesh) into wall-clock phases:

| phase (262,144 cells) | time |
|---|---|
| mesh geometry, first build | 13.5 s |
| operator assembly (per-face Python loop) | 15.2 s |
| LHS + Dirichlet rows (LIL loop) | 0.8 s |
| sparse LU factorization | 3.3 s |
| 100 right-hand sides | 1.0 s |
| 100 LU back-solves | 8.0 s |
| **`solve()` + setup total** | **43.6 s** |

Only the factorization and back-solves (11.3 s, 26%) are compiled linear algebra.
Three facts follow:

- **Assembly.** `solve()` reassembles the diffusion matrix on every call, with a
  per-face loop that writes into a LIL matrix. Thermalio already has a vectorized
  COO path, `_assemble_diffusion_vectorized`, used for temperature-dependent
  conductivity. Fed a constant conductivity, it produces the same matrix bit for bit
  in 0.044 s: 342x faster at 262k cells and 260–470x across sizes.
- **Geometry.** Geometry built once and then cached (as a mesh loaded from the mesh
  library is) costs 0.68 s instead of 13.5 s.
- **Total.** The same algorithm with cached geometry, vectorized assembly and a lean
  time loop takes 13.8 s instead of 43.6 s, 3.2x faster. At 4,096 cells it is 8.2x
  faster, because fixed overheads matter more there. The lean loop agrees with
  `solve()` to round-off (6e-16).

The time-order study shows the same effect from another angle. A complete
`solve()` on 48² cells costs about 0.2 s whether it takes 5 steps or 80. The
per-call assembly sets the floor, not the time stepping.

**Lesson.** Profile before optimizing. The first win here is removing Python loops
from per-call work. It needs no new algorithm and changes no result.

## 3. Choose the right linear solver

`linear_solvers.png` solves Thermalio's backward-Euler system with four strategies:

- refactorize every step (`spsolve`);
- factorize once, then back-solve each step;
- conjugate gradient (CG) with a Jacobi preconditioner;
- CG with smoothed-aggregation algebraic multigrid (AMG).

The system is `(h² I + Δt α L) u = rhs` on the interior cells, which matches
Thermalio's assembled matrix exactly. Sizes run up to 1,044,484 unknowns, in two
regimes:

| per time step, N ≈ 260k | refactorize | factor once | CG + Jacobi | CG + AMG |
|---|---|---|---|---|
| transient, Δt = h²/(2α) | 2.84 s | 0.068 s (+2.8 s once) | 0.054 s (19 its) | 0.19 s (4 its) |
| large steps, Δt = 1 | 1.83 s | 0.061 s (+1.8 s once) | 1.98 s (714 its) | 0.37 s (7 its) |

At N ≈ 1M:

| per time step, N ≈ 1M | CG + Jacobi | CG + AMG |
|---|---|---|
| transient | 0.28 s (19 its) | 0.80 s (4 its) |
| large steps | 17.2 s (1,166 its) | 1.44 s (7 its) |

- **Never refactorize every step for a linear problem.** Reusing the factorization is
  42x cheaper per step at 260k unknowns. Thermalio already does this
  (`reuse_linear_lhs=True`).
- **The time step sets the conditioning.** With Δt ∝ h², which is accuracy-balanced
  for backward Euler with a second-order space discretization, the mass matrix
  dominates. CG + Jacobi then converges in 19 iterations at every size. It is the
  cheapest method at large N and needs no setup.
- **Large time steps need multigrid.** For near-steady problems, implicit large steps
  or Stefan problems, the Jacobi iteration count grows like 1/h: 136 → 1,166 from 4k
  to 1M unknowns. AMG stays at 6–7 iterations, 12x faster at 1M.
- **Scaling exponents decide the future.** Fitted over the largest meshes, the cost
  of a 200-step run grows like:
  - AMG: N^1.02–1.11;
  - CG + Jacobi with large steps: N^1.43;
  - factorize once: N^1.23–1.27, with memory growing superlinearly as well.

  Direct solvers are the right choice up to a few hundred thousand unknowns; AMG
  beyond that.

## 4. Higher order is acceleration

`accuracy_as_acceleration.png`:

- **Time.** On the same 48² mesh, against a fine-Δt reference:
  - Crank–Nicolson (order 2) reaches a temporal error of 1e-5 in 28 steps.
  - Backward Euler (order 1) needs about 9,700 steps (extrapolated with its observed
    order 1.00), roughly 350x more.
  - Each Crank–Nicolson step costs one extra sparse mat-vec.
- **Space.** At about 4,000 unknowns:
  - scikit-fem P1 is at 2e-3 and P2 at about 1e-4.
  - Second-order-accurate finite-volume schemes are in between.
  - The P2 error falls like h³ in this L² norm. Each halving of h buys 8x accuracy
    instead of 4x, so P2 wins once the tolerance is tight.

Thermalio's `time_scheme="crank_nicolson"` and `flux_discretization="quadratic"`
already exist. Using them is the cheapest acceleration available.

## 5. Parallelism: what it buys, and when

`parallelism.png`:

- **(a) Task parallelism.** 16 independent Thermalio solves are farmed out with
  `concurrent.futures.ProcessPoolExecutor`.
  - Speedup: 1.65–1.99x on 2 workers and 3.32–3.47x on 4 (two runs).
  - 8 workers on 4 cores gain nothing more.
  - This is the easiest parallelism there is, and it fits parameter sweeps,
    uncertainty quantification, mesh-convergence studies and inverse-problem
    gradient batches.
- **(b) Data parallelism inside kernels.** Row blocks or BLAS threads, 4 threads:
  - vector triad: 3.5x, memory bandwidth 9 → 31 GB/s;
  - sparse mat-vec with 4M unknowns: 3.2x;
  - dense matrix multiply: 3.1–3.7x.

  On this VM, memory bandwidth has not saturated at 4 cores. On a typical
  workstation socket, memory-bound kernels such as sparse mat-vec, CG and explicit
  updates stop scaling well below the core count, because they run at memory speed.
  Compute-bound kernels keep scaling. This is the roofline model.
- **(c) Amdahl's law.** At 65k cells, with the measured serial fraction:
  - **Thermalio today:** the parallelizable fraction is f = 0.21 (LU and back-solves
    only), so the limit is 1.26x on any number of cores.
  - **Lean pipeline** (cached geometry, vectorized assembly): f = 0.84, limit 6.3x.
  - **Engineered parallel code:** f = 0.99 gives 100x.

  The serial fraction, not the core count, decides the ceiling. Removing serial
  Python overhead is a precondition for in-solve parallelism.

**Not measured here.**
- **GPUs.** They help memory-bound sparse solvers roughly in proportion to their
  memory bandwidth, typically an order of magnitude above one CPU socket. That pays
  only once problems exceed about 10^5–10^6 unknowns per device, and it requires the
  whole time loop to live on the device.
- **Distributed memory (MPI domain decomposition).** It is what lets 3D codes such as
  FEniCS, deal.II and MOOSE go beyond one node, for 10^7–10^10 unknowns or more
  memory than one machine has.

A 2D Thermalio problem rarely reaches that range.

## 6. What to do, in order of measured payoff

1. **Vectorize constant-coefficient assembly and the Dirichlet-row construction, and
   cache the assembled operator across `solve()` calls.**
   - Gain: 3–8x end-to-end; up to 470x on assembly alone.
   - Results bit-identical, so no QA baselines change.
2. **Default to second-order time stepping for accuracy studies.** Crank–Nicolson,
   or BDF2 for stiff problems. Gain: 10–350x fewer steps at tight tolerances.
3. **Pick the linear solver by regime.**
   - Factorize once below about 250k unknowns.
   - CG + Jacobi for Δt ∝ h².
   - CG + AMG for large steps, steady or nonlinear problems, and 10^6+ unknowns.
4. **Run independent solves in a process pool.** Sweeps, uncertainty quantification
   and inverse problems get 3.3–3.5x on 4 cores now, roughly linear in cores.
5. **Only then add threads or a GPU.** With f ≈ 0.84 after steps 1–3, 4 cores give
   about 2.7x and the ceiling is about 6x. Going further needs the remaining serial
   steps moved into compiled code as well.

## Reproduce

```bash
pip install -r requirements-external.txt   # FiPy, scikit-fem, pyamg, threadpoolctl
MPLCONFIGDIR=/tmp/matplotlib python external_code_comparison.py            # ~5 min
MPLCONFIGDIR=/tmp/matplotlib python external_code_comparison.py --replot   # redraw only
MPLCONFIGDIR=/tmp/matplotlib python performance_study.py                   # ~4 min
```

Timings are single runs, except that kernel timings take the best of 3–5. The
two-worker ensemble speedup varied from 1.65x to 1.99x between runs; treat
differences under about 20% as noise.
