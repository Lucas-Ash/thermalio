# Performance guide: accuracy, timing, acceleration and parallelism

This guide answers three questions with measurements from this repository:

1. How do Thermalio's accuracy and speed compare with FiPy and scikit-fem?
2. Where does Thermalio's run time actually go?
3. How much do acceleration techniques and parallelism matter, and in what order?

The first edition of this guide measured the solver as it was and ranked its
improvements by payoff. The first three are now implemented. This edition reports
before-and-after numbers:

1. Vectorized assembly with operator caching.
2. BDF2 as the default time scheme of the verification drivers.
3. Factorization reuse with Robin and radiative boundaries, and a vectorized
   first-time setup (mesh generators, gradient stencils).

All timings come from one 4-core x86-64 cloud VM with no GPU, running Python 3.13,
NumPy 2.5, SciPy 1.18 and pyamg 5.4.
- Every code uses SciPy's single-threaded sparse kernels unless a panel says
  otherwise.
- Absolute times will differ on other machines. The ratios and scaling exponents are
  the part that transfers.
- Timings vary by about ±30% between sessions on this VM. FiPy's isotropic run took
  6.5 s, 4.9 s and 3.9 s in three sessions, and Thermalio's legacy assembly loop 15.2 s
  and 10.8 s. Ratios measured within one session are the reliable numbers, and this
  guide quotes those wherever it compares before with after.

Scripts and figures:

| script | figures (`test_plots/benchmark/...`) |
|---|---|
| `external_code_comparison.py` | `external_codes/external_code_comparison.png`, `external_codes/external_code_timing.png` |
| `performance_study.py` | `performance/where_time_goes.png`, `boundary_closures.png`, `linear_solvers.png`, `parallelism.png`, `accuracy_as_acceleration.png` |

## The short answer

- **Serial acceleration came first, and it was large.**
  - The diffusion operator used to be assembled by a per-face Python loop. It is now
    assembled from per-face arrays: 10–45x faster on a mesh's first assembly and
    about 50x once the face arrays are cached.
  - The operator and its LU factorization are reused across `solve()` calls.
  - Robin and radiative boundaries reuse their factorization instead of
    refactorizing every step: 13x and 24x faster on 65k cells. Pennes + Robin in the
    cross-code comparison went from 2.3 s to 0.20 s.
  - First-time setup on a never-seen mesh (generator plus solver construction):
    18.4 s → 3.8 s at 262k cells.
  - A 100-step Dirichlet run on a never-seen 262k-cell mesh now takes 14.9 s, mesh
    generation included. The same run with the legacy code takes about 40 s in this
    session (2.7x). A repeated solve takes 8.9 s, which is the linear-algebra floor (LU
    back-solves). At 4k cells, runs are 4–11x faster.
  - Results are unchanged:
    - Dirichlet, Neumann, Robin and flux are bit-identical to the previous commit,
      or agree to round-off (≤ 3.8e-15) after the assembly change.
    - Radiative agrees with full Newton to ≤ 2e-12.
    - QA passes 75/75, bit-identical.
- **Second-order time stepping is the cheapest accuracy there is.** At the same Δt,
  the verification drivers' new BDF2 default is:
  - 1,700x more accurate on the radiative case (square mesh);
  - 15–39x more accurate on temperature-dependent conductivity;
  - 3–9x more accurate on the time-dependent QA cases where the time error dominated
    (heat kernel, anisotropic kernel, sine mode, point source, off-axis wave);
  - 1.0–1.7x on cases limited by the spatial error (hot block, Nyquist oscillations,
    Green's function, driven sine).

  For a 1e-5 time error, BDF2 needs 88 steps, Crank–Nicolson 28 and backward Euler
  about 9,700.
- **Parallelism now has room to pay.** Before, only 21% of a run was parallelizable
  linear algebra, so Amdahl's law capped any number of cores at 1.26x. Now 96% is,
  so 4 cores could give about 3.6x and the ceiling is 28x.
  - Independent solves (sweeps, uncertainty quantification, inverse problems) ran
    3.2–3.5x faster on 4 cores. The same 16 solves now take 40 ms each, and that
    drops to 2.5x, because process start-up and data transfer become a large share.
    Task parallelism pays for tasks that are long compared with that overhead.
  - Threaded kernels give 2.0–3.8x on 4 cores (range over sessions).
- **Thermalio is 2D, so problems stay small.** h = 1/1000 is one million cells, and
  that takes 0.3 s per time step on one core. GPUs and distributed memory (MPI)
  matter for 3D codes and for 10^7+ unknowns.

## 1. Thermalio vs FiPy vs scikit-fem

The same manufactured problems are solved by all codes:

- backward Euler with the same time step, 200 steps (Thermalio included, for a
  like-for-like comparison);
- meshes from n = 8 to n = 64 per side;
- relative L² error against the exact solution.

Finest mesh (n = 64), wall time including setup:

| problem | Thermalio FV | FiPy FV | scikit-fem P1 | scikit-fem P2 |
|---|---|---|---|---|
| isotropic Dirichlet | 3.3e-4 in **0.16 s** (was 1.5 s) | 8.3e-5 in 3.9 s | 2.0e-3 in 2.5 s | 1.5e-5 in 27 s |
| anisotropic tensor (ratio 5, 30°) | 3.1e-5 in **0.20 s** (was 1.6 s) | 5.3e-4 in 3.7 s | 1.9e-3 in 2.7 s | 1.5e-5 in 26 s |
| Pennes perfusion + Robin | 2.5e-5 in **0.20 s** (was 2.3 s) | — | 1.0e-4 in 2.6 s | 5.5e-6 in 28 s |

All codes ran in the same session. The "was" values are from the first edition,
which ran in a slower session.

The scikit-fem P2 column has about 4x as many unknowns as the others. Thermalio's
quadratic flux on the skewed anisotropic mesh reaches 1.45e-5 in 3.0 s.

The Pennes + Robin run sped up only once item 3 landed (section 2b): its Robin
closure made every step refactorize the matrix.

How to read `external_code_timing.png`:

- **Per-DOF cost (panel b).** Per time step per unknown:
  - Thermalio: 0.2–0.7 µs (was 2–3 µs);
  - scikit-fem: 3–8 µs;
  - FiPy: 5–120 µs.
  - FiPy rebuilds its matrix in Python every step, a fixed overhead of about 15 ms
    per step.
  - Our scikit-fem adapter re-assembles the load by quadrature and refactorizes the
    matrix every step.
- **Time to a target error of 1e-3 (panel c).**
  - Isotropic: Thermalio 0.05 s, scikit-fem P2 0.64 s, FiPy 1.6 s. scikit-fem P1 does
    not get there by n = 64.
  - Anisotropic: Thermalio 0.013 s, FiPy 2.0 s. FiPy's anisotropic flux converges at
    order 0.9.
  - Pennes + Robin: Thermalio 0.04 s, scikit-fem P2 0.22 s, P1 0.32 s.
- **Work–precision (panels d–f).**
  - Thermalio has the lowest cost per digit on the isotropic and anisotropic
    problems, and on the Pennes problem down to about 1e-4.
  - On the isotropic Dirichlet problem, FiPy is still 4x more accurate on the same
    mesh. FiPy imposes Dirichlet data on boundary faces; Thermalio fixes a band of
    cells, which shifts the boundary by h/2 (gap-analysis item 4).
  - scikit-fem P2 wins below about 1e-5. That is the higher-order effect of
    section 4.

These timings compare implementations as driven by our adapters, not the best each
library can do. The scikit-fem adapter, for example, could reuse its factorization.

## 2. Measure first: where the time goes

`where_time_goes.png` splits a 100-step Thermalio solve (`PolygonalHeatSolver`,
Dirichlet, square mesh) into wall-clock phases. At 262,144 cells:

| phase | before (same session) | now |
|---|---|---|
| mesh generator + first solver construction | 18.4 s | 3.8 s (0.6 s with geometry cached) |
| operator assembly | 10.8 s (per-face loop) | 0.83 s first time on a mesh, about 0.3 s after |
| LHS + Dirichlet rows | 0.65 s (LIL loop) | 0.03 s |
| sparse LU factorization | 2.6 s | 2.6 s (reused by later `solve()` calls) |
| 100 right-hand sides | | 0.86 s |
| 100 LU back-solves | | 7.2 s |
| **100-step run on a never-seen mesh, generator included** | **about 40.5 s** (sum of the legacy phases) | **14.9 s** (2.7x) |
| 100-step run with cached geometry | about 22.7 s | 11.8 s (1.9x) |
| repeated `solve()` | everything rebuilt | **8.9 s** |

The first edition measured 43.6 s (without the generator) in a slower session. LU
back-solves did not change and now dominate, which is why the end-to-end gain at this
size is smaller than at 4k cells.

What changed, and why it was safe:

- **Vectorized assembly** (`heat_solver/fv_assembly.py`). Per-face geometry is
  extracted once per mesh and cached with the mesh's derived geometry. Each
  assembly is then a few array operations.
  - Per-entry sums run in the legacy order through `np.bincount`, which adds
    sequentially, and exact zeros are dropped as LIL drops them. The sparsity
    pattern is therefore the legacy one.
  - The only difference is the last bit of some 2-vector dot products. The old code
    called BLAS on length-2 vectors, and BLAS uses fused multiply-adds on this CPU.
  - Matrices agree to at most 4e-16 relative over 6 mesh types × 4 conductivity
    kinds, with and without non-orthogonal correction and for the reconstructed
    flux. They are bit-identical where the arithmetic is exact.
  - Solutions agree to at most 3.8e-15 over 150 solver scenarios.
  - The legacy loops remain as `_assemble_tpfa_loop` and
    `_assemble_diffusion_reconstructed_loop`, so `tests/test_fv_assembly.py` can keep
    checking this.
- **Dirichlet rows** are built by array slicing with the same pattern as the LIL
  version (18x faster). The per-step `rhs[i] = g[i]` loops are gone.
- **Caching across `solve()` calls.** The operator and each `(Δt, scheme)`
  factorization are kept while `solver.A` is the operator `solve()` assembled.
  Replacing `solver.A` invalidates them.
- **First-time setup** (item 3), both bit-identical:
  - The mesh generators computed cell centroids one cell at a time. They now share
    the vectorized routine the mesh library already used.
  - The least-squares gradient stencils were built cell by cell, with a tiny SVD and
    solve for each cell. Cells are now batched by neighbour count. NumPy's stacked
    `matmul`, `matrix_rank` and `solve` call the same BLAS/LAPACK routines for each
    cell, so the stencils are bit-identical to the loop (kept as
    `_build_gradient_reconstruction_loop`).

Smaller meshes gain more, because fixed overheads weighed more there. At 4,096 cells
a run on a never-seen mesh takes 0.15 s, against 0.56 s in the first edition, and a
repeated solve takes 0.05 s. The unit-test suite went from 210 s to 87 s over items
1–3.

### 2b. Boundary conditions that look time-dependent

With Robin or radiative boundaries, the matrix is `base + Δt B`, where `B` is the
boundary closure. The solvers treated it as changing every step and refactorized it
every step, and for radiation every Newton iteration. Profiling showed this was the
whole cost: rebuilding `B` was cheap, and the factorization was not.

`boundary_closures.png`, 100 steps, same session:

| cells | Robin: refactorize → reuse | radiative: full Newton → frozen Jacobian |
|---|---|---|
| 4,096 | 0.77 s → 0.11 s (7x) | 1.75 s → 0.18 s (10x) |
| 16,384 | 4.9 s → 0.40 s (12x) | 8.5 s → 0.48 s (18x) |
| 65,536 | 28.9 s → 2.2 s (13x) | 59.3 s → 2.4 s (24x) |

- **Robin** (`heat_solver/factor_cache.py`). `B` only changes when β changes, and in
  practice β is constant. `FactorCache` keeps the factorization and reuses it while
  the operator and the values of `B` are unchanged.
  - SciPy's `spsolve` on a CSR matrix factorizes the transpose and solves
    transposed. `spsolve_factor` does exactly that, so a reused factorization gives
    the same bits as a fresh `spsolve`.
  - The cache key uses the exact Δt. The step schedule produces Δt values that differ
    in the last bit, and a rounded key would reuse a slightly different matrix.
  - Results are bit-identical in 120 polygonal, 40 transport and 72 curvilinear runs.
  - Time-dependent β still refactorizes, as before.
- **Radiative.** `B` depends on the temperature, so it does change. Newton's method
  refactorizes at every iteration. Frozen-Jacobian Newton reuses one factorization:
  - across iterations and steps, refreshing it only when an update shrinks by less
    than half;
  - converging to the same fixed point, here to ≤ 2e-12;
  - taking 2 iterations per step and no new factorization after the first step.

  `nonlinear_options={"radiative_jacobian": "newton"}` restores full Newton.
- **Curvilinear** boundary rows used to be built by converting the whole matrix to
  LIL every step. They are now spliced into the CSR matrix with array operations, and
  are bit-identical to the old rows.

All three solver families (polygonal, transport, curvilinear) use this.

**Lesson.** Profile before optimizing. The win came from removing Python loops from
per-call work, without a new algorithm and without changing results.

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
| transient, Δt = h²/(2α) | 2.83 s | 0.066 s (+2.8 s once) | 0.054 s (19 its) | 0.22 s (4 its) |
| large steps, Δt = 1 | 1.87 s | 0.063 s (+1.8 s once) | 1.91 s (714 its) | 0.35 s (7 its) |

At N ≈ 1M:

| per time step, N ≈ 1M | CG + Jacobi | CG + AMG |
|---|---|---|
| transient | 0.35 s (19 its) | 0.87 s (4 its) |
| large steps | 21.4 s (1,166 its) | 1.56 s (7 its) |

- **Never refactorize every step for a linear problem.** Reusing the factorization is
  about 43x cheaper per step at 260k unknowns. Thermalio does this, and now also
  across `solve()` calls.
- **The time step sets the conditioning.** With Δt ∝ h², the mass matrix dominates
  and CG + Jacobi converges in 19 iterations at every size. It is the cheapest
  method at large N and needs no setup.
- **Large time steps need multigrid.** With near-steady problems, large implicit
  steps or Stefan problems, Jacobi's iteration count grows like 1/h: 136 → 1,166
  from 4k to 1M unknowns. AMG stays at 6–7 iterations and is 14x faster at 1M.
- **Scaling exponents decide the future.** Fitted over the largest meshes, a
  200-step run grows like:
  - AMG: about N^1.0–1.1;
  - factorize once: N^1.2–1.3, with memory growing superlinearly too.

  Direct solvers are right up to a few hundred thousand unknowns; AMG beyond that.

## 4. Higher order is acceleration

`accuracy_as_acceleration.png`, panel (a): time error on a 48² mesh against a
fine-Δt reference.

| scheme | steps for a 1e-5 time error | notes |
|---|---|---|
| backward Euler (order 1) | ~9,700 (extrapolated) | |
| BDF2 (order 2, damping) | 88 | error constant about 12x Crank–Nicolson's |
| Crank–Nicolson (order 2) | 28 | not damping |

**Why BDF2 is the driver default rather than Crank–Nicolson.** Crank–Nicolson's
amplification factor tends to −1 for stiff modes, so rough data rings.
- On a hot square with large steps, Crank–Nicolson produced −0.35 in a problem whose
  solution stays non-negative.
- BDF2 damps those modes and stayed positive there (`tests/test_bdf2.py`). It is
  A-stable, not positivity-preserving in general.

Several verification cases have rough data (`hot_block`, `nyquist_oscillations`,
`point_source`).

**What the default changed** (same Δt as before, finest mesh):

| verification case | backward Euler | BDF2 |
|---|---|---|
| radiative, square polygonal | 6.1e-4 | 3.6e-7 |
| radiative, tiled non-orthogonal | 4.4e-4 | 7.1e-6 |
| temperature-dependent conductivity | 1.7–1.9e-5 | 0.5–1.2e-6 |
| Stefan (apparent capacity), curvilinear: observed rate | 0.69 | 1.96 |
| QA, curvilinear: heat kernel / anisotropic kernel / sine mode | 1.5e-2 / 1.2e-2 / 4.9e-5 | 1.6e-3 / 2.7e-3 / 1.5e-5 |
| steady cases | unchanged (±3% through the transient) | |

- The polygonal, curvilinear and triangular solvers all accept
  `time_scheme="bdf2"`. It has variable steps and a backward-Euler first step. The
  observed order is 2.03–2.04 on cached, uncached and nonlinear paths and with every
  boundary type.
- The solver classes keep backward Euler as their own default. The cross-code
  comparison in section 1 also uses backward Euler everywhere, to compare like with
  like.

Panel (b), space: at about 4,000 unknowns, scikit-fem P1 is at 2e-3 and P2 at about
1e-4. P2's error falls like h³ in this norm, so it wins at tight tolerances.
`flux_discretization="quadratic"` is Thermalio's route there on skewed meshes.

## 5. Parallelism: what it buys, and when

`parallelism.png`:

- **(a) Task parallelism.** 16 independent Thermalio solves are farmed out with
  `concurrent.futures.ProcessPoolExecutor`.
  - Speedup: 1.9x on 2 workers and 3.2–3.5x on 4 while each solve took 0.15–0.35 s.
  - 8 workers on 4 cores gain nothing more.
  - After items 1–3 each solve takes about 40 ms. The 16-solve ensemble takes 0.66 s
    serially, and 4 workers give only 2.5x: process start-up and data transfer are now
    a large share. That is Amdahl's law at the task level. Batch several solves per
    task, or keep a pool alive across sweeps.
  - This is the easiest parallelism there is, and it fits parameter sweeps,
    uncertainty quantification, convergence studies and inverse-problem gradient
    batches.
- **(b) Data parallelism inside kernels**, 4 threads:
  - vector triad: 3.4–3.6x, memory bandwidth 9 → 31 GB/s;
  - sparse mat-vec with 4M unknowns: 2.0–3.2x (range over sessions);
  - dense matrix multiply: 3.1–3.8x.

  On this VM, memory bandwidth has not saturated at 4 cores. On a typical
  workstation socket, memory-bound kernels such as sparse mat-vec and CG stop scaling
  below the core count, while compute-bound kernels keep scaling. This is the roofline
  model.
- **(c) Amdahl's law** at 65k cells, with the measured serial fraction:
  - **Before:** f = 0.21 (only LU and back-solves parallel), limit 1.26x.
  - **Now:** f = 0.96 (assembly, right-hand sides and LU), limit 28x; 4 cores would
    give about 3.6x.
  - The remaining serial work is cached geometry and Dirichlet-row construction.
  - Caveat: LU back-solves are counted as parallelizable, but sparse triangular
    solves parallelize poorly. Iterative solvers (CG + Jacobi or AMG, section 3) are
    the parallel-friendly route.

**Not measured here.**
- **GPUs.** They help memory-bound sparse solvers roughly in proportion to memory
  bandwidth, typically an order of magnitude above one CPU socket. That pays only
  above about 10^5–10^6 unknowns per device, with the whole time loop on the device.
- **Distributed memory (MPI domain decomposition).** It lets 3D codes such as FEniCS,
  deal.II and MOOSE reach 10^7–10^10 unknowns.

A 2D Thermalio problem rarely gets there.

## 6. What to do, in order of measured payoff

1. ~~**Vectorize assembly and the Dirichlet rows, and cache the operator across
   `solve()` calls.**~~ **Done.**
   - 3.2x end-to-end at 262k cells, 7–10x at 4k cells; up to 50x on assembly.
   - Agrees with the old code to round-off.
2. ~~**Default to second-order time stepping for accuracy studies.**~~ **Done**
   (BDF2).
   - 3–1,700x more accurate at the same step count.
   - Equivalently, about 100x fewer steps for a 1e-5 time error.
3. ~~**Remove the remaining per-step cost with Robin / radiative boundaries, and
   speed up first-time setup.**~~ **Done** (section 2b).
   - The cost turned out to be refactorization, not assembly. Robin is 13x faster
     and radiative 24x at 65k cells; Pennes + Robin went from 2.3 s to 0.20 s.
   - Setup is 4.9x faster on a fresh 262k-cell mesh.
   - Results are bit-identical, except radiative, which agrees to ≤ 2e-12.
4. **Pick the linear solver by regime.** This is now the largest lever: LU back-solves
   are about 80% of a repeated solve.
   - Factorize once below about 250k unknowns.
   - CG + Jacobi for Δt ∝ h².
   - CG + AMG (pyamg) for large steps, steady or nonlinear problems, and 10^6+
     unknowns.
   - `PolygonalHeatSolver` offers `cg` and `bicgstab`, but no AMG preconditioner yet.
5. **Run independent solves in a process pool**, with tasks long compared with the
   pool overhead (batch solves per task). Sweeps, uncertainty quantification and
   inverse problems got 3.2–3.5x on 4 cores with 0.2–0.3 s tasks.
6. **Then add threads.** With f ≈ 0.96, 4 cores give about 3.6x, provided the solves
   use parallel-friendly kernels (item 4) rather than sparse triangular solves.

## Reproduce

```bash
pip install -r requirements-external.txt   # FiPy, scikit-fem, pyamg, threadpoolctl
MPLCONFIGDIR=/tmp/matplotlib python external_code_comparison.py            # ~4 min
MPLCONFIGDIR=/tmp/matplotlib python external_code_comparison.py --replot   # redraw only
MPLCONFIGDIR=/tmp/matplotlib python performance_study.py                   # ~4 min
MPLCONFIGDIR=/tmp/matplotlib python performance_study.py --replot          # redraw only
```

The "before" curves in `where_time_goes.png` come from
`performance/summary_before_vectorization.json`, measured at commit 45dd494 in a
slower session. `boundary_closures.png` measures both modes in one session:
refactorizing is forced by disabling the factor cache, and full Newton by
`radiative_jacobian="newton"`. Timings are single runs, except that kernel timings take
the best of 3–5. Treat differences under about 20% as noise.
