# Cross-code comparison: Thermalio against seven independent codes

`external_code_comparison_extended.py` solves the same three manufactured problems
with Thermalio and with every independent heat-equation code that installs here
with pip (versions pinned in `requirements-external.txt`):

| code | method | what it isolates |
|---|---|---|
| FiPy 4.0.3 (NIST) | cell-centred FV, structured grids | a mature FV code in pure Python |
| PorePy 1.13 | cell-centred FV, TPFA and MPFA-O, general polygonal grids | the same flux families as Thermalio, on the *same* skewed meshes |
| py-pde 0.60.0 | finite differences, compiled with Numba | a just-in-time compiled stencil code |
| scikit-fem 12.0.2 | P1 / P2 finite elements, assembled in Python | finite elements without compiled kernels |
| NGSolve 6.2 | P1 / P2 / P3 finite elements, C++ core, task-parallel | a compiled, multithreaded production FE library |

Devito (generated-C finite differences) was also tried. Its current release requires
NumPy ≤ 2.4.3, which conflicts with the pinned NumPy 2.5.3, so it was left out.
FEniCSx, Firedrake and deal.II have no pip wheels.

All codes use backward Euler with the same step (Δt = 2.5e-4, 200 steps to t = 0.05),
so differences between codes isolate the spatial discretization. Thermalio uses its
recommended configuration: two-point flux with non-orthogonal correction, and on skewed
meshes also the quadratic flux. Everything runs on one core, except the NGSolve
"4 threads" series.

Figures: `test_plots/benchmark/external_codes_extended/cross_code_accuracy.png` and
`cross_code_cost.png`. Data: `summary.json`, `convergence.csv` in the same directory.
Tests: `tests/test_external_codes_more.py` (CI job `external-validation`).

## Accuracy (`cross_code_accuracy.png`)

Relative L² error at n = 64 (4096 cells, or about 4–37k FE unknowns):

| code | isotropic | anisotropic | Pennes + Robin |
|---|---|---|---|
| Thermalio FV (band) | 8.33e-5 | **1.52e-5** | 2.54e-5 |
| Thermalio FV, face Dirichlet | 8.34e-5 | 1.59e-5 | – |
| FiPy | 8.34e-5 | 5.25e-4 | – |
| PorePy TPFA | 8.34e-5 | 3.40e-3 (no convergence) | – |
| PorePy MPFA-O | 8.34e-5 | 2.17e-4 | – |
| py-pde FD | 8.34e-5 | 1.56e-5 | 2.54e-5 |
| scikit-fem P1 / P2 | 2.0e-3 / 1.5e-5 | 1.9e-3 / 1.5e-5 | 1.0e-4 / 5.5e-6 |
| NGSolve P1 / P2 / P3 | 9.4e-4 / 1.9e-5 / 6.1e-6 | 8.3e-4 / 1.9e-5 / 6.4e-6 | 5.8e-5 / 5.5e-6 / 5.4e-6 |

How to read it:

- **(a) Isotropic.** All four finite-volume codes and py-pde give the same answer:
  8.34e-5. That is expected, because on a Cartesian grid they are all the same
  five-point scheme with the data on the boundary faces. Panel (e) confirms it to
  round-off: Thermalio (face mode) differs from FiPy by 1.3e-14, from PorePy by 4e-15
  and from py-pde by 1e-14. These are three independent implementations of
  Thermalio's discretization, a stronger check than any single comparison.
- **(b) Anisotropic tensor.** Here the codes separate.
  - Thermalio (1.52e-5) is as accurate as the P2 finite elements (1.5e-5 to 1.9e-5)
    with a quarter of their unknowns.
  - py-pde's central differences match it on this Cartesian grid.
  - FiPy's tensor diffusion is 35x less accurate (order 0.9). PorePy's TPFA is
    inconsistent for a rotated tensor (the error does not decrease). Its MPFA-O is
    14x less accurate than Thermalio.
- **(c) Pennes + Robin.** Thermalio and py-pde agree to 4e-13. The same face
  elimination of the Robin condition was reached independently.
- **(d) Identical skewed meshes.** PorePy runs on exactly Thermalio's skewed cells
  (the node coordinates are copied). On the anisotropic problem at n = 64:
  - Thermalio TPFA + correction: 1.52e-5;
  - Thermalio quadratic flux: 1.45e-5;
  - PorePy MPFA-O: 1.89e-4 (12x larger, order 1.0 over this window);
  - PorePy TPFA: 1.85e-3, and does not converge.
  PorePy's MPFA is second order at steady state on the same problem (checked
  separately: orders 2.00–2.01). Its first-order behaviour here is specific to this
  short transient: the scheme's O(h) local truncation error has not yet averaged out
  ("supraconvergence" needs the solution to settle).
- **(f) Accuracy per unknown.** High-order finite elements win per unknown.
  NGSolve P3 reaches 6e-6 with 37k unknowns, but that is the backward-Euler time
  error, which caps P2 and P3 in panels (a)–(c) (the dash-dotted floor line).
  Second-order FV sits between P1 and P2.

## Cost (`cross_code_cost.png`)

Every run is timed cold, the way a user meets it: a fresh process state, no cached
geometry or factorization. Two warm runs (one step, and all steps) then split that
time into a one-off part (mesh, assembly, factorization, JIT compilation) and a
per-step part.

Time to reach 1e-4 relative error (log-log interpolation along the refinement path):

| code | isotropic | anisotropic | Pennes + Robin |
|---|---|---|---|
| Thermalio FV (band) | 0.15 s | **0.025 s** | 0.093 s |
| Thermalio FV, face Dirichlet | 0.099 s | 0.031 s | – |
| PorePy TPFA | **0.091 s** | not reached | – |
| PorePy MPFA-O | 0.15 s | not reached | – |
| NGSolve P3 (1 thread) | 0.15 s | 0.18 s | **≤ 0.067 s** |
| NGSolve P2, 4 threads | 0.33 s | 0.38 s | 0.071 s |
| NGSolve P2 (1 thread) | 0.74 s | 0.83 s | 0.10 s |
| FiPy | 4.2 s | not reached | – |
| scikit-fem P2 | 4.9 s | 4.8 s | 0.48 s |
| py-pde | 20 s | 34 s | 21 s |

What the panels show:

- **(d) Where the time goes.** The codes differ most in fixed costs, not stepping.
  - py-pde spends 20–35 s compiling its Numba kernels for every new equation, then
    steps about as fast as Thermalio (0.6 ms per step at 4096 cells).
  - FiPy spends 23 ms per step rebuilding its matrix from its term objects.
  - scikit-fem re-assembles the load vector in Python every step (15 ms for P1,
    146 ms for P2).
  - Thermalio, PorePy and NGSolve factorize once and reuse the factorization.
- **(e) Per-step cost against size.** At 65k cells (256²):

  | code | per step |
  |---|---|
  | Thermalio, face Dirichlet | 12.5 ms (isolated re-timing: 8.4–9.9 ms) |
  | PorePy TPFA | 12.1 ms (isolated: 9.7–10.4 ms) |
  | Thermalio, band Dirichlet | 24 ms in the study run (isolated: 13–15 ms) |
  | NGSolve P1 | 105 ms; 28 ms on 4 threads (3.7x) |
  | NGSolve P2 | 234 ms; 77 ms on 4 threads (3.0x) |
  | scikit-fem P1 | 728 ms |

  The isolated values are three repeats of each code back to back in one process,
  which removes most of the run-to-run noise of the long study run.
  - **Why NGSolve gains from threads and the FV codes would not.** NGSolve's
    per-step cost is dominated by re-assembling the source and projecting the
    boundary data every step, which parallelizes (3x on 4 cores). The FV codes do
    only one back-substitution and a vector update per step. Their stepping is
    already close to the cost of the triangular solves, which SuperLU runs on one
    thread.
  - **Band against face.** Band Dirichlet costs 1.3–1.5x more per step than face
    Dirichlet. Its identity rows make the matrix unsymmetric, and the resulting LU
    factors are slower to apply.
  - **py-pde fails at 65k cells with this Δt.** Its "implicit" Euler is a
    fixed-point iteration that needs Δt·α/h² well below 1 (0.4 here). Its explicit
    Euler is unstable there too.
- **(f) Time to accuracy.** For a single solve, Thermalio is the fastest on the
  anisotropic problem, 7x ahead of NGSolve P3, the next fastest. On the isotropic
  problem, PorePy TPFA (0.091 s) and Thermalio in face mode (0.099 s) are tied. On
  the smooth Robin problem, NGSolve's P2/P3 elements reach 1e-4 on coarse meshes and
  are as fast as Thermalio.

## What this says about Thermalio

- **Correctness.** Its five-point scheme is reproduced to round-off by three
  independent codes: FiPy, PorePy and py-pde (differences 4e-15 to 1.3e-14). Its
  Robin elimination is reproduced by py-pde (4e-13).
- **Accuracy where it matters.** On anisotropic and skewed problems it is the most
  accurate finite-volume code tested: 12–35x better than PorePy MPFA-O and FiPy.
  It is as accurate as P2 finite elements with a quarter of their unknowns.
- **Speed.** Per step, face mode matches the leanest FV loop (PorePy TPFA), and band
  mode is 1.3–1.5x behind it. On one core, Thermalio is 8x faster than NGSolve P1
  and 60x faster than scikit-fem P1, because it reuses one factorization and
  assembles nothing per step.
- **Remaining gaps.**
  - High-order elements reach a given accuracy with fewer unknowns on smooth
    problems.
  - NGSolve gets about 3x from 4 threads; Thermalio has no threaded stepping (its
    SuperLU solves are single-threaded).
  - Backward Euler's time error (about 6e-6 here) is the floor for every code.
    BDF2, Thermalio's driver default, removes it (see `docs/performance_guide.md`).

**Found and fixed by this comparison.** Face-mode Dirichlet refactorized whenever the
step size differed in its last bits (Δt comes from differences of accumulated times).
In a 40-step run at 65k cells it factorized 3 times instead of once, at 0.28 s each.
The face closure now keys its factorization on the rounded Δt, as the band path does,
so it factorizes once per solve, saving about 0.56 s per solve at 65k cells. Results
change only at round-off: the FiPy agreement is unchanged at 1e-14.

## Reproduce

```bash
sudo apt-get install -y libglu1-mesa        # PorePy's gmsh dependency
pip install -r requirements-external.txt
MPLCONFIGDIR=/tmp/matplotlib python external_code_comparison_extended.py            # ~25 min
MPLCONFIGDIR=/tmp/matplotlib python external_code_comparison_extended.py --replot   # redraw only
```

Timings vary by about ±30% between sessions on this VM (see `docs/performance_guide.md`);
the ratios measured within one run are the reliable numbers.
