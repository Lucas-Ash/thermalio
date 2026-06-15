# Thermalio research backlog

Researched directions for adapting Thermalio toward novel, impactful work. Each
note follows `TEMPLATE.md`. These are planning documents only — they do not
affect any test, sweep, or regression run.

| ID | Direction | Status | Note |
|----|-----------|--------|------|
| A | Open V&V benchmark suite for non-classical heat transport | **landed** | [direction_A_vv_benchmarks.md](direction_A_vv_benchmarks.md) |
| B | Monotone / bound-preserving polygonal & MPFA FV for non-Fourier / fractional / anisotropic models | idea | [direction_B_monotone_polygonal_fv.md](direction_B_monotone_polygonal_fv.md) |
| C | 2D non-Fourier / fractional phase change (hyperbolic & fractional Stefan) | idea | [direction_C_nonfourier_phase_change.md](direction_C_nonfourier_phase_change.md) |
| D | Inverse-problem / parameter-identification testbed (bioheat, non-Fourier) | idea | [direction_D_inverse_problems.md](direction_D_inverse_problems.md) |

## Why these

Thermalio is unusual as a research platform: it unifies a broad set of physics
(classical, anisotropic, temperature-dependent, phase change, radiation, Cattaneo
hyperbolic, advection, fractional subdiffusion, Pennes bioheat, functionally
graded) behind one verification-first finite-volume framework with
manufactured-solution convergence sweeps across multiple mesh paradigms. The
directions above each exploit that breadth + rigor against a gap confirmed in the
current literature. Direction A is in progress because it is the highest-leverage,
lowest-risk starting point and strengthens the foundation the others build on.
