# Direction B: Monotone / bound-preserving polygonal & MPFA FV for non-classical models

- **Status:** idea
- **Owner:** —
- **Last updated:** 2026-06-14

## Motivation & V&V question
MPFA / polygonal finite volume is mature for *classical* anisotropic diffusion
but rarely applied to non-Fourier (Cattaneo), fractional, or anisotropic-bioheat
transport — those typically use meshfree/BEM/FEM/1D-FD. Linear MPFA/MFD schemes
are known to lose monotonicity (discrete maximum principle, DMP) at high
anisotropy ratio and mesh skew. A **bound-preserving conservative FV** for the
transport models on skewed polygonal meshes, with a systematic study of where and
why the DMP is violated, is an open numerical-analysis contribution.

V&V question: *under what anisotropy ratio and mesh skew do the schemes preserve
discrete bounds / the maximum principle, and can a monotone variant restore them
without destroying accuracy?*

## Scope
- **In scope (candidate first increment):** a DMP/overshoot diagnostic that sweeps
  anisotropy ratio × skew across the transport models using the existing
  `hot_block` maximum-principle case and the skewed/tiled meshes; quantify
  violations (min/max excursions beyond initial bounds).
- **Out of scope (later):** a full nonlinear monotone FV scheme; extending MPFA
  beyond Dirichlet/classical (currently MPFA is Dirichlet-favored and was observed
  to go singular on the mixed tiled mesh — see direction A's N-version finding).

## Design sketch
- Likely new: `heat_solver/monotone.py` (nonlinear two-point/limited FV variant);
  a DMP-diagnostic harness reusing `nversion`-style sweeps.
- Reuse: `hot_block` case (already a maximum-principle probe), anisotropic tensor
  alpha, skewed/tiled mesh generators, `polygonal.py` flux assembly.

## Reuse map
- `heat_solver/meshes.py` skewed/tiled generators; `process_alpha` tensor support;
  `cases.get_analytical_case("hot_block")`; `nversion.run_nversion` pattern.

## Verification plan
- Anisotropy×skew sweep reporting overshoot/undershoot; a monotone variant should
  drive excursions to ~0 while retaining observed order on smooth cases (tie to
  direction A's order-of-accuracy reporting).

## Risks & open questions
- Monotone schemes are often only first-order; quantify the accuracy/robustness
  trade-off. MPFA robustness on mixed-polygon meshes is limited (singular cases).

## PR breakdown
- PR1: DMP/overshoot diagnostic + sweep on existing schemes.
- PR2+: nonlinear monotone FV variant; analysis.

## References
- Le Potier; Lipnikov, Svyatskiy, Vassilevski (monotone/nonlinear FV).
- Nordbotten & Aavatsmark (MPFA monotonicity).
