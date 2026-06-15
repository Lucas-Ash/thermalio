# Direction B: Monotone / bound-preserving polygonal & MPFA FV for non-classical models

- **Status:** in-progress (PR1 diagnostic + linear monotone projection landed; PR2 NTPFA next)
- **Owner:** —
- **Last updated:** 2026-06-15

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
- **Delivered (PR1):** `heat_solver/dmp.py` — DMP diagnostics (`m_matrix_metrics`,
  `bound_excursion`), a conservative **linear monotone scheme** (`make_monotone`,
  a symmetric M-matrix projection), the `monotone=True` option on
  `PolygonalHeatSolver`, an anisotropy×skew sweep harness (`run_dmp_study`), and
  the `dmp_study.py` script. Findings below.
- **Next (PR2):** a **nonlinear** monotone scheme (NTPFA, Le Potier /
  Lipnikov–Svyatskiy–Vassilevski) that is *both* consistent and DMP-preserving on
  K-non-orthogonal meshes — needed because the linear projection (PR1) restores
  the DMP but reverts to the inconsistent two-point behavior on skewed anisotropic
  meshes (see findings).
- **Out of scope (later):** extending MPFA beyond Dirichlet/classical (MPFA is
  Dirichlet-favored and goes singular on the mixed tiled mesh — direction A
  N-version finding).

## Findings (PR1)
From `dmp_study.py` on a bounded indicator block (values in [0, 1], zero
Dirichlet data):
- **Base TPFA** (`nonorthogonal_correction=False`) is an M-matrix for any SPD
  diffusivity → **zero** over/undershoot, but is ~O(1) **inconsistent** on
  K-non-orthogonal (skewed/anisotropic) meshes.
- **Nonorthogonal-corrected and reconstructed** fluxes are consistent/accurate
  but introduce positive off-diagonals → DMP violations growing with skew and
  anisotropy ratio (up to ~0.11 excursion on the skewed tiled mesh).
- **`make_monotone` (M-matrix projection)** restores the M-matrix (zero
  excursion) and preserves conservation/zero-row-sums, but on severely skewed
  anisotropic meshes adds enough artificial diffusion to revert to first-order /
  inconsistent behavior — the classic linear-monotone-scheme limitation that
  motivates PR2.

## Design sketch
- Landed: `heat_solver/dmp.py`; `monotone` flag in `heat_solver/polygonal.py`
  (`_apply_monotone` applied to each assembled diffusion matrix).
- PR2 (planned): `heat_solver/ntpfa.py` — nonlinear two-point flux with auxiliary
  harmonic/vertex values, conormal decomposition, nonnegative nonlinear weights,
  and a Picard iteration; reuse `_TransportBase`/solver assembly hooks.

## Reuse map
- `heat_solver/meshes.py` skewed/tiled generators; `process_alpha` tensor support;
  `heat_solver/verification.py` for observed order; `polygonal.py` flux assembly.

## Verification plan
- PR1 (done): `tests/test_dmp.py` — M-matrix metrics, monotone-projection
  properties (M-matrix, zero row sums, symmetry, no-op on M-matrices), base TPFA
  monotone under anisotropy, reconstructed violates DMP, monotone restores it,
  monotone is a no-op on orthogonal isotropic meshes.
- PR2: NTPFA should drive excursions to ~0 **and** retain ~2nd-order accuracy on
  skewed anisotropic meshes (tie to direction A's order-of-accuracy reporting).

## PR breakdown
- PR1 (done): DMP/overshoot diagnostic + linear monotone projection + sweep.
- PR2: nonlinear monotone FV (NTPFA) for consistency + DMP.

## References
- Le Potier; Lipnikov, Svyatskiy, Vassilevski (monotone/nonlinear FV).
- Nordbotten & Aavatsmark (MPFA monotonicity).
