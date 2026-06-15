# Direction B: Monotone / bound-preserving polygonal & MPFA FV for non-classical models

- **Status:** landed (PR1 diagnostic, PR2 nonlinear AFC, PR3 smoothness-relaxed limiter, PR4 global maximum-principle limiter)
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
- **Delivered (PR2):** a **nonlinear** bound-preserving high-resolution scheme via
  **algebraic flux correction / FCT** (`heat_solver/afc.py`, `AFCMonotoneSolver`).
  It limits the anti-diffusive flux between the high-order operator ``A_H`` and
  the low-order M-matrix operator ``A_L = make_monotone(A_H)`` with a node-based
  Zalesak limiter (Picard-iterated on the RHS, so the implicit operator stays the
  monotone M-matrix). Chosen over geometric NTPFA because it reuses the verified
  `make_monotone` and is purely algebraic (lower bug surface). Findings:
    - **bound-preserving**: zero over/undershoot on the steep anisotropic front
      where the high-order scheme overshoots ~0.04–0.06;
    - **accuracy**: ~2–3x lower error than the linear M-matrix projection on a
      smooth anisotropic solution, approaching the high-order scheme;
    - **limitation (addressed by PR3)**: the basic Zalesak limiter clips smooth
      extrema, capping accuracy on smooth solutions.
- **Delivered (PR3):** a **smoothness-relaxed / linearity-preserving** limiter
  (`AFCMonotoneSolver(..., smoothness_factor>0)`) using a Venkatakrishnan-style
  mesh-vanishing tolerance ``eps ~ factor * U_ref * h^1.5`` that lets small
  (smooth) anti-diffusive fluxes pass unlimited while O(1) front jumps stay
  strictly limited. Findings: it **fully recovers the high-order scheme's
  accuracy** on the smooth anisotropic case (error drops from the clipped strict
  value to exactly the reconstructed value), and the steep-front overshoot is
  **essentially non-oscillatory** — it vanishes monotonically under refinement
  (faster than the slope-1.5 reference). ``smoothness_factor=0`` recovers PR2's
  strictly bound-preserving limiter.
- **Delivered (PR4):** a **global maximum-principle (MPP) limiter**
  (`AFCMonotoneSolver(..., limiter="global")`, Zhang--Shu style). It enforces the
  physical bounds ``[m, M]`` (auto-detected from initial + boundary data, or
  given) for every node instead of local neighbour bounds. Because a smooth
  interior extremum lies inside ``[m, M]`` it is never clipped, so PR4 achieves
  **both** goals at once: strict bound preservation even at coarse resolution
  (zero excursion where PR3's relaxed limiter overshoots ~0.04) **and** full
  high-order accuracy on smooth solutions (error matches the reconstructed
  scheme; order 0.89 vs the strict local limiter's 0.07). Rigorous when the
  continuous problem obeys a maximum principle (source-free or sign-definite
  forcing). PR4 strictly dominates PR2 (bounds, but clips smooth) and PR3
  (accurate, but coarse overshoot).
- **Open (future):** MPP for problems without an a-priori maximum principle
  (general sign-changing sources); geometric NTPFA for local-extremum control
  within ``[m, M]``.
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
- PR2 (done): nonlinear bound-preserving high-resolution scheme via algebraic
  flux correction (`AFCMonotoneSolver`) + demonstrative graphs
  (`dmp_afc_demo.py` -> `test_plots/dmp/monotonicity_showcase.png`).
- PR3 (done): smoothness-relaxed (linearity-preserving) limiter
  (`smoothness_factor`) + graph (`test_plots/dmp/smoothness_relaxation.png`).
- PR4 (done): global maximum-principle limiter (`limiter="global"`) -- strict
  bounds AND high-order accuracy + graph (`test_plots/dmp/mpp_limiter.png`).

## References
- Le Potier; Lipnikov, Svyatskiy, Vassilevski (monotone/nonlinear FV).
- Nordbotten & Aavatsmark (MPFA monotonicity).
