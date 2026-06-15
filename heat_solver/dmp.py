"""Discrete-maximum-principle (DMP) diagnostics and a monotone scheme.

Cell-centered finite-volume diffusion can violate the discrete maximum principle
(produce over/undershoots beyond physical bounds) on skewed meshes and under
strong anisotropy.  This module provides:

* **Diagnostics** -- ``m_matrix_metrics`` quantifies the algebraic root cause
  (positive off-diagonal entries of the diffusion matrix; an M-matrix has none),
  and ``bound_excursion`` measures solution over/undershoot beyond given bounds.

* **A monotone scheme** -- ``make_monotone`` applies a symmetric *M-matrix
  projection* (a.k.a. discrete upwinding / the low-order operator of algebraic
  flux correction): it adds the minimal symmetric artificial diffusion that
  removes every positive off-diagonal, while preserving symmetry and zero row
  sums.  The result is provably an M-matrix (hence DMP-preserving) and remains
  conservative and exact for constant fields, at the cost of extra numerical
  diffusion where it activates.

Background: the base two-point flux (``flux_scheme='tpfa'`` with
``nonorthogonal_correction=False``) already yields an M-matrix for any SPD
diffusivity, because the conormal transmissibility ``n^T K n / d`` is
non-negative.  The non-orthogonal correction and the reconstructed-gradient flux
restore consistency on skewed/anisotropic meshes but introduce positive
off-diagonals -- the accuracy-vs-monotonicity trade-off this module quantifies.

References: Le Potier (2005); Lipnikov, Svyatskiy & Vassilevski (2009);
Kuzmin (algebraic flux correction); Nordbotten & Aavatsmark (MPFA monotonicity).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, diags


def m_matrix_metrics(A):
    """Quantify M-matrix violations of an assembled (diffusion) matrix.

    Returns a dict with the number and magnitude of positive off-diagonal
    entries (an M-matrix has none), normalized by the typical diagonal scale.
    """
    coo = A.tocoo()
    offdiag = coo.row != coo.col
    od_vals = coo.data[offdiag]
    pos = od_vals[od_vals > 0.0]
    diag = np.abs(A.diagonal())
    typ = float(np.median(diag[diag > 0])) if np.any(diag > 0) else 1.0
    return {
        "num_offdiag": int(od_vals.size),
        "num_positive_offdiag": int(pos.size),
        "max_positive_offdiag": float(pos.max()) if pos.size else 0.0,
        "violation_ratio": float(pos.max() / typ) if pos.size else 0.0,
        "is_m_matrix": bool(pos.size == 0),
    }


def bound_excursion(u, lo=0.0, hi=1.0, tol=0.0):
    """Over/undershoot of ``u`` beyond ``[lo, hi]`` (0 if within bounds)."""
    u = np.asarray(u, dtype=float)
    return {
        "min": float(u.min()),
        "max": float(u.max()),
        "undershoot": float(max(0.0, lo - u.min() - tol)),
        "overshoot": float(max(0.0, u.max() - hi - tol)),
    }


def make_monotone(A):
    """Symmetric M-matrix projection of a zero-row-sum diffusion matrix.

    Adds the minimal symmetric artificial diffusion ``D`` (with ``d_ij =
    max(0, a_ij, a_ji)`` off-diagonal and ``d_ii = -sum_j d_ij``) so that
    ``A_mono = A - D`` has non-positive off-diagonals.  Preserves zero row sums
    (hence conservation and exactness for constants), and symmetry when ``A`` is
    symmetric; becomes more diffusive only on the entries it repairs.  A matrix
    that is already an M-matrix (e.g. base TPFA) is returned unchanged.
    """
    coo = A.tocoo()
    offdiag = {}
    for i, j, v in zip(coo.row, coo.col, coo.data):
        if i != j and v != 0.0:
            offdiag[(int(i), int(j))] = offdiag.get((int(i), int(j)), 0.0) + float(v)

    pairs = {(min(i, j), max(i, j)) for (i, j) in offdiag}
    rows, cols, data = [], [], []
    diag_add = np.zeros(A.shape[0])
    for (i, j) in pairs:
        a_ij = offdiag.get((i, j), 0.0)
        a_ji = offdiag.get((j, i), 0.0)
        d = max(0.0, a_ij, a_ji)
        if d > 0.0:
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([-d, -d])  # remove the positive off-diagonal coupling
            diag_add[i] += d
            diag_add[j] += d

    if not rows:
        return A.tocsr()
    correction = coo_matrix((data, (rows, cols)), shape=A.shape)
    return (A + correction + diags(diag_add)).tocsr()


def anisotropic_tensor(ratio=1.0, angle=0.0, kappa=0.1):
    """SPD 2x2 diffusivity with principal values ``kappa`` and ``kappa/ratio``,
    rotated by ``angle`` (radians).  ``ratio=1`` is isotropic."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    D = np.diag([kappa, kappa / ratio])
    return R @ D @ R.T


# Flux-scheme variants probed by the DMP study: label -> solver kwargs.
DMP_SCHEMES = {
    "tpfa_base": {"flux_scheme": "tpfa", "flux_discretization": "tpfa", "nonorthogonal_correction": False},
    "tpfa_corrected": {"flux_scheme": "tpfa", "flux_discretization": "tpfa", "nonorthogonal_correction": True},
    "reconstructed": {"flux_scheme": "tpfa", "flux_discretization": "reconstructed", "nonorthogonal_correction": True},
    "reconstructed_monotone": {"flux_scheme": "tpfa", "flux_discretization": "reconstructed", "nonorthogonal_correction": True, "monotone": True},
}


def _block_initial_condition(centers, block):
    x, y = centers[:, 0], centers[:, 1]
    inside = (x >= block[0]) & (x <= block[1]) & (y >= block[2]) & (y <= block[3])
    return inside.astype(float)


def run_dmp_case(vertices, polygons, centers, alpha, scheme_kwargs, *, dt, t_end,
                 block=(-0.35, 0.35, -0.35, 0.35)):
    """Diffuse a bounded indicator block (values in [0, 1]) with zero Dirichlet
    data and report the M-matrix metrics of the operator and the solution's
    over/undershoot beyond [0, 1].

    The continuous solution of pure diffusion of data in [0, 1] with boundary in
    [0, 1] stays in [0, 1] for any SPD diffusivity, so any excursion is a purely
    numerical DMP violation.
    """
    from .polygonal import PolygonalHeatSolver

    solver = PolygonalHeatSolver(
        vertices, polygons, alpha, dt, bc_type="dirichlet",
        bc_func=lambda x, y, t: np.zeros_like(np.asarray(x, dtype=float)),
        source_func=lambda x, y, t: np.zeros_like(np.asarray(x, dtype=float)),
        **scheme_kwargs,
    )
    solver._assemble_system()
    metrics = m_matrix_metrics(solver.A)
    u0 = _block_initial_condition(centers, block)
    _, u = solver.solve(u0, 0.0, t_end)
    return {"m_matrix": metrics, "bounds": bound_excursion(u, 0.0, 1.0)}


def run_dmp_study(mesh_builders, anisotropy, schemes=None, *, dt=1.5e-3, t_end=9e-3):
    """Sweep meshes x anisotropy x flux schemes; return a list of records.

    Parameters
    ----------
    mesh_builders : dict ``name -> callable() -> (vertices, polygons, centers)``.
    anisotropy : dict ``label -> 2x2 diffusivity`` (e.g. from ``anisotropic_tensor``).
    schemes : dict of scheme kwargs (defaults to ``DMP_SCHEMES``).
    """
    schemes = schemes if schemes is not None else DMP_SCHEMES
    records = []
    for mesh_name, builder in mesh_builders.items():
        vertices, polygons, centers = builder()
        for aniso_label, alpha in anisotropy.items():
            for scheme_label, kwargs in schemes.items():
                res = run_dmp_case(vertices, polygons, centers, alpha, kwargs, dt=dt, t_end=t_end)
                records.append({
                    "mesh": mesh_name,
                    "anisotropy": aniso_label,
                    "scheme": scheme_label,
                    "is_m_matrix": res["m_matrix"]["is_m_matrix"],
                    "num_positive_offdiag": res["m_matrix"]["num_positive_offdiag"],
                    "overshoot": res["bounds"]["overshoot"],
                    "undershoot": res["bounds"]["undershoot"],
                    "min": res["bounds"]["min"],
                    "max": res["bounds"]["max"],
                })
    return records
