"""Vectorized assembly of the cell-centred diffusion operators.

The legacy assembly in :class:`heat_solver.polygonal.PolygonalHeatSolver` loops
over faces in Python and accumulates into a LIL matrix, which dominated run
time (15 s for 262k cells; see ``docs/performance_guide.md``).  The functions
here build the same matrices from per-face arrays:

* the per-entry sums are accumulated in the legacy order (faces in
  ``edge_to_cells`` order; within a face, the two-point terms before the
  correction terms) with ``np.bincount``, which adds sequentially, and entries
  that sum to exactly zero are dropped (LIL does not store them), so the
  sparsity pattern is the legacy one;
* the remaining difference is the last bit of some 2-vector dot products: the
  legacy code calls BLAS on length-2 vectors, which may use fused multiply-adds
  depending on the CPU, while the vectorized arithmetic rounds each product.
  On Cartesian meshes the arithmetic is exact and the matrices are identical;
  in general they agree to a few units in the last place.

Face geometry and the flattened gradient stencils depend only on the mesh and
are cached in the mesh's derived-geometry ``extras`` (shared by every solver on
the same mesh).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


def interior_faces(vertices, cell_centers, edge_to_cells):
    """Per-face arrays for the interior faces, in ``edge_to_cells`` order.

    Returns a dict with ``i < j`` (cells), ``normal`` (unit, pointing from ``i``
    to ``j``), ``dn`` (normal distance between the centres, clipped as in the
    legacy assembly), ``length``, ``midpoint`` and ``d`` (``c_j - c_i``).
    Zero-length faces are dropped.
    """
    pairs = [(edge, cells) for edge, cells in edge_to_cells.items() if len(cells) == 2]
    if not pairs:
        empty = np.zeros(0)
        return {"i": np.zeros(0, np.intp), "j": np.zeros(0, np.intp), "normal": np.zeros((0, 2)),
                "dn": empty, "length": empty, "midpoint": np.zeros((0, 2)), "d": np.zeros((0, 2))}
    edges = np.array([e for e, _ in pairs], dtype=np.intp)
    cells = np.array([c for _, c in pairs], dtype=np.intp)
    ci = np.minimum(cells[:, 0], cells[:, 1])
    cj = np.maximum(cells[:, 0], cells[:, 1])
    v0 = vertices[edges[:, 0]]
    v1 = vertices[edges[:, 1]]
    ev = v1 - v0
    length = np.sqrt(ev[:, 0] * ev[:, 0] + ev[:, 1] * ev[:, 1])
    keep = length != 0
    ci, cj, v0, v1, ev, length = ci[keep], cj[keep], v0[keep], v1[keep], ev[keep], length[keep]
    tangent = ev / length[:, None]
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    d = cell_centers[cj] - cell_centers[ci]
    dn = d[:, 0] * normal[:, 0] + d[:, 1] * normal[:, 1]
    flip = dn < 0
    normal[flip] = -normal[flip]
    dn[flip] = -dn[flip]
    small = dn <= 1e-12
    if np.any(small):
        dn[small] = np.maximum(np.sqrt(d[small, 0] ** 2 + d[small, 1] ** 2), 1e-12)
    out = {"i": ci, "j": cj, "normal": normal, "dn": dn, "length": length, "midpoint": 0.5 * (v0 + v1), "d": d}
    for arr in out.values():
        arr.setflags(write=False)
    return out


def flatten_stencils(gradient_coeffs):
    """``(ptr, idx, vec)``: CSR-style form of the per-cell gradient stencils (dict order kept)."""
    sizes = np.fromiter((len(c) for c in gradient_coeffs), dtype=np.intp, count=len(gradient_coeffs))
    ptr = np.concatenate([[0], np.cumsum(sizes)]).astype(np.intp)
    idx = np.fromiter((k for c in gradient_coeffs for k in c), dtype=np.intp, count=int(ptr[-1]))
    vec = np.array([np.asarray(v, dtype=float) for c in gradient_coeffs for v in c.values()], dtype=float)
    vec = vec.reshape(-1, 2)
    for arr in (ptr, idx, vec):
        arr.setflags(write=False)
    return ptr, idx, vec


def _stencil_entries(cells, ptr):
    """For each listed cell, its stencil entry positions; returns ``(owner, position)``."""
    sizes = ptr[cells + 1] - ptr[cells]
    owner = np.repeat(np.arange(cells.size), sizes)
    start = np.repeat(ptr[cells], sizes)
    offset = np.arange(owner.size) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    return owner, start + offset


def accumulate(rows, cols, vals, n):
    """CSR matrix whose entries are the in-order sums of the contributions (exact zeros dropped)."""
    if rows.size == 0:
        return csr_matrix((n, n))
    keys = rows.astype(np.int64) * n + cols.astype(np.int64)
    uniq, inverse = np.unique(keys, return_inverse=True)
    sums = np.bincount(inverse, weights=vals, minlength=uniq.size)
    nz = sums != 0.0
    uniq, sums = uniq[nz], sums[nz]
    r = uniq // n
    c = (uniq % n).astype(np.intp)
    indptr = np.concatenate([[0], np.cumsum(np.bincount(r, minlength=n))]).astype(np.intp)
    return csr_matrix((sums, c, indptr), shape=(n, n))


def _matvec(K, v):
    """Batched ``K @ v`` for ``K`` (m, 2, 2) and ``v`` (m, 2)."""
    return np.column_stack([K[:, 0, 0] * v[:, 0] + K[:, 0, 1] * v[:, 1],
                            K[:, 1, 0] * v[:, 0] + K[:, 1, 1] * v[:, 1]])


def _dot(a, b):
    return a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]


def assemble_tpfa(faces, K, n_cells, stencils=None):
    """Two-point flux operator, plus the explicit non-orthogonal correction when ``stencils`` is given.

    ``K`` is the face conductivity tensor, shape ``(n_faces, 2, 2)``.
    """
    i, j, n, dn, L, d = faces["i"], faces["j"], faces["normal"], faces["dn"], faces["length"], faces["d"]
    m = i.size
    Kn = _matvec(K, n)
    alpha_n = _dot(n, Kn)
    base = alpha_n * L / dn
    face = np.arange(m)
    rows = [i, i, j, j]
    cols = [i, j, i, j]
    vals = [base, -base, -base, base]
    owner_face = [face] * 4
    rank = [np.zeros(m, np.intp), np.ones(m, np.intp), np.full(m, 2, np.intp), np.full(m, 3, np.intp)]
    if stencils is not None and m:
        ptr, sidx, svec = stencils
        corr = Kn - (alpha_n[:, None] * d) / dn[:, None]
        active = np.flatnonzero(np.sqrt(corr[:, 0] ** 2 + corr[:, 1] ** 2) > 1e-14)
        if active.size:
            # Stencil entries of cell i, then of cell j, for every active face (i first: the
            # legacy code accumulates the per-face coefficient over (i, j) in that order).
            oi, pi = _stencil_entries(i[active], ptr)
            oj, pj = _stencil_entries(j[active], ptr)
            owner = np.concatenate([oi, oj])
            pos = np.concatenate([pi, pj])
            fidx = active[owner]
            part = 0.5 * _dot(corr[fidx], svec[pos])
            group_key = fidx.astype(np.int64) * n_cells + sidx[pos]
            groups, inverse = np.unique(group_key, return_inverse=True)
            coeff = np.bincount(inverse, weights=part, minlength=groups.size)
            gface = (groups // n_cells).astype(np.intp)
            gidx = (groups % n_cells).astype(np.intp)
            scaled = L[gface] * coeff
            rows += [i[gface], j[gface]]
            cols += [gidx, gidx]
            vals += [-scaled, scaled]
            owner_face += [gface, gface]
            rank += [np.full(gface.size, 4, np.intp), np.full(gface.size, 5, np.intp)]
    rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    order = np.lexsort((np.concatenate(rank), np.concatenate(owner_face)))
    return accumulate(rows[order], cols[order], vals[order], n_cells)


def assemble_reconstructed(faces, K, n_cells, stencils):
    """Flux ``-|f| n.K (grad_i + grad_j)/2`` with least-squares cell gradients.

    Faces whose reconstructed flux vanishes identically fall back to the
    two-point flux, as in the legacy assembly.
    """
    i, j, n, dn, L = faces["i"], faces["j"], faces["normal"], faces["dn"], faces["length"]
    m = i.size
    ptr, sidx, svec = stencils
    face = np.arange(m)
    oi, pi = _stencil_entries(i, ptr)
    oj, pj = _stencil_entries(j, ptr)
    owner = np.concatenate([oi, oj])
    pos = np.concatenate([pi, pj])
    key = owner.astype(np.int64) * n_cells + sidx[pos]
    groups, inverse = np.unique(key, return_inverse=True)
    wsum = np.column_stack([np.bincount(inverse, weights=svec[pos, 0], minlength=groups.size),
                            np.bincount(inverse, weights=svec[pos, 1], minlength=groups.size)])
    gface = (groups // n_cells).astype(np.intp)
    gidx = (groups % n_cells).astype(np.intp)
    keep = np.sqrt(wsum[:, 0] ** 2 + wsum[:, 1] ** 2) > 1e-30
    gface, gidx, wsum = gface[keep], gidx[keep], wsum[keep]
    tk = (-0.5 * L[gface]) * _dot(n[gface], _matvec(K[gface], wsum))
    keep = np.abs(tk) > 1e-30
    gface, gidx, tk = gface[keep], gidx[keep], tk[keep]
    any_flux = np.zeros(m, dtype=bool)
    any_flux[gface] = True
    fb = face[~any_flux]
    base = _dot(n[fb], _matvec(K[fb], n[fb])) * L[fb] / dn[fb]
    rows = np.concatenate([i[gface], j[gface], i[fb], i[fb], j[fb], j[fb]])
    cols = np.concatenate([gidx, gidx, i[fb], j[fb], i[fb], j[fb]])
    vals = np.concatenate([tk, -tk, base, -base, -base, base])
    owners = np.concatenate([gface, gface, fb, fb, fb, fb])
    order = np.argsort(owners, kind="stable")
    return accumulate(rows[order], cols[order], vals[order], n_cells)


def identity_rows(K, mask):
    """Copy of CSR ``K`` with the rows in ``mask`` replaced by identity rows.

    Same values and sparsity pattern as replacing the rows of a LIL copy
    (``lil.rows[i] = [i]; lil.data[i] = [1.0]``), without the Python loop.
    """
    K = K.tocsr()
    if not K.has_sorted_indices:
        K = K.sorted_indices()
    n = K.shape[0]
    mask = np.asarray(mask, dtype=bool)
    counts = np.diff(K.indptr)
    row_of = np.repeat(np.arange(n), counts)
    keep = ~mask[row_of]
    new_counts = np.where(mask, 1, counts)
    indptr = np.concatenate([[0], np.cumsum(new_counts)]).astype(K.indptr.dtype)
    indices = np.empty(indptr[-1], dtype=K.indices.dtype)
    data = np.empty(indptr[-1], dtype=K.data.dtype)
    # Kept rows: copy their entries at the same offset within the row; masked rows: a single diagonal 1.
    offset = np.arange(K.nnz) - K.indptr[row_of]
    dest = indptr[row_of[keep]] + offset[keep]
    indices[dest] = K.indices[keep]
    data[dest] = K.data[keep]
    rows_m = np.flatnonzero(mask)
    indices[indptr[rows_m]] = rows_m
    data[indptr[rows_m]] = 1.0
    return csr_matrix((data, indices, indptr), shape=K.shape)
