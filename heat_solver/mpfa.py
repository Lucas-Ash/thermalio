"""
MPFA-O style diffusion on polygonal meshes via piecewise-linear triangles (centroid, vertex, centroid).

Triangles connect consecutive cell centroids around each mesh vertex (CCW fan). A linear FEM
stiffness is assembled with tensor K(x), then interior vertex DOFs are Schur-complemented. With
homogeneous Dirichlet data on all mesh vertices (u = 0 on ∂Ω including corners), the remaining
cell-cell block S_cc is the condensed diffusion operator. This matches common manufactured cases
with zero boundary data (e.g. sine modes on a box). For general Dirichlet data, vertex values would
need an additional RHS coupling (not implemented here).

For Neumann or Robin BCs, ``PolygonalHeatSolver`` does not use this module for the diffusion block;
it falls back to the same two-point TPFA stencil plus ``_assemble_boundary_system`` so boundary
fluxes stay consistent with the rest of the solver.

TPFA in ``polygonal.py`` is the two-point flux with optional nonorthogonal correction.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import splu

from .materials import process_alpha


def _boundary_vertex_mask(edge_to_cells: dict, n_vertices: int) -> np.ndarray:
    on_b = np.zeros(n_vertices, dtype=bool)
    for edge, cells in edge_to_cells.items():
        if len(cells) == 1:
            on_b[edge[0]] = True
            on_b[edge[1]] = True
    return on_b


def _vertex_cell_fans(vertices: np.ndarray, polygons: list, cell_centers: np.ndarray) -> dict[int, list[int]]:
    from collections import defaultdict

    v_to_cells: dict[int, set] = defaultdict(set)
    for ci, poly in enumerate(polygons):
        for vid in poly:
            v_to_cells[vid].add(ci)
    fans: dict[int, list[int]] = {}
    for vid, cells in v_to_cells.items():
        v = vertices[vid]
        clist = list(cells)
        angles = [np.arctan2(cell_centers[c][1] - v[1], cell_centers[c][0] - v[0]) for c in clist]
        order = np.argsort(angles)
        fans[vid] = [clist[i] for i in order]
    return fans


def _triangle_stiffness(x: np.ndarray, k_tensor: np.ndarray) -> np.ndarray:
    """3x3 stiffness int K grad phi_i · grad phi_j dA on a linear triangle."""
    x0, x1, x2 = x[0], x[1], x[2]
    area2 = (x1[0] - x0[0]) * (x2[1] - x0[1]) - (x2[0] - x0[0]) * (x1[1] - x0[1])
    a = 0.5 * abs(area2)
    if a <= 1e-30:
        return np.zeros((3, 3))
    g = np.array(
        [
            [x1[1] - x2[1], x2[1] - x0[1], x0[1] - x1[1]],
            [x2[0] - x1[0], x0[0] - x2[0], x1[0] - x0[0]],
        ],
        dtype=float,
    ) / (2.0 * a)
    return a * (g.T @ k_tensor @ g)


def assemble_mpfa_diffusion(
    vertices: np.ndarray,
    polygons: list,
    cell_centers: np.ndarray,
    alpha,
    edge_to_cells: dict,
) -> csr_matrix:
    """
    Condensed cell-centered diffusion operator for homogeneous u = 0 on all mesh vertices.

    Returns S_cc (n_cells x n_cells) after eliminating interior vertices with Schur complement
    and assuming boundary vertex values are zero.
    """
    n_cells = len(polygons)
    n_v = len(vertices)
    n_dof = n_cells + n_v

    bmask = _boundary_vertex_mask(edge_to_cells, n_v)
    fans = _vertex_cell_fans(vertices, polygons, cell_centers)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    def add_loc(ii: list[int], jj: list[int], ke: np.ndarray):
        for r in range(3):
            for c in range(3):
                rows.append(ii[r])
                cols.append(jj[c])
                data.append(float(ke[r, c]))

    for vid, fan in fans.items():
        m = len(fan)
        if m < 2:
            continue
        xv = vertices[vid]
        for k in range(m):
            c0 = fan[k]
            c1 = fan[(k + 1) % m]
            xtri = np.array([cell_centers[c0], xv, cell_centers[c1]], dtype=float)
            tc = np.mean(xtri, axis=0)
            k_t = process_alpha(alpha, np.array(tc[0]), np.array(tc[1]))
            if k_t.ndim >= 2:
                k_mat = np.asarray(k_t, dtype=float).reshape(2, 2)
            else:
                k_mat = np.eye(2) * float(k_t)
            ke = _triangle_stiffness(xtri, k_mat)
            if not np.any(ke):
                continue
            ii = [c0, n_cells + vid, c1]
            add_loc(ii, ii, ke)

    a_full = coo_matrix((data, (rows, cols)), shape=(n_dof, n_dof)).tocsr()
    a_full = (a_full + a_full.T) * 0.5

    # Global dof order: [all cells 0..N-1 | vertices N..N+Nv-1]
    vi_local = [i for i in range(n_v) if not bmask[i]]
    vb_local = [i for i in range(n_v) if bmask[i]]
    n_i = len(vi_local)

    # Permutation: [cells | interior verts | boundary verts]
    perm: list[int] = list(range(n_cells))
    for i in vi_local:
        perm.append(n_cells + i)
    for i in vb_local:
        perm.append(n_cells + i)
    a_perm = a_full[perm, :][:, perm].tocsr()

    n_c = n_cells
    sl_c = slice(0, n_c)
    sl_i = slice(n_c, n_c + n_i)

    a_cc = a_perm[sl_c, sl_c]
    if n_i == 0:
        return a_cc.tocsr()

    a_ci = a_perm[sl_c, sl_i]
    a_ic = a_perm[sl_i, sl_c]
    a_ii = a_perm[sl_i, sl_i]

    # Schur complement out interior vertices i
    # S = A_cc - A_ci inv(A_ii) A_ic  etc.
    a_ii_csc = a_ii.tocsc()
    lu_ii = splu(a_ii_csc)

    # X = inv(A_ii) @ A_ic  with shape (n_i, n_c)
    x_ic = lu_ii.solve(a_ic.toarray())
    s_cc = a_cc - a_ci @ x_ic

    # Homogeneous boundary vertex values => diffusion operator on cells is S_cc
    return csr_matrix(s_cc)

