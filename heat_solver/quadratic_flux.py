"""Quadratically consistent diffusion fluxes for general polygonal meshes.

Why this exists
---------------
Thermalio's ``flux_discretization='reconstructed'`` builds the face gradient as
the *plain average* of two least-squares (LSQ) cell gradients.  Two defects
make it lose its second-order accuracy on non-orthogonal meshes:

1. **No compact coupling.**  The averaged-gradient stencil is wide and weakly
   coupled (near-checkerboard modes), so on sheared meshes the error constant
   blows up and the deep-interior error stops converging.
2. **Only linear exactness.**  A linearly exact LSQ gradient has an ``O(h)``
   error on irregular stencils, so face fluxes are only ``O(h)`` consistent and
   the solution converges at first order on rough meshes (this also limits
   the TPFA + non-orthogonal-correction path).

This module fixes both:

* **2-exact reconstruction** (Barth & Frederickson): on each cell a quadratic
  ``p_i(x) = c_i + g_i.(x - x_i) + 1/2 (x - x_i)^T H_i (x - x_i)`` is fitted by
  weighted least squares to the *cell averages* of a vertex-ring stencil, with
  the cell second moments ``M_j`` in the fitting rows, so the fit is exact for
  quadratic fields.  ``g_i`` is then ``O(h^2)`` accurate.
* **Face-point gradients**: each side evaluates ``grad p(x_f) = g + H (x_f - x)``
  at the face midpoint; the face gradient is their average (``O(h^2)``).
* **Consistent compact stabilization**: a two-point penalty
  ``T_f [ (u_j - u_i) - J_f ]`` with ``T_f = n.K n / n.d > 0``, where ``J_f`` is
  the jump in cell averages *predicted* by the two quadratics.  The penalty
  restores compact coupling and vanishes identically on quadratic fields, so it
  does not spoil consistency.

The flux is conservative (one value per face, added to one cell and
subtracted from its neighbour) and exact for quadratic solutions with constant
``K``, which gives second-order convergence on smooth, sheared, perturbed and
mixed polygonal meshes alike.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix


def polygon_second_moments(points):
    """Area, centroid and central second moments ``M = (1/A) int (x-c)(x-c)^T``."""
    pts = np.asarray(points, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cr = x * yn - xn * y
    a = 0.5 * np.sum(cr)
    cx = np.sum((x + xn) * cr) / (6.0 * a)
    cy = np.sum((y + yn) * cr) / (6.0 * a)
    ixx = np.sum(cr * (x * x + x * xn + xn * xn)) / 12.0
    iyy = np.sum(cr * (y * y + y * yn + yn * yn)) / 12.0
    ixy = np.sum(cr * (x * yn + 2 * x * y + 2 * xn * yn + xn * y)) / 24.0
    mxx = ixx / a - cx * cx
    myy = iyy / a - cy * cy
    mxy = ixy / a - cx * cy
    return abs(a), np.array([cx, cy]), np.array([[mxx, mxy], [mxy, myy]])


def _vertex_neighbors(polygons, n_cells):
    vert_cells = {}
    for c, poly in enumerate(polygons):
        for v in poly:
            vert_cells.setdefault(int(v), set()).add(c)
    nbrs = [set() for _ in range(n_cells)]
    for cells in vert_cells.values():
        for c in cells:
            nbrs[c] |= cells
    for c in range(n_cells):
        nbrs[c].discard(c)
    return [sorted(s) for s in nbrs]


def _fit_row(delta, m_diff):
    """Row of the 2-exact fit: ubar_j - ubar_i = g.D + 1/2 D^T H D + 1/2 H:(M_j - M_i)."""
    dx, dy = delta
    return np.array([
        dx,
        dy,
        0.5 * dx * dx + 0.5 * m_diff[0, 0],
        dx * dy + m_diff[0, 1],
        0.5 * dy * dy + 0.5 * m_diff[1, 1],
    ])


class QuadraticReconstruction:
    """2-exact least-squares reconstruction on a polygonal mesh.

    For every cell ``i`` the coefficients ``z_i = (g_x, g_y, H_xx, H_xy, H_yy)``
    are linear in the cell averages: ``z_i = sum_k C_i[k] u_k`` with
    ``C_i`` stored as ``(indices, 5 x n matrix)``.
    """

    def __init__(self, vertices, polygons, min_stencil=7):
        self.vertices = np.asarray(vertices, dtype=float)
        self.polygons = [list(p) for p in polygons]
        n = len(self.polygons)
        geo = [polygon_second_moments(self.vertices[p]) for p in self.polygons]
        self.areas = np.array([g[0] for g in geo])
        self.centers = np.array([g[1] for g in geo])
        self.moments = np.array([g[2] for g in geo])
        ring1 = _vertex_neighbors(self.polygons, n)
        self.stencils = []
        self.coeffs = []
        self.degree = np.zeros(n, dtype=int)
        for i in range(n):
            stencil = list(ring1[i])
            if len(stencil) < min_stencil:  # boundary cells: add the second vertex ring
                extra = set(stencil)
                for j in ring1[i]:
                    extra |= set(ring1[j])
                extra.discard(i)
                stencil = sorted(extra)
            self.stencils.append(np.asarray(stencil, dtype=int))
            self.coeffs.append(self._fit(i, self.stencils[-1]))

    def _fit(self, i, stencil):
        xi, mi = self.centers[i], self.moments[i]
        deltas = self.centers[stencil] - xi
        h = np.sqrt(self.areas[i])
        rows = np.array([_fit_row(d, self.moments[j] - mi) for d, j in zip(deltas, stencil)])
        w = 1.0 / np.maximum(np.sum(deltas**2, axis=1), 1e-300)
        scale = np.array([h, h, h * h, h * h, h * h])  # column scaling for conditioning
        R = rows * scale
        W = np.sqrt(w)[:, None]
        degree = 2
        if len(stencil) < 5 or np.linalg.matrix_rank(W * R) < 5:
            degree = 1
            R = R[:, :2]
            scale = scale[:2]
        pinv = np.linalg.pinv(W * R) * W.T  # (k x n): z_scaled = pinv @ (u_S - u_i)
        z = pinv * scale[:, None]  # R = rows*scale  =>  z = scale * y
        if degree == 1:
            z = np.vstack([z, np.zeros((3, z.shape[1]))])
        self.degree[i] = degree
        # Express as coefficients on [stencil..., i]: z = Z_S u_S - (sum Z_S) u_i
        return np.hstack([z, -z.sum(axis=1, keepdims=True)])

    def indices(self, i):
        return np.append(self.stencils[i], i)


class QuadraticDiffusionOperator:
    """Precomputed quadratic-flux operator: ``A(K) = sum_f sum_{ab} K_f[a, b] G_{f,ab}``.

    Every entry of the quadratic flux (consistent part and penalty) is linear
    in the four components of the face conductivity tensor, so the geometric
    coefficients are computed once and re-assembly for new conductivities --
    e.g. temperature-dependent ``K(x, u)`` inside a nonlinear iteration -- is a
    vectorized product plus one sparse conversion.

    ``face_values(u)`` evaluates the average of the two one-sided quadratic
    reconstructions at each face midpoint (``O(h^3)`` accurate), which is used
    for the face temperature of nonlinear conductivities; the plain two-cell
    average would only be ``O(h)`` accurate on irregular meshes.
    """

    def __init__(self, vertices, polygons, edge_to_cells, reconstruction=None, penalty=1.0):
        rec = reconstruction or QuadraticReconstruction(vertices, polygons)
        self.reconstruction = rec
        self.n = len(rec.polygons)
        verts = np.asarray(vertices, dtype=float)
        rows, cols, geom, face_id = [], [], [], []
        fv_rows, fv_cols, fv_vals = [], [], []
        mids, lengths, normals = [], [], []
        f = 0
        for edge, cells in edge_to_cells.items():
            if len(cells) != 2:
                continue
            i, j = sorted(cells)
            v0, v1 = verts[edge[0]], verts[edge[1]]
            ev = v1 - v0
            length = float(np.hypot(*ev))
            if length == 0.0:
                continue
            normal = np.array([ev[1], -ev[0]]) / length
            d = rec.centers[j] - rec.centers[i]
            if normal @ d < 0:
                normal = -normal
            xf = 0.5 * (v0 + v1)
            idx_i, Ci = rec.indices(i), rec.coeffs[i]
            idx_j, Cj = rec.indices(j), rec.coeffs[j]
            ri, rj = xf - rec.centers[i], xf - rec.centers[j]
            # one-sided gradients at x_f, as coefficient vectors (2 x n_stencil)
            Gi = np.vstack([Ci[0] + Ci[2] * ri[0] + Ci[3] * ri[1], Ci[1] + Ci[3] * ri[0] + Ci[4] * ri[1]])
            Gj = np.vstack([Cj[0] + Cj[2] * rj[0] + Cj[3] * rj[1], Cj[1] + Cj[3] * rj[0] + Cj[4] * rj[1]])
            idx = np.concatenate([idx_i, idx_j, [i, j], idx_i, idx_j])
            row_ij = _fit_row(d, rec.moments[j] - rec.moments[i])
            row_ji = _fit_row(-d, rec.moments[i] - rec.moments[j])
            pen = np.concatenate([np.zeros(len(idx_i) + len(idx_j)), [1.0, -1.0],
                                  0.5 * (row_ij @ Ci), -0.5 * (row_ji @ Cj)])
            nd = float(normal @ d)
            g4 = np.zeros((len(idx), 4))  # components (0,0), (0,1), (1,0), (1,1) of K
            for c, (a, b) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
                # n.K.grad u = sum_ab n_a K_ab (grad u)_b ; penalty T = n.K.n / n.d
                cons = np.concatenate([Gi[b], Gj[b], [0.0, 0.0], np.zeros(len(idx_i) + len(idx_j))])
                g4[:, c] = -0.5 * length * normal[a] * cons + length * normal[a] * normal[b] / nd * pen
            for sign, row in ((1.0, i), (-1.0, j)):
                rows.append(np.full(len(idx), row))
                cols.append(idx)
                geom.append(sign * g4)
                face_id.append(np.full(len(idx), f))
            # face value: 1/2 [p_i(x_f) + p_j(x_f)], p(x) = ubar - 1/2 H:M + g.r + 1/2 r^T H r
            for C, r, ii, M, idxc in ((Ci, ri, i, rec.moments[i], idx_i), (Cj, rj, j, rec.moments[j], idx_j)):
                vals = (C[0] * r[0] + C[1] * r[1]
                        + 0.5 * (C[2] * (r[0] ** 2 - M[0, 0]) + 2 * C[3] * (r[0] * r[1] - M[0, 1])
                                 + C[4] * (r[1] ** 2 - M[1, 1])))
                fv_rows.append(np.full(len(idxc) + 1, f))
                fv_cols.append(np.append(idxc, ii))
                fv_vals.append(0.5 * np.append(vals, 1.0))
            mids.append(xf)
            lengths.append(length)
            normals.append(normal)
            f += 1
        self.n_faces = f
        self.face_midpoints = np.asarray(mids)
        self.face_lengths = np.asarray(lengths)
        self.face_normals = np.asarray(normals)
        self._rows = np.concatenate(rows)
        self._cols = np.concatenate(cols)
        self._geom = np.concatenate(geom)
        self._face = np.concatenate(face_id)
        if penalty != 1.0:
            raise NotImplementedError("penalty != 1 is not supported by the precomputed operator.")
        self._face_value = coo_matrix((np.concatenate(fv_vals), (np.concatenate(fv_rows), np.concatenate(fv_cols))),
                                      shape=(f, self.n)).tocsr()

    def face_values(self, u):
        return self._face_value @ np.asarray(u, dtype=float)

    def assemble(self, K_faces):
        """``K_faces``: scalar, (2, 2), (n_faces,) or (n_faces, 2, 2) conductivities at face midpoints."""
        K = np.asarray(K_faces, dtype=float)
        if K.ndim == 0:
            K = np.broadcast_to(np.eye(2) * float(K), (self.n_faces, 2, 2))
        elif K.shape == (2, 2):
            K = np.broadcast_to(K, (self.n_faces, 2, 2))
        elif K.shape == (self.n_faces,):
            K = K[:, None, None] * np.eye(2)
        comps = K.reshape(self.n_faces, 4)[self._face]
        data = np.einsum("ec,ec->e", self._geom, comps)
        return coo_matrix((data, (self._rows, self._cols)), shape=(self.n, self.n)).tocsr()


def assemble_quadratic_diffusion(vertices, polygons, edge_to_cells, kappa_at, reconstruction=None,
                                 penalty=1.0, operator=None):
    """Assemble ``A`` with ``(A u)_i = sum_faces outward flux of -K grad u``.

    ``kappa_at(x, y) -> (..., 2, 2)`` gives the conductivity at face midpoints
    (vectorized).  Boundary faces are skipped (closures are added by the
    solver).  Returns ``(A, operator)``; pass ``operator`` back to reuse the
    precomputed geometry.
    """
    op = operator or QuadraticDiffusionOperator(vertices, polygons, edge_to_cells, reconstruction, penalty)
    xf = op.face_midpoints
    K = np.asarray(kappa_at(xf[:, 0], xf[:, 1]), dtype=float)
    return op.assemble(K), op


# ---------------------------------------------------------------------------
# Cell-average data (consistent finite-volume sampling)
# ---------------------------------------------------------------------------

# Degree-4 (6-point) Dunavant rule on the reference triangle (barycentric l1, l2; weights sum to 1).
_TRI_POINTS = np.array([
    [0.445948490915965, 0.445948490915965], [0.445948490915965, 0.108103018168070],
    [0.108103018168070, 0.445948490915965], [0.091576213509771, 0.091576213509771],
    [0.091576213509771, 0.816847572980459], [0.816847572980459, 0.091576213509771],
])
_TRI_WEIGHTS = np.array([0.223381589678011] * 3 + [0.109951743655322] * 3)


class CellQuadrature:
    """Triangle-fan quadrature (degree 4) of every polygonal cell."""

    def __init__(self, vertices, polygons):
        verts = np.asarray(vertices, dtype=float)
        pts, wts, owner = [], [], []
        for c, poly in enumerate(polygons):
            P = verts[list(poly)]
            ctr = P.mean(axis=0)
            for k in range(len(P)):
                a, b = P[k], P[(k + 1) % len(P)]
                area = 0.5 * abs((a[0] - ctr[0]) * (b[1] - ctr[1]) - (b[0] - ctr[0]) * (a[1] - ctr[1]))
                for (l1, l2), w in zip(_TRI_POINTS, _TRI_WEIGHTS):
                    pts.append(ctr + l1 * (a - ctr) + l2 * (b - ctr))
                    wts.append(w * area)
                    owner.append(c)
        self.points = np.asarray(pts)
        self.weights = np.asarray(wts)
        self.owner = np.asarray(owner)
        self.n_cells = len(polygons)
        self.areas = np.bincount(self.owner, weights=self.weights, minlength=self.n_cells)

    def average(self, f, *args):
        vals = np.broadcast_to(np.asarray(f(self.points[:, 0], self.points[:, 1], *args), dtype=float),
                               (self.points.shape[0],))
        return np.bincount(self.owner, weights=self.weights * vals, minlength=self.n_cells) / self.areas


class CellAveragedFunction:
    """Wrap ``f(x, y, t)`` so that calls at the cell centroids return *cell averages*.

    Any other evaluation points (e.g. boundary-face midpoints) fall back to
    point evaluation.  Used for the source and Dirichlet data of the
    quadratic flux, whose 2-exact reconstruction treats unknowns as cell
    averages: centroid sampling is only an ``O(h^2)`` inconsistency, but the
    reconstruction differentiates it twice.
    """

    def __init__(self, f, quadrature, centers):
        self.f = f
        self.quadrature = quadrature
        self.centers = np.asarray(centers, dtype=float)

    def __call__(self, x, y, *args):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.shape == (self.centers.shape[0],) and np.allclose(x, self.centers[:, 0]) and np.allclose(
                y, self.centers[:, 1]):
            return self.quadrature.average(self.f, *args)
        return self.f(x, y, *args)
