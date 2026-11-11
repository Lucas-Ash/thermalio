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


def assemble_quadratic_diffusion(vertices, polygons, edge_to_cells, kappa_at, reconstruction=None,
                                 penalty=1.0):
    """Assemble ``A`` with ``(A u)_i = sum_faces outward flux of -K grad u``.

    ``kappa_at(x, y) -> 2x2`` gives the (possibly variable) conductivity at a
    face midpoint.  Faces on the domain boundary are skipped (boundary closures
    are added by the solver).  Returns ``(A, reconstruction)``.
    """
    rec = reconstruction or QuadraticReconstruction(vertices, polygons)
    verts = np.asarray(vertices, dtype=float)
    rows, cols, vals = [], [], []
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
        K = np.asarray(kappa_at(xf[0], xf[1]), dtype=float)
        if K.ndim == 0:
            K = np.eye(2) * float(K)
        Kn = K.T @ normal  # n.K grad u = (K^T n).grad u
        idx_i, Ci = rec.indices(i), rec.coeffs[i]
        idx_j, Cj = rec.indices(j), rec.coeffs[j]

        def grad_at(C, xc, x):
            r = x - xc
            # grad p(x) = g + H r ; rows of C: gx, gy, Hxx, Hxy, Hyy
            gx = C[0] + C[2] * r[0] + C[3] * r[1]
            gy = C[1] + C[3] * r[0] + C[4] * r[1]
            return gx, gy

        gix, giy = grad_at(Ci, rec.centers[i], xf)
        gjx, gjy = grad_at(Cj, rec.centers[j], xf)
        # Consistent part: -|e| n.K * average face gradient.
        flux_i = -0.5 * length * (Kn[0] * gix + Kn[1] * giy)
        flux_j = -0.5 * length * (Kn[0] * gjx + Kn[1] * gjy)
        entries = {}
        for idx, coef in ((idx_i, flux_i), (idx_j, flux_j)):
            for k, c in zip(idx, coef):
                entries[k] = entries.get(k, 0.0) + c
        # Compact penalty T [ (u_j - u_i) - J ], J = predicted jump, exact for quadratics.
        T = penalty * float(normal @ K @ normal) / float(normal @ d)
        row_ij = _fit_row(d, rec.moments[j] - rec.moments[i])
        row_ji = _fit_row(-d, rec.moments[i] - rec.moments[j])
        pred_i = row_ij @ Ci  # p_i predicts ubar_j - ubar_i
        pred_j = -(row_ji @ Cj)  # p_j predicts ubar_j - ubar_i
        entries[j] = entries.get(j, 0.0) - T * length
        entries[i] = entries.get(i, 0.0) + T * length
        for idx, coef in ((idx_i, pred_i), (idx_j, pred_j)):
            for k, c in zip(idx, coef):
                entries[k] = entries.get(k, 0.0) + 0.5 * T * length * c
        for k, c in entries.items():
            if c != 0.0:
                rows.extend((i, j))
                cols.extend((k, k))
                vals.extend((c, -c))
    n = len(rec.polygons)
    A = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    return A, rec


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
