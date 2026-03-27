import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve

from .geometry import polygon_area_and_centroid
from .phase_change import ApparentHeatCapacityModel


class NonUniformHeatSolver:
    """
    Finite Volume heat equation solver on an unstructured triangular mesh.
    Vertex-centered unknowns with median dual control volumes.
    """

    def __init__(
        self,
        points,
        tris,
        alpha,
        dt,
        bc_type="dirichlet",
        bc_func=None,
        source_func=None,
        phase_change_model=None,
        phase_change_options=None,
    ):
        self.points = np.asarray(points, dtype=float)
        self.tris = np.asarray(tris, dtype=int)
        self.N = self.points.shape[0]
        self.alpha = alpha
        self.dt = float(dt)
        self.bc_type = bc_type
        self.bc_func = bc_func if bc_func is not None else (lambda x, y, t: 0.0)
        self.source_func = source_func if source_func is not None else (lambda x, y, t: 0.0)
        if phase_change_model is not None and not isinstance(phase_change_model, ApparentHeatCapacityModel):
            raise TypeError("phase_change_model must be an ApparentHeatCapacityModel instance or None.")
        self.phase_change_model = phase_change_model
        self.phase_change_options = {"max_iters": 30, "tol": 1e-9, "relaxation": 1.0}
        if phase_change_options is not None:
            self.phase_change_options.update(dict(phase_change_options))
        self._build_topology()
        self._build_dual_geometry()
        self._assemble_diffusion_matrix()
        self._build_mass_matrix()
        self.t = 0.0
        self.u = np.zeros(self.N)

    def _build_topology(self):
        edges = {}
        for k, (a, b, c) in enumerate(self.tris):
            for i, j in ((a, b), (b, c), (c, a)):
                edge = (i, j) if i < j else (j, i)
                edges.setdefault(edge, []).append(k)
        boundary_edges = [edge for edge, adj in edges.items() if len(adj) == 1]
        self.boundary_edges = np.array(boundary_edges, dtype=int)
        boundary_vertices = set()
        for i, j in boundary_edges:
            boundary_vertices.add(i)
            boundary_vertices.add(j)
        self.is_boundary = np.zeros(self.N, dtype=bool)
        if boundary_vertices:
            self.is_boundary[list(boundary_vertices)] = True
        neighbors = [[] for _ in range(self.N)]
        for i, j in edges:
            neighbors[i].append(j)
            neighbors[j].append(i)
        self.neighbors = [sorted(set(nbs)) for nbs in neighbors]
        self.tri_centroids = np.mean(self.points[self.tris], axis=1)
        self.tri_areas = self._triangle_areas(self.points[self.tris])
        self.edge_to_tris = edges

    @staticmethod
    def _triangle_areas(tri_points):
        a = tri_points[:, 0, :]
        b = tri_points[:, 1, :]
        c = tri_points[:, 2, :]
        return 0.5 * np.abs(np.cross(b - a, c - a))

    @staticmethod
    def _triangle_circumcenter(a, b, c):
        ax, ay = a
        bx, by = b
        cx, cy = c
        det = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if np.isclose(det, 0.0):
            return (a + b + c) / 3.0
        a2 = ax * ax + ay * ay
        b2 = bx * bx + by * by
        c2 = cx * cx + cy * cy
        ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / det
        uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / det
        return np.array([ux, uy], dtype=float)

    def _build_dual_geometry(self):
        circumcenters = np.array([
            self._triangle_circumcenter(self.points[a], self.points[b], self.points[c])
            for a, b, c in self.tris
        ])
        dual_length = {}
        vertex_to_points = [[] for _ in range(self.N)]

        for tri_idx, tri in enumerate(self.tris):
            cc = circumcenters[tri_idx]
            for vertex in tri:
                vertex_to_points[vertex].append(cc)

        for edge, tri_ids in self.edge_to_tris.items():
            i, j = edge
            midpoint = 0.5 * (self.points[i] + self.points[j])
            if len(tri_ids) == 2:
                dual_length[edge] = np.linalg.norm(circumcenters[tri_ids[0]] - circumcenters[tri_ids[1]])
            else:
                dual_length[edge] = np.linalg.norm(circumcenters[tri_ids[0]] - midpoint)
            if len(tri_ids) == 1:
                vertex_to_points[i].append(midpoint)
                vertex_to_points[j].append(midpoint)

        cv_area = np.zeros(self.N, dtype=float)
        for i, points in enumerate(vertex_to_points):
            if self.is_boundary[i]:
                points = points + [self.points[i]]
            if len(points) < 3:
                # Fallback to the median-dual area in degenerate cases.
                for tri_idx, tri in enumerate(self.tris):
                    if i not in tri:
                        continue
                    a, b, c = tri
                    A = self.points[a]
                    B = self.points[b]
                    C = self.points[c]
                    centroid = self.tri_centroids[tri_idx]
                    mab = 0.5 * (A + B)
                    mbc = 0.5 * (B + C)
                    mca = 0.5 * (C + A)
                    if i == a:
                        cv_area[i] += 0.5 * abs(np.cross(mab - A, centroid - A)) + 0.5 * abs(np.cross(centroid - A, mca - A))
                    elif i == b:
                        cv_area[i] += 0.5 * abs(np.cross(mbc - B, centroid - B)) + 0.5 * abs(np.cross(centroid - B, mab - B))
                    else:
                        cv_area[i] += 0.5 * abs(np.cross(mca - C, centroid - C)) + 0.5 * abs(np.cross(centroid - C, mbc - C))
                continue
            pts = np.array(points, dtype=float)
            angles = np.arctan2(pts[:, 1] - self.points[i, 1], pts[:, 0] - self.points[i, 0])
            order = np.argsort(angles)
            area, _ = polygon_area_and_centroid(pts[order])
            cv_area[i] = area

        self.circumcenters = circumcenters
        self.cv_area = cv_area
        self.dual_length = dual_length
        self.cc_distance = {}
        for i in range(self.N):
            for j in self.neighbors[i]:
                if i < j:
                    self.cc_distance[(i, j)] = max(np.linalg.norm(self.points[j] - self.points[i]), 1e-14)

    def _assemble_diffusion_matrix(self):
        from .materials import process_alpha
        Alpha_tri = process_alpha(self.alpha, self.tri_centroids[:, 0], self.tri_centroids[:, 1])

        rows = []
        cols = []
        vals = []

        for tri_idx, (a, b, c) in enumerate(self.tris):
            p1 = self.points[a]
            p2 = self.points[b]
            p3 = self.points[c]

            dx2, dy2 = p2 - p1
            dx3, dy3 = p3 - p1
            detJ = dx2 * dy3 - dx3 * dy2
            if detJ < 0:
                detJ = -detJ
                dx2, dx3 = dx3, dx2
                dy2, dy3 = dy3, dy2
                b, c = c, b

            area = 0.5 * detJ
            invJ = 1.0 / detJ
            
            grad_phi2 = np.array([dy3 * invJ, -dx3 * invJ])
            grad_phi3 = np.array([-dy2 * invJ, dx2 * invJ])
            grad_phi1 = -grad_phi2 - grad_phi3

            B = np.vstack((grad_phi1, grad_phi2, grad_phi3))
            A_matrix = Alpha_tri[tri_idx]
            
            K = area * B @ A_matrix @ B.T
            
            nodes = [a, b, c]
            for i in range(3):
                for j in range(3):
                    rows.append(nodes[i])
                    cols.append(nodes[j])
                    vals.append(K[i, j])

        from scipy.sparse import coo_matrix
        self.A = coo_matrix((vals, (rows, cols)), shape=(self.N, self.N)).tocsr()

    def _build_mass_matrix(self):
        self.M = diags(self.cv_area, format="csr")
        self.M_over_dt = diags(self.cv_area / self.dt, format="csr")
        self.LHS_base = (self.M_over_dt + self.A).tocsr()

    def _effective_heat_capacity(self, temperature):
        if self.phase_change_model is None:
            return np.ones_like(temperature, dtype=float)
        return np.broadcast_to(
            np.asarray(self.phase_change_model.effective_heat_capacity(temperature), dtype=float),
            temperature.shape,
        )

    def apply_dirichlet(self, A, rhs, t, fixed=None):
        if fixed is None:
            mask = self.is_boundary
        else:
            mask = np.zeros(self.N, dtype=bool)
            mask[np.asarray(fixed, dtype=int)] = True
        A_mod = A.tolil()
        for i in np.where(mask)[0]:
            rhs[i] = self.bc_func(*self.points[i], t)
            A_mod.rows[i] = [i]
            A_mod.data[i] = [1.0]
        return A_mod.tocsr(), rhs

    def step(self, u, t, dt_step):
        source_vals = np.broadcast_to(
            np.asarray(self.source_func(self.points[:, 0], self.points[:, 1], t + dt_step), dtype=float),
            (self.N,),
        )
        if self.phase_change_model is None:
            mass_over_dt = diags(self.cv_area / dt_step, format="csr")
            rhs = (mass_over_dt @ u).copy() + self.cv_area * source_vals
            lhs = (mass_over_dt + self.A).tocsr()
            if self.bc_type.lower() == "dirichlet":
                lhs, rhs = self.apply_dirichlet(lhs, rhs, t + dt_step)
            return spsolve(lhs, rhs)

        max_iters = int(self.phase_change_options["max_iters"])
        tol = float(self.phase_change_options["tol"])
        relaxation = float(self.phase_change_options["relaxation"])
        u_iter = u.copy()
        for _ in range(max_iters):
            cp_eff = self._effective_heat_capacity(u_iter)
            mass_over_dt = diags((self.cv_area * cp_eff) / dt_step, format="csr")
            rhs = (mass_over_dt @ u).copy() + self.cv_area * source_vals
            lhs = (mass_over_dt + self.A).tocsr()
            if self.bc_type.lower() == "dirichlet":
                lhs, rhs = self.apply_dirichlet(lhs, rhs, t + dt_step)
            u_next = spsolve(lhs, rhs)
            if relaxation != 1.0:
                u_next = relaxation * u_next + (1.0 - relaxation) * u_iter
            err = np.max(np.abs(u_next - u_iter))
            scale = max(1.0, np.max(np.abs(u_next)))
            u_iter = u_next
            if err <= tol * scale:
                return u_iter
        raise RuntimeError(
            "Phase-change nonlinear solve did not converge. "
            "Try smaller dt or larger phase_change_options['max_iters']."
        )

    def solve(self, u0, t0, t_end, callback=None):
        self.u = np.asarray(u0, dtype=float).copy()
        self.t = float(t0)
        nsteps = int(np.ceil((t_end - t0) / self.dt))
        for k in range(nsteps):
            t_next = min(self.t + self.dt, t_end)
            dt_step = t_next - self.t
            self.u = self.step(self.u, self.t, dt_step)
            self.t = t_next
            if callback is not None:
                callback(k + 1, self.t, self.u)
        return self.t, self.u
