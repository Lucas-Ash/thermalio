import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import factorized, spsolve

from .geometry import cross2d, polygon_area_and_centroid
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
        reuse_linear_lhs=True,
    ):
        self.points = np.asarray(points, dtype=float)
        self.tris = np.asarray(tris, dtype=int)
        self.N = self.points.shape[0]
        self.alpha = alpha
        self.dt = float(dt)
        from .boundary_conditions import callback_accepts_normals, normalize_bc_type

        self.bc_type = normalize_bc_type(bc_type)
        self.bc_func = bc_func if bc_func is not None else (lambda x, y, t: 0.0)
        self._bc_accepts_normals = self.bc_type != "dirichlet" and callback_accepts_normals(self.bc_func)
        self.source_func = source_func if source_func is not None else (lambda x, y, t: 0.0)
        if phase_change_model is not None and not isinstance(phase_change_model, ApparentHeatCapacityModel):
            raise TypeError("phase_change_model must be an ApparentHeatCapacityModel instance or None.")
        self.phase_change_model = phase_change_model
        self.phase_change_options = {"max_iters": 30, "tol": 1e-9, "relaxation": 1.0}
        if phase_change_options is not None:
            self.phase_change_options.update(dict(phase_change_options))
        self._build_topology()
        self._build_boundary_edges()
        self._build_dual_geometry()
        self._assemble_diffusion_matrix()
        if self.bc_type != "dirichlet":
            # Boundary vertices are unknowns for natural boundary conditions.  Their
            # circumcentre-based control volumes can extend outside the domain
            # (obtuse boundary triangles); use the lumped-P1 volume (one third of
            # each incident triangle) there.  Dirichlet runs never use these
            # volumes, so their results are unchanged.
            lumped = np.zeros(self.N)
            np.add.at(lumped, self.tris.ravel(), np.repeat(self.tri_areas / 3.0, 3))
            self.cv_area = self.cv_area.copy()
            self.cv_area[self.is_boundary] = lumped[self.is_boundary]
        self._build_mass_matrix()
        self.t = 0.0
        self.u = np.zeros(self.N)
        self.reuse_linear_lhs = bool(reuse_linear_lhs)

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

    def _build_boundary_edges(self):
        """Boundary edges with lengths and outward unit normals (away from the owning triangle)."""
        if self.boundary_edges.size == 0:
            self._bedge_len = np.zeros(0)
            self._bedge_normal = np.zeros((0, 2))
            return
        lengths, normals = [], []
        for i, j in self.boundary_edges:
            tri = self.tris[self.edge_to_tris[(int(i), int(j))][0]]
            k = int([v for v in tri if v != i and v != j][0])
            e = self.points[j] - self.points[i]
            length = float(np.hypot(*e))
            n = np.array([e[1], -e[0]]) / length
            if np.dot(n, self.points[k] - self.points[i]) > 0:
                n = -n
            lengths.append(length)
            normals.append(n)
        self._bedge_len = np.asarray(lengths)
        self._bedge_normal = np.asarray(normals)

    def boundary_closure(self, t, temperature=None):
        """Lumped natural-boundary terms ``(diag, rhs)`` for non-Dirichlet conditions.

        The weak form contributes ``int_dOmega q_in phi_i ds`` with the inward
        flux ``q_in = alpha du/dn ~ c - a u`` (see
        :mod:`heat_solver.boundary_conditions`).  Each boundary edge gives half
        its length to each endpoint (trapezoid rule), with the data evaluated
        at the endpoint using that edge's outward normal.
        """
        from .boundary_conditions import evaluate, inflow_linearization
        from .materials import process_alpha

        diag = np.zeros(self.N)
        rhs = np.zeros(self.N)
        if self.bc_type == "dirichlet" or self.boundary_edges.size == 0:
            return diag, rhs
        ends = self.boundary_edges.reshape(-1)  # (2E,) endpoints
        normals = np.repeat(self._bedge_normal, 2, axis=0)
        weights = np.repeat(0.5 * self._bedge_len, 2)
        x, y = self.points[ends, 0], self.points[ends, 1]
        data = evaluate(self.bc_func, x, y, t, normals=normals, accepts_normals=self._bc_accepts_normals)
        alpha_n = None
        if self.bc_type == "neumann":
            K = process_alpha(self.alpha, x, y)
            alpha_n = np.einsum("ij,ij->i", normals, np.einsum("ijk,ik->ij", K, normals))
        temp = None if temperature is None else np.asarray(temperature, dtype=float)[ends]
        a, c = inflow_linearization(self.bc_type, data, ends.size, alpha_n=alpha_n, temperature=temp)
        np.add.at(diag, ends, weights * a)
        np.add.at(rhs, ends, weights * c)
        return diag, rhs

    def _solve_with_boundary(self, lhs, rhs, t, u_guess, max_iters=60, tol=1e-12):
        """Add the natural-boundary closure and solve (Newton iteration for radiation)."""
        if self.bc_type == "dirichlet":
            lhs, rhs = self.apply_dirichlet(lhs, rhs, t)
            return spsolve(lhs, rhs)
        if self.bc_type != "radiative":
            d, b = self.boundary_closure(t)
            return spsolve((lhs + diags(d)).tocsr(), rhs + b)
        u = np.asarray(u_guess, dtype=float).copy()
        for _ in range(max_iters):
            d, b = self.boundary_closure(t, temperature=u)
            u_new = spsolve((lhs + diags(d)).tocsr(), rhs + b)
            err = float(np.max(np.abs(u_new - u)))
            u = u_new
            if err <= tol * max(1.0, float(np.max(np.abs(u)))):
                return u
        raise RuntimeError("radiative boundary iteration did not converge; reduce dt.")

    @staticmethod
    def _triangle_areas(tri_points):
        a = tri_points[:, 0, :]
        b = tri_points[:, 1, :]
        c = tri_points[:, 2, :]
        return 0.5 * np.abs(cross2d(b - a, c - a))

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
                        cv_area[i] += 0.5 * abs(cross2d(mab - A, centroid - A)) + 0.5 * abs(cross2d(centroid - A, mca - A))
                    elif i == b:
                        cv_area[i] += 0.5 * abs(cross2d(mbc - B, centroid - B)) + 0.5 * abs(cross2d(centroid - B, mab - B))
                    else:
                        cv_area[i] += 0.5 * abs(cross2d(mca - C, centroid - C)) + 0.5 * abs(cross2d(centroid - C, mbc - C))
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

    def _apply_dirichlet_matrix_only(self, A_csr, fixed=None):
        if fixed is None:
            mask = self.is_boundary
        else:
            mask = np.zeros(self.N, dtype=bool)
            mask[np.asarray(fixed, dtype=int)] = True
        A_mod = A_csr.tolil()
        for i in np.where(mask)[0]:
            A_mod.rows[i] = [i]
            A_mod.data[i] = [1.0]
        return A_mod.tocsr()

    @staticmethod
    def _dt_schedule(t0, t_end, dt):
        t = t0
        nsteps = int(np.ceil((t_end - t0) / dt))
        schedule = []
        for _ in range(nsteps):
            t_next = min(t + dt, t_end)
            schedule.append(float(t_next - t))
            t = t_next
        return schedule

    def step(self, u, t, dt_step):
        source_vals = np.broadcast_to(
            np.asarray(self.source_func(self.points[:, 0], self.points[:, 1], t + dt_step), dtype=float),
            (self.N,),
        )
        if self.phase_change_model is None:
            mass_over_dt = diags(self.cv_area / dt_step, format="csr")
            rhs = (mass_over_dt @ u).copy() + self.cv_area * source_vals
            lhs = (mass_over_dt + self.A).tocsr()
            return self._solve_with_boundary(lhs, rhs, t + dt_step, u)

        max_iters = int(self.phase_change_options["max_iters"])
        tol = float(self.phase_change_options["tol"])
        relaxation = float(self.phase_change_options["relaxation"])
        u_iter = u.copy()
        for _ in range(max_iters):
            cp_eff = self._effective_heat_capacity(u_iter)
            mass_over_dt = diags((self.cv_area * cp_eff) / dt_step, format="csr")
            rhs = (mass_over_dt @ u).copy() + self.cv_area * source_vals
            lhs = (mass_over_dt + self.A).tocsr()
            if self.bc_type == "dirichlet":
                lhs, rhs = self.apply_dirichlet(lhs, rhs, t + dt_step)
            else:
                d, b = self.boundary_closure(t + dt_step, temperature=u_iter)
                lhs = (lhs + diags(d)).tocsr()
                rhs = rhs + b
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
        dts = self._dt_schedule(t0, t_end, self.dt)
        nsteps = len(dts)
        use_cache = (
            self.reuse_linear_lhs
            and self.phase_change_model is None
            and self.bc_type in ("dirichlet", "neumann", "flux")  # constant LHS
        )
        lhs_by_dt = {}
        factor_by_dt = {}
        if use_cache:
            for key in sorted({round(float(d), 12) for d in dts}):
                mass_over_dt = diags(self.cv_area / key, format="csr")
                lhs = (mass_over_dt + self.A).tocsr()
                lhs_by_dt[key] = self._apply_dirichlet_matrix_only(lhs) if self.bc_type == "dirichlet" else lhs
                factor_by_dt[key] = factorized(lhs_by_dt[key])

        for k in range(nsteps):
            t_next = min(self.t + self.dt, t_end)
            dt_step = dts[k]
            if use_cache:
                dt_key = round(float(dt_step), 12)
                source_vals = np.broadcast_to(
                    np.asarray(
                        self.source_func(self.points[:, 0], self.points[:, 1], t_next),
                        dtype=float,
                    ),
                    (self.N,),
                )
                mass_over_dt = diags(self.cv_area / dt_step, format="csr")
                rhs = (mass_over_dt @ self.u).copy() + self.cv_area * source_vals
                if self.bc_type == "dirichlet":
                    for i in np.where(self.is_boundary)[0]:
                        rhs[i] = self.bc_func(*self.points[i], t_next)
                else:
                    rhs = rhs + self.boundary_closure(t_next)[1]
                self.u = factor_by_dt[dt_key](rhs)
            else:
                self.u = self.step(self.u, self.t, dt_step)
            self.t = t_next
            if callback is not None:
                callback(k + 1, self.t, self.u)
        return self.t, self.u
