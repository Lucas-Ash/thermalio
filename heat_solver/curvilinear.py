import inspect

import numpy as np
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import factorized, spsolve

from .phase_change import ApparentHeatCapacityModel


class CurvilinearHeatSolver:
    """
    Solves the heat equation u_t = div(alpha * grad(u)) + S on a structured curvilinear grid.
    Grid is given by X(eta, xi) and Y(eta, xi) where dimensions are (ny, nx).
    Uses conservative mapped finite differences.

    ``time_scheme='backward_euler'`` (default) is first-order accurate in time.
    ``time_scheme='crank_nicolson'`` uses the trapezoidal step for the semi-discrete
    system ``u' = A u + S`` and is second-order accurate in time for smooth solutions.
    """

    def __init__(
        self,
        X,
        Y,
        alpha,
        dt,
        bc_type="dirichlet",
        bc_func=None,
        source_func=None,
        time_scheme="backward_euler",
        phase_change_model=None,
        phase_change_options=None,
        nonlinear_options=None,
        reuse_linear_lhs=True,
    ):
        self.X = np.asarray(X, dtype=float)
        self.Y = np.asarray(Y, dtype=float)
        if self.X.shape != self.Y.shape:
            raise ValueError("X and Y must have the same shape")
        self.ny, self.nx = self.X.shape
        self.alpha = alpha
        self.dt = float(dt)
        self.bc_func = bc_func if bc_func is not None else (lambda x, y, t: np.zeros_like(x, dtype=float))
        self.source_func = source_func if source_func is not None else (lambda x, y, t: np.zeros_like(x, dtype=float))
        self.bc_type = str(bc_type).lower()
        if self.bc_type not in {"dirichlet", "neumann", "robin", "radiative"}:
            raise ValueError("bc_type must be one of: dirichlet, neumann, robin, radiative")
        self.time_scheme = str(time_scheme).lower().strip()
        if self.time_scheme not in {"backward_euler", "crank_nicolson"}:
            raise ValueError("time_scheme must be 'backward_euler' or 'crank_nicolson'.")
        if phase_change_model is not None and not isinstance(phase_change_model, ApparentHeatCapacityModel):
            raise TypeError("phase_change_model must be an ApparentHeatCapacityModel instance or None.")
        self.phase_change_model = phase_change_model
        self.nonlinear_options = {
            "max_iters": 30,
            "tol": 1e-9,
            "relaxation": 1.0,
        }
        if nonlinear_options is not None:
            self.nonlinear_options.update(dict(nonlinear_options))
        if phase_change_options is not None:
            self.nonlinear_options.update(dict(phase_change_options))
        self.phase_change_options = self.nonlinear_options
        self.N = self.nx * self.ny
        self.J = np.ones((self.ny, self.nx), dtype=float)
        self.M_diag, self.A = self._build_matrices()
        self._bc_accepts_normals = self._bc_func_accepts_normals()
        self._boundary_data = self._build_boundary_data()
        self._boundary_idx = self._boundary_data["boundary_idx"]
        self.reuse_linear_lhs = bool(reuse_linear_lhs)

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

    def _idx(self, j, i):
        return j * self.nx + i

    def _build_matrices(self):
        ny, nx = self.ny, self.nx
        X, Y = self.X, self.Y
        
        # 1. Compute metrics via central difference for interior, and one-sided for boundaries
        x_xi = np.zeros_like(X)
        x_eta = np.zeros_like(X)
        y_xi = np.zeros_like(Y)
        y_eta = np.zeros_like(Y)
        
        # interior
        x_xi[:, 1:-1] = 0.5 * (X[:, 2:] - X[:, :-2])
        x_eta[1:-1, :] = 0.5 * (X[2:, :] - X[:-2, :])
        y_xi[:, 1:-1] = 0.5 * (Y[:, 2:] - Y[:, :-2])
        y_eta[1:-1, :] = 0.5 * (Y[2:, :] - Y[:-2, :])
        
        # boundaries (one-sided)
        x_xi[:, 0] = X[:, 1] - X[:, 0]
        x_xi[:, -1] = X[:, -1] - X[:, -2]
        y_xi[:, 0] = Y[:, 1] - Y[:, 0]
        y_xi[:, -1] = Y[:, -1] - Y[:, -2]
        
        x_eta[0, :] = X[1, :] - X[0, :]
        x_eta[-1, :] = X[-1, :] - X[-2, :]
        y_eta[0, :] = Y[1, :] - Y[0, :]
        y_eta[-1, :] = Y[-1, :] - Y[-2, :]
        
        J = x_xi * y_eta - x_eta * y_xi
        self.J = np.abs(J)
        
        from .materials import process_alpha
        Alpha = process_alpha(self.alpha, X, Y)
        a11 = Alpha[..., 0, 0]
        a12 = Alpha[..., 0, 1]
        a22 = Alpha[..., 1, 1]
        
        q11 = (a11 * y_eta**2 - 2 * a12 * x_eta * y_eta + a22 * x_eta**2) / J
        q22 = (a11 * y_xi**2 - 2 * a12 * x_xi * y_xi + a22 * x_xi**2) / J
        q12 = (-a11 * y_xi * y_eta + a12 * (x_xi * y_eta + x_eta * y_xi) - a22 * x_xi * x_eta) / J
        
        self.q11 = q11
        self.q22 = q22
        self.q12 = q12
        
        A = lil_matrix((self.N, self.N))
        
        # Assemble Laplacian for interior points
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                idx = self._idx(j, i)
                
                # Flux at i+1/2, j
                q11_e = 0.5 * (q11[j, i+1] + q11[j, i])
                q12_e = 0.5 * (q12[j, i+1] + q12[j, i])
                # du / dxi
                dxi_u_e = {self._idx(j, i+1): 1.0, self._idx(j, i): -1.0}
                # du / deta (averaged at i+1/2, j)
                deta_u_e = {
                    self._idx(j+1, i+1): 0.25, self._idx(j+1, i): 0.25,
                    self._idx(j-1, i+1): -0.25, self._idx(j-1, i): -0.25
                }
                
                # Flux at i-1/2, j
                q11_w = 0.5 * (q11[j, i] + q11[j, i-1])
                q12_w = 0.5 * (q12[j, i] + q12[j, i-1])
                dxi_u_w = {self._idx(j, i): 1.0, self._idx(j, i-1): -1.0}
                deta_u_w = {
                    self._idx(j+1, i): 0.25, self._idx(j+1, i-1): 0.25,
                    self._idx(j-1, i): -0.25, self._idx(j-1, i-1): -0.25
                }
                
                # Flux at i, j+1/2
                q22_n = 0.5 * (q22[j+1, i] + q22[j, i])
                q12_n = 0.5 * (q12[j+1, i] + q12[j, i])
                deta_u_n = {self._idx(j+1, i): 1.0, self._idx(j, i): -1.0}
                dxi_u_n = {
                    self._idx(j+1, i+1): 0.25, self._idx(j, i+1): 0.25,
                    self._idx(j+1, i-1): -0.25, self._idx(j, i-1): -0.25
                }
                
                # Flux at i, j-1/2
                q22_s = 0.5 * (q22[j, i] + q22[j-1, i])
                q12_s = 0.5 * (q12[j, i] + q12[j-1, i])
                deta_u_s = {self._idx(j, i): 1.0, self._idx(j-1, i): -1.0}
                dxi_u_s = {
                    self._idx(j, i+1): 0.25, self._idx(j-1, i+1): 0.25,
                    self._idx(j, i-1): -0.25, self._idx(j-1, i-1): -0.25
                }
                
                # Accumulate term:
                # (1/J) * [ (q11 u_xi + q12 u_eta)_e - (q11 u_xi + q12 u_eta)_w +
                #           (q12 u_xi + q22 u_eta)_n - (q12 u_xi + q22 u_eta)_s ]
                inv_J = 1.0 / self.J[j, i]
                scale = inv_J
                
                # Adds to A[idx, col]
                def add_coeff(flux_dict, mult):
                    for col, val in flux_dict.items():
                        A[idx, col] += scale * mult * val
                        
                add_coeff(dxi_u_e, q11_e)
                add_coeff(deta_u_e, q12_e)
                add_coeff(dxi_u_w, -q11_w)
                add_coeff(deta_u_w, -q12_w)
                
                add_coeff(dxi_u_n, q12_n)
                add_coeff(deta_u_n, q22_n)
                add_coeff(dxi_u_s, -q12_s)
                add_coeff(deta_u_s, -q22_s)
                
        # Identity matrix for M_diag (mass is technically just J in the finite volume sense, 
        # but since we did 1/J above, we are solving du/dt = Laplace(u). So M can be Identity).
        # Actually 1/J was multiplied to the RHS, so du/dt = alpha/J * (...) + S.
        # This implies standard mass matrix is just Identity.
        M_diag = diags(np.ones(self.N), format="csr")
        
        return M_diag, A.tocsr()

    def _bc_func_accepts_normals(self):
        try:
            signature = inspect.signature(self.bc_func)
        except (TypeError, ValueError):
            return True

        for param in signature.parameters.values():
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                return True

        positional_params = [
            param
            for param in signature.parameters.values()
            if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        return len(positional_params) >= 5

    def _evaluate_bc(self, x, y, t, normals=None):
        if normals is not None and self._bc_accepts_normals:
            return self.bc_func(x, y, t, normals[:, 0], normals[:, 1])
        return self.bc_func(x, y, t)

    def _parse_robin_data(self, data, count):
        if isinstance(data, dict):
            beta = data.get("beta")
            value = data.get("value", data.get("gamma", data.get("rhs")))
        else:
            try:
                beta, value = data
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Robin boundary conditions require bc_func to return (beta, value) "
                    "for alpha * du/dn + beta * u = value."
                ) from exc

        if beta is None or value is None:
            raise ValueError("Robin boundary conditions require bc_func to provide both beta and value.")

        beta = np.broadcast_to(np.asarray(beta, dtype=float), (count,))
        value = np.broadcast_to(np.asarray(value, dtype=float), (count,))
        return beta, value

    def _parse_radiative_data(self, data, count):
        if isinstance(data, dict):
            emissivity = data.get("epsilon", data.get("emissivity"))
            t_inf = data.get("t_inf", data.get("ambient_temperature", data.get("T_inf")))
            sigma = data.get("sigma", 5.670374419e-8)
        else:
            try:
                if len(data) == 2:
                    emissivity, t_inf = data
                    sigma = 5.670374419e-8
                else:
                    emissivity, t_inf, sigma = data
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Radiative boundary conditions require bc_func to return "
                    "(epsilon, T_inf) or (epsilon, T_inf, sigma), or a dict with "
                    "'epsilon'/'emissivity' and 't_inf'/'ambient_temperature'."
                ) from exc

        if emissivity is None or t_inf is None:
            raise ValueError("Radiative boundary conditions require both emissivity and ambient temperature.")

        emissivity = np.broadcast_to(np.asarray(emissivity, dtype=float), (count,))
        t_inf = np.broadcast_to(np.asarray(t_inf, dtype=float), (count,))
        sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (count,))
        if np.any(emissivity < 0.0):
            raise ValueError("Radiative emissivity must be non-negative.")
        if np.any(sigma <= 0.0):
            raise ValueError("Stefan-Boltzmann coefficient sigma must be positive.")
        return emissivity, t_inf, sigma

    def _boundary_tangent(self, side, j, i):
        if side in {"left", "right"}:
            if self.ny <= 1:
                return np.array([0.0, 1.0], dtype=float)
            j0 = max(j - 1, 0)
            j1 = min(j + 1, self.ny - 1)
            tangent = np.array(
                [self.X[j1, i] - self.X[j0, i], self.Y[j1, i] - self.Y[j0, i]],
                dtype=float,
            )
        else:
            if self.nx <= 1:
                return np.array([1.0, 0.0], dtype=float)
            i0 = max(i - 1, 0)
            i1 = min(i + 1, self.nx - 1)
            tangent = np.array(
                [self.X[j, i1] - self.X[j, i0], self.Y[j, i1] - self.Y[j, i0]],
                dtype=float,
            )
        if np.linalg.norm(tangent) <= 1e-14:
            return np.array([1.0, 0.0], dtype=float)
        return tangent

    def _boundary_stencil_indices(self, side, j, i):
        candidates = []
        if side == "left":
            j_range = range(max(0, j - 1), min(self.ny, j + 2))
            i_range = range(0, min(self.nx, 3))
        elif side == "right":
            j_range = range(max(0, j - 1), min(self.ny, j + 2))
            i_range = range(max(0, self.nx - 3), self.nx)
        elif side == "bottom":
            j_range = range(0, min(self.ny, 3))
            i_range = range(max(0, i - 1), min(self.nx, i + 2))
        else:
            j_range = range(max(0, self.ny - 3), self.ny)
            i_range = range(max(0, i - 1), min(self.nx, i + 2))

        for jj in j_range:
            for ii in i_range:
                if jj == j and ii == i:
                    continue
                candidates.append((jj, ii))
        return candidates

    def _build_normal_derivative_coeffs(self, side, j, i, normal):
        boundary_point = np.array([self.X[j, i], self.Y[j, i]], dtype=float)
        candidates = self._boundary_stencil_indices(side, j, i)
        offsets = []
        node_ids = []
        for jj, ii in candidates:
            delta = np.array([self.X[jj, ii], self.Y[jj, ii]], dtype=float) - boundary_point
            if np.linalg.norm(delta) <= 1e-14:
                continue
            offsets.append(delta)
            node_ids.append(self._idx(jj, ii))

        if len(offsets) < 2:
            raise ValueError("Curvilinear boundary stencil is too small to reconstruct a normal derivative.")

        D = np.asarray(offsets, dtype=float)
        distances = np.linalg.norm(D, axis=1)
        weights = 1.0 / np.maximum(distances**2, 1e-12)
        weighted_D = D * weights[:, None]
        gram = D.T @ weighted_D
        if np.linalg.matrix_rank(gram) < 2:
            raise ValueError("Curvilinear boundary stencil is rank-deficient; cannot reconstruct a normal derivative.")

        # Weighted least-squares gradient reconstruction exact for linear fields.
        row = normal @ np.linalg.solve(gram, weighted_D.T)
        coeffs = {node_id: float(coeff) for node_id, coeff in zip(node_ids, row)}
        owner_idx = self._idx(j, i)
        coeffs[owner_idx] = coeffs.get(owner_idx, 0.0) - float(np.sum(row))
        return coeffs

    def _append_boundary_component(self, components, side, j, i):
        if side == "left":
            if self.nx < 2:
                return
        elif side == "right":
            if self.nx < 2:
                return
        elif side == "bottom":
            if self.ny < 2:
                return
        else:
            if self.ny < 2:
                return

        boundary_point = np.array([self.X[j, i], self.Y[j, i]], dtype=float)
        tangent = self._boundary_tangent(side, j, i)
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        normal_norm = np.linalg.norm(normal)
        if normal_norm <= 1e-14:
            return
        normal /= normal_norm

        stencil = self._boundary_stencil_indices(side, j, i)
        if not stencil:
            return
        stencil_points = np.asarray([[self.X[jj, ii], self.Y[jj, ii]] for jj, ii in stencil], dtype=float)
        centroid = np.mean(stencil_points, axis=0)
        inward_vec = centroid - boundary_point
        if np.dot(inward_vec, normal) > 0.0:
            normal = -normal

        from .materials import process_alpha

        alpha_face = process_alpha(
            self.alpha,
            np.array([boundary_point[0]], dtype=float),
            np.array([boundary_point[1]], dtype=float),
        )[0]
        alpha_n = float(normal @ alpha_face @ normal)
        owner_idx = self._idx(j, i)
        deriv_coeffs = self._build_normal_derivative_coeffs(side, j, i, normal)
        components.append(
            {
                "owner": owner_idx,
                "x": boundary_point[0],
                "y": boundary_point[1],
                "normal": normal,
                "alpha_n": alpha_n,
                "deriv_coeffs": deriv_coeffs,
            }
        )

    def _build_boundary_data(self):
        components = []
        for j in range(self.ny):
            self._append_boundary_component(components, "left", j, 0)
            if self.nx > 1:
                self._append_boundary_component(components, "right", j, self.nx - 1)
        for i in range(self.nx):
            self._append_boundary_component(components, "bottom", 0, i)
            if self.ny > 1:
                self._append_boundary_component(components, "top", self.ny - 1, i)

        owners = np.asarray([comp["owner"] for comp in components], dtype=int)
        boundary_idx = np.unique(owners)
        return {
            "components": components,
            "owners": owners,
            "boundary_idx": boundary_idx,
            "x_nodes": self.X.flatten()[boundary_idx],
            "y_nodes": self.Y.flatten()[boundary_idx],
        }

    def _apply_dirichlet_rows(self, lhs, rhs, t):
        bc_vals = np.broadcast_to(
            np.asarray(self.bc_func(self._boundary_data["x_nodes"], self._boundary_data["y_nodes"], t), dtype=float),
            self._boundary_idx.shape,
        )
        for row_idx, value in zip(self._boundary_idx, bc_vals):
            lhs.rows[row_idx] = [int(row_idx)]
            lhs.data[row_idx] = [1.0]
            rhs[row_idx] = value

    def _apply_flux_boundary_rows(self, lhs, rhs, t, temperature=None):
        components = self._boundary_data["components"]
        if not components:
            return

        x = np.asarray([comp["x"] for comp in components], dtype=float)
        y = np.asarray([comp["y"] for comp in components], dtype=float)
        normals = np.asarray([comp["normal"] for comp in components], dtype=float)
        owners = np.asarray([comp["owner"] for comp in components], dtype=int)
        alpha_n = np.asarray([comp["alpha_n"] for comp in components], dtype=float)
        deriv_coeffs = [comp["deriv_coeffs"] for comp in components]

        if self.bc_type == "neumann":
            normal_derivative = np.broadcast_to(
                np.asarray(self._evaluate_bc(x, y, t, normals=normals), dtype=float),
                owners.shape,
            )
            bc_rhs = alpha_n * normal_derivative
        elif self.bc_type == "robin":
            beta, value = self._parse_robin_data(
                self._evaluate_bc(x, y, t, normals=normals),
                owners.size,
            )
            bc_rhs = value
        else:
            if temperature is None:
                raise ValueError("Radiative boundary conditions require the current temperature iterate.")
            emissivity, t_inf, sigma = self._parse_radiative_data(
                self._evaluate_bc(x, y, t, normals=normals),
                owners.size,
            )
            t_boundary = np.maximum(np.asarray(temperature, dtype=float)[owners], 0.0)
            coeff = 4.0 * emissivity * sigma * t_boundary**3
            const = 3.0 * emissivity * sigma * t_boundary**4 + emissivity * sigma * np.maximum(t_inf, 0.0) ** 4
            bc_rhs = const

        row_coeffs = {}
        row_rhs = {}
        for comp_idx, (row, rhs_value) in enumerate(zip(owners, bc_rhs)):
            coeffs = row_coeffs.setdefault(int(row), {})
            for col, deriv_coeff in deriv_coeffs[comp_idx].items():
                coeffs[int(col)] = coeffs.get(int(col), 0.0) + float(alpha_n[comp_idx] * deriv_coeff)
            if self.bc_type == "robin":
                coeffs[int(row)] = coeffs.get(int(row), 0.0) + float(beta[comp_idx])
            elif self.bc_type == "radiative":
                coeffs[int(row)] = coeffs.get(int(row), 0.0) + float(coeff[comp_idx])
            row_rhs[int(row)] = row_rhs.get(int(row), 0.0) + float(rhs_value)

        for row, coeffs in row_coeffs.items():
            cols = sorted(coeffs)
            lhs.rows[row] = cols
            lhs.data[row] = [coeffs[col] for col in cols]
            rhs[row] = row_rhs[row]

    def _dirichlet_lhs_csr(self, dt_eff, is_boundary_flat):
        theta = 1.0 if self.time_scheme == "backward_euler" else 0.5
        lhs = (self.M_diag - theta * dt_eff * self.A).copy().tolil()
        for i in np.where(is_boundary_flat)[0]:
            lhs.rows[i] = [i]
            lhs.data[i] = [1.0]
        return lhs.tocsr()

    def _effective_heat_capacity(self, temperature):
        if self.phase_change_model is None:
            return np.ones_like(temperature, dtype=float)
        return np.broadcast_to(
            np.asarray(self.phase_change_model.effective_heat_capacity(temperature), dtype=float),
            temperature.shape,
        )

    def solve(self, u0, t0, t_end):
        u = np.asarray(u0, dtype=float).flatten()
        t = t0
        dts = self._dt_schedule(t0, t_end, self.dt)
        nsteps = len(dts)

        X_flat = self.X.flatten()
        Y_flat = self.Y.flatten()
        theta = 1.0 if self.time_scheme == "backward_euler" else 0.5
        requires_nonlinear = self.phase_change_model is not None or self.bc_type == "radiative"

        factor_by_dt = {}
        if self.reuse_linear_lhs and self.bc_type == "dirichlet" and not requires_nonlinear:
            is_boundary_flat = np.zeros(self.N, dtype=bool)
            is_boundary_flat[self._boundary_idx] = True
            for key in sorted({round(float(d), 12) for d in dts}):
                factor_by_dt[key] = factorized(self._dirichlet_lhs_csr(key, is_boundary_flat))

        for k in range(nsteps):
            t_next = min(t + self.dt, t_end)
            dt_eff = dts[k]
            source_vals = np.broadcast_to(
                np.asarray(self.source_func(X_flat, Y_flat, t_next), dtype=float),
                (self.N,),
            )
            if self.time_scheme == "backward_euler":
                rhs = self.M_diag @ u + dt_eff * source_vals
            else:
                source_prev = np.broadcast_to(
                    np.asarray(self.source_func(X_flat, Y_flat, t), dtype=float),
                    (self.N,),
                )
                rhs = (self.M_diag + 0.5 * dt_eff * self.A) @ u + 0.5 * dt_eff * (source_prev + source_vals)

            if self.bc_type == "dirichlet" and self.reuse_linear_lhs and not requires_nonlinear:
                bc_vals = np.broadcast_to(
                    np.asarray(self.bc_func(self._boundary_data["x_nodes"], self._boundary_data["y_nodes"], t_next), dtype=float),
                    self._boundary_idx.shape,
                )
                dt_key = round(float(dt_eff), 12)
                rhs[self._boundary_idx] = bc_vals
                u = factor_by_dt[dt_key](rhs)
            elif requires_nonlinear:
                max_iters = int(self.nonlinear_options["max_iters"])
                tol = float(self.nonlinear_options["tol"])
                relaxation = float(self.nonlinear_options["relaxation"])
                u_iter = u.copy()
                for _ in range(max_iters):
                    cp_eff = self._effective_heat_capacity(u_iter)
                    mass_diag = diags(cp_eff, format="csr")
                    if self.time_scheme == "backward_euler":
                        rhs_iter = mass_diag @ u + dt_eff * source_vals
                    else:
                        source_prev = np.broadcast_to(
                            np.asarray(self.source_func(X_flat, Y_flat, t), dtype=float),
                            (self.N,),
                        )
                        rhs_iter = (mass_diag + 0.5 * dt_eff * self.A) @ u + 0.5 * dt_eff * (source_prev + source_vals)
                    lhs = (mass_diag - theta * dt_eff * self.A).copy().tolil()
                    if self.bc_type == "dirichlet":
                        self._apply_dirichlet_rows(lhs, rhs_iter, t_next)
                    else:
                        self._apply_flux_boundary_rows(lhs, rhs_iter, t_next, temperature=u_iter)
                    u_raw = spsolve(lhs.tocsr(), rhs_iter)
                    if relaxation != 1.0:
                        u_next = relaxation * u_raw + (1.0 - relaxation) * u_iter
                    else:
                        u_next = u_raw
                    err = np.max(np.abs(u_next - u_iter))
                    scale = max(1.0, np.max(np.abs(u_next)))
                    u_iter = u_next
                    if err <= tol * scale:
                        break
                else:
                    raise RuntimeError(
                        "Radiative boundary solve did not converge. "
                        "Try a smaller dt or larger nonlinear_options['max_iters']."
                    )
                u = u_iter
            else:
                lhs = (self.M_diag - theta * dt_eff * self.A).copy().tolil()
                if self.bc_type == "dirichlet":
                    self._apply_dirichlet_rows(lhs, rhs, t_next)
                else:
                    self._apply_flux_boundary_rows(lhs, rhs, t_next)
                u = spsolve(lhs.tocsr(), rhs)
            t = t_next

        return t, u.reshape((self.ny, self.nx))
