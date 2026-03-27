import numpy as np
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import factorized, spsolve


class CurvilinearHeatSolver:
    """
    Solves the heat equation u_t = div(alpha * grad(u)) + S on a structured curvilinear grid.
    Grid is given by X(eta, xi) and Y(eta, xi) where dimensions are (ny, nx).
    Uses conservative mapped finite differences.
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
        self.bc_type = bc_type
        if self.bc_type != "dirichlet":
            raise ValueError("CurvilinearHeatSolver currently only supports Dirichlet boundary conditions.")
        
        self.N = self.nx * self.ny
        self.J = np.ones((self.ny, self.nx), dtype=float)
        self.M_diag, self.A = self._build_matrices()
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

    def _dirichlet_lhs_csr(self, dt_eff, is_boundary_flat):
        lhs = (self.M_diag - dt_eff * self.A).copy().tolil()
        for i in np.where(is_boundary_flat)[0]:
            lhs.rows[i] = [i]
            lhs.data[i] = [1.0]
        return lhs.tocsr()

    def solve(self, u0, t0, t_end):
        u = np.asarray(u0, dtype=float).flatten()
        t = t0
        dts = self._dt_schedule(t0, t_end, self.dt)
        nsteps = len(dts)

        X_flat = self.X.flatten()
        Y_flat = self.Y.flatten()

        is_boundary = np.zeros((self.ny, self.nx), dtype=bool)
        is_boundary[0, :] = True
        is_boundary[-1, :] = True
        is_boundary[:, 0] = True
        is_boundary[:, -1] = True
        is_boundary_flat = is_boundary.flatten()

        factor_by_dt = {}
        if self.reuse_linear_lhs:
            for key in sorted({round(float(d), 12) for d in dts}):
                factor_by_dt[key] = factorized(self._dirichlet_lhs_csr(key, is_boundary_flat))

        for k in range(nsteps):
            t_next = min(t + self.dt, t_end)
            dt_eff = dts[k]
            rhs = self.M_diag @ u
            source_vals = np.asarray(self.source_func(X_flat, Y_flat, t_next), dtype=float)
            rhs = rhs + dt_eff * source_vals
            bc_vals = self.bc_func(X_flat, Y_flat, t_next)
            if self.reuse_linear_lhs:
                dt_key = round(float(dt_eff), 12)
                for i in np.where(is_boundary_flat)[0]:
                    rhs[i] = bc_vals[i]
                u = factor_by_dt[dt_key](rhs)
            else:
                lhs = (self.M_diag - dt_eff * self.A).copy().tolil()
                for i in np.where(is_boundary_flat)[0]:
                    lhs.rows[i] = [i]
                    lhs.data[i] = [1.0]
                    rhs[i] = bc_vals[i]
                u = spsolve(lhs.tocsr(), rhs)
            t = t_next

        return t, u.reshape((self.ny, self.nx))
