import inspect

import numpy as np
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import bicgstab, cg, spsolve

from .geometry import polygon_area_and_centroid
from .phase_change import ApparentHeatCapacityModel


def _solve_sparse_linear_system(lhs, rhs, method, options, x0=None):
    """
    Solve ``lhs @ x = rhs`` with either a sparse direct factorization or an iterative Krylov method.

    Parameters
    ----------
    lhs : scipy.sparse matrix
    rhs : ndarray
    method : str
        ``"direct"``, ``"bicgstab"``, or ``"cg"``.
    options : dict
        Optional keys: ``rtol``, ``atol``, ``maxiter``.
    x0 : ndarray, optional
        Initial guess for iterative methods (defaults to zeros in SciPy if None).
    """
    if method == "direct":
        return spsolve(lhs, rhs)

    opts = options or {}
    rtol = float(opts.get("rtol", 1e-12))
    atol = float(opts.get("atol", 0.0))
    maxiter = opts.get("maxiter", None)
    if maxiter is not None:
        maxiter = int(maxiter)

    if method == "bicgstab":
        x, info = bicgstab(lhs, rhs, x0=x0, rtol=rtol, atol=atol, maxiter=maxiter)
    elif method == "cg":
        x, info = cg(lhs, rhs, x0=x0, rtol=rtol, atol=atol, maxiter=maxiter)
    else:
        raise ValueError(f"Unknown linear solver method: {method!r}")

    if info != 0:
        raise RuntimeError(
            f"Iterative linear solver {method!r} failed (SciPy info code {info}). "
            "Try tightening rtol/atol, increasing maxiter, or using linear_solver='direct'."
        )
    return x


class PolygonalHeatSolver:
    """
    Finite Volume heat equation solver on a conforming polygonal mesh.
    Cell-centered unknowns with implicit time integration.

    ``time_scheme='backward_euler'`` (default) is first-order accurate in time. ``time_scheme='crank_nicolson'``
    uses the Crank–Nicolson (θ=1/2) trapezoidal step for the semi-discrete system ``M u' + A u = M Q``,
    which is second-order accurate in time for smooth data (spatial accuracy is unchanged).

    Each step solves a sparse linear system; use ``linear_solver='bicgstab'`` or ``'cg'`` for iterative solves.

    ``flux_scheme='tpfa'`` (default) is the two-point flux with optional nonorthogonal correction.
    ``flux_scheme='mpfa'`` uses an MPFA-O style subcell linear FEM (``heat_solver.mpfa``) for the
    diffusion block when ``bc_type='dirichlet'``. For ``bc_type`` Neumann or Robin, the solver uses
    the same two-point TPFA diffusion stencil as ``flux_scheme='tpfa'`` together with
    ``_assemble_boundary_system``, because the condensed MPFA assembly assumes fixed vertex values on
    the boundary trace.

    ``flux_discretization='tpfa'`` (default) uses the classical two-point transmissibility between
    cell centers, optionally with ``nonorthogonal_correction``. ``flux_discretization='reconstructed'``
    (only with ``flux_scheme='tpfa'``) builds the diffusive flux from a face-average of least-squares
    cell gradients, giving a wider stencil that often reduces error on skewed meshes for smooth
    solutions.
    """

    def __init__(
        self,
        vertices,
        polygons,
        alpha,
        dt,
        bc_type="dirichlet",
        bc_func=None,
        source_func=None,
        nonorthogonal_correction=True,
        linear_solver="direct",
        linear_solver_options=None,
        time_scheme="backward_euler",
        flux_scheme="tpfa",
        flux_discretization="tpfa",
        phase_change_model=None,
        phase_change_options=None,
        temperature_dependent_diffusivity=None,
        nonlinear_options=None,
    ):
        self.vertices = np.asarray(vertices, dtype=float)
        self.polygons = [list(poly) for poly in polygons]
        self.alpha = alpha
        self.dt = float(dt)
        self.bc_type = str(bc_type).lower()
        self.bc_func = bc_func if bc_func is not None else (lambda x, y, t: np.zeros_like(x, dtype=float))
        self.source_func = source_func if source_func is not None else (lambda x, y, t: np.zeros_like(x, dtype=float))
        if self.bc_type not in {"dirichlet", "neumann", "robin", "radiative"}:
            raise ValueError("bc_type must be one of: dirichlet, neumann, robin, radiative")
        self.nonorthogonal_correction = bool(nonorthogonal_correction)
        self.linear_solver = str(linear_solver).lower().strip()
        if self.linear_solver not in {"direct", "bicgstab", "cg"}:
            raise ValueError(
                "linear_solver must be one of: 'direct', 'bicgstab', 'cg' "
                "(use 'bicgstab' for general meshes; 'cg' only if the assembled system is symmetric positive definite)."
            )
        self.linear_solver_options = dict(linear_solver_options) if linear_solver_options else {}
        self.time_scheme = str(time_scheme).lower().strip()
        if self.time_scheme not in {"backward_euler", "crank_nicolson"}:
            raise ValueError("time_scheme must be 'backward_euler' or 'crank_nicolson'.")
        self.flux_scheme = str(flux_scheme).lower().strip()
        if self.flux_scheme not in {"tpfa", "mpfa"}:
            raise ValueError("flux_scheme must be 'tpfa' or 'mpfa'.")
        self.flux_discretization = str(flux_discretization).lower().strip()
        if self.flux_discretization not in {"tpfa", "reconstructed"}:
            raise ValueError("flux_discretization must be 'tpfa' or 'reconstructed'.")
        if self.flux_scheme == "mpfa" and self.flux_discretization == "reconstructed":
            raise ValueError("flux_discretization='reconstructed' is only supported with flux_scheme='tpfa'.")
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
        # Backward-compatible alias retained for existing callers/tests.
        self.phase_change_options = self.nonlinear_options
        self.M = len(self.polygons)
        self.cell_centers = self._compute_cell_centers()
        self.cell_areas = self._compute_cell_areas()
        self.edge_to_cells = self._build_edge_to_cells()
        self.boundary_faces = self._build_boundary_faces()
        self.neighbors = self._build_neighbors()
        self.is_boundary = self._detect_boundary_cells()
        self._bc_accepts_normals = self._bc_func_accepts_normals()
        self._alpha_accepts_temperature = self._alpha_func_accepts_temperature()
        self.temperature_dependent_diffusivity = (
            self._alpha_accepts_temperature
            if temperature_dependent_diffusivity is None
            else bool(temperature_dependent_diffusivity)
        )
        if self.temperature_dependent_diffusivity and self.flux_scheme == "mpfa":
            raise ValueError("temperature-dependent diffusivity is currently supported only with flux_scheme='tpfa'.")
        need_tpfa_gradients = (
            (self.flux_scheme == "tpfa" and self.flux_discretization == "reconstructed")
            or (
                self.flux_scheme == "tpfa"
                and self.flux_discretization == "tpfa"
                and self.nonorthogonal_correction
            )
            or (self.flux_scheme == "mpfa" and self.bc_type != "dirichlet")
        )
        self.gradient_coeffs = self._build_gradient_reconstruction() if need_tpfa_gradients else None
        self.M_diag = None
        self.A = None
        self.u = np.zeros(self.M)

    def _effective_heat_capacity(self, temperature):
        if self.phase_change_model is None:
            return np.ones_like(temperature, dtype=float)
        return np.broadcast_to(
            np.asarray(self.phase_change_model.effective_heat_capacity(temperature), dtype=float),
            temperature.shape,
        )

    def _build_mass_matrix(self, heat_capacity):
        return diags(self.cell_areas * heat_capacity, format="csr")

    def _compute_cell_centers(self):
        return np.array([polygon_area_and_centroid(self.vertices[poly])[1] for poly in self.polygons])

    def _compute_cell_areas(self):
        return np.array([polygon_area_and_centroid(self.vertices[poly])[0] for poly in self.polygons])

    def _build_edge_to_cells(self):
        edge_to_cells = {}
        for idx, poly in enumerate(self.polygons):
            for i in range(len(poly)):
                edge = tuple(sorted((poly[i], poly[(i + 1) % len(poly)])))
                edge_to_cells.setdefault(edge, []).append(idx)
        return edge_to_cells

    def _build_boundary_faces(self):
        cells = []
        midpoints = []
        normals = []
        lengths = []
        distances = []
        verts = self.vertices

        for cell_idx, poly in enumerate(self.polygons):
            center = self.cell_centers[cell_idx]
            for i in range(len(poly)):
                a = poly[i]
                b = poly[(i + 1) % len(poly)]
                edge = tuple(sorted((a, b)))
                if len(self.edge_to_cells[edge]) != 1:
                    continue

                v0 = verts[a]
                v1 = verts[b]
                edge_vec = v1 - v0
                edge_len = np.linalg.norm(edge_vec)
                if edge_len <= 1e-14:
                    continue

                tangent = edge_vec / edge_len
                normal = np.array([-tangent[1], tangent[0]])
                midpoint = 0.5 * (v0 + v1)
                if np.dot(midpoint - center, normal) < 0:
                    normal = -normal

                distance = abs(np.dot(midpoint - center, normal))
                if distance <= 1e-12:
                    distance = max(np.linalg.norm(midpoint - center), 1e-12)

                cells.append(cell_idx)
                midpoints.append(midpoint)
                normals.append(normal)
                lengths.append(edge_len)
                distances.append(distance)

        if not cells:
            return {
                "cells": np.zeros(0, dtype=int),
                "midpoints": np.zeros((0, 2), dtype=float),
                "normals": np.zeros((0, 2), dtype=float),
                "lengths": np.zeros(0, dtype=float),
                "distances": np.zeros(0, dtype=float),
            }

        return {
            "cells": np.asarray(cells, dtype=int),
            "midpoints": np.asarray(midpoints, dtype=float),
            "normals": np.asarray(normals, dtype=float),
            "lengths": np.asarray(lengths, dtype=float),
            "distances": np.asarray(distances, dtype=float),
        }

    def _build_neighbors(self):
        neighbors = [[] for _ in range(self.M)]
        for cells in self.edge_to_cells.values():
            if len(cells) == 2:
                i, j = cells
                neighbors[i].append(j)
                neighbors[j].append(i)
        return [sorted(set(nbs)) for nbs in neighbors]

    def _detect_boundary_cells(self):
        is_boundary = np.zeros(self.M, dtype=bool)
        if self.boundary_faces["cells"].size:
            is_boundary[self.boundary_faces["cells"]] = True
        return is_boundary

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

    def _alpha_func_accepts_temperature(self):
        if not callable(self.alpha):
            return False
        try:
            signature = inspect.signature(self.alpha)
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
        return len(positional_params) >= 3

    def _evaluate_bc(self, x, y, t, normals=None):
        if normals is not None and self._bc_accepts_normals:
            return self.bc_func(x, y, t, normals[:, 0], normals[:, 1])
        return self.bc_func(x, y, t)

    def _process_alpha(self, x, y, temperature=None):
        from .materials import process_alpha

        temp = temperature if self.temperature_dependent_diffusivity else None
        return process_alpha(self.alpha, x, y, temperature=temp)

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
            raise ValueError(
                "Robin boundary conditions require bc_func to provide both beta and value."
            )

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
            raise ValueError(
                "Radiative boundary conditions require both emissivity and ambient temperature."
            )

        emissivity = np.broadcast_to(np.asarray(emissivity, dtype=float), (count,))
        t_inf = np.broadcast_to(np.asarray(t_inf, dtype=float), (count,))
        sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (count,))
        if np.any(emissivity < 0.0):
            raise ValueError("Radiative emissivity must be non-negative.")
        if np.any(sigma <= 0.0):
            raise ValueError("Stefan-Boltzmann coefficient sigma must be positive.")
        return emissivity, t_inf, sigma

    def _assemble_boundary_system(self, t, temperature=None):
        cells = self.boundary_faces["cells"]
        if cells.size == 0:
            return diags(np.zeros(self.M), format="csr"), np.zeros(self.M)

        lengths = self.boundary_faces["lengths"]
        distances = self.boundary_faces["distances"]
        midpoints = self.boundary_faces["midpoints"]
        normals = self.boundary_faces["normals"]

        boundary_temperature = None if temperature is None else np.asarray(temperature, dtype=float)[cells]
        Alpha_faces = self._process_alpha(midpoints[:, 0], midpoints[:, 1], temperature=boundary_temperature)
        alpha_n = np.einsum("ij,ij->i", normals, np.einsum("ijk,ik->ij", Alpha_faces, normals))

        if self.bc_type == "neumann":
            normal_derivative = np.broadcast_to(
                np.asarray(
                    self._evaluate_bc(midpoints[:, 0], midpoints[:, 1], t, normals=normals),
                    dtype=float,
                ),
                (cells.size,),
            )
            rhs = np.bincount(cells, weights=alpha_n * lengths * normal_derivative, minlength=self.M)
            return diags(np.zeros(self.M), format="csr"), rhs

        if self.bc_type == "radiative":
            if temperature is None:
                raise ValueError("Radiative boundary assembly requires the current temperature iterate.")
            emissivity, t_inf, sigma = self._parse_radiative_data(
                self._evaluate_bc(midpoints[:, 0], midpoints[:, 1], t, normals=normals),
                cells.size,
            )
            t_face = np.asarray(temperature, dtype=float)[cells]
            # Tangent linearization around current iterate:
            # q(T) = eps*sigma*(T^4 - T_inf^4) ≈ a*T - b,
            # a = 4*eps*sigma*T_k^3, b = 3*eps*sigma*T_k^4 + eps*sigma*T_inf^4
            # and alpha*du/dn = -q contributes +a*T on lhs and +b on rhs.
            coeff = 4.0 * emissivity * sigma * np.maximum(t_face, 0.0) ** 3
            const = 3.0 * emissivity * sigma * np.maximum(t_face, 0.0) ** 4 + emissivity * sigma * np.maximum(t_inf, 0.0) ** 4
            diag = np.bincount(cells, weights=lengths * coeff, minlength=self.M)
            rhs = np.bincount(cells, weights=lengths * const, minlength=self.M)
            return diags(diag, format="csr"), rhs

        robin_beta, robin_value = self._parse_robin_data(
            self._evaluate_bc(midpoints[:, 0], midpoints[:, 1], t, normals=normals),
            cells.size,
        )
        denom = alpha_n + robin_beta * distances
        if np.any(np.abs(denom) <= 1e-14):
            raise ValueError("Robin boundary condition is singular because alpha + beta * d is too small.")

        effective_beta = alpha_n * robin_beta / denom
        effective_value = alpha_n * robin_value / denom
        diag = np.bincount(cells, weights=lengths * effective_beta, minlength=self.M)
        rhs = np.bincount(cells, weights=lengths * effective_value, minlength=self.M)
        return diags(diag, format="csr"), rhs

    def _build_gradient_reconstruction(self):
        coeffs = []
        for i in range(self.M):
            nbs = self.neighbors[i]
            if len(nbs) < 2:
                coeffs.append({i: np.zeros(2)})
                continue
            offsets = self.cell_centers[nbs] - self.cell_centers[i]
            distances = np.linalg.norm(offsets, axis=1)
            weights = 1.0 / np.maximum(distances**2, 1e-12)
            weighted_offsets = offsets * weights[:, None]
            normal_matrix = offsets.T @ weighted_offsets
            if np.linalg.matrix_rank(normal_matrix) < 2:
                coeffs.append({i: np.zeros(2)})
                continue
            recon = np.linalg.solve(normal_matrix, weighted_offsets.T)
            cell_coeffs = {}
            for nb, vec in zip(nbs, recon.T):
                cell_coeffs[nb] = vec
            cell_coeffs[i] = -np.sum(recon.T, axis=0)
            coeffs.append(cell_coeffs)
        return coeffs

    def _assemble_diffusion_reconstructed(self, temperature=None):
        """
        Diffusion matrix from face flux F = -edge_len * n^T K grad_face u, with
        grad_face = 0.5 * (grad u_i + grad u_j) and cell gradients from least squares.
        """
        if self.gradient_coeffs is None:
            raise RuntimeError("flux_discretization='reconstructed' requires gradient coefficients.")

        diffusion = lil_matrix((self.M, self.M))
        centers = self.cell_centers
        verts = self.vertices
        for edge, cells in self.edge_to_cells.items():
            if len(cells) != 2:
                continue
            i, j = cells
            if j < i:
                i, j = j, i
            ci = centers[i]
            cj = centers[j]
            v0, v1 = verts[edge[0]], verts[edge[1]]
            edge_vec = v1 - v0
            edge_len = np.linalg.norm(edge_vec)
            if edge_len == 0:
                continue
            tangent = edge_vec / edge_len
            normal = np.array([-tangent[1], tangent[0]])
            d_vec = cj - ci
            dn = np.dot(d_vec, normal)
            if dn < 0:
                normal = -normal
                dn = -dn
            if dn <= 1e-12:
                dn = max(np.linalg.norm(d_vec), 1e-12)

            midpoint = 0.5 * (v0 + v1)
            face_temperature = None
            if temperature is not None:
                face_temperature = 0.5 * (temperature[i] + temperature[j])
            Alpha_face = self._process_alpha(midpoint[0], midpoint[1], temperature=face_temperature)
            gi = self.gradient_coeffs[i]
            gj = self.gradient_coeffs[j]
            idxs = set(gi.keys()) | set(gj.keys())

            any_flux = False
            for k in idxs:
                wi = gi.get(k)
                if wi is None:
                    wi = np.zeros(2, dtype=float)
                else:
                    wi = np.asarray(wi, dtype=float)
                wj = gj.get(k)
                if wj is None:
                    wj = np.zeros(2, dtype=float)
                else:
                    wj = np.asarray(wj, dtype=float)
                wsum = wi + wj
                if np.linalg.norm(wsum) <= 1e-30:
                    continue
                tk = -0.5 * edge_len * float(np.dot(normal, Alpha_face @ wsum))
                if abs(tk) <= 1e-30:
                    continue
                any_flux = True
                diffusion[i, k] += tk
                diffusion[j, k] -= tk

            if not any_flux:
                alpha_n = np.dot(normal, Alpha_face @ normal)
                base = alpha_n * edge_len / dn
                diffusion[i, i] += base
                diffusion[i, j] -= base
                diffusion[j, i] -= base
                diffusion[j, j] += base

        self.A = diffusion.tocsr()

    def _assemble_system(self, temperature=None):
        mass = diags(self.cell_areas, format="csr")
        # MPFA condensed FEM is only used with Dirichlet: Neumann/Robin need a flux closure
        # consistent with unprescribed boundary values; use the same TPFA two-point stencil as
        # ``flux_scheme='tpfa'`` for the diffusion block in that case.
        if self.flux_scheme == "mpfa" and self.bc_type == "dirichlet":
            from .mpfa import assemble_mpfa_diffusion

            self.M_diag = mass
            self.A = assemble_mpfa_diffusion(
                self.vertices,
                self.polygons,
                self.cell_centers,
                self.alpha,
                self.edge_to_cells,
            )
            return

        if self.flux_discretization == "reconstructed":
            self.M_diag = mass
            self._assemble_diffusion_reconstructed(temperature=temperature)
            return

        diffusion = lil_matrix((self.M, self.M))
        centers = self.cell_centers
        verts = self.vertices
        for edge, cells in self.edge_to_cells.items():
            if len(cells) != 2:
                continue
            i, j = cells
            if j < i:
                i, j = j, i
            ci = centers[i]
            cj = centers[j]
            v0, v1 = verts[edge[0]], verts[edge[1]]
            edge_vec = v1 - v0
            edge_len = np.linalg.norm(edge_vec)
            if edge_len == 0:
                continue
            tangent = edge_vec / edge_len
            normal = np.array([-tangent[1], tangent[0]])
            d_vec = cj - ci
            dn = np.dot(d_vec, normal)
            if dn < 0:
                normal = -normal
                dn = -dn
            if dn <= 1e-12:
                dn = max(np.linalg.norm(d_vec), 1e-12)
            
            midpoint = 0.5 * (v0 + v1)
            face_temperature = None
            if temperature is not None:
                face_temperature = 0.5 * (temperature[i] + temperature[j])
            Alpha_face = self._process_alpha(midpoint[0], midpoint[1], temperature=face_temperature)
            alpha_n = np.dot(normal, Alpha_face @ normal)
            base = alpha_n * edge_len / dn
            diffusion[i, i] += base
            diffusion[i, j] -= base
            diffusion[j, i] -= base
            diffusion[j, j] += base

            if self.nonorthogonal_correction:
                v = Alpha_face @ normal
                correction_direction = v - alpha_n * d_vec / dn
                if np.linalg.norm(correction_direction) <= 1e-14:
                    continue
                face_coeffs = {}
                for cell in (i, j):
                    for idx, grad_coeff in self.gradient_coeffs[cell].items():
                        face_coeffs[idx] = face_coeffs.get(idx, 0.0) + 0.5 * np.dot(correction_direction, grad_coeff)
                correction_scale = edge_len
                for idx, coeff in face_coeffs.items():
                    diffusion[i, idx] -= correction_scale * coeff
                    diffusion[j, idx] += correction_scale * coeff
        self.M_diag = mass
        self.A = diffusion.tocsr()

    def solve(self, u0, t0, t_end):
        u = np.array(u0, dtype=float)
        t = t0
        nsteps = int(np.ceil((t_end - t0) / self.dt))
        has_temp_dependent_diffusion = bool(self.temperature_dependent_diffusivity)
        has_nonlinear_radiation = self.bc_type == "radiative"
        requires_nonlinear = (
            self.phase_change_model is not None
            or has_temp_dependent_diffusion
            or has_nonlinear_radiation
        )

        if not has_temp_dependent_diffusion:
            self._assemble_system()

        for _ in range(nsteps):
            t_next = min(t + self.dt, t_end)
            dt_eff = t_next - t
            cx = self.cell_centers[:, 0]
            cy = self.cell_centers[:, 1]
            source_next = np.broadcast_to(
                np.asarray(self.source_func(cx, cy, t_next), dtype=float),
                (self.M,),
            )

            if not requires_nonlinear:
                if self.time_scheme == "backward_euler":
                    rhs = self.M_diag @ u + dt_eff * (self.cell_areas * source_next)
                    if self.bc_type == "dirichlet":
                        bc_vals = self.bc_func(cx, cy, t_next)
                        lhs = (self.M_diag + dt_eff * self.A).copy().tolil()
                        for i in np.where(self.is_boundary)[0]:
                            lhs.rows[i] = [i]
                            lhs.data[i] = [1.0]
                            rhs[i] = bc_vals[i]
                        lhs = lhs.tocsr()
                    else:
                        boundary_matrix, boundary_rhs = self._assemble_boundary_system(t_next)
                        lhs = self.M_diag + dt_eff * (self.A + boundary_matrix)
                        rhs = rhs + dt_eff * boundary_rhs
                else:
                    source_prev = np.broadcast_to(
                        np.asarray(self.source_func(cx, cy, t), dtype=float),
                        (self.M,),
                    )
                    src_avg = self.cell_areas * (source_next + source_prev)
                    if self.bc_type == "dirichlet":
                        rhs = (self.M_diag - 0.5 * dt_eff * self.A) @ u + 0.5 * dt_eff * src_avg
                        bc_vals = self.bc_func(cx, cy, t_next)
                        for i in np.where(self.is_boundary)[0]:
                            rhs[i] = bc_vals[i]
                        lhs = (self.M_diag + 0.5 * dt_eff * self.A).copy().tolil()
                        for i in np.where(self.is_boundary)[0]:
                            lhs.rows[i] = [i]
                            lhs.data[i] = [1.0]
                        lhs = lhs.tocsr()
                    else:
                        b_mat_prev, b_rhs_prev = self._assemble_boundary_system(t)
                        b_mat_next, b_rhs_next = self._assemble_boundary_system(t_next)
                        k_prev = self.A + b_mat_prev
                        k_next = self.A + b_mat_next
                        rhs = (self.M_diag - 0.5 * dt_eff * k_prev) @ u + 0.5 * dt_eff * src_avg
                        rhs = rhs + 0.5 * dt_eff * (b_rhs_prev + b_rhs_next)
                        lhs = self.M_diag + 0.5 * dt_eff * k_next

                x0 = u if self.linear_solver != "direct" else None
                u = _solve_sparse_linear_system(
                    lhs,
                    rhs,
                    self.linear_solver,
                    self.linear_solver_options,
                    x0=x0,
                )
                t = t_next
                continue

            max_iters = int(self.nonlinear_options["max_iters"])
            tol = float(self.nonlinear_options["tol"])
            relaxation = float(self.nonlinear_options["relaxation"])
            u_iter = u.copy()

            source_prev = None
            src_avg = None
            if self.time_scheme == "crank_nicolson":
                source_prev = np.broadcast_to(
                    np.asarray(self.source_func(cx, cy, t), dtype=float),
                    (self.M,),
                )
                src_avg = self.cell_areas * (source_next + source_prev)

            for _ in range(max_iters):
                if has_temp_dependent_diffusion:
                    self._assemble_system(temperature=u_iter)
                cp_eff = self._effective_heat_capacity(u_iter)
                mass_matrix = self._build_mass_matrix(cp_eff)

                if self.time_scheme == "backward_euler":
                    rhs = mass_matrix @ u + dt_eff * (self.cell_areas * source_next)
                    if self.bc_type == "dirichlet":
                        bc_vals = self.bc_func(cx, cy, t_next)
                        lhs = (mass_matrix + dt_eff * self.A).copy().tolil()
                        for i in np.where(self.is_boundary)[0]:
                            lhs.rows[i] = [i]
                            lhs.data[i] = [1.0]
                            rhs[i] = bc_vals[i]
                        lhs = lhs.tocsr()
                    else:
                        boundary_matrix, boundary_rhs = self._assemble_boundary_system(t_next, temperature=u_iter)
                        lhs = mass_matrix + dt_eff * (self.A + boundary_matrix)
                        rhs = rhs + dt_eff * boundary_rhs
                elif self.bc_type == "dirichlet":
                    rhs = (mass_matrix - 0.5 * dt_eff * self.A) @ u + 0.5 * dt_eff * src_avg
                    bc_vals = self.bc_func(cx, cy, t_next)
                    for i in np.where(self.is_boundary)[0]:
                        rhs[i] = bc_vals[i]
                    lhs = (mass_matrix + 0.5 * dt_eff * self.A).copy().tolil()
                    for i in np.where(self.is_boundary)[0]:
                        lhs.rows[i] = [i]
                        lhs.data[i] = [1.0]
                    lhs = lhs.tocsr()
                else:
                    if has_temp_dependent_diffusion or has_nonlinear_radiation:
                        b_mat_next, b_rhs_next = self._assemble_boundary_system(t_next, temperature=u_iter)
                        k_next = self.A + b_mat_next
                        rhs = (mass_matrix - 0.5 * dt_eff * k_next) @ u + 0.5 * dt_eff * src_avg
                        rhs = rhs + dt_eff * b_rhs_next
                        lhs = mass_matrix + 0.5 * dt_eff * k_next
                    else:
                        b_mat_prev, b_rhs_prev = self._assemble_boundary_system(t, temperature=u)
                        b_mat_next, b_rhs_next = self._assemble_boundary_system(t_next, temperature=u_iter)
                        k_prev = self.A + b_mat_prev
                        k_next = self.A + b_mat_next
                        rhs = (mass_matrix - 0.5 * dt_eff * k_prev) @ u + 0.5 * dt_eff * src_avg
                        rhs = rhs + 0.5 * dt_eff * (b_rhs_prev + b_rhs_next)
                        lhs = mass_matrix + 0.5 * dt_eff * k_next

                x0 = u_iter if self.linear_solver != "direct" else None
                u_next = _solve_sparse_linear_system(
                    lhs,
                    rhs,
                    self.linear_solver,
                    self.linear_solver_options,
                    x0=x0,
                )
                if relaxation != 1.0:
                    u_next = relaxation * u_next + (1.0 - relaxation) * u_iter
                err = np.max(np.abs(u_next - u_iter))
                scale = max(1.0, np.max(np.abs(u_next)))
                u_iter = u_next
                if err <= tol * scale:
                    break
            else:
                raise RuntimeError(
                    "Nonlinear solve did not converge. "
                    "Try a smaller dt or larger nonlinear_options['max_iters']."
                )

            u = u_iter
            t = t_next
        return t, u
