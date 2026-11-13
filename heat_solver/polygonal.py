import inspect

import numpy as np
from scipy.sparse import coo_matrix, diags, lil_matrix
from scipy.sparse.linalg import bicgstab, cg, factorized, spsolve

from .factor_cache import FactorCache, frozen_newton
from .linear_solvers import KRYLOV_METHODS, IterativeSolver, choose_linear_solver
from .fv_assembly import assemble_reconstructed, assemble_tpfa, flatten_stencils, identity_rows, interior_faces
from .geometry import polygon_area_and_centroid
from .phase_change import ApparentHeatCapacityModel


def _solve_sparse_linear_system(lhs, rhs, method, options, x0=None, direct_factor=None):
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
    direct_factor : callable, optional
        If ``method == "direct"``, optional ``factorized(lhs)`` from SciPy to reuse the numeric
        factorization when the matrix is fixed across steps.
    """
    if method == "direct":
        if direct_factor is not None:
            return direct_factor(rhs)
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


class _AndersonAccelerator:
    """Anderson acceleration (type-I) for fixed-point iterations.

    Stores a sliding window of iterates and their fixed-point residuals,
    then computes an optimized linear combination via least-squares
    to accelerate convergence of the Picard map.
    """

    __slots__ = ("_depth", "_g_hist", "_f_hist")

    def __init__(self, depth):
        self._depth = max(int(depth), 1)
        self._g_hist = []
        self._f_hist = []

    def step(self, x_k, g_k):
        """Return accelerated iterate.  *x_k* is the current guess,
        *g_k* = G(x_k) is the unaccelerated fixed-point evaluation."""
        f_k = g_k - x_k
        self._g_hist.append(g_k.copy())
        self._f_hist.append(f_k.copy())
        k = len(self._f_hist)
        if k < 2:
            return g_k.copy()
        m_k = min(k - 1, self._depth)
        dF = np.empty((f_k.size, m_k))
        dG = np.empty((f_k.size, m_k))
        for j in range(m_k):
            idx = k - 1 - j
            dF[:, j] = self._f_hist[idx] - self._f_hist[idx - 1]
            dG[:, j] = self._g_hist[idx] - self._g_hist[idx - 1]
        gamma, _, _, _ = np.linalg.lstsq(dF, f_k, rcond=None)
        x_new = g_k - dG @ gamma
        excess = k - (self._depth + 1)
        if excess > 0:
            del self._g_hist[:excess]
            del self._f_hist[:excess]
        return x_new

    def reset(self):
        self._g_hist.clear()
        self._f_hist.clear()


def _apply_dirichlet_rows_csr(K, boundary_indices):
    """Zero boundary rows in a CSR matrix and set their diagonal to 1.0 (in-place, pattern kept)."""
    rows = np.asarray(boundary_indices, dtype=np.intp)
    if rows.size == 0:
        return
    starts = K.indptr[rows]
    counts = K.indptr[rows + 1] - starts
    entry = np.repeat(starts, counts) + (np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts))
    K.data[entry] = 0.0
    K.data[entry[K.indices[entry] == np.repeat(rows, counts)]] = 1.0


class PolygonalHeatSolver:
    """
    Finite Volume heat equation solver on a conforming polygonal mesh.
    Cell-centered unknowns with implicit time integration.

    ``time_scheme='backward_euler'`` (default) is first-order accurate in time. ``time_scheme='crank_nicolson'``
    uses the Crank–Nicolson (θ=1/2) trapezoidal step for the semi-discrete system ``M u' + A u = M Q``,
    which is second-order accurate in time for smooth data (spatial accuracy is unchanged).
    ``time_scheme='bdf2'`` is the variable-step second-order backward difference formula (first step
    backward Euler): second order like Crank–Nicolson but strongly damping stiff modes, so it does not
    ring on rough initial or boundary data.  It is the default of the verification drivers.

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
    cell gradients.  It has no compact two-point coupling and only linearly exact gradients, so it
    loses second order on sheared meshes (order ~0.8 measured); kept for reproducibility.

    ``flux_discretization='quadratic'`` (recommended for non-orthogonal / anisotropic problems) uses a
    2-exact least-squares reconstruction on a vertex-ring stencil, face-midpoint gradients and a
    consistent compact jump penalty (``heat_solver.quadratic_flux``).  It is exact for quadratic
    solutions on arbitrary polygonal meshes and second order on skewed and hexagonal meshes.  Its
    unknowns are cell averages, so source and Dirichlet data are integrated over cells by default
    (``cell_average_data``); use :meth:`cell_average` for initial data.
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
        reuse_linear_lhs=True,
        monotone=False,
        cell_average_data=None,
        dirichlet_mode="band",
    ):
        from .mesh_io import Mesh2D

        if isinstance(vertices, Mesh2D):
            raise TypeError("pass a Mesh2D unpacked: PolygonalHeatSolver(*mesh, alpha, dt, ...).")
        self.vertices = np.asarray(vertices, dtype=float)
        self.polygons = [list(poly) for poly in polygons]
        self.alpha = alpha
        self.dt = float(dt)
        self.bc_type = str(bc_type).lower()
        self.bc_func = bc_func if bc_func is not None else (lambda x, y, t: np.zeros_like(x, dtype=float))
        self.source_func = source_func if source_func is not None else (lambda x, y, t: np.zeros_like(x, dtype=float))
        from .boundary_conditions import normalize_bc_type

        self.bc_type = normalize_bc_type(self.bc_type)
        self.dirichlet_mode = str(dirichlet_mode).lower().strip()
        if self.dirichlet_mode not in ("band", "face"):
            raise ValueError("dirichlet_mode must be 'band' or 'face'.")
        # "band": boundary cells are held at the data (u_c = g at the centroid, legacy).
        # "face": every cell is an unknown and g is imposed on the boundary faces through
        # the face flux alpha_n (g - u_c) / d, as FiPy does (the Robin closure with beta -> inf).
        self._band_dirichlet = self.bc_type == "dirichlet" and self.dirichlet_mode == "band"
        self.nonorthogonal_correction = bool(nonorthogonal_correction)
        self.monotone = bool(monotone)
        self.linear_solver = str(linear_solver).lower().strip()
        if self.linear_solver not in {"direct", "auto", *KRYLOV_METHODS}:
            raise ValueError(
                "linear_solver must be one of: 'direct', 'auto', 'cg', 'bicgstab', 'gmres' "
                "(use 'bicgstab' or 'gmres' for general meshes; 'cg' only if the assembled system is symmetric "
                "positive definite; 'auto' chooses by problem size and stiffness)."
            )
        self.linear_solver_options = dict(linear_solver_options) if linear_solver_options else {}
        pc = self.linear_solver_options.get("preconditioner")
        self.linear_solver_options["preconditioner"] = None if pc in (None, "none") else str(pc).lower()
        if self.linear_solver_options["preconditioner"] not in (None, "jacobi", "amg"):
            raise ValueError("linear_solver_options['preconditioner'] must be None, 'jacobi' or 'amg'.")
        self._iterative = None  # IterativeSolver reused across solve() calls (keeps preconditioners)
        self._auto_plans = {}
        self.time_scheme = str(time_scheme).lower().strip()
        if self.time_scheme not in {"backward_euler", "crank_nicolson", "bdf2"}:
            raise ValueError("time_scheme must be 'backward_euler', 'crank_nicolson' or 'bdf2'.")
        self.flux_scheme = str(flux_scheme).lower().strip()
        if self.flux_scheme not in {"tpfa", "mpfa"}:
            raise ValueError("flux_scheme must be 'tpfa' or 'mpfa'.")
        self.flux_discretization = str(flux_discretization).lower().strip()
        if self.flux_discretization not in {"tpfa", "reconstructed", "quadratic"}:
            raise ValueError("flux_discretization must be 'tpfa', 'reconstructed' or 'quadratic'.")
        if self.flux_scheme == "mpfa" and self.flux_discretization != "tpfa":
            raise ValueError(f"flux_discretization={self.flux_discretization!r} is only supported with flux_scheme='tpfa'.")
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
        # Derived geometry comes from the mesh-content cache (populated by
        # heat_solver.mesh_io when a mesh is loaded from file, or computed once,
        # vectorized, on first use) so repeated problems on one mesh skip it.
        from .mesh_io import geometry_for

        self._mesh_key, self._geometry = geometry_for(self.vertices, self.polygons)
        geo = self._geometry
        self.cell_centers = geo.cell_centers
        self.cell_areas = geo.cell_areas
        self.edge_to_cells = geo.edge_to_cells
        self.boundary_faces = dict(geo.boundary_faces)
        self.neighbors = geo.neighbors
        self.is_boundary = geo.is_boundary
        # Cell-average data: integrate the source (and Dirichlet data) over each
        # cell instead of sampling at the centroid.  Default on for the
        # quadratic flux (whose 2-exact reconstruction needs consistent
        # averages), off otherwise so existing schemes are unchanged.
        if cell_average_data is None:
            cell_average_data = self.flux_discretization == "quadratic"
        self.cell_average_data = bool(cell_average_data)
        if self.cell_average_data and self.bc_type == "dirichlet" and not self._band_dirichlet:
            raise ValueError(
                "dirichlet_mode='face' is not available with cell-averaged unknowns "
                "(flux_discretization='quadratic' or cell_average_data=True): the face closure treats the "
                "cell value as a point value, which is first-order accurate there. Use dirichlet_mode='band'.")
        self._cell_quadrature = None
        if self.cell_average_data:
            from .quadratic_flux import CellAveragedFunction, CellQuadrature

            self._cell_quadrature = CellQuadrature(self.vertices, self.polygons)
            self.source_func = CellAveragedFunction(self.source_func, self._cell_quadrature, self.cell_centers)
            if self._band_dirichlet:  # band data are cell values; face data stay point values
                self.bc_func = CellAveragedFunction(self.bc_func, self._cell_quadrature, self.cell_centers)
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
            or (self.flux_scheme == "mpfa" and not self._band_dirichlet)
            or (self.flux_discretization == "quadratic" and self.bc_type in ("robin", "radiative"))
        )
        self.gradient_coeffs = None
        if need_tpfa_gradients:
            self.gradient_coeffs = geo.extras.get("gradient_coeffs")
            if self.gradient_coeffs is None:
                self.gradient_coeffs = self._build_gradient_reconstruction()
                geo.extras["gradient_coeffs"] = self.gradient_coeffs
        self.M_diag = None
        self.A = None
        self.u = np.zeros(self.M)
        self.reuse_linear_lhs = bool(reuse_linear_lhs)
        self._vec_assembly = None
        self._constant_A = None  # operator from the last temperature-independent assembly (reused by solve())
        self._linear_cache = {}  # (dt, time-scheme coefficients) -> (lhs, factor), valid for _linear_cache_A
        self._linear_cache_A = None
        # Factorizations of matrices with a boundary closure (Robin / radiative), reused
        # while the operator and the closure are unchanged (heat_solver/factor_cache.py).
        self._factor_cache = FactorCache()
        self.last_solve_stats = {}
        self._boundary_idx = np.where(self.is_boundary)[0]
        if (
            self.temperature_dependent_diffusivity
            and self.flux_scheme == "tpfa"
            and self.flux_discretization == "tpfa"
        ):
            self._precompute_vectorized_assembly()

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
        from .boundary_conditions import parse_robin

        return parse_robin(data, count)

    def _parse_radiative_data(self, data, count):
        from .boundary_conditions import parse_radiative

        return parse_radiative(data, count)

    def _boundary_tangential_operator(self):
        """Sparse ``(n_faces x M)`` map ``u -> u_c + s_f . grad_h u_c`` for boundary faces.

        ``s_f`` is the tangential part of the offset from the cell centroid to
        the face midpoint.  On orthogonal boundary cells ``s_f = 0`` and the map
        is the plain cell value; on skewed cells it removes the O(h) error that
        the one-sided closure would otherwise make in the face value (the
        boundary analogue of the interior non-orthogonal correction).
        """
        cached = getattr(self, "_tangential_op", None)
        if cached is not None:
            return cached
        from scipy.sparse import coo_matrix

        faces = self.boundary_faces
        cells, mids, normals = faces["cells"], faces["midpoints"], faces["normals"]
        rows, cols, vals = list(range(cells.size)), cells.tolist(), [1.0] * cells.size
        if self.gradient_coeffs is not None:
            for f, c in enumerate(cells.tolist()):
                r = mids[f] - self.cell_centers[c]
                s_vec = r - np.dot(r, normals[f]) * normals[f]
                if np.hypot(*s_vec) <= 1e-14 * max(1.0, np.hypot(*r)):
                    continue
                for k, g in self.gradient_coeffs[c].items():
                    rows.append(f)
                    cols.append(int(k))
                    vals.append(float(np.dot(s_vec, g)))
        op = coo_matrix((vals, (rows, cols)), shape=(cells.size, self.M)).tocsr()
        self._tangential_op = op
        return op

    def _boundary_face_relations(self, t, temperature=None, tangential=True):
        """Per-face boundary relations in terms of the corrected cell value ``v = T u``.

        Returns ``(T, a, c, lam, mu)`` with the inward flux ``q_in ~ c - a v``
        and the face value ``u_f ~ lam v + mu`` for every non-Dirichlet type
        (linear types exactly; radiation Newton-linearized about ``temperature``
        after solving the scalar face balance).  ``T`` applies the tangential
        correction for Robin/radiative closures when gradients are available
        (non-orthogonal correction or the quadratic flux); otherwise it is the
        plain face-to-cell map.
        """
        from scipy.sparse import coo_matrix

        faces = self.boundary_faces
        cells = faces["cells"]
        lengths, distances = faces["lengths"], faces["distances"]
        midpoints, normals = faces["midpoints"], faces["normals"]
        n_f = cells.size
        boundary_temperature = None if temperature is None else np.asarray(temperature, dtype=float)[cells]
        Alpha_faces = self._process_alpha(midpoints[:, 0], midpoints[:, 1], temperature=boundary_temperature)
        alpha_n = np.einsum("ij,ij->i", normals, np.einsum("ijk,ik->ij", Alpha_faces, normals))
        data = self._evaluate_bc(midpoints[:, 0], midpoints[:, 1], t, normals=normals)
        closure_types = ("robin", "radiative", "dirichlet")  # dirichlet only reaches here in face mode
        if tangential and self.bc_type in closure_types and self.gradient_coeffs is not None:
            T = self._boundary_tangential_operator()
        else:
            T = coo_matrix((np.ones(n_f), (np.arange(n_f), cells)), shape=(n_f, self.M)).tocsr()
        if self.bc_type in ("neumann", "flux"):
            g = np.broadcast_to(np.asarray(data, dtype=float), (n_f,))
            q_in = alpha_n * g if self.bc_type == "neumann" else g
            return T, np.zeros(n_f), q_in, np.ones(n_f), distances * q_in / alpha_n
        if self.bc_type == "dirichlet":
            if self._band_dirichlet:
                raise ValueError("band Dirichlet data are imposed on cells, not through a face closure.")
            # Face value fixed: q_in = alpha_n (g - v) / d  (Robin with beta -> infinity).
            g = np.broadcast_to(np.asarray(data, dtype=float), (n_f,))
            w = alpha_n / distances
            return T, w, w * g, np.zeros(n_f), g.copy()
        if self.bc_type == "robin":
            beta, value = self._parse_robin_data(data, n_f)
            denom = alpha_n + beta * distances
            if np.any(np.abs(denom) <= 1e-14):
                raise ValueError("Robin boundary condition is singular because alpha + beta * d is too small.")
            a = alpha_n * beta / denom
            c = alpha_n * value / denom
            return T, a, c, alpha_n / denom, distances * value / denom
        # radiative
        if temperature is None:
            raise ValueError("Radiative boundary assembly requires the current temperature iterate.")
        emissivity, t_inf, sigma = self._parse_radiative_data(data, n_f)
        v = T @ np.asarray(temperature, dtype=float)
        es = emissivity * sigma
        t_inf4 = np.maximum(t_inf, 0.0) ** 4
        w = alpha_n / distances
        u_f = v.copy()
        for _ in range(50):
            uf_pos = np.maximum(u_f, 0.0)
            step = (w * (u_f - v) + es * (uf_pos**4 - t_inf4)) / (w + 4.0 * es * uf_pos**3)
            u_f = u_f - step
            if np.max(np.abs(step)) <= 1e-14 * max(1.0, float(np.max(np.abs(u_f)))):
                break
        uf_pos = np.maximum(u_f, 0.0)
        q = -es * (uf_pos**4 - t_inf4)
        dfdv = w / (w + 4.0 * es * uf_pos**3)
        a = 4.0 * es * uf_pos**3 * dfdv
        return T, a, q + a * v, dfdv, u_f - dfdv * v

    def _assemble_boundary_system(self, t, temperature=None, tangential=True):
        """Boundary closure ``(B, b)``: ``(A + B) u = ... + b`` for natural types and face Dirichlet.

        Built from :meth:`_boundary_face_relations`: ``B = E^T diag(|f| a) T`` and
        ``b = E^T (|f| c)`` with ``E`` the face-to-cell incidence.  ``B`` is
        diagonal on orthogonal boundary cells and gains the tangential-correction
        stencil on skewed ones.
        """
        cells = self.boundary_faces["cells"]
        if cells.size == 0:
            return diags(np.zeros(self.M), format="csr"), np.zeros(self.M)
        T, a, c, _, _ = self._boundary_face_relations(t, temperature=temperature, tangential=tangential)
        lengths = self.boundary_faces["lengths"]
        rhs = np.bincount(cells, weights=lengths * c, minlength=self.M)
        if not np.any(a):
            return diags(np.zeros(self.M), format="csr"), rhs
        from scipy.sparse import coo_matrix

        E = coo_matrix((lengths * a, (cells, np.arange(cells.size))), shape=(self.M, cells.size)).tocsr()
        return (E @ T).tocsr(), rhs

    def _build_gradient_reconstruction(self):
        """Weighted least-squares gradient stencils, one dict ``{cell: coeff}`` per cell.

        Cells are batched by neighbour count; the stacked ``matmul``,
        ``matrix_rank`` and ``solve`` call the same BLAS / LAPACK routines per
        cell as the per-cell loop (``_build_gradient_reconstruction_loop``), so
        the coefficients are bit-identical (``tests/test_fv_assembly.py``).
        """
        centers = self.cell_centers
        nb_lists = self.neighbors
        counts = np.fromiter(map(len, nb_lists), dtype=np.intp, count=self.M)
        coeffs = [None] * self.M
        for k in np.unique(counts):
            cells = np.flatnonzero(counts == k)
            if k < 2:
                for i in cells:
                    coeffs[i] = {int(i): np.zeros(2)}
                continue
            nbs = np.array([nb_lists[i] for i in cells], dtype=np.intp)
            offsets = centers[nbs] - centers[cells][:, None, :]
            distances = np.linalg.norm(offsets, axis=2)
            weights = 1.0 / np.maximum(distances**2, 1e-12)
            weighted = offsets * weights[:, :, None]
            normal = np.matmul(offsets.transpose(0, 2, 1), weighted)
            ok = np.linalg.matrix_rank(normal) >= 2
            recon = np.empty((cells.size, 2, k))
            if np.any(ok):
                recon[ok] = np.linalg.solve(normal[ok], weighted[ok].transpose(0, 2, 1))
            own = -np.sum(recon, axis=2)
            for j, i in enumerate(cells):
                if not ok[j]:
                    coeffs[i] = {int(i): np.zeros(2)}
                    continue
                cell_coeffs = dict(zip(nb_lists[i], recon[j].T))
                cell_coeffs[int(i)] = own[j]
                coeffs[i] = cell_coeffs
        return coeffs

    def _build_gradient_reconstruction_loop(self):
        """Legacy per-cell loop (reference for ``_build_gradient_reconstruction``)."""
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

    def _interior_face_arrays(self):
        """Interior-face geometry for the vectorized assembly (cached per mesh)."""
        extras = self._geometry.extras
        faces = extras.get("interior_faces")
        if faces is None:
            faces = interior_faces(self.vertices, self.cell_centers, self.edge_to_cells)
            extras["interior_faces"] = faces
        return faces

    def _flat_gradient_stencils(self):
        if self.gradient_coeffs is None:
            return None
        extras = self._geometry.extras
        if extras.get("gradient_coeffs") is self.gradient_coeffs:
            flat = extras.get("gradient_coeffs_flat")
            if flat is None:
                flat = flatten_stencils(self.gradient_coeffs)
                extras["gradient_coeffs_flat"] = flat
            return flat
        return flatten_stencils(self.gradient_coeffs)

    def _face_tensors(self, faces, temperature=None):
        """Conductivity tensors ``(n_faces, 2, 2)`` at the face midpoints."""
        mid = faces["midpoint"]
        m = mid.shape[0]
        face_t = None if temperature is None else 0.5 * (temperature[faces["i"]] + temperature[faces["j"]])
        try:
            K = np.asarray(self._process_alpha(mid[:, 0], mid[:, 1], temperature=face_t), dtype=float)
            if K.shape == (m, 2, 2):
                return K
        except Exception:  # noqa: BLE001 - alpha callables written for scalar points only
            pass
        return np.array([
            np.asarray(self._process_alpha(mid[k, 0], mid[k, 1],
                                           temperature=None if face_t is None else face_t[k]), dtype=float).reshape(2, 2)
            for k in range(m)
        ]).reshape(m, 2, 2)

    def _assemble_diffusion_reconstructed(self, temperature=None):
        """
        Diffusion matrix from face flux F = -edge_len * n^T K grad_face u, with
        grad_face = 0.5 * (grad u_i + grad u_j) and cell gradients from least squares.
        """
        if self.gradient_coeffs is None:
            raise RuntimeError("flux_discretization='reconstructed' requires gradient coefficients.")
        faces = self._interior_face_arrays()
        self.A = assemble_reconstructed(faces, self._face_tensors(faces, temperature), self.M,
                                        self._flat_gradient_stencils())

    def _assemble_diffusion_reconstructed_loop(self, temperature=None):
        """Legacy per-face loop for the reconstructed flux (reference for the vectorized assembly)."""
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

        return diffusion.tocsr()

    def cell_average(self, f, *args):
        """Cell averages of ``f(x, y, *args)`` (degree-4 triangle-fan quadrature).

        Use for initial data and for error measurement with
        ``flux_discretization='quadratic'``, whose unknowns are cell averages.
        """
        from .quadratic_flux import CellQuadrature

        if self._cell_quadrature is None:
            self._cell_quadrature = CellQuadrature(self.vertices, self.polygons)
        return self._cell_quadrature.average(f, *args)

    def _apply_monotone(self, A):
        """Optionally project the diffusion matrix onto an M-matrix (DMP-preserving)."""
        if not self.monotone:
            return A
        from .dmp import make_monotone

        return make_monotone(A)

    def _assemble_system(self, temperature=None):
        self._assemble_operator(temperature)
        # solve() reuses a temperature-independent operator while it is still current.
        self._constant_A = self.A if temperature is None else None

    def _assemble_operator(self, temperature=None):
        if temperature is not None and self._vec_assembly is not None:
            self._assemble_diffusion_vectorized(temperature)
            self.A = self._apply_monotone(self.A)
            return
        mass = diags(self.cell_areas, format="csr")
        # MPFA condensed FEM is only used with Dirichlet: Neumann/Robin need a flux closure
        # consistent with unprescribed boundary values; use the same TPFA two-point stencil as
        # ``flux_scheme='tpfa'`` for the diffusion block in that case.
        if self.flux_scheme == "mpfa" and self._band_dirichlet:
            from .mpfa import assemble_mpfa_diffusion

            self.M_diag = mass
            self.A = self._apply_monotone(assemble_mpfa_diffusion(
                self.vertices,
                self.polygons,
                self.cell_centers,
                self.alpha,
                self.edge_to_cells,
            ))
            return

        if self.flux_discretization == "reconstructed":
            self.M_diag = mass
            self._assemble_diffusion_reconstructed(temperature=temperature)
            self.A = self._apply_monotone(self.A)
            return

        if self.flux_discretization == "quadratic":
            from .quadratic_flux import QuadraticDiffusionOperator

            self.M_diag = mass
            op = getattr(self, "_quadratic_operator", None) or self._geometry.extras.get("quadratic_operator")
            if op is None:
                op = QuadraticDiffusionOperator(self.vertices, self.polygons, self.edge_to_cells)
                self._geometry.extras["quadratic_operator"] = op  # shared by all solvers on this mesh
            self._quadratic_operator = op
            xf = op.face_midpoints
            # Face temperature from the quadratic reconstruction (O(h^3)), not the
            # two-cell average (O(h) on irregular meshes).
            face_t = None if temperature is None else op.face_values(temperature)
            K = self._process_alpha(xf[:, 0], xf[:, 1], temperature=face_t)
            self.A = self._apply_monotone(op.assemble(K))
            return

        faces = self._interior_face_arrays()
        stencils = self._flat_gradient_stencils() if self.nonorthogonal_correction else None
        self.M_diag = mass
        self.A = self._apply_monotone(assemble_tpfa(faces, self._face_tensors(faces, temperature), self.M, stencils))

    def _assemble_tpfa_loop(self, temperature=None):
        """Legacy per-face loop for the two-point flux (reference for the vectorized assembly)."""
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
        return diffusion.tocsr()

    def _precompute_vectorized_assembly(self):
        """Pre-compute edge topology and COO structure for vectorized diffusion assembly.

        For temperature-dependent *isotropic* (scalar) diffusivity the diffusion
        matrix has the form ``A = sum_edges alpha_e * G_e`` where ``G_e`` depends
        only on mesh geometry.  This method pre-computes ``G_e`` (as COO weights)
        so that each subsequent assembly is a single vectorised alpha evaluation
        followed by a weighted COO scatter -- no Python edge loop required.
        """
        cells_i_list, cells_j_list, v0_list, v1_list = [], [], [], []
        for edge, cells in self.edge_to_cells.items():
            if len(cells) != 2:
                continue
            i, j = cells
            if j < i:
                i, j = j, i
            cells_i_list.append(i)
            cells_j_list.append(j)
            v0_list.append(self.vertices[edge[0]])
            v1_list.append(self.vertices[edge[1]])

        if not cells_i_list:
            return

        cells_i = np.array(cells_i_list, dtype=np.intp)
        cells_j = np.array(cells_j_list, dtype=np.intp)
        v0 = np.asarray(v0_list, dtype=float)
        v1 = np.asarray(v1_list, dtype=float)

        edge_vecs = v1 - v0
        edge_lens = np.linalg.norm(edge_vecs, axis=1)

        valid = edge_lens > 0
        cells_i = cells_i[valid]
        cells_j = cells_j[valid]
        v0 = v0[valid]
        v1 = v1[valid]
        edge_vecs = edge_vecs[valid]
        edge_lens = edge_lens[valid]
        n_edges = cells_i.shape[0]

        tangents = edge_vecs / edge_lens[:, None]
        normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
        midpoints = 0.5 * (v0 + v1)
        d_vecs = self.cell_centers[cells_j] - self.cell_centers[cells_i]
        dn = np.einsum("ij,ij->i", d_vecs, normals)

        flip = dn < 0
        normals[flip] *= -1
        dn[flip] *= -1
        small = dn <= 1e-12
        dn[small] = np.maximum(np.linalg.norm(d_vecs[small], axis=1), 1e-12)

        # Verify the alpha callable returns isotropic scalars.
        n_test = min(3, n_edges)
        try:
            _tv = np.asarray(
                self.alpha(midpoints[:n_test, 0], midpoints[:n_test, 1], np.ones(n_test)),
                dtype=float,
            )
            if _tv.ndim > 1 or _tv.shape != (n_test,):
                return
        except Exception:
            return

        tpfa_trans = edge_lens / dn

        base_rows = np.concatenate([cells_i, cells_i, cells_j, cells_j])
        base_cols = np.concatenate([cells_i, cells_j, cells_i, cells_j])
        base_edge_map = np.tile(np.arange(n_edges), 4)
        base_geom = np.concatenate([tpfa_trans, -tpfa_trans, -tpfa_trans, tpfa_trans])

        cr, cc, cg, ce = [], [], [], []
        if self.nonorthogonal_correction and self.gradient_coeffs is not None:
            corr_unit = normals - d_vecs / dn[:, None]
            corr_norms = np.linalg.norm(corr_unit, axis=1)
            for e in range(n_edges):
                if corr_norms[e] <= 1e-14:
                    continue
                cu = corr_unit[e]
                L = edge_lens[e]
                i_cell, j_cell = int(cells_i[e]), int(cells_j[e])
                fc = {}
                for cell in (i_cell, j_cell):
                    for idx, gc in self.gradient_coeffs[cell].items():
                        fc[idx] = fc.get(idx, 0.0) + 0.5 * float(np.dot(cu, gc))
                for idx, c in fc.items():
                    if abs(c) <= 1e-30:
                        continue
                    cr.extend([i_cell, j_cell])
                    cc.extend([idx, idx])
                    cg.extend([-L * c, L * c])
                    ce.extend([e, e])

        if cr:
            all_rows = np.concatenate([base_rows, np.array(cr, dtype=np.intp)])
            all_cols = np.concatenate([base_cols, np.array(cc, dtype=np.intp)])
            all_geom = np.concatenate([base_geom, np.array(cg, dtype=float)])
            all_emap = np.concatenate([base_edge_map, np.array(ce, dtype=np.intp)])
        else:
            all_rows = base_rows
            all_cols = base_cols
            all_geom = base_geom
            all_emap = base_edge_map

        self.M_diag = diags(self.cell_areas, format="csr")
        self._vec_assembly = {
            "n_edges": n_edges,
            "cells_i": cells_i,
            "cells_j": cells_j,
            "midpoints": midpoints,
            "all_rows": all_rows,
            "all_cols": all_cols,
            "all_geom": all_geom,
            "all_emap": all_emap,
        }

    def _assemble_diffusion_vectorized(self, temperature):
        """Build diffusion matrix from pre-computed COO structure (isotropic scalar alpha)."""
        va = self._vec_assembly
        face_temps = 0.5 * (temperature[va["cells_i"]] + temperature[va["cells_j"]])
        alpha_vals = np.asarray(
            self.alpha(va["midpoints"][:, 0], va["midpoints"][:, 1], face_temps),
            dtype=float,
        ).ravel()
        data = alpha_vals[va["all_emap"]] * va["all_geom"]
        self.A = coo_matrix(
            (data, (va["all_rows"], va["all_cols"])), shape=(self.M, self.M)
        ).tocsr()

    def _step_coefficients(self, dts):
        """Per-step ``(a0, a1, a2)``: ``u_t ~ (a0 u^{n+1} - a1 u^n + a2 u^{n-1}) / dt_n``.

        Backward Euler (and the first BDF2 step) is ``(1, 1, 0)``; BDF2 uses the
        variable-step coefficients with ``w = dt_n / dt_{n-1}``.
        """
        coeffs = []
        for n, k in enumerate(dts):
            if self.time_scheme == "bdf2" and n > 0:
                w = k / dts[n - 1]
                coeffs.append(((1.0 + 2.0 * w) / (1.0 + w), 1.0 + w, w * w / (1.0 + w)))
            else:
                coeffs.append((1.0, 1.0, 0.0))
        return coeffs

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

    def _linear_operator_reusable(self):
        """True when the assembled linear system matrix is independent of time step index."""
        if self.phase_change_model is not None:
            return False
        if self.temperature_dependent_diffusivity:
            return False
        if self.bc_type == "radiative":
            return False
        if self.bc_type == "robin":
            return False
        if self.bc_type == "dirichlet" and not self._band_dirichlet:
            return False  # face Dirichlet: the closure B enters the matrix (factor cache reuses it)
        return True

    def _apply_dirichlet_identity_rows(self, K_csr):
        return identity_rows(K_csr, self.is_boundary)

    def _dirichlet_mask(self):
        return self.is_boundary if self._band_dirichlet else None

    def _linear_plan(self, dts):
        """``(method, preconditioner, reason)`` for this solve; sets up the iterative solver if needed."""
        opts = self.linear_solver_options
        if self.linear_solver == "direct":
            return "direct", None, "linear_solver='direct'"
        if self.linear_solver == "auto":
            theta = 0.5 if self.time_scheme == "crank_nicolson" else 1.0
            nonlinear = self.phase_change_model is not None or bool(self.temperature_dependent_diffusivity)
            # radiative boundaries: ~2 solves per step, one factorization (frozen Jacobian)
            per_step = 3 if nonlinear else (2 if self.bc_type == "radiative" else 1)
            n_solves = max(1, len(dts)) * per_step
            key = (round(float(dts[0]), 12) if dts else 0.0, self.time_scheme, int(np.log2(n_solves)))
            if key not in self._auto_plans:
                if self.A is None:
                    self._assemble_system()
                probe = self.M_diag + (theta * (dts[0] if dts else self.dt)) * self.A
                self._auto_plans[key] = choose_linear_solver(probe, opts, n_solves=n_solves,
                                                             matrix_changes=nonlinear)
            method, pc, reason = self._auto_plans[key]
        else:
            method, pc, reason = self.linear_solver, opts.get("preconditioner"), "user choice"
        if method != "direct" and (self._iterative is None or (self._iterative.method, self._iterative.preconditioner)
                                   != (method, pc)):
            self._iterative = IterativeSolver(method, pc, opts)
        return method, pc, reason

    def _face_dirichlet_closure(self):
        """Time-independent part of the face-Dirichlet closure: ``B`` and the face weights."""
        B, _ = self._assemble_boundary_system(0.0)
        _, w, _, _, _ = self._boundary_face_relations(0.0)
        faces = self.boundary_faces
        return {"B": B, "w": w, "cells": faces["cells"], "lengths": faces["lengths"],
                "midpoints": faces["midpoints"], "normals": faces["normals"]}

    def _face_dirichlet_values(self, t, fc):
        """Dirichlet data ``g`` at the boundary-face midpoints at time ``t``."""
        mid = fc["midpoints"]
        data = self._evaluate_bc(mid[:, 0], mid[:, 1], t, normals=fc["normals"])
        return np.broadcast_to(np.asarray(data, dtype=float), (fc["cells"].size,))

    def _face_dirichlet_rhs(self, t, fc):
        """Right-hand side of the face-Dirichlet closure at time ``t`` (same arithmetic as
        ``_assemble_boundary_system``, so results are unchanged)."""
        c = fc["w"] * self._face_dirichlet_values(t, fc)
        return np.bincount(fc["cells"], weights=fc["lengths"] * c, minlength=self.M)

    def _boundary_factor(self, key, B, build, stats):
        """Factorization of ``build()`` (a matrix containing the boundary closure ``B``), reused while unchanged."""
        if self._factor_cache.lookup(key, (self.A, self.M_diag), B) is not None:
            stats["factorizations_reused"] += 1
        else:
            stats["boundary_factorizations"] += 1
        return self._factor_cache.factor(key, (self.A, self.M_diag), B, build)

    def _radiative_step(self, u, hist, t, t_next, dt, a0, source_next, src_avg, lin_key, max_iters, tol, stats):
        """One implicit step with radiative boundaries by frozen-Jacobian Newton.

        Solves the same nonlinear system as the generic Picard/Newton loop
        (``B(v) v - b(v)`` is the exact radiative boundary term), but reuses one
        factorization across iterations and steps; see :func:`frozen_newton`.
        ``nonlinear_options["radiative_jacobian"] = "newton"`` restores full Newton.
        """
        M, A = self.M_diag, self.A
        if self.time_scheme == "crank_nicolson":
            theta, lead = 0.5, M
            b_mat_prev, b_rhs_prev = self._assemble_boundary_system(t, temperature=u)
            rhs = (M - 0.5 * dt * (A + b_mat_prev)) @ u + 0.5 * dt * src_avg + 0.5 * dt * b_rhs_prev
        else:
            theta, lead = 1.0, self._mass(a0)
            rhs = M @ hist + dt * (self.cell_areas * source_next)
        coef = theta * dt
        u_new, iters, n_factor = frozen_newton(
            u0=u,
            boundary=lambda v: self._assemble_boundary_system(t_next, temperature=v),
            apply_base=lambda v: lead @ v + coef * (A @ v),
            rhs=rhs, coef=coef,
            build_jacobian=lambda B: lead + coef * (A + B),
            cache=self._factor_cache, key=("radiative", lin_key, self.time_scheme), deps=(A, M),
            tol=tol, max_iters=max_iters)
        stats["radiative_iterations"] += iters
        stats["boundary_factorizations"] += n_factor
        return u_new

    def _mass(self, coeff):
        return self.M_diag if coeff == 1.0 else coeff * self.M_diag

    def _dirichlet_values(self, x, y, t):
        return np.broadcast_to(np.asarray(self.bc_func(x, y, t), dtype=float), (self.M,))

    def _build_linear_lhs_csr(self, dt_eff, mass_coeff=1.0):
        """LHS ``c M + theta dt A`` for linear Dirichlet/Neumann problems (Robin excluded).

        ``mass_coeff`` is the BDF2 leading coefficient (1 for backward Euler and Crank–Nicolson).
        """
        theta = 0.5 if self.time_scheme == "crank_nicolson" else 1.0
        mass = self.M_diag if mass_coeff == 1.0 else mass_coeff * self.M_diag
        K = mass + (theta * dt_eff) * self.A
        if self._band_dirichlet:
            return self._apply_dirichlet_identity_rows(K)
        return K.tocsr()

    def solve(self, u0, t0, t_end):
        u = np.array(u0, dtype=float)
        t = t0
        dts = self._dt_schedule(t0, t_end, self.dt)
        nsteps = len(dts)
        has_temp_dependent_diffusion = bool(self.temperature_dependent_diffusivity)
        has_nonlinear_radiation = self.bc_type == "radiative"
        requires_nonlinear = (
            self.phase_change_model is not None
            or has_temp_dependent_diffusion
            or has_nonlinear_radiation
        )

        if not has_temp_dependent_diffusion and (self.A is None or self.A is not self._constant_A):
            self._assemble_system()  # assembled once; reused by later solve() calls

        _pc_fast = (
            self.phase_change_model is not None
            and not has_temp_dependent_diffusion
            and not has_nonlinear_radiation
            and self._band_dirichlet
            and self.time_scheme == "backward_euler"
        )
        _pc_lhs = None
        _pc_diag_pos = None
        _pc_a2l = None
        _pc_A_data = None
        if _pc_fast:
            _pc_A_csr = self.A.tocsr()
            _pc_lhs = (diags(self.cell_areas, format="csr") + _pc_A_csr).tocsr()
            _pc_lhs_indptr = _pc_lhs.indptr
            _pc_lhs_indices = _pc_lhs.indices
            _pc_diag_pos = np.empty(self.M, dtype=np.intp)
            for _i in range(self.M):
                _s = _pc_lhs_indptr[_i]
                _pc_diag_pos[_i] = _s + np.searchsorted(
                    _pc_lhs_indices[_s:_pc_lhs_indptr[_i + 1]], _i
                )
            _pc_a2l = np.empty(_pc_A_csr.nnz, dtype=np.intp)
            for _i in range(self.M):
                _as = _pc_A_csr.indptr[_i]
                _ae = _pc_A_csr.indptr[_i + 1]
                _ls = _pc_lhs_indptr[_i]
                _pc_a2l[_as:_ae] = _ls + np.searchsorted(
                    _pc_lhs_indices[_ls:_pc_lhs_indptr[_i + 1]],
                    _pc_A_csr.indices[_as:_ae],
                )
            _pc_A_data = _pc_A_csr.data
            _pc_A_cols = _pc_A_csr.indices.copy()

        use_linear_cache = (
            self.reuse_linear_lhs
            and (not requires_nonlinear)
            and self._linear_operator_reusable()
        )
        step_coeffs = self._step_coefficients(dts)
        lin_keys = [(round(float(d), 12), round(c[0], 12)) for d, c in zip(dts, step_coeffs)]
        plan = self._linear_plan(dts)
        direct = plan[0] == "direct"
        lhs_by_dt = None
        factor_by_dt = None
        if use_linear_cache:
            uniq_dt = []
            seen = set()
            for key in lin_keys:
                if key not in seen:
                    seen.add(key)
                    uniq_dt.append(key)
            if self._linear_cache_A is not self.A:  # operator changed: drop stale systems
                self._linear_cache = {}
                self._linear_cache_A = self.A
            lhs_by_dt = {}
            factor_by_dt = {} if direct else None
            for key in uniq_dt:
                cache_key = (key, self.time_scheme, direct)
                if cache_key not in self._linear_cache:
                    a0 = step_coeffs[lin_keys.index(key)][0]
                    lhs_new = self._build_linear_lhs_csr(key[0], mass_coeff=a0)
                    self._linear_cache[cache_key] = (lhs_new, factorized(lhs_new) if direct else None)
                lhs_by_dt[key], factor = self._linear_cache[cache_key]
                if factor_by_dt is not None:
                    factor_by_dt[key] = factor

        u_prev = None
        stats = {"factorizations_reused": 0, "boundary_factorizations": 0, "radiative_iterations": 0,
                 "linear_solver": plan[0], "preconditioner": plan[1], "linear_solver_reason": plan[2]}
        self.last_solve_stats = stats
        it_before = dict(self._iterative.stats) if self._iterative is not None else None
        # Face Dirichlet: the closure matrix does not depend on time (only its right-hand side
        # does), so build it once per solve and reuse its factorizations without re-checking.
        face_closure = None
        if self.bc_type == "dirichlet" and not self._band_dirichlet and not has_temp_dependent_diffusion:
            face_closure = self._face_dirichlet_closure()
        face_factors = {}
        for step_idx in range(nsteps):
            t_next = min(t + self.dt, t_end)
            dt_eff = dts[step_idx]
            step_factor = None
            cx = self.cell_centers[:, 0]
            cy = self.cell_centers[:, 1]
            source_next = np.broadcast_to(
                np.asarray(self.source_func(cx, cy, t_next), dtype=float),
                (self.M,),
            )
            # Implicit one-/two-step history: u^n for backward Euler, a1 u^n - a2 u^{n-1} for BDF2.
            a0, a1, a2 = step_coeffs[step_idx]
            hist = u if a2 == 0.0 else a1 * u - a2 * u_prev
            u_prev = u

            if not requires_nonlinear:
                dt_key = lin_keys[step_idx]
                if self.time_scheme != "crank_nicolson":
                    rhs = self.M_diag @ hist + dt_eff * (self.cell_areas * source_next)
                    if self._band_dirichlet:
                        bidx = self._boundary_idx
                        rhs[bidx] = self._dirichlet_values(cx, cy, t_next)[bidx]
                        if use_linear_cache:
                            lhs = lhs_by_dt[dt_key]
                        else:
                            lhs = identity_rows(self._mass(a0) + dt_eff * self.A, self.is_boundary)
                    else:
                        if face_closure is not None:
                            boundary_matrix = face_closure["B"]
                            boundary_rhs = self._face_dirichlet_rhs(t_next, face_closure)
                        else:
                            boundary_matrix, boundary_rhs = self._assemble_boundary_system(t_next)
                        rhs = rhs + dt_eff * boundary_rhs
                        fkey = ("implicit", float(dt_eff), a0, self.time_scheme)
                        if face_closure is not None:
                            # Constant closure: steps whose dt differs only by round-off (dt from
                            # differences of accumulated times) share one factorization, as the band
                            # path does through its rounded cache key.
                            fkey = ("implicit", dt_key, self.time_scheme)
                        if use_linear_cache:
                            lhs = lhs_by_dt[dt_key]
                        elif direct and fkey in face_factors:
                            lhs, step_factor = None, face_factors[fkey]
                            stats["factorizations_reused"] += 1
                        elif direct:
                            # Robin: reuse the factorization while the closure is unchanged.
                            lhs = None
                            step_factor = self._boundary_factor(
                                # exact dt / a0 (not the rounded cache key): reuse only an identical matrix
                                fkey, boundary_matrix,
                                lambda B=boundary_matrix, c=a0, k=dt_eff: self._mass(c) + k * (self.A + B), stats)
                            if face_closure is not None:
                                face_factors[fkey] = step_factor
                        else:
                            lhs = self._mass(a0) + dt_eff * (self.A + boundary_matrix)
                else:
                    source_prev = np.broadcast_to(
                        np.asarray(self.source_func(cx, cy, t), dtype=float),
                        (self.M,),
                    )
                    src_avg = self.cell_areas * (source_next + source_prev)
                    if self._band_dirichlet:
                        rhs = (self.M_diag - 0.5 * dt_eff * self.A) @ u + 0.5 * dt_eff * src_avg
                        bidx = self._boundary_idx
                        rhs[bidx] = self._dirichlet_values(cx, cy, t_next)[bidx]
                        if use_linear_cache:
                            lhs = lhs_by_dt[dt_key]
                        else:
                            lhs = identity_rows(self.M_diag + 0.5 * dt_eff * self.A, self.is_boundary)
                    else:
                        if face_closure is not None:
                            b_mat_prev = b_mat_next = face_closure["B"]
                            b_rhs_prev = self._face_dirichlet_rhs(t, face_closure)
                            b_rhs_next = self._face_dirichlet_rhs(t_next, face_closure)
                        else:
                            b_mat_prev, b_rhs_prev = self._assemble_boundary_system(t)
                            b_mat_next, b_rhs_next = self._assemble_boundary_system(t_next)
                        k_prev = self.A + b_mat_prev
                        k_next = self.A + b_mat_next
                        rhs = (self.M_diag - 0.5 * dt_eff * k_prev) @ u + 0.5 * dt_eff * src_avg
                        rhs = rhs + 0.5 * dt_eff * (b_rhs_prev + b_rhs_next)
                        fkey = ("crank_nicolson", dt_key if face_closure is not None else float(dt_eff))
                        if use_linear_cache:
                            lhs = lhs_by_dt[dt_key]
                        elif direct and fkey in face_factors:
                            lhs, step_factor = None, face_factors[fkey]
                            stats["factorizations_reused"] += 1
                        elif direct:
                            lhs = None
                            step_factor = self._boundary_factor(
                                fkey, b_mat_next,
                                lambda K=k_next, k=dt_eff: self.M_diag + 0.5 * k * K, stats)
                            if face_closure is not None:
                                face_factors[fkey] = step_factor
                        else:
                            lhs = self.M_diag + 0.5 * dt_eff * k_next

                if direct:
                    if step_factor is not None:
                        direct_factor = step_factor
                    else:
                        direct_factor = factor_by_dt[dt_key] if factor_by_dt is not None else None
                    u = _solve_sparse_linear_system(lhs, rhs, "direct", None, direct_factor=direct_factor)
                else:
                    u = self._iterative.solve(lhs, rhs, x0=u, key=("linear", dt_key, self.time_scheme),
                                              dirichlet_mask=self._dirichlet_mask())
                t = t_next
                continue

            max_iters = int(self.nonlinear_options["max_iters"])
            tol = float(self.nonlinear_options["tol"])
            relaxation = float(self.nonlinear_options["relaxation"])
            u_iter = u.copy()

            # ---- Fast vectorized path for temp-dependent scalar diffusivity ----
            _use_fast = (
                has_temp_dependent_diffusion
                and self._vec_assembly is not None
                and self.phase_change_model is None
                and self._band_dirichlet
                and self.time_scheme == "backward_euler"
            )
            if _use_fast:
                anderson_depth = int(self.nonlinear_options.get("anderson_depth", 0))
                accel = _AndersonAccelerator(anderson_depth) if anderson_depth > 0 else None
                bc_vals = np.broadcast_to(
                    np.asarray(self.bc_func(cx, cy, t_next), dtype=float), (self.M,)
                )
                rhs_base = self.M_diag @ u + dt_eff * np.asarray(
                    self.cell_areas * source_next, dtype=float
                )
                bidx = self._boundary_idx
                rhs_base[bidx] = bc_vals[bidx]

                for _ in range(max_iters):
                    self._assemble_diffusion_vectorized(u_iter)
                    lhs = self.M_diag + dt_eff * self.A
                    _apply_dirichlet_rows_csr(lhs, bidx)
                    u_raw = spsolve(lhs, rhs_base)

                    if accel is not None:
                        u_next = accel.step(u_iter, u_raw)
                        if not np.all(np.isfinite(u_next)):
                            u_next = u_raw
                            accel.reset()
                    elif relaxation != 1.0:
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
                        "Nonlinear solve did not converge. "
                        "Try a smaller dt or larger nonlinear_options['max_iters']."
                    )
                u = u_iter
                t = t_next
                continue
            # ---- End fast path ----

            # ---- Fast path for phase-change with constant diffusivity ----
            if _pc_fast:
                bc_vals = np.broadcast_to(
                    np.asarray(self.bc_func(cx, cy, t_next), dtype=float), (self.M,)
                )
                dt_area_src = dt_eff * np.asarray(
                    self.cell_areas * source_next, dtype=float
                )
                bidx = self._boundary_idx
                linearize_cp = bool(self.nonlinear_options.get("linearize_cp", False))

                if linearize_cp:
                    pcm = self.phase_change_model
                    cp_eff = self._effective_heat_capacity(u)
                    inv_cp = 1.0 / cp_eff
                    H_n = np.asarray(pcm.enthalpy(u), dtype=float)

                    _pc_lhs.data[:] = 0.0
                    _pc_lhs.data[_pc_a2l] = (
                        dt_eff * _pc_A_data * inv_cp[_pc_A_cols]
                    )
                    _pc_lhs.data[_pc_diag_pos] += self.cell_areas

                    rhs = dt_area_src - dt_eff * (self.A @ u)
                    H_bc = np.asarray(pcm.enthalpy(bc_vals), dtype=float)
                    rhs[bidx] = H_bc[bidx] - H_n[bidx]
                    _apply_dirichlet_rows_csr(_pc_lhs, bidx)

                    dH = spsolve(_pc_lhs, rhs)
                    u = np.asarray(
                        pcm.temperature_from_enthalpy(H_n + dH), dtype=float
                    )
                    t = t_next
                    continue

                anderson_depth = int(self.nonlinear_options.get("anderson_depth", 0))
                accel = _AndersonAccelerator(anderson_depth) if anderson_depth > 0 else None

                for _ in range(max_iters):
                    cp_eff = self._effective_heat_capacity(u_iter)
                    rhs = self.cell_areas * cp_eff * u + dt_area_src
                    rhs[bidx] = bc_vals[bidx]

                    _pc_lhs.data[:] = 0.0
                    _pc_lhs.data[_pc_a2l] = dt_eff * _pc_A_data
                    _pc_lhs.data[_pc_diag_pos] += self.cell_areas * cp_eff
                    _apply_dirichlet_rows_csr(_pc_lhs, bidx)

                    u_raw = spsolve(_pc_lhs, rhs)

                    if accel is not None:
                        u_next = accel.step(u_iter, u_raw)
                        if not np.all(np.isfinite(u_next)):
                            u_next = u_raw
                            accel.reset()
                    elif relaxation != 1.0:
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
                        "Nonlinear solve did not converge. "
                        "Try a smaller dt or larger nonlinear_options['max_iters']."
                    )
                u = u_iter
                t = t_next
                continue
            # ---- End fast phase-change path ----

            source_prev = None
            src_avg = None
            if self.time_scheme == "crank_nicolson":
                source_prev = np.broadcast_to(
                    np.asarray(self.source_func(cx, cy, t), dtype=float),
                    (self.M,),
                )
                src_avg = self.cell_areas * (source_next + source_prev)

            if (has_nonlinear_radiation and not has_temp_dependent_diffusion and self.phase_change_model is None
                    and direct and relaxation == 1.0
                    and self.nonlinear_options.get("radiative_jacobian", "frozen") == "frozen"):
                u = self._radiative_step(u, hist, t, t_next, dt_eff, a0, source_next, src_avg,
                                         lin_keys[step_idx], max_iters, tol, stats)
                t = t_next
                continue

            # Crank--Nicolson needs the *old-level* operator A(u^n) on the explicit
            # half; using the new-level iterate for both halves (as before) makes the
            # scheme first order in time for temperature-dependent conductivity.
            A_old = None
            if self.time_scheme == "crank_nicolson" and has_temp_dependent_diffusion:
                self._assemble_system(temperature=u)
                A_old = self.A.copy()

            for _ in range(max_iters):
                if has_temp_dependent_diffusion:
                    self._assemble_system(temperature=u_iter)
                cp_eff = self._effective_heat_capacity(u_iter)
                mass_matrix = self._build_mass_matrix(cp_eff)

                if self.time_scheme != "crank_nicolson":
                    rhs = mass_matrix @ hist + dt_eff * (self.cell_areas * source_next)
                    lead = mass_matrix if a0 == 1.0 else a0 * mass_matrix
                    if self._band_dirichlet:
                        bidx = self._boundary_idx
                        rhs[bidx] = self._dirichlet_values(cx, cy, t_next)[bidx]
                        lhs = identity_rows(lead + dt_eff * self.A, self.is_boundary)
                    else:
                        boundary_matrix, boundary_rhs = self._assemble_boundary_system(t_next, temperature=u_iter)
                        lhs = lead + dt_eff * (self.A + boundary_matrix)
                        rhs = rhs + dt_eff * boundary_rhs
                elif self._band_dirichlet:
                    A_prev = self.A if A_old is None else A_old
                    rhs = (mass_matrix - 0.5 * dt_eff * A_prev) @ u + 0.5 * dt_eff * src_avg
                    bidx = self._boundary_idx
                    rhs[bidx] = self._dirichlet_values(cx, cy, t_next)[bidx]
                    lhs = identity_rows(mass_matrix + 0.5 * dt_eff * self.A, self.is_boundary)
                else:
                    if has_temp_dependent_diffusion or has_nonlinear_radiation:
                        # Old level evaluated at u^n (operator and nonlinear boundary
                        # closure), new level at the current iterate.
                        b_mat_prev, b_rhs_prev = self._assemble_boundary_system(t, temperature=u)
                        b_mat_next, b_rhs_next = self._assemble_boundary_system(t_next, temperature=u_iter)
                        k_prev = (self.A if A_old is None else A_old) + b_mat_prev
                        k_next = self.A + b_mat_next
                        rhs = (mass_matrix - 0.5 * dt_eff * k_prev) @ u + 0.5 * dt_eff * src_avg
                        rhs = rhs + 0.5 * dt_eff * (b_rhs_prev + b_rhs_next)
                        lhs = mass_matrix + 0.5 * dt_eff * k_next
                    else:
                        b_mat_prev, b_rhs_prev = self._assemble_boundary_system(t, temperature=u)
                        b_mat_next, b_rhs_next = self._assemble_boundary_system(t_next, temperature=u_iter)
                        k_prev = self.A + b_mat_prev
                        k_next = self.A + b_mat_next
                        rhs = (mass_matrix - 0.5 * dt_eff * k_prev) @ u + 0.5 * dt_eff * src_avg
                        rhs = rhs + 0.5 * dt_eff * (b_rhs_prev + b_rhs_next)
                        lhs = mass_matrix + 0.5 * dt_eff * k_next

                if direct:
                    u_next = _solve_sparse_linear_system(lhs, rhs, "direct", None)
                else:
                    u_next = self._iterative.solve(lhs, rhs, x0=u_iter, key=("nonlinear", lin_keys[step_idx][0]),
                                                   dirichlet_mask=self._dirichlet_mask())
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
        if it_before is not None or self._iterative is not None:
            before = it_before or {k: 0 for k in self._iterative.stats}
            for k, v in self._iterative.stats.items():
                stats["krylov_" + k] = v - before.get(k, 0)
        return t, u
