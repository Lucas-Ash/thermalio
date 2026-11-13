"""Adapters to independent, external heat-equation codes for cross-code validation.

Direction A parked "validation against an external package".  This module
wraps independently developed, pip-installable codes so the *same*
manufactured problems can be solved by Thermalio and by codes that share no
source with it:

* **FiPy** (NIST, cell-centred finite volumes on structured grids, its own
  face-flux and boundary machinery, solved through its SciPy backend);
* **scikit-fem** (continuous Galerkin finite elements, P1/P2 Lagrange on
  triangles; a different discretization family altogether);
* **NGSolve** (compiled C++ high-order finite elements with its own sparse
  direct solvers and optional task-parallel assembly);
* **PorePy** (cell-centred finite volumes with two-point (TPFA) and
  multi-point (MPFA-O) fluxes on general polygonal grids, the closest peer of
  Thermalio's own flux choices);
* **py-pde** (finite differences on Cartesian grids, compiled just in time
  with Numba).

All are optional (``requirements-external.txt``) and imported lazily, so the
core package never depends on them.  All solvers use backward Euler in time so
that, at matched time steps, differences between codes isolate the spatial
discretizations.

Problem interface (shared by every adapter)::

    kappa   : float or 2x2 array (constant, possibly anisotropic conductivity)
    u0      : f(x, y) -> initial values
    source  : f(x, y, t) -> Q
    reaction: float k >= 0 (Pennes perfusion term ``+ k u``)
    dirichlet : f(x, y, t) -> boundary values   (on the whole boundary), or
    robin   : (h, value) with value(x, y, t, nx, ny) -> kappa du/dn + h u

NGSolve and py-pde take their data as symbolic expressions (they compile them
into coefficient functions / Numba kernels), so their adapters take a
:class:`SymbolicProblem` instead of NumPy callables.
"""

from __future__ import annotations

import os

import numpy as np


def _kappa_matrix(kappa):
    k = np.asarray(kappa, dtype=float)
    if k.ndim == 0:
        return np.array([[float(k), 0.0], [0.0, float(k)]])
    if k.shape != (2, 2):
        raise ValueError("kappa must be a scalar or a 2x2 matrix.")
    return k


def fipy_available():
    try:
        import fipy  # noqa: F401
    except Exception:  # pragma: no cover - depends on the environment
        return False
    return True


def skfem_available():
    try:
        import skfem  # noqa: F401
    except Exception:  # pragma: no cover
        return False
    return True


def _importable(name):
    try:
        __import__(name)
    except Exception:  # pragma: no cover - depends on the environment
        return False
    return True


def ngsolve_available():
    return _importable("ngsolve")


def porepy_available():
    return _importable("porepy")


def pypde_available():
    return _importable("pde")


class SymbolicProblem:
    """A manufactured problem in SymPy: ``u_t - div(K grad u) + r u = Q`` on ``bbox``.

    ``exact`` is a SymPy expression in the symbols ``x, y, t``.  The source and
    the boundary data follow from it, in whichever form a code needs: NumPy
    callables, NGSolve coefficient functions or py-pde expression strings.
    ``robin_h`` switches the boundary to ``K grad u . n + h u = value``
    (``K`` scalar).
    """

    def __init__(self, exact, bbox, kappa, reaction=0.0, robin_h=None):
        import sympy as sp

        self.x, self.y, self.t = sp.symbols("x y t")
        self.exact = exact
        self.bbox = tuple(float(b) for b in bbox)
        self.K = _kappa_matrix(kappa)
        self.reaction = float(reaction)
        self.robin_h = None if robin_h is None else float(robin_h)
        x, y, t, K = self.x, self.y, self.t, self.K
        u = exact
        div = (K[0, 0] * sp.diff(u, x, 2) + (K[0, 1] + K[1, 0]) * sp.diff(u, x, y) + K[1, 1] * sp.diff(u, y, 2))
        self.source = sp.diff(u, t) - div + self.reaction * u
        self.grad = (sp.diff(u, x), sp.diff(u, y))

    def numpy(self, expr):
        import sympy as sp

        f = sp.lambdify((self.x, self.y, self.t), expr, "numpy")
        return lambda x, y, t: np.broadcast_to(f(x, y, t), np.shape(x)).astype(float)

    def robin_value(self, nx, ny):
        """``K grad u . n + h u`` for a constant outward normal ``(nx, ny)`` (SymPy)."""
        k = self.K[0, 0]
        return k * (self.grad[0] * nx + self.grad[1] * ny) + self.robin_h * self.exact


# ---------------------------------------------------------------------------
# FiPy
# ---------------------------------------------------------------------------


def fipy_heat_solve(n, bbox, kappa, dt, t_end, u0, source, dirichlet, reaction=0.0):
    """Solve on an ``n x n`` FiPy ``Grid2D`` with time-dependent Dirichlet faces.

    Returns ``(cell_centers (M, 2), u (M,))``.  Grid cell centres coincide with
    Thermalio's ``generate_square_polygonal_mesh(nx=n, ny=n, bbox)`` centres.
    """
    os.environ.setdefault("FIPY_SOLVERS", "scipy")
    import fipy as fp

    x0, x1, y0, y1 = bbox
    dx, dy = (x1 - x0) / n, (y1 - y0) / n
    mesh = fp.Grid2D(dx=dx, dy=dy, nx=n, ny=n) + ((x0,), (y0,))
    xc, yc = (np.asarray(v) for v in mesh.cellCenters)
    xf, yf = (np.asarray(v) for v in mesh.faceCenters)
    phi = fp.CellVariable(mesh=mesh, value=np.asarray(u0(xc, yc), dtype=float))
    bc = fp.FaceVariable(mesh=mesh, value=0.0)
    phi.constrain(bc, where=mesh.exteriorFaces)
    src = fp.CellVariable(mesh=mesh, value=0.0)
    K = _kappa_matrix(kappa)
    coeff = fp.FaceVariable(mesh=mesh, rank=2, value=tuple(map(tuple, K)))
    rhs = fp.DiffusionTerm(coeff=coeff) + src
    if reaction:
        rhs = rhs - fp.ImplicitSourceTerm(coeff=float(reaction))
    eq = fp.TransientTerm() == rhs
    nsteps = max(int(round(t_end / dt)), 1)
    dt = t_end / nsteps
    t = 0.0
    for _ in range(nsteps):
        t += dt
        src.setValue(np.asarray(source(xc, yc, t), dtype=float))
        bc.setValue(np.asarray(dirichlet(xf, yf, t), dtype=float))
        eq.solve(var=phi, dt=dt)
    return np.column_stack([xc, yc]), np.asarray(phi.value, dtype=float)


# ---------------------------------------------------------------------------
# scikit-fem
# ---------------------------------------------------------------------------


class SkfemResult:
    """scikit-fem solution with point evaluation and exact-solution L2 errors."""

    def __init__(self, mesh, basis, u):
        self.mesh, self.basis, self.u = mesh, basis, u

    @property
    def ndofs(self):
        return int(self.basis.N)

    def evaluate(self, points):
        pts = np.asarray(points, dtype=float)
        return self.basis.probes(pts.T) @ self.u

    def l2_error(self, exact, t):
        from skfem import Functional

        uh = self.basis.interpolate(self.u)

        @Functional
        def err2(w):
            return (w["uh"] - exact(w.x[0], w.x[1], t)) ** 2

        @Functional
        def ex2(w):
            return exact(w.x[0], w.x[1], t) ** 2

        return float(np.sqrt(err2.assemble(self.basis, uh=uh) / ex2.assemble(self.basis)))


def skfem_heat_solve(n, bbox, kappa, dt, t_end, u0, source, dirichlet=None, robin=None,
                     reaction=0.0, order=1):
    """Backward-Euler P1/P2 Galerkin solve on a structured triangulation."""
    from skfem import (
        Basis, BilinearForm, ElementTriP1, ElementTriP2, FacetBasis, LinearForm, MeshTri, asm,
        condense, solve,
    )

    if (dirichlet is None) == (robin is None):
        raise ValueError("give exactly one of dirichlet / robin.")
    x0, x1, y0, y1 = bbox
    mesh = MeshTri.init_tensor(np.linspace(x0, x1, n + 1), np.linspace(y0, y1, n + 1))
    element = {1: ElementTriP1(), 2: ElementTriP2()}[int(order)]
    basis = Basis(mesh, element)
    K = _kappa_matrix(kappa)

    @BilinearForm
    def stiffness(u, v, w):
        gu, gv = u.grad, v.grad
        return (K[0, 0] * gu[0] + K[0, 1] * gu[1]) * gv[0] + (K[1, 0] * gu[0] + K[1, 1] * gu[1]) * gv[1]

    @BilinearForm
    def mass(u, v, w):
        return u * v

    A = asm(stiffness, basis)
    Mm = asm(mass, basis)
    if reaction:
        A = A + float(reaction) * Mm
    fbasis = None
    if robin is not None:
        h, robin_value = robin
        fbasis = FacetBasis(mesh, element)

        @BilinearForm
        def robin_lhs(u, v, w):
            return float(h) * u * v

        A = A + asm(robin_lhs, fbasis)
    nsteps = max(int(round(t_end / dt)), 1)
    dt = t_end / nsteps
    lhs = (Mm + dt * A).tocsr()
    doflocs = basis.doflocs
    u = np.asarray(u0(doflocs[0], doflocs[1]), dtype=float)
    bdofs = basis.get_dofs().all() if dirichlet is not None else None
    t = 0.0
    for _ in range(nsteps):
        t += dt

        @LinearForm
        def load(v, w, _t=t):
            return source(w.x[0], w.x[1], _t) * v

        rhs = Mm @ u + dt * asm(load, basis)
        if robin is not None:
            @LinearForm
            def robin_rhs(v, w, _t=t):
                return robin_value(w.x[0], w.x[1], _t, w.n[0], w.n[1]) * v

            rhs = rhs + dt * asm(robin_rhs, fbasis)
            u = solve(lhs, rhs)
        else:
            g = np.asarray(dirichlet(doflocs[0], doflocs[1], t), dtype=float)
            u = solve(*condense(lhs, rhs, x=g, D=bdofs))
    return SkfemResult(mesh, basis, u)


# ---------------------------------------------------------------------------
# NGSolve
# ---------------------------------------------------------------------------


class NgsolveResult:
    def __init__(self, mesh, gfu, exact_cf, order, ndofs):
        self.mesh, self.gfu, self.exact_cf, self.order, self.ndofs = mesh, gfu, exact_cf, order, ndofs

    def evaluate(self, points):
        pts = np.asarray(points, dtype=float)
        return np.asarray(self.gfu(self.mesh(pts[:, 0], pts[:, 1]))).ravel()

    def l2_error(self):
        import ngsolve as ng

        q = 2 * self.order + 4
        e2 = ng.Integrate((self.gfu - self.exact_cf) ** 2, self.mesh, order=q)
        u2 = ng.Integrate(self.exact_cf ** 2, self.mesh, order=q)
        return float(np.sqrt(e2 / u2))


def ngsolve_heat_solve(n, problem, dt, t_end, order=1, threads=1):
    """Backward-Euler H1 (P``order``) Galerkin solve with NGSolve on a structured triangulation.

    ``threads > 1`` runs assembly, boundary projection and matrix-vector products
    under NGSolve's task manager.  The sparse Cholesky factorization is built
    once and reused for every step.  Dirichlet data are set on the boundary dofs
    every step (``GridFunction.Set(..., BND)``).
    """
    import sympy as sp
    import ngsolve as ng
    from ngsolve.meshes import MakeStructured2DMesh

    x0, x1, y0, y1 = problem.bbox
    mesh = MakeStructured2DMesh(quads=False, nx=n, ny=n,
                                mapping=lambda a, b: (x0 + (x1 - x0) * a, y0 + (y1 - y0) * b))
    tpar = ng.Parameter(0.0)
    modules = [{"sin": ng.sin, "cos": ng.cos, "exp": ng.exp}, "math"]
    to_cf = lambda e: sp.lambdify((problem.x, problem.y, problem.t), e, modules=modules)(ng.x, ng.y, tpar)
    exact_cf = to_cf(problem.exact)
    source_cf = to_cf(problem.source)
    robin = problem.robin_h is not None
    fes = ng.H1(mesh, order=int(order), dirichlet="" if robin else ".*")
    u, v = fes.TnT()
    K = ng.CF(tuple(problem.K.ravel()), dims=(2, 2))
    nsteps = max(int(round(t_end / dt)), 1)
    dt = t_end / nsteps
    ng.SetNumThreads(int(threads))

    def run():
        mass = ng.BilinearForm(u * v * ng.dx).Assemble()
        lhs = ng.BilinearForm(fes)
        lhs += (u * v + dt * ((K * ng.grad(u)) * ng.grad(v) + problem.reaction * u * v)) * ng.dx
        f = ng.LinearForm(fes)
        f += source_cf * v * ng.dx
        if robin:
            nrm = ng.specialcf.normal(2)
            k = float(problem.K[0, 0])
            gx, gy = to_cf(problem.grad[0]), to_cf(problem.grad[1])
            value_cf = k * (gx * nrm[0] + gy * nrm[1]) + problem.robin_h * exact_cf
            lhs += dt * problem.robin_h * u * v * ng.ds
            f += value_cf * v * ng.ds
        lhs.Assemble()
        inv = lhs.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky")
        gfu = ng.GridFunction(fes)
        tpar.Set(0.0)
        gfu.Set(exact_cf)
        old, res = gfu.vec.CreateVector(), gfu.vec.CreateVector()
        for step in range(nsteps):
            old.data = gfu.vec
            tpar.Set((step + 1) * dt)
            f.Assemble()
            if not robin:
                gfu.Set(exact_cf, ng.BND)
            res.data = mass.mat * old + dt * f.vec - lhs.mat * gfu.vec
            gfu.vec.data += inv * res
        return gfu

    if threads > 1:
        with ng.TaskManager():
            gfu = run()
    else:
        gfu = run()
    return NgsolveResult(mesh, gfu, exact_cf, int(order), int(fes.ndof))


# ---------------------------------------------------------------------------
# PorePy
# ---------------------------------------------------------------------------


def porepy_heat_solve(n, bbox, kappa, dt, t_end, u0, source, dirichlet, reaction=0.0, method="tpfa",
                      nodes=None):
    """Backward-Euler cell-centred FV solve with PorePy's TPFA or MPFA-O flux.

    The grid is an ``n x n`` ``CartGrid`` on ``bbox``; ``nodes`` (``(n+1)^2 x 2``,
    x fastest) moves its nodes, e.g. to Thermalio's skewed mesh, so both codes
    discretize exactly the same cells.  Dirichlet data are imposed on boundary
    faces.  The flux matrices come from PorePy (``Tpfa`` / ``Mpfa``
    ``discretize``); the time loop is the plain backward-Euler balance
    ``|K| (u - u_old) / dt + div(F u + F_b g) = |K| Q``.  Returns
    ``(centers, u, volumes)``.
    """
    import porepy as pp
    import scipy.sparse as sps
    from scipy.sparse.linalg import splu

    x0, x1, y0, y1 = bbox
    sd = pp.CartGrid([n, n], physdims=[x1 - x0, y1 - y0])
    sd.nodes[0] += x0
    sd.nodes[1] += y0
    if nodes is not None:
        sd.nodes[:2] = np.asarray(nodes, dtype=float).T
    sd.compute_geometry()
    nc = sd.num_cells
    K = _kappa_matrix(kappa)
    perm = pp.SecondOrderTensor(kxx=np.full(nc, K[0, 0]), kyy=np.full(nc, K[1, 1]), kxy=np.full(nc, K[0, 1]))
    bfaces = sd.get_all_boundary_faces()
    data = pp.initialize_data({}, "heat", {"second_order_tensor": perm,
                                           "bc": pp.BoundaryCondition(sd, bfaces, "dir"),
                                           "bc_values": np.zeros(sd.num_faces)})
    {"tpfa": pp.Tpfa, "mpfa": pp.Mpfa}[method]("heat").discretize(sd, data)
    mats = data[pp.DISCRETIZATION_MATRICES]["heat"]
    div = sd.cell_faces.T.tocsr()
    A = (div @ mats["flux"]).tocsr()
    Bb = (div @ mats["bound_flux"]).tocsr()
    vol = sd.cell_volumes
    nsteps = max(int(round(t_end / dt)), 1)
    dt = t_end / nsteps
    lu = splu((sps.diags(vol * (1.0 + dt * float(reaction))) + dt * A).tocsc())
    xc, yc = sd.cell_centers[0], sd.cell_centers[1]
    xf, yf = sd.face_centers[0, bfaces], sd.face_centers[1, bfaces]
    u = np.asarray(u0(xc, yc), dtype=float)
    g = np.zeros(sd.num_faces)
    t = 0.0
    for _ in range(nsteps):
        t += dt
        g[bfaces] = dirichlet(xf, yf, t)
        u = lu.solve(vol * u + dt * vol * np.asarray(source(xc, yc, t), dtype=float) - dt * (Bb @ g))
    return np.column_stack([xc, yc]), u, vol


# ---------------------------------------------------------------------------
# py-pde
# ---------------------------------------------------------------------------


def _pde_str(expr):
    import sympy as sp

    return sp.sstr(sp.sympify(expr))


def pypde_heat_solve(n, problem, dt, t_end, scheme="implicit"):
    """Finite-difference solve with py-pde on an ``n x n`` cell-centred Cartesian grid.

    ``scheme="implicit"`` is py-pde's backward Euler (fixed-point iterated to
    1e-12); ``"euler"`` its explicit Euler.  Dirichlet data are imposed on the
    faces through ghost cells; Robin data through py-pde's mixed condition
    ``du/dn + (h/k) u = value/k``.

    py-pde compiles the right-hand side and the time stepper with Numba before
    the first step, so the cost is measured in two parts: ``compile`` (building
    the compiled stepper) and ``stepping`` (running it for every step).
    ``total`` is their sum, the wall time of one fresh solve.  Returns
    ``(centers, u, times)``.
    """
    import time

    import pde
    from pde.solvers import EulerSolver, ImplicitSolver

    x0, x1, y0, y1 = problem.bbox
    grid = pde.CartesianGrid([[x0, x1], [y0, y1]], [n, n])
    K, r = problem.K.tolist(), problem.reaction
    terms = [f"{K[0][0]!r}*d2_dx2(u)", f"{K[1][1]!r}*d2_dy2(u)"]
    if K[0][1] + K[1][0] != 0.0:
        terms.append(f"{K[0][1] + K[1][0]!r}*d_dx(d_dy(u))")
    if r:
        terms.append(f"-{r!r}*u")
    rhs = " + ".join(terms) + f" + ({_pde_str(problem.source)})"
    if problem.robin_h is None:
        bc = {"value_expression": _pde_str(problem.exact)}
    else:
        k = float(K[0][0])
        side = lambda nx, ny: {"type": "mixed_expression", "value": problem.robin_h / k,
                               "const": _pde_str(problem.robin_value(nx, ny) / k)}
        bc = {"x-": side(-1, 0), "x+": side(1, 0), "y-": side(0, -1), "y+": side(0, 1)}
    eq = pde.PDE({"u": rhs}, bc=bc)
    state = pde.ScalarField.from_expression(grid, _pde_str(problem.exact.subs(problem.t, 0)))
    nsteps = max(int(round(t_end / dt)), 1)
    dt = t_end / nsteps
    t0 = time.perf_counter()
    if scheme == "implicit":
        solver = ImplicitSolver(eq, maxerror=1e-12, maxiter=500)
    else:
        solver = EulerSolver(eq, adaptive=False)
    stepper = solver.make_stepper(state, dt=dt)
    stepper(state.copy(), 0.0, dt)  # first call finishes the compilation
    compile_time = time.perf_counter() - t0
    res = state.copy()
    t0 = time.perf_counter()
    stepper(res, 0.0, t_end)
    stepping = time.perf_counter() - t0
    times = {"compile": compile_time, "stepping": stepping, "total": compile_time + stepping}
    xy = grid.cell_coords  # (n, n, 2) indexed [ix, iy]
    centers = np.column_stack([xy[..., 0].T.ravel(), xy[..., 1].T.ravel()])  # x fastest, as Thermalio
    return centers, np.asarray(res.data, dtype=float).T.ravel(), times
