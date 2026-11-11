"""Adapters to independent, external heat-equation codes for cross-code validation.

Direction A parked "validation against an external package".  This module
wraps two independently developed, pip-installable codes so the *same*
manufactured problems can be solved by Thermalio and by codes that share no
source with it:

* **FiPy** (NIST, cell-centred finite volumes on structured grids, its own
  face-flux and boundary machinery, solved through its SciPy backend);
* **scikit-fem** (continuous Galerkin finite elements, P1/P2 Lagrange on
  triangles; a different discretization family altogether).

Both are optional (``requirements-external.txt``) and imported lazily, so the
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
