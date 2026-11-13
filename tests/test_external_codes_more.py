"""Cross-code validation against NGSolve, PorePy and py-pde (each skipped if not installed).

On Cartesian meshes PorePy's TPFA and py-pde's finite differences discretize
the isotropic problem exactly as Thermalio does with ``dirichlet_mode="face"``
(five-point stencil, data on the boundary faces), so the solutions must agree
to round-off.  NGSolve is a different method (high-order finite elements) and
is checked against the exact solution.
"""

import numpy as np
import pytest
import sympy as sp

from heat_solver.dmp import anisotropic_tensor
from heat_solver.external_codes import SymbolicProblem
from heat_solver.meshes import generate_nonorthogonal_polygonal_mesh, generate_square_polygonal_mesh
from heat_solver.polygonal import PolygonalHeatSolver
from heat_solver.transport import ReactionDiffusionHeatSolver

x, y, t = sp.symbols("x y t")
BBOX = (-1.0, 1.0, -1.0, 1.0)
DT, T_END = 1e-3, 0.02
SINE = sp.exp(-t) * sp.sin(sp.pi * x) * sp.sin(sp.pi * y)
ISO = SymbolicProblem(SINE, BBOX, 0.1)


def _thermalio(problem, n, mesh=None, **kw):
    v, p, _ = mesh if mesh is not None else generate_square_polygonal_mesh(nx=n, ny=n, bbox=problem.bbox)
    ex = problem.numpy(problem.exact)
    s = PolygonalHeatSolver(v, p, problem.K, DT, bc_func=ex, source_func=problem.numpy(problem.source), **kw)
    c = s.cell_centers
    _, u = s.solve(ex(c[:, 0], c[:, 1], 0.0), 0.0, T_END)
    return c, u, s.cell_areas


def _rel(u, ue, w):
    return float(np.sqrt(np.sum(w * (u - ue) ** 2) / np.sum(w * ue**2)))


def _porepy(problem, n, method, nodes=None):
    from heat_solver.external_codes import porepy_heat_solve

    ex = problem.numpy(problem.exact)
    return porepy_heat_solve(n, problem.bbox, problem.K, DT, T_END, lambda a, b: ex(a, b, 0.0),
                             problem.numpy(problem.source), ex, reaction=problem.reaction, method=method, nodes=nodes)


def test_porepy_tpfa_matches_thermalio_face_dirichlet():
    pytest.importorskip("porepy")
    for n in (8, 16):
        c, u, _ = _thermalio(ISO, n, dirichlet_mode="face")
        c_pp, u_pp, _ = _porepy(ISO, n, "tpfa")
        assert np.allclose(c_pp, c)
        assert np.max(np.abs(u_pp - u)) <= 1e-12 * np.max(np.abs(u))
        c_pp, u_mpfa, _ = _porepy(ISO, n, "mpfa")  # MPFA-O reduces to TPFA on K-orthogonal grids
        assert np.max(np.abs(u_mpfa - u)) <= 1e-11 * np.max(np.abs(u))


def test_porepy_on_thermalio_skewed_mesh():
    pytest.importorskip("porepy")
    prob = SymbolicProblem(SINE, BBOX, anisotropic_tensor(ratio=5.0, angle=np.pi / 6) * 0.1)
    errs = {"mpfa": [], "thermalio": []}
    for n in (8, 16, 32):
        mesh = generate_nonorthogonal_polygonal_mesh(nx=n, ny=n, bbox=BBOX, skew=0.3)
        c_pp, u_pp, vol = _porepy(prob, n, "mpfa", nodes=mesh[0])
        c, u, areas = _thermalio(prob, n, mesh=mesh, nonorthogonal_correction=True)
        assert np.allclose(c_pp, c) and np.allclose(vol, areas)  # identical cells
        ex = prob.numpy(prob.exact)
        errs["mpfa"].append(_rel(u_pp, ex(c[:, 0], c[:, 1], T_END), vol))
        errs["thermalio"].append(_rel(u, ex(c[:, 0], c[:, 1], T_END), areas))
    assert all(np.diff(np.log(errs["mpfa"])) < 0)  # converges
    assert errs["thermalio"][-1] < errs["mpfa"][-1]


def test_pypde_matches_thermalio_face_dirichlet_and_robin():
    pytest.importorskip("pde")
    from heat_solver.external_codes import pypde_heat_solve

    n = 8
    c, u, _ = _thermalio(ISO, n, dirichlet_mode="face")
    c_pd, u_pd, times = pypde_heat_solve(n, ISO, DT, T_END)
    assert np.allclose(c_pd, c)
    assert np.max(np.abs(u_pd - u)) <= 1e-9 * np.max(np.abs(u))
    assert times["compile"] > 0.0 and times["stepping"] > 0.0
    # Robin + perfusion: py-pde's mixed ghost-cell condition is Thermalio's Robin face closure.
    pen = SymbolicProblem(sp.exp(-t) * (2 + sp.sin(sp.pi * x) * sp.cos(sp.pi * y)), (0, 1, 0, 1), 0.1,
                          reaction=2.0, robin_h=3.0)
    v, p, _ = generate_square_polygonal_mesh(nx=n, ny=n, bbox=pen.bbox)
    value = {(nx_, ny_): pen.numpy(pen.robin_value(nx_, ny_)) for nx_, ny_ in ((1, 0), (-1, 0), (0, 1), (0, -1))}

    def robin(a, b, tt, nx_, ny_):
        out = np.zeros_like(a)
        for (ex_, ey_), f in value.items():
            m = np.isclose(nx_, ex_) & np.isclose(ny_, ey_)
            out[m] = f(a[m], b[m], tt)
        return 3.0 * np.ones_like(a), out

    s = ReactionDiffusionHeatSolver(v, p, 0.1, DT, 2.0, bc_type="robin", bc_func=robin,
                                    source_func=pen.numpy(pen.source))
    ex = pen.numpy(pen.exact)
    cc = s.cell_centers
    _, u = s.solve(ex(cc[:, 0], cc[:, 1], 0.0), 0.0, T_END)
    _, u_pd, _ = pypde_heat_solve(n, pen, DT, T_END)
    assert np.max(np.abs(u_pd - u)) <= 1e-9 * np.max(np.abs(u))


@pytest.mark.parametrize("robin", [False, True])
def test_ngsolve_converges_at_the_expected_order(robin):
    pytest.importorskip("ngsolve")
    from heat_solver.external_codes import ngsolve_heat_solve

    prob = ISO if not robin else SymbolicProblem(sp.exp(-t) * (2 + sp.sin(sp.pi * x) * sp.cos(sp.pi * y)),
                                                 (0, 1, 0, 1), 0.1, reaction=2.0, robin_h=3.0)
    for order, min_rate in ((1, 1.8), (2, 2.6)):
        errs = [ngsolve_heat_solve(n, prob, 1e-4, 0.005, order=order).l2_error() for n in (8, 16)]
        assert np.log2(errs[0] / errs[1]) > min_rate, (order, errs)
    one = ngsolve_heat_solve(16, prob, 1e-4, 0.005, order=2)
    four = ngsolve_heat_solve(16, prob, 1e-4, 0.005, order=2, threads=4)
    pts = np.array([[0.3, 0.2], [0.55, 0.7]]) * (prob.bbox[1] - prob.bbox[0]) + prob.bbox[0]
    np.testing.assert_allclose(four.evaluate(pts), one.evaluate(pts), rtol=1e-10)
