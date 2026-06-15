import numpy as np
import pytest

pytest.importorskip("sympy")

from heat_solver.geometry import polygon_area_and_centroid
from heat_solver.meshes import (
    generate_nonorthogonal_tiled_polygonal_mesh,
    generate_square_polygonal_mesh,
)
from heat_solver.mms import manufactured_case
from heat_solver.polygonal import PolygonalHeatSolver
from heat_solver.reference_fd import relative_l2, solve_fd_reference

BBOX = (-1.0, 1.0, -1.0, 1.0)
ALPHA = 0.1


def _mms_case():
    return manufactured_case("exp(-t)*sin(pi*x)*sin(pi*y)", alpha=ALPHA, model="diffusion")


def _fv_solve(vertices, polygons, centers, case, dt=1e-4, t_end=0.02):
    solver = PolygonalHeatSolver(
        vertices, polygons, ALPHA, dt, bc_type="dirichlet",
        bc_func=case["boundary"], source_func=case["source"],
    )
    u0 = case["solution"](centers[:, 0], centers[:, 1], 0.0)
    _, u = solver.solve(u0, 0.0, t_end)
    return u


def test_fd_reference_converges_second_order():
    case = _mms_case()
    errs = []
    for n in (20, 40):
        X, Y, u = solve_fd_reference(
            BBOX, n, ALPHA, dt=1e-4, t_end=0.02,
            solution=case["solution"], source=case["source"], bc_func=case["boundary"],
        )
        errs.append(relative_l2(u, case["solution"](X, Y, 0.02)))
    assert np.log2(errs[0] / errs[1]) > 1.8
    assert errs[1] < 1e-3


def _rel_l2_vs_exact(u, centers, case, t_end):
    ue = case["solution"](centers[:, 0], centers[:, 1], t_end)
    return float(np.sqrt(np.sum((u - ue) ** 2)) / (np.sqrt(np.sum(ue**2)) + 1e-300))


def test_fv_matches_independent_fd_on_uniform_grid():
    # On a uniform grid, TPFA finite volume and 5-point finite difference are two
    # independent implementations; they must achieve the same accuracy against the
    # exact solution (artifact-free comparison: errors vs exact, no interpolation).
    case = _mms_case()
    n = 40
    X, Y, u_fd = solve_fd_reference(
        BBOX, n, ALPHA, dt=1e-4, t_end=0.02,
        solution=case["solution"], source=case["source"], bc_func=case["boundary"],
    )
    fd_err = relative_l2(u_fd, case["solution"](X, Y, 0.02))
    v, p, c = generate_square_polygonal_mesh(nx=n, ny=n, bbox=BBOX)
    fv_err = _rel_l2_vs_exact(_fv_solve(v, p, c, case), c, case, 0.02)
    assert fd_err < 2e-4 and fv_err < 2e-4
    # The two independent codes coincide on a uniform grid (errors within 5%).
    assert abs(fv_err - fd_err) < 0.05 * max(fv_err, fd_err)


def test_fv_tiled_agrees_with_independent_fd():
    # A genuinely different FV discretization (skewed tiled mesh) and the
    # independent FD reference both converge to the same exact solution.
    case = _mms_case()
    X, Y, u_fd = solve_fd_reference(
        BBOX, 60, ALPHA, dt=1e-4, t_end=0.02,
        solution=case["solution"], source=case["source"], bc_func=case["boundary"],
    )
    fd_err = relative_l2(u_fd, case["solution"](X, Y, 0.02))
    v, p, c = generate_nonorthogonal_tiled_polygonal_mesh(nx_tiles=12, ny_tiles=12, bbox=BBOX)
    fv_err = _rel_l2_vs_exact(_fv_solve(v, p, c, case), c, case, 0.02)
    assert fd_err < 5e-3 and fv_err < 5e-3
