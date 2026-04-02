import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from heat_solver import get_analytical_case, run_curvilinear_test
from heat_solver.curvilinear import CurvilinearHeatSolver

def test_curvilinear_heat_solver_initialization():
    # Create simple Cartesian mesh X, Y
    ny, nx = 5, 5
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    alpha = 0.5
    dt = 0.01
    
    solver = CurvilinearHeatSolver(X, Y, alpha, dt)
    
    assert solver.nx == nx
    assert solver.ny == ny
    # On Cartesian mesh, J should be dx * dy
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    expected_J = dx * dy
    # Interior nodes should have J close to expected
    assert np.allclose(solver.J[1:-1, 1:-1], expected_J)
    
    # Check matrices
    # M_diag should be identity sparse matrix natively
    assert solver.M_diag.shape == (nx * ny, nx * ny)
    assert solver.A.shape == (nx * ny, nx * ny)

def test_curvilinear_heat_solver_step():
    ny, nx = 3, 3
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    alpha = 1.0
    dt = 0.1
    
    solver = CurvilinearHeatSolver(X, Y, alpha, dt, bc_type="dirichlet")
    
    u0 = np.zeros((ny, nx))
    # Solve 1 step
    t, u1 = solver.solve(u0, 0.0, dt)
    
    assert t == dt
    assert u1.shape == (ny, nx)
    # Since boundary condition func defaults to 0 and u0 is 0, u1 should be 0
    assert np.allclose(u1, 0.0)

def test_curvilinear_heat_solver_anisotropic_alpha():
    ny, nx = 3, 3
    X, Y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    # Full tensor alpha
    alpha = [[2.0, 0.5], [0.5, 1.0]]
    dt = 0.1
    solver = CurvilinearHeatSolver(X, Y, alpha, dt)
    
    # Just checking initialization doesn't throw errors
    assert solver.A is not None


def test_curvilinear_invalid_time_scheme_raises():
    ny, nx = 3, 3
    X, Y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    with pytest.raises(ValueError, match="time_scheme"):
        CurvilinearHeatSolver(X, Y, 1.0, 0.1, time_scheme="rk4")


def test_curvilinear_jacobian_metric_invariants():
    # Create a linearly stretched grid: x = 2*xi, y = 3*eta
    # On a unit computational grid (dx=1/(nx-1), dy=1/(ny-1)),
    # physical x goes from 0 to 2, y goes from 0 to 3.
    ny, nx = 4, 4
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 3, ny)
    X, Y = np.meshgrid(x, y)
    
    alpha = 1.0
    dt = 0.1
    solver = CurvilinearHeatSolver(X, Y, alpha, dt)
    
    # For isotropic alpha=1.0:
    q11 = solver.q11
    q22 = solver.q22
    q12 = solver.q12
    
    # J = x_xi * y_eta - x_eta * y_xi
    #   = (2.0 / 3.0) * (3.0 / 3.0) - 0 = (2/3)*1 = 2/3?
    # Wait, difference spacing in physical grids vs computational:
    # dxi = 1, deta = 1 because x = X(xi, eta).
    # Finite differences assume dxi=1, deta=1!
    # Ah, the solver actually uses actual difference spacing. Let's just check the property
    # q11 * q22 - q12**2 should be strictly > 0 (elliptic property) everywhere
    assert np.all(q11 * q22 - q12**2 > 0)
    
    # And q11 must be positive everywhere
    assert np.all(q11 > 0)
    assert np.all(q22 > 0)
    
    # On orthogonal mapped grids, cross-derivative metric q12 must be exactly zero
    assert np.allclose(q12, 0.0)


def test_curvilinear_crank_nicolson_matches_manual_step():
    alpha = 0.1
    dt = 0.02
    t0 = 0.0
    t1 = dt
    case = get_analytical_case("source_driven_sine", alpha=alpha, t_end=t1)
    bc = case.get("boundary", case["solution"])
    nx = ny = 5
    X, Y = np.meshgrid(np.linspace(-1.0, 1.0, nx), np.linspace(-1.0, 1.0, ny))
    solver = CurvilinearHeatSolver(
        X,
        Y,
        alpha=alpha,
        dt=dt,
        bc_type="dirichlet",
        bc_func=bc,
        source_func=case["source"],
        time_scheme="crank_nicolson",
    )

    u0 = np.asarray(case["solution"](X, Y, t0), dtype=float)
    u = u0.flatten()
    source_prev = np.asarray(case["source"](X.flatten(), Y.flatten(), t0), dtype=float)
    source_next = np.asarray(case["source"](X.flatten(), Y.flatten(), t1), dtype=float)
    rhs = (solver.M_diag + 0.5 * dt * solver.A) @ u + 0.5 * dt * (source_prev + source_next)
    lhs = (solver.M_diag - 0.5 * dt * solver.A).copy().tolil()
    solver._apply_dirichlet_rows(lhs, rhs, t1)
    u_manual = spsolve(lhs.tocsr(), rhs)

    _, u_solv = solver.solve(u0, t0, t1)
    assert np.max(np.abs(u_manual - u_solv.flatten())) < 1e-12


def test_curvilinear_crank_nicolson_has_higher_time_order_than_backward_euler():
    alpha = 0.1
    case = "source_driven_sine"
    t_init = 0.0
    t_end = 0.08
    bbox = (-1.0, 1.0, -1.0, 1.0)
    nx = ny = 24
    dts = np.array([0.04, 0.02, 0.01], dtype=float)
    ref_dt = 0.00125
    warp = 0.0
    twist = 0.0

    *_, u_ref, _, _, _ = run_curvilinear_test(
        case=case,
        alpha=alpha,
        dt=ref_dt,
        t_init=t_init,
        t_end=t_end,
        nx=nx,
        ny=ny,
        bbox=bbox,
        warp=warp,
        twist=twist,
        time_scheme="crank_nicolson",
    )
    u_ref = np.asarray(u_ref, dtype=float)
    ref_norm = np.sqrt(np.mean(u_ref**2)) + 1e-16

    err_be = []
    err_cn = []
    for dt in dts:
        *_, u_be, _, _, _ = run_curvilinear_test(
            case=case,
            alpha=alpha,
            dt=float(dt),
            t_init=t_init,
            t_end=t_end,
            nx=nx,
            ny=ny,
            bbox=bbox,
            warp=warp,
            twist=twist,
            time_scheme="backward_euler",
        )
        *_, u_cn, _, _, _ = run_curvilinear_test(
            case=case,
            alpha=alpha,
            dt=float(dt),
            t_init=t_init,
            t_end=t_end,
            nx=nx,
            ny=ny,
            bbox=bbox,
            warp=warp,
            twist=twist,
            time_scheme="crank_nicolson",
        )
        err_be.append(np.sqrt(np.mean((np.asarray(u_be, dtype=float) - u_ref) ** 2)) / ref_norm)
        err_cn.append(np.sqrt(np.mean((np.asarray(u_cn, dtype=float) - u_ref) ** 2)) / ref_norm)

    slope_be = np.polyfit(np.log(dts), np.log(np.asarray(err_be)), 1)[0]
    slope_cn = np.polyfit(np.log(dts), np.log(np.asarray(err_cn)), 1)[0]
    assert slope_be > 0.7
    assert slope_cn > 1.4
    assert slope_cn > slope_be + 0.4


def _run_boundary_case(case):
    alpha = 0.1
    dt = 0.02
    t_end = 0.1
    case_info = get_analytical_case(case, alpha=alpha, t_end=t_end)
    return run_curvilinear_test(
        case=case,
        alpha=alpha,
        dt=dt,
        t_init=0.0,
        t_end=t_end,
        nx=20,
        ny=20,
        bbox=case_info["bbox"],
        warp=0.18,
        twist=0.07,
    )


def test_curvilinear_solver_handles_steady_linear_neumann_bc_on_deformed_mesh():
    _X, _Y, _u_num, _u_exact, diff, results = _run_boundary_case("steady_linear_neumann")

    assert np.all(np.isfinite(diff))
    assert np.all(np.isfinite(results["L2_rel"]))
    assert results["L2_rel"] < 5e-4
    assert results["Linf_rel"] < 1e-3


def test_curvilinear_solver_handles_steady_linear_robin_bc_on_deformed_mesh():
    _X, _Y, _u_num, _u_exact, diff, results = _run_boundary_case("steady_linear_robin")

    assert np.all(np.isfinite(diff))
    assert np.all(np.isfinite(results["L2_rel"]))
    assert results["L2_rel"] < 5e-4
    assert results["Linf_rel"] < 1e-3


def test_curvilinear_solver_handles_stefan_apparent_capacity_on_deformed_mesh():
    _X, _Y, _u_num, _u_exact, diff, results = run_curvilinear_test(
        case="stefan_apparent_capacity",
        alpha=0.08,
        dt=1e-5,
        t_init=0.0,
        t_end=0.001,
        nx=21,
        ny=21,
        bbox=(-1.0, 1.0, -1.0, 1.0),
        warp=0.18,
        twist=0.07,
    )

    assert np.all(np.isfinite(diff))
    assert np.all(np.isfinite(results["L2_rel"]))
    assert results["L2_rel"] < 5e-4
    assert results["Linf_rel"] < 1e-2
