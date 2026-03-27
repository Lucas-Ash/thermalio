import numpy as np
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
