import numpy as np

from heat_solver.cases import get_analytical_case
from heat_solver.meshes import generate_nonuniform_delaunay
from heat_solver.triangular import NonUniformHeatSolver

def test_triangular_heat_solver_initialization():
    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    tris = np.array([[0, 1, 2]])
    alpha = 0.5
    dt = 0.1
    
    solver = NonUniformHeatSolver(points, tris, alpha, dt, bc_type="dirichlet")
    
    assert solver.N == 3
    # Check cv_area
    assert len(solver.cv_area) == 3
    assert solver.A is not None
    assert solver.M is not None

def test_triangular_heat_solver_tensor_alpha():
    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    tris = np.array([[0, 1, 2]])
    alpha = [[2.0, 0.5], [0.5, 1.0]]
    dt = 0.1
    
    # Just verifies the FEM tensor assembly doesn't crash
    solver = NonUniformHeatSolver(points, tris, alpha, dt, bc_type="dirichlet")
    assert solver.N == 3

def test_triangular_heat_solver_step():
    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    tris = np.array([[0, 1, 2]])
    alpha = 1.0
    dt = 0.1
    
    solver = NonUniformHeatSolver(points, tris, alpha, dt, bc_type="dirichlet")
    u0 = np.array([1.0, 1.0, 1.0])
    t, u = solver.solve(u0, 0.0, 0.1)
    
    assert t == 0.1
    assert len(u) == 3


def test_triangular_domain_area_conservation():
    from heat_solver.meshes import generate_nonuniform_delaunay
    # 2x2 square box domain: Area = 4.0
    points, tris = generate_nonuniform_delaunay(nx=5, ny=5, bbox=(0, 2, 0, 2), jitter=0.0)
    solver = NonUniformHeatSolver(points, tris, alpha=1.0, dt=0.1)
    
    # The sum of all triangle areas should equal the total domain area exactly
    total_tri_area = np.sum(solver.tri_areas)
    assert np.isclose(total_tri_area, 4.0)

def test_triangular_stiffness_matrix_nullspace():
    from heat_solver.meshes import generate_nonuniform_delaunay
    points, tris = generate_nonuniform_delaunay(nx=3, ny=3, bbox=(0, 1, 0, 1), jitter=0.0)
    solver = NonUniformHeatSolver(points, tris, alpha=[[1.0, 0.5],[0.5, 1.0]], dt=0.1)
    
    # The pure stiffness matrix A (before boundary condition replacement)
    # represents the divergence of flux. For a uniform field (constant),
    # the flux is zero, so A * ones = 0.
    # Therefore, the row sums of A must be precisely zero!
    A_dense = solver.A.toarray()
    row_sums = np.sum(A_dense, axis=1)
    assert np.allclose(row_sums, 0.0, atol=1e-12)


def test_triangular_stefan_apparent_capacity_reduces_error():
    alpha = 0.08
    dt = 1e-3
    t_init = 0.0
    t_end = 0.05
    case = get_analytical_case("stefan_apparent_capacity", alpha=alpha, t_end=t_end)
    points, tris = generate_nonuniform_delaunay(nx=18, ny=18, bbox=case["bbox"], jitter=0.0, seed=3)
    u0 = case["solution"](points[:, 0], points[:, 1], t_init)
    u_exact = case["solution"](points[:, 0], points[:, 1], t_end)

    solver_no_phase = NonUniformHeatSolver(
        points,
        tris,
        alpha=alpha,
        dt=dt,
        bc_type="dirichlet",
        bc_func=case["solution"],
        source_func=case["source"],
    )
    _, u_no_phase = solver_no_phase.solve(u0, t_init, t_end)
    err_no_phase = np.sqrt(np.mean((u_no_phase - u_exact) ** 2))

    solver_phase = NonUniformHeatSolver(
        points,
        tris,
        alpha=alpha,
        dt=dt,
        bc_type="dirichlet",
        bc_func=case["solution"],
        source_func=case["source"],
        phase_change_model=case["phase_change_model"],
        phase_change_options=case["phase_change_options"],
    )
    _, u_phase = solver_phase.solve(u0, t_init, t_end)
    err_phase = np.sqrt(np.mean((u_phase - u_exact) ** 2))
    assert np.isfinite(err_phase)
    assert err_phase <= err_no_phase + 1e-12
    assert err_phase < 5e-3
