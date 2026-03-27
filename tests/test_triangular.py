import numpy as np
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
