import numpy as np
from heat_solver.polygonal import PolygonalHeatSolver

def test_polygonal_heat_solver_initialization():
    vertices = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ])
    polygons = [[0, 1, 2, 3]]
    alpha = 0.5
    dt = 0.1
    solver = PolygonalHeatSolver(vertices, polygons, alpha, dt, bc_type="dirichlet")
    
    assert solver.M == 1
    assert len(solver.cell_areas) == 1
    assert solver.cell_areas[0] == 1.0
    assert np.allclose(solver.cell_centers[0], [0.5, 0.5])
    
    # Check boundaries were parsed
    assert len(solver.boundary_faces["cells"]) == 4

def test_polygonal_heat_solver_tensor_alpha():
    vertices = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ])
    polygons = [[0, 1, 2, 3]]
    alpha = [[2.0, 0.5], [0.5, 1.0]]
    dt = 0.1
    solver = PolygonalHeatSolver(vertices, polygons, alpha, dt, bc_type="dirichlet", nonorthogonal_correction=True)
    
    # Verify gradient reconstruction executes successfully and doesn't crash on single cell
    assert solver.gradient_coeffs is not None
    assert len(solver.gradient_coeffs) == 1

def test_polygonal_heat_solver_solve():
    vertices = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ])
    polygons = [[0, 1, 2, 3]]
    alpha = 1.0
    dt = 0.1
    solver = PolygonalHeatSolver(vertices, polygons, alpha, dt, bc_type="dirichlet")
    u0 = np.array([1.0])
    t, u = solver.solve(u0, 0.0, 0.1)
    
    assert t == 0.1
    assert len(u) == 1
    # Dirichlet defaults to zero, should decay
    assert u[0] < 1.0


def test_polygonal_gradient_reconstruction_linear_exactness():
    # Test gradient reconstruction on a 2x2 square grid (4 cells)
    # The gradient reconstruction must strictly be exact for linear fields
    from heat_solver.meshes import generate_square_polygonal_mesh
    vertices, polygons, _ = generate_square_polygonal_mesh(nx=2, ny=2, bbox=(0, 2, 0, 2))
    solver = PolygonalHeatSolver(vertices, polygons, alpha=1.0, dt=0.1, bc_type="dirichlet")
    
    # Establish a linear field u(x, y) = 3x - 2y + 1
    # Gradient should be strictly [3, -2].
    u = 3.0 * solver.cell_centers[:, 0] - 2.0 * solver.cell_centers[:, 1] + 1.0
    
    # Apply least-squares reconstruction
    grads = np.zeros((4, 2))
    for i in range(4):
        for nb, coeff in solver.gradient_coeffs[i].items():
            grads[i] += coeff * u[nb]
    
    # Assert exact uniform gradient
    assert grads.shape == (4, 2)
    assert np.allclose(grads[:, 0], 3.0)
    assert np.allclose(grads[:, 1], -2.0)

def test_polygonal_boundary_geometry_invariants():
    from heat_solver.meshes import generate_square_polygonal_mesh
    vertices, polygons, _ = generate_square_polygonal_mesh(nx=1, ny=1, bbox=(0, 1, 0, 1))
    solver = PolygonalHeatSolver(vertices, polygons, alpha=1.0, dt=0.1, bc_type="dirichlet")
    
    # 1 square cell, 4 boundary faces
    assert len(solver.boundary_faces["normals"]) == 4
    assert len(solver.boundary_faces["lengths"]) == 4
    
    # Face lengths should all be 1.0
    assert np.allclose(solver.boundary_faces["lengths"], 1.0)
    
    # Boundary normals out of the single cell should sum to [0, 0] by the divergence theorem
    normals = solver.boundary_faces["normals"]
    assert np.allclose(np.sum(normals, axis=0), [0.0, 0.0])
