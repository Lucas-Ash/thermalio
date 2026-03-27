import numpy as np
import pytest
from scipy.sparse import eye, random as sparse_random

from heat_solver.drivers import run_square_polygonal_test
from heat_solver.polygonal import PolygonalHeatSolver, _solve_sparse_linear_system

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


def test_polygonal_invalid_time_scheme_raises():
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    polygons = [[0, 1, 2, 3]]
    with pytest.raises(ValueError, match="time_scheme"):
        PolygonalHeatSolver(vertices, polygons, 1.0, 0.1, time_scheme="rk4")


def test_polygonal_crank_nicolson_matches_manual_one_step():
    """CN step must match explicit (M ± 0.5*dt*A) assembly (same as solver)."""
    from scipy.sparse.linalg import spsolve

    from heat_solver.cases import get_analytical_case
    from heat_solver.meshes import generate_square_polygonal_mesh

    alpha = 0.1
    case = "source_driven_sine"
    bbox = (0.0, 1.0, 0.0, 1.0)
    vertices, polygons, _ = generate_square_polygonal_mesh(nx=4, ny=4, bbox=bbox)
    info = get_analytical_case(case, alpha=alpha, t_end=0.02)
    bc = info.get("boundary", info["solution"])
    src = info.get("source", lambda x, y, t: 0.0)
    sol = info["solution"]

    solver = PolygonalHeatSolver(
        vertices,
        polygons,
        alpha,
        dt=0.02,
        bc_type="dirichlet",
        bc_func=bc,
        source_func=src,
        nonorthogonal_correction=True,
        time_scheme="crank_nicolson",
    )
    solver._assemble_system()
    M, A = solver.M_diag, solver.A
    cx, cy = solver.cell_centers[:, 0], solver.cell_centers[:, 1]
    t0, t1 = 0.0, 0.02
    u = np.asarray(sol(cx, cy, t0), dtype=float)
    dt = 0.02
    sn = np.asarray(src(cx, cy, t1), dtype=float)
    sp = np.asarray(src(cx, cy, t0), dtype=float)
    src_avg = solver.cell_areas * (sn + sp)
    rhs = (M - 0.5 * dt * A) @ u + 0.5 * dt * src_avg
    bc_vals = bc(cx, cy, t1)
    for i in np.where(solver.is_boundary)[0]:
        rhs[i] = bc_vals[i]
    lhs = (M + 0.5 * dt * A).tolil()
    for i in np.where(solver.is_boundary)[0]:
        lhs.rows[i] = [i]
        lhs.data[i] = [1.0]
    lhs = lhs.tocsr()
    u_manual = spsolve(lhs, rhs)

    _, u_solv = solver.solve(u, t0, t1)
    assert np.max(np.abs(u_manual - u_solv)) < 1e-12


def test_polygonal_crank_nicolson_matches_backward_euler_small_dt():
    kwargs = dict(
        case="sine_mode",
        alpha=0.1,
        dt=1e-4,
        t_init=0.0,
        t_end=2e-4,
        nx=6,
        ny=6,
        bbox=(-1.0, 1.0, -1.0, 1.0),
        nonorthogonal_correction=True,
    )
    *_, u_be, _, _, _ = run_square_polygonal_test(time_scheme="backward_euler", **kwargs)
    *_, u_cn, _, _, _ = run_square_polygonal_test(time_scheme="crank_nicolson", **kwargs)
    assert np.allclose(u_be, u_cn, rtol=1e-5, atol=1e-7)


def test_polygonal_invalid_linear_solver_raises():
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    polygons = [[0, 1, 2, 3]]
    with pytest.raises(ValueError, match="linear_solver"):
        PolygonalHeatSolver(vertices, polygons, 1.0, 0.1, linear_solver="gmres")


def test_sparse_linear_solver_helper_matches_direct():
    rng = np.random.default_rng(0)
    n = 32
    R = sparse_random(n, n, density=0.08, random_state=rng, dtype=float)
    A = (R + R.T) * 0.5 + 2.0 * eye(n, format="csr")
    x_true = rng.standard_normal(n)
    b = A @ x_true
    x_dir = _solve_sparse_linear_system(A, b, "direct", {}, x0=None)
    x_bc = _solve_sparse_linear_system(
        A, b, "bicgstab", {"rtol": 1e-12, "maxiter": 10_000}, x0=x_true.copy()
    )
    assert np.allclose(x_dir, x_bc, rtol=1e-8, atol=1e-9)
    x_cg = _solve_sparse_linear_system(A, b, "cg", {"rtol": 1e-12, "maxiter": 10_000}, x0=x_true.copy())
    assert np.allclose(x_dir, x_cg, rtol=1e-8, atol=1e-9)


def test_polygonal_bicgstab_matches_direct_small_mesh():
    kwargs = dict(
        case="sine_mode",
        alpha=0.1,
        dt=5e-3,
        t_init=0.0,
        t_end=0.02,
        nx=8,
        ny=8,
        bbox=(-1.0, 1.0, -1.0, 1.0),
        nonorthogonal_correction=True,
    )
    *_, u_direct, _, _, _ = run_square_polygonal_test(linear_solver="direct", **kwargs)
    *_, u_iter, _, _, _ = run_square_polygonal_test(
        linear_solver="bicgstab",
        linear_solver_options={"rtol": 1e-12, "atol": 0.0},
        **kwargs,
    )
    assert np.allclose(u_direct, u_iter, rtol=1e-9, atol=1e-10)


def test_polygonal_iterative_matches_direct_high_resolution_analytical():
    """BiCGSTAB implicit steps agree with the direct sparse solve on a fine square mesh."""
    kwargs = dict(
        case="source_driven_sine",
        alpha=0.1,
        dt=2e-3,
        t_init=0.0,
        t_end=0.05,
        nx=48,
        ny=48,
        bbox=(0.0, 1.0, 0.0, 1.0),
        nonorthogonal_correction=True,
    )
    *_, u_direct, _, _, _ = run_square_polygonal_test(linear_solver="direct", **kwargs)
    *_, u_iter, _, _, _ = run_square_polygonal_test(
        linear_solver="bicgstab",
        linear_solver_options={"rtol": 1e-11, "maxiter": 50_000},
        **kwargs,
    )
    assert np.allclose(u_direct, u_iter, rtol=1e-8, atol=1e-9)


def test_polygonal_stefan_apparent_capacity_reduces_error():
    from heat_solver.cases import get_analytical_case
    from heat_solver.meshes import generate_square_polygonal_mesh

    alpha = 0.08
    dt = 1e-3
    t_init = 0.0
    t_end = 0.05
    case = get_analytical_case("stefan_apparent_capacity", alpha=alpha, t_end=t_end)
    vertices, polygons, _ = generate_square_polygonal_mesh(nx=20, ny=20, bbox=case["bbox"])
    centers = np.array([np.mean(vertices[np.asarray(poly, dtype=int)], axis=0) for poly in polygons])
    u0 = case["solution"](centers[:, 0], centers[:, 1], t_init)
    u_exact = case["solution"](centers[:, 0], centers[:, 1], t_end)

    solver_no_phase = PolygonalHeatSolver(
        vertices,
        polygons,
        alpha=alpha,
        dt=dt,
        bc_type="dirichlet",
        bc_func=case["solution"],
        source_func=case["source"],
    )
    _, u_no_phase = solver_no_phase.solve(u0, t_init, t_end)
    err_no_phase = np.sqrt(np.mean((u_no_phase - u_exact) ** 2))

    solver_phase = PolygonalHeatSolver(
        vertices,
        polygons,
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

    assert err_phase < 0.4 * err_no_phase


def test_polygonal_stefan_crank_nicolson_has_higher_time_order_than_backward_euler():
    from heat_solver.geometry import polygon_area_and_centroid
    from heat_solver.meshes import generate_square_polygonal_mesh

    alpha = 0.08
    case = "stefan_apparent_capacity"
    t_init = 0.0
    t_end = 0.02
    bbox = (-1.0, 1.0, -1.0, 1.0)
    nx = ny = 12
    dts = np.array([0.01, 0.005, 0.0025], dtype=float)
    phase_opts = {"max_iters": 80, "tol": 1e-10, "relaxation": 0.7}

    verts, polys, _centers, u_ref, _, _, _ = run_square_polygonal_test(
        case=case,
        alpha=alpha,
        dt=6.25e-4,
        t_init=t_init,
        t_end=t_end,
        nx=nx,
        ny=ny,
        bbox=bbox,
        nonorthogonal_correction=True,
        time_scheme="crank_nicolson",
        phase_change_options=phase_opts,
    )
    areas = np.array([polygon_area_and_centroid(verts[p])[0] for p in polys], dtype=float)
    ref_norm = np.sqrt(np.sum(areas * u_ref**2)) + 1e-16

    err_be = []
    err_cn = []
    for dt in dts:
        *_, u_be, _, _, _ = run_square_polygonal_test(
            case=case,
            alpha=alpha,
            dt=float(dt),
            t_init=t_init,
            t_end=t_end,
            nx=nx,
            ny=ny,
            bbox=bbox,
            nonorthogonal_correction=True,
            time_scheme="backward_euler",
            phase_change_options=phase_opts,
        )
        *_, u_cn, _, _, _ = run_square_polygonal_test(
            case=case,
            alpha=alpha,
            dt=float(dt),
            t_init=t_init,
            t_end=t_end,
            nx=nx,
            ny=ny,
            bbox=bbox,
            nonorthogonal_correction=True,
            time_scheme="crank_nicolson",
            phase_change_options=phase_opts,
        )
        err_be.append(np.sqrt(np.sum(areas * (u_be - u_ref) ** 2)) / ref_norm)
        err_cn.append(np.sqrt(np.sum(areas * (u_cn - u_ref) ** 2)) / ref_norm)

    slope_be = np.polyfit(np.log(dts), np.log(np.asarray(err_be)), 1)[0]
    slope_cn = np.polyfit(np.log(dts), np.log(np.asarray(err_cn)), 1)[0]
    assert slope_be > 0.7
    assert slope_cn > 1.4
    assert slope_cn > slope_be + 0.4


def test_polygonal_temperature_dependent_diffusivity_manufactured_accuracy():
    *_, results = run_square_polygonal_test(
        case="temperature_dependent_diffusivity",
        alpha=0.12,
        dt=1e-3,
        t_init=0.0,
        t_end=0.05,
        nx=36,
        ny=36,
        bbox=(0.0, 1.0, 0.0, 1.0),
        nonorthogonal_correction=True,
    )
    assert float(results["L2_rel"]) < 1.5e-2
    assert float(results["Linf_rel"]) < 2.5e-2


def test_polygonal_radiative_manufactured_accuracy():
    *_, results = run_square_polygonal_test(
        case="radiative_manufactured",
        alpha=0.1,
        dt=1e-3,
        t_init=0.0,
        t_end=0.03,
        nx=32,
        ny=32,
        bbox=(0.0, 1.0, 0.0, 1.0),
        nonorthogonal_correction=True,
    )
    assert float(results["L2_rel"]) < 5e-2
    assert float(results["Linf_rel"]) < 8e-2


def test_polygonal_temperature_dependent_diffusivity_disallows_mpfa():
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    polygons = [[0, 1, 2, 3]]

    def alpha_temp(x, y, T):
        del x, y
        return 0.1 * (1.0 + 0.2 * T)

    with pytest.raises(ValueError, match="temperature-dependent diffusivity"):
        PolygonalHeatSolver(
            vertices,
            polygons,
            alpha=alpha_temp,
            dt=0.1,
            bc_type="dirichlet",
            flux_scheme="mpfa",
        )
