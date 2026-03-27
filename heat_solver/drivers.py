import matplotlib.pyplot as plt
import numpy as np

from .cases import build_error_report, get_analytical_case
from .geometry import polygon_area_and_centroid
from .meshes import (
    generate_hexagonal_polygonal_mesh,
    generate_mixed_polygonal_mesh,
    generate_nonuniform_delaunay,
    generate_nonorthogonal_polygonal_mesh,
    generate_nonorthogonal_tiled_polygonal_mesh,
    generate_square_polygonal_mesh,
)
from .plotting import create_delaunay_figure, create_polygonal_figure
from .polygonal import PolygonalHeatSolver
from .triangular import NonUniformHeatSolver
from .curvilinear import CurvilinearHeatSolver


def run_curvilinear_test(
    alpha=0.1,
    dt=5e-3,
    t_init=0.0,
    t_end=0.02,
    case="sine_mode",
    nx=40,
    ny=40,
    warp=0.1,
    bbox=(-1.0, 1.0, -1.0, 1.0),
    bc_type=None,
    bc_func=None,
):
    case_info = get_analytical_case(case, alpha=alpha, t_end=t_end)
    if bc_type is None:
        bc_type = case_info.get("bc_type", "dirichlet")
    if bc_type != "dirichlet":
        raise ValueError("Curvilinear solver only supports Dirichlet boundary conditions.")
    
    exact_solution = case_info["solution"]
    if bc_func is None:
        bc_func = case_info.get("boundary", exact_solution)
        
    x_min, x_max, y_min, y_max = bbox
    xi = np.linspace(x_min, x_max, nx)
    eta = np.linspace(y_min, y_max, ny)
    XI, ETA = np.meshgrid(xi, eta)
    
    L_x = x_max - x_min
    L_y = y_max - y_min
    X = XI + warp * L_x * np.sin(np.pi * (XI - x_min) / L_x) * np.sin(np.pi * (ETA - y_min) / L_y)
    Y = ETA + warp * L_y * np.sin(np.pi * (XI - x_min) / L_x) * np.sin(np.pi * (ETA - y_min) / L_y)
    
    source_func = case_info.get("source", lambda x, y, t: 0.0)
    
    def g(x, y, t):
        return bc_func(x, y, t)
        
    solver = CurvilinearHeatSolver(X, Y, alpha=alpha, dt=dt, bc_type=bc_type, bc_func=g, source_func=source_func)
    u0 = exact_solution(X, Y, t_init)
    t_final, u_num_grid = solver.solve(u0=u0, t0=t_init, t_end=t_end)
    
    u_exact_grid = exact_solution(X, Y, t_final)
    diff_grid = u_num_grid - u_exact_grid
    
    dxi = L_x / (nx - 1)
    deta = L_y / (ny - 1)
    weights = solver.J * dxi * deta
    
    results = build_error_report(
        weights=weights.flatten(),
        diff=diff_grid.flatten(),
        u_exact=u_exact_grid.flatten(),
        t_final=t_final,
        bbox=bbox,
        case=case_info["name"],
        N=nx * ny,
        M=(nx - 1) * (ny - 1),
    )
    
    return X, Y, u_num_grid, u_exact_grid, diff_grid, results

def run_polygonal_mesh_test(
    vertices,
    polygons,
    alpha=0.1,
    dt=5e-3,
    t_init=0.05,
    t_end=0.15,
    case="heat_kernel",
    bbox=None,
    nonorthogonal_correction=True,
    bc_type=None,
    bc_func=None,
    linear_solver="direct",
    linear_solver_options=None,
    time_scheme="backward_euler",
    flux_scheme="tpfa",
    flux_discretization="tpfa",
):
    case_info = get_analytical_case(case, alpha=alpha, t_end=t_end)
    if bbox is None:
        verts = np.asarray(vertices, dtype=float)
        bbox = (float(verts[:, 0].min()), float(verts[:, 0].max()), float(verts[:, 1].min()), float(verts[:, 1].max()))
    vertices = np.asarray(vertices, dtype=float)
    polygons = [list(poly) for poly in polygons]
    centers = np.array([polygon_area_and_centroid(vertices[poly])[1] for poly in polygons])
    exact_solution = case_info["solution"]
    source_func = case_info.get("source", lambda x, y, t: 0.0)

    if bc_type is None:
        bc_type = case_info.get("bc_type", "dirichlet")
    if bc_func is None:
        bc_func = case_info.get("boundary", exact_solution)

    solver = PolygonalHeatSolver(
        vertices,
        polygons,
        alpha=alpha,
        dt=dt,
        bc_type=bc_type,
        bc_func=bc_func,
        source_func=source_func,
        nonorthogonal_correction=nonorthogonal_correction,
        linear_solver=linear_solver,
        linear_solver_options=linear_solver_options,
        time_scheme=time_scheme,
        flux_scheme=flux_scheme,
        flux_discretization=flux_discretization,
    )
    u0 = exact_solution(centers[:, 0], centers[:, 1], t_init)
    t_final, u_num = solver.solve(u0=u0, t0=t_init, t_end=t_end)
    u_exact = exact_solution(centers[:, 0], centers[:, 1], t_final)
    diff = u_num - u_exact
    results = build_error_report(
        weights=solver.cell_areas,
        diff=diff,
        u_exact=u_exact,
        t_final=t_final,
        bbox=bbox,
        case=case_info["name"],
        N=len(vertices),
        M=len(polygons),
    )
    return vertices, polygons, centers, u_num, u_exact, diff, results


def run_polygonal_test(
    alpha=0.1,
    nx=10,
    ny=10,
    jitter=0.05,
    dt=5e-3,
    t_init=0.05,
    t_end=0.15,
    seed=2,
    spacing=0.2,
    case="heat_kernel",
    bbox=None,
    nonorthogonal_correction=True,
    bc_type=None,
    bc_func=None,
    linear_solver="direct",
    linear_solver_options=None,
    time_scheme="backward_euler",
    flux_scheme="tpfa",
    flux_discretization="tpfa",
):
    case_info = get_analytical_case(case, alpha=alpha, t_end=t_end)
    if bbox is None:
        bbox = case_info["bbox"]
    vertices, polygons, _ = generate_hexagonal_polygonal_mesh(nx=nx, ny=ny, spacing=spacing, jitter=jitter, bbox=bbox, seed=seed)
    return run_polygonal_mesh_test(
        vertices,
        polygons,
        alpha=alpha,
        dt=dt,
        t_init=t_init,
        t_end=t_end,
        case=case,
        bbox=bbox,
        nonorthogonal_correction=nonorthogonal_correction,
        bc_type=bc_type,
        bc_func=bc_func,
        linear_solver=linear_solver,
        linear_solver_options=linear_solver_options,
        time_scheme=time_scheme,
        flux_scheme=flux_scheme,
        flux_discretization=flux_discretization,
    )


def run_mixed_polygonal_test(
    alpha=0.1,
    dt=5e-3,
    t_init=0.0,
    t_end=0.02,
    case="sine_mode",
    nx_tiles=4,
    ny_tiles=4,
    bbox=(-1.0, 1.0, -1.0, 1.0),
    nonorthogonal_correction=True,
    bc_type=None,
    bc_func=None,
    linear_solver="direct",
    linear_solver_options=None,
    time_scheme="backward_euler",
    flux_scheme="tpfa",
    flux_discretization="tpfa",
):
    vertices, polygons, _ = generate_mixed_polygonal_mesh(nx_tiles=nx_tiles, ny_tiles=ny_tiles, bbox=bbox)
    return run_polygonal_mesh_test(
        vertices,
        polygons,
        alpha=alpha,
        dt=dt,
        t_init=t_init,
        t_end=t_end,
        case=case,
        bbox=bbox,
        nonorthogonal_correction=nonorthogonal_correction,
        bc_type=bc_type,
        bc_func=bc_func,
        linear_solver=linear_solver,
        linear_solver_options=linear_solver_options,
        time_scheme=time_scheme,
        flux_scheme=flux_scheme,
        flux_discretization=flux_discretization,
    )


def run_square_polygonal_test(
    alpha=0.1,
    dt=5e-3,
    t_init=0.0,
    t_end=0.02,
    case="sine_mode",
    nx=24,
    ny=24,
    bbox=(-1.0, 1.0, -1.0, 1.0),
    nonorthogonal_correction=True,
    bc_type=None,
    bc_func=None,
    linear_solver="direct",
    linear_solver_options=None,
    time_scheme="backward_euler",
    flux_scheme="tpfa",
    flux_discretization="tpfa",
):
    vertices, polygons, _ = generate_square_polygonal_mesh(nx=nx, ny=ny, bbox=bbox)
    return run_polygonal_mesh_test(
        vertices,
        polygons,
        alpha=alpha,
        dt=dt,
        t_init=t_init,
        t_end=t_end,
        case=case,
        bbox=bbox,
        nonorthogonal_correction=nonorthogonal_correction,
        bc_type=bc_type,
        bc_func=bc_func,
        linear_solver=linear_solver,
        linear_solver_options=linear_solver_options,
        time_scheme=time_scheme,
        flux_scheme=flux_scheme,
        flux_discretization=flux_discretization,
    )


def run_nonorthogonal_polygonal_test(
    alpha=0.1,
    dt=5e-3,
    t_init=0.0,
    t_end=0.02,
    case="sine_mode",
    nx=24,
    ny=24,
    bbox=(-1.0, 1.0, -1.0, 1.0),
    skew=0.35,
    nonorthogonal_correction=True,
    bc_type=None,
    bc_func=None,
    linear_solver="direct",
    linear_solver_options=None,
    time_scheme="backward_euler",
    flux_scheme="tpfa",
    flux_discretization="tpfa",
):
    vertices, polygons, _ = generate_nonorthogonal_polygonal_mesh(nx=nx, ny=ny, bbox=bbox, skew=skew)
    return run_polygonal_mesh_test(
        vertices,
        polygons,
        alpha=alpha,
        dt=dt,
        t_init=t_init,
        t_end=t_end,
        case=case,
        bbox=bbox,
        nonorthogonal_correction=nonorthogonal_correction,
        bc_type=bc_type,
        bc_func=bc_func,
        linear_solver=linear_solver,
        linear_solver_options=linear_solver_options,
        time_scheme=time_scheme,
        flux_scheme=flux_scheme,
        flux_discretization=flux_discretization,
    )


def run_nonorthogonal_tiled_polygonal_test(
    alpha=0.1,
    dt=5e-3,
    t_init=0.0,
    t_end=0.02,
    case="sine_mode",
    nx_tiles=4,
    ny_tiles=4,
    bbox=(-1.0, 1.0, -1.0, 1.0),
    skew=0.35,
    nonorthogonal_correction=True,
    bc_type=None,
    bc_func=None,
    linear_solver="direct",
    linear_solver_options=None,
    time_scheme="backward_euler",
    flux_scheme="tpfa",
    flux_discretization="tpfa",
):
    vertices, polygons, _ = generate_nonorthogonal_tiled_polygonal_mesh(
        nx_tiles=nx_tiles,
        ny_tiles=ny_tiles,
        bbox=bbox,
        skew=skew,
    )
    return run_polygonal_mesh_test(
        vertices,
        polygons,
        alpha=alpha,
        dt=dt,
        t_init=t_init,
        t_end=t_end,
        case=case,
        bbox=bbox,
        nonorthogonal_correction=nonorthogonal_correction,
        bc_type=bc_type,
        bc_func=bc_func,
        linear_solver=linear_solver,
        linear_solver_options=linear_solver_options,
        time_scheme=time_scheme,
        flux_scheme=flux_scheme,
        flux_discretization=flux_discretization,
    )


def run_test(alpha=0.1, nx=40, ny=40, jitter=0.2, dt=5e-3, t_init=0.05, t_end=0.15, seed=2, mesh_type="delaunay", hex_spacing=0.2, case="heat_kernel", bbox=None):
    if mesh_type != "delaunay":
        raise ValueError("run_test supports only 'delaunay'. Use polygonal test drivers for polygonal meshes.")
    case_info = get_analytical_case(case, alpha=alpha, t_end=t_end)
    bc_type = case_info.get("bc_type", "dirichlet")
    if bc_type != "dirichlet":
        raise ValueError(
            f"Case '{case}' requires {bc_type} boundary conditions, but the triangular solver supports only Dirichlet."
        )
    if bbox is None:
        bbox = case_info["bbox"]
    points, tris = generate_nonuniform_delaunay(nx=nx, ny=ny, jitter=jitter, bbox=bbox, seed=seed)
    exact_solution = case_info["solution"]
    source_func = case_info.get("source", lambda x, y, t: 0.0)

    def g(x, y, t):
        return exact_solution(x, y, t)

    solver = NonUniformHeatSolver(points, tris, alpha=alpha, dt=dt, bc_type="dirichlet", bc_func=g, source_func=source_func)
    u0 = exact_solution(points[:, 0], points[:, 1], t_init)
    t_final, u_num = solver.solve(u0=u0, t0=t_init, t_end=t_end)
    u_exact = exact_solution(points[:, 0], points[:, 1], t_final)
    diff = u_num - u_exact
    results = build_error_report(
        weights=solver.cv_area,
        diff=diff,
        u_exact=u_exact,
        t_final=t_final,
        bbox=bbox,
        case=case_info["name"],
        N=points.shape[0],
        M=tris.shape[0],
    )
    return points, tris, u_num, u_exact, diff, results


def run_verification_suite(mesh_type="polygonal", cases=("heat_kernel", "sine_mode", "harmonic_polynomial"), **kwargs):
    suite = {}
    for case in cases:
        if mesh_type == "polygonal":
            *_, results = run_polygonal_test(case=case, **kwargs)
        elif mesh_type == "mixed_polygonal":
            *_, results = run_mixed_polygonal_test(case=case, **kwargs)
        elif mesh_type == "square_polygonal":
            *_, results = run_square_polygonal_test(case=case, **kwargs)
        elif mesh_type == "nonorthogonal_polygonal":
            *_, results = run_nonorthogonal_polygonal_test(case=case, **kwargs)
        elif mesh_type == "nonorthogonal_tiled_polygonal":
            *_, results = run_nonorthogonal_tiled_polygonal_test(case=case, **kwargs)
        elif mesh_type == "delaunay":
            *_, results = run_test(case=case, mesh_type="delaunay", **kwargs)
        elif mesh_type == "curvilinear":
            *_, results = run_curvilinear_test(case=case, **kwargs)
        else:
            raise ValueError(
                "mesh_type must be one of: polygonal, mixed_polygonal, square_polygonal, "
                "nonorthogonal_polygonal, nonorthogonal_tiled_polygonal, delaunay, curvilinear"
            )
        suite[case] = results
    return suite


def visualize_polygonal(
    alpha=0.1,
    nx=10,
    ny=10,
    jitter=0.05,
    dt=5e-3,
    t_init=0.05,
    t_end=0.15,
    seed=2,
    spacing=0.2,
    case="heat_kernel",
    bbox=None,
    nonorthogonal_correction=True,
):
    vertices, polygons, _, u_num, u_exact, diff, results = run_polygonal_test(
        alpha=alpha,
        nx=nx,
        ny=ny,
        jitter=jitter,
        dt=dt,
        t_init=t_init,
        t_end=t_end,
        seed=seed,
        spacing=spacing,
        case=case,
        bbox=bbox,
        nonorthogonal_correction=nonorthogonal_correction,
    )
    fig = create_polygonal_figure(vertices, polygons, u_num, u_exact, diff, results["case"], "Hexagonal Polygonal Mesh")
    fig.tight_layout()
    print("Computed results:")
    for key, value in results.items():
        print(f"  {key}: {value:.6e}" if isinstance(value, float) else f"  {key}: {value}")
    plt.show()


def visualize(alpha=0.1, nx=40, ny=40, jitter=0.05, dt=5e-3, t_init=0.05, t_end=0.15, seed=2, mesh_type="delaunay", hex_spacing=0.2, case="heat_kernel", bbox=None):
    points, tris, u_num, u_exact, diff, results = run_test(
        alpha=alpha, nx=nx, ny=ny, jitter=jitter, dt=dt, t_init=t_init, t_end=t_end, seed=seed, mesh_type=mesh_type, hex_spacing=hex_spacing, case=case, bbox=bbox
    )
    fig = create_delaunay_figure(points, tris, u_num, u_exact, diff, results["case"])
    fig.tight_layout()
    print("Computed results:")
    for key, value in results.items():
        print(f"  {key}: {value:.6e}" if isinstance(value, float) else f"  {key}: {value}")
    plt.show()
