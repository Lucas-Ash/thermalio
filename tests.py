import csv
from math import log
from pathlib import Path
import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from heat_solver import (
    create_delaunay_figure,
    create_polygonal_figure,
    create_curvilinear_figure,
    get_analytical_case,
    run_mixed_polygonal_test,
    run_nonorthogonal_polygonal_test,
    run_nonorthogonal_tiled_polygonal_test,
    run_polygonal_test,
    run_square_polygonal_test,
    run_test as run_delaunay_test,
    run_curvilinear_test,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "test_plots"

CASE_SETTINGS = {
    "heat_kernel": {
        "alpha": 0.1,
        "dt": 5e-3,
        "t_init": 0.05,
        "t_end": 0.15,
    },
    "sine_mode": {
        "alpha": 0.1,
        "dt": 1e-3,
        "t_init": 0.0,
        "t_end": 0.02,
    },
    "harmonic_polynomial": {
        "alpha": 0.1,
        "dt": 1e-3,
        "t_init": 0.0,
        "t_end": 0.02,
    },
    "source_driven_sine": {
        "alpha": 0.1,
        "dt": 1e-3,
        "t_init": 0.0,
        "t_end": 0.05,
    },
    "steady_linear_neumann": {
        "alpha": 0.1,
        "dt": 2e-2,
        "t_init": 0.0,
        "t_end": 0.1,
    },
    "steady_linear_robin": {
        "alpha": 0.1,
        "dt": 2e-2,
        "t_init": 0.0,
        "t_end": 0.1,
    },
    "linear_patch": {
        "alpha": 0.1,
        "dt": 1e-3,
        "t_init": 0.0,
        "t_end": 0.05,
    },
    "hot_block": {
        "alpha": 0.1,
        "dt": 1e-3,
        "t_init": 0.0,
        "t_end": 0.05,
    },
    "off_axis_wave": {
        "alpha": 0.1,
        "dt": 1e-3,
        "t_init": 0.0,
        "t_end": 0.05,
    },
    "nyquist_oscillations": {
        "alpha": 0.1,
        "dt": 1e-4,
        "t_init": 0.0,
        "t_end": 0.002,
    },
    "point_source": {
        "alpha": 0.1,
        "dt": 1e-3,
        "t_init": 0.01,
        "t_end": 0.05,
    },
    "green_function_source": {
        "alpha": 0.1,
        "dt": 1e-3,
        "t_init": 0.0,
        "t_end": 0.05,
    },
    "laplace_equation": {
        "alpha": 0.1,
        "dt": 1e-3,
        "t_init": 0.0,
        "t_end": 0.02,
    },
    "anisotropic_heat_kernel": {
        "alpha": [[0.2, 0.05], [0.05, 0.1]],
        "dt": 1e-3,
        "t_init": 0.01,
        "t_end": 0.05,
    },
    "stefan_apparent_capacity": {
        "alpha": 0.08,
        "dt": 1e-3,
        "t_init": 0.0,
        "t_end": 0.05,
    },
    "temperature_dependent_diffusivity": {
        "alpha": 0.12,
        "dt": 1e-3,
        "t_init": 0.0,
        "t_end": 0.05,
    },
    "radiative_manufactured": {
        "alpha": 0.1,
        "dt": 1e-3,
        "t_init": 0.0,
        "t_end": 0.03,
    },
}

RESOLUTION_LEVELS = [
    {
        "name": "level_01_coarse",
        "nonorthogonal_tiled_tiles": 2,
        "polygonal_spacing": 0.5,
        "mixed_tiles": 2,
        "square_nx": 4,
        "nonorthogonal_nx": 4,
        "delaunay_nx": 4,
        "curvilinear_nx": 4,
    },
    {
        "name": "level_02_medium",
        "nonorthogonal_tiled_tiles": 4,
        "polygonal_spacing": 0.25,
        "mixed_tiles": 4,
        "square_nx": 8,
        "nonorthogonal_nx": 8,
        "delaunay_nx": 8,
        "curvilinear_nx": 8,
    },
    {
        "name": "level_03_fine",
        "nonorthogonal_tiled_tiles": 6,
        "polygonal_spacing": 0.1666666,
        "mixed_tiles": 6,
        "square_nx": 12,
        "nonorthogonal_nx": 12,
        "delaunay_nx": 12,
        "curvilinear_nx": 12,
    },
    {
        "name": "level_04_finer",
        "nonorthogonal_tiled_tiles": 8,
        "polygonal_spacing": 0.125,
        "mixed_tiles": 8,
        "square_nx": 16,
        "nonorthogonal_nx": 16,
        "delaunay_nx": 16,
        "curvilinear_nx": 16,
    },
    {
        "name": "level_05_superfine",
        "nonorthogonal_tiled_tiles": 12,
        "polygonal_spacing": 0.0833333,
        "mixed_tiles": 12,
        "square_nx": 24,
        "nonorthogonal_nx": 24,
        "delaunay_nx": 24,
        "curvilinear_nx": 24,
    },
]

MESH_ORDER = (
    "curvilinear",
    "polygonal",
    "square_polygonal",
    "mixed_polygonal",
    "nonorthogonal_polygonal",
    "nonorthogonal_tiled_polygonal",
    "delaunay",
)


def _case_bbox(case, settings):
    return get_analytical_case(case, alpha=settings["alpha"], t_end=settings["t_end"])["bbox"]


def _bbox_width_height(bbox):
    return bbox[1] - bbox[0], bbox[3] - bbox[2]


def _polygonal_config(case, settings, level):
    bbox = _case_bbox(case, settings)
    width, height = _bbox_width_height(bbox)
    spacing = level["polygonal_spacing"]
    return {
        **settings,
        "bbox": bbox,
        "spacing": spacing,
        "nx": max(2, int(width / spacing)),
        "ny": max(2, int(height / spacing)),
        "jitter": 0.05,
        "seed": 2,
        "nonorthogonal_correction": True,
    }


def _square_config(case, settings, level):
    return {
        **settings,
        "bbox": _case_bbox(case, settings),
        "nx": level["square_nx"],
        "ny": level["square_nx"],
        "nonorthogonal_correction": True,
    }


def _mixed_config(case, settings, level):
    return {
        **settings,
        "bbox": _case_bbox(case, settings),
        "nx_tiles": level["mixed_tiles"],
        "ny_tiles": level["mixed_tiles"],
        "nonorthogonal_correction": True,
    }


def _nonorthogonal_config(case, settings, level):
    return {
        **settings,
        "bbox": _case_bbox(case, settings),
        "nx": level["nonorthogonal_nx"],
        "ny": level["nonorthogonal_nx"],
        "skew": 0.35,
        "nonorthogonal_correction": True,
    }


def _nonorthogonal_tiled_config(case, settings, level):
    return {
        **settings,
        "bbox": _case_bbox(case, settings),
        "nx_tiles": level["nonorthogonal_tiled_tiles"],
        "ny_tiles": level["nonorthogonal_tiled_tiles"],
        "skew": 0.35,
        "nonorthogonal_correction": True,
    }

def _delaunay_config(case, settings, level):
    bbox = _case_bbox(case, settings)
    nx_param = level["delaunay_nx"]
    return {
        **settings,
        "bbox": bbox,
        "nx": nx_param,
        "ny": nx_param,
        "jitter": 0.2, # same default as in run_test signature
        "seed": 2,
        "mesh_type": "delaunay"
    }


def _curvilinear_config(case, settings, level):
    bbox = _case_bbox(case, settings)
    nx_param = level["curvilinear_nx"]
    return {
        **settings,
        "bbox": bbox,
        "nx": nx_param,
        "ny": nx_param,
        "warp": 0.1,
    }


def _resolution_metric(mesh_name, config):
    bbox = config["bbox"]
    width, height = _bbox_width_height(bbox)
    if mesh_name == "polygonal":
        return config["spacing"]
    if mesh_name == "square_polygonal":
        return max(width / config["nx"], height / config["ny"])
    if mesh_name == "mixed_polygonal":
        return max(width / (3 * config["nx_tiles"]), height / (3 * config["ny_tiles"]))
    if mesh_name == "nonorthogonal_polygonal":
        return max(width / config["nx"], height / config["ny"])
    if mesh_name == "nonorthogonal_tiled_polygonal":
        return max(width / (3 * config["nx_tiles"]), height / (3 * config["ny_tiles"]))
    if mesh_name == "delaunay":
        return max(width / config["nx"], height / config["ny"])
    if mesh_name == "curvilinear":
        return max(width / config["nx"], height / config["ny"])
    raise ValueError(f"Unknown mesh '{mesh_name}'")


def _save_polygonal_plot(output_path, vertices, polygons, u_num, u_exact, diff, case_name, mesh_title, figure_title):
    fig = create_polygonal_figure(vertices, polygons, u_num, u_exact, diff, case_name, mesh_title)
    fig.suptitle(figure_title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def _save_delaunay_plot(output_path, points, tris, u_num, u_exact, diff, case_name, mesh_title, figure_title):
    fig = create_delaunay_figure(points, tris, u_num, u_exact, diff, case_name)
    fig.suptitle(figure_title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def _run_polygonal_case(case, config):
    return run_polygonal_test(case=case, **config)

def _save_polygonal_case(case, output_path, config):
    vertices, polygons, _centers, u_num, u_exact, diff, results = _run_polygonal_case(case, config)
    _save_polygonal_plot(output_path, vertices, polygons, u_num, u_exact, diff, results["case"], "Hexagonal Polygonal Mesh", f"Hexagonal Verification: {results['case']}")
    return results

def _run_square_polygonal_case(case, config):
    return run_square_polygonal_test(case=case, **config)

def _save_square_case(case, output_path, config):
    vertices, polygons, _centers, u_num, u_exact, diff, results = _run_square_polygonal_case(case, config)
    _save_polygonal_plot(output_path, vertices, polygons, u_num, u_exact, diff, results["case"], "Square Polygonal Mesh", f"Square Verification: {results['case']}")
    return results

def _run_mixed_polygonal_case(case, config):
    return run_mixed_polygonal_test(case=case, **config)

def _save_mixed_case(case, output_path, config):
    vertices, polygons, _centers, u_num, u_exact, diff, results = _run_mixed_polygonal_case(case, config)
    _save_polygonal_plot(output_path, vertices, polygons, u_num, u_exact, diff, results["case"], "Mixed Polygonal Mesh", f"Mixed Verification: {results['case']}")
    return results

def _run_nonorthogonal_polygonal_case(case, config):
    return run_nonorthogonal_polygonal_test(case=case, **config)

def _save_nonorthogonal_case(case, output_path, config):
    vertices, polygons, _centers, u_num, u_exact, diff, results = _run_nonorthogonal_polygonal_case(case, config)
    _save_polygonal_plot(output_path, vertices, polygons, u_num, u_exact, diff, results["case"], "Non-Orthogonal Polygonal Mesh", f"Non-Orthogonal Verification: {results['case']}")
    return results

def _run_nonorthogonal_tiled_case(case, config):
    return run_nonorthogonal_tiled_polygonal_test(case=case, **config)

def _save_nonorthogonal_tiled_case(case, output_path, config):
    vertices, polygons, _centers, u_num, u_exact, diff, results = _run_nonorthogonal_tiled_case(case, config)
    _save_polygonal_plot(output_path, vertices, polygons, u_num, u_exact, diff, results["case"], "Non-Orthogonal Tiled Polygonal Mesh", f"Non-Orthogonal Tiled Verification: {results['case']}")
    return results

def _run_delaunay_case(case, config):
    points, tris, u_num, u_exact, diff, results = run_delaunay_test(case=case, **config)
    centers = (points[tris[:, 0]] + points[tris[:, 1]] + points[tris[:, 2]]) / 3.0
    return points, tris, centers, u_num, u_exact, diff, results

def _save_delaunay_case(case, output_path, config):
    points, tris, _centers, u_num, u_exact, diff, results = _run_delaunay_case(case, config)
    _save_delaunay_plot(output_path, points, tris, u_num, u_exact, diff, results["case"], "Delaunay Mesh", f"Delaunay Verification: {results['case']}")
    return results

def _run_curvilinear_case(case, config):
    X, Y, u_num, u_exact, diff, results = run_curvilinear_test(case=case, **config)
    ny, nx = X.shape
    nodes = np.column_stack([X.flatten(), Y.flatten()])
    quads = []
    centers = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            n1 = j * nx + i + 1
            n2 = (j + 1) * nx + i + 1
            n3 = (j + 1) * nx + i
            quads.append([n0, n1, n2, n3])
            
            c_x = (X[j,i] + X[j,i+1] + X[j+1,i+1] + X[j+1,i]) / 4.0
            c_y = (Y[j,i] + Y[j,i+1] + Y[j+1,i+1] + Y[j+1,i]) / 4.0
            centers.append([c_x, c_y])
            
    quads = np.array(quads)
    centers = np.array(centers)
    results["X"] = X
    results["Y"] = Y
    results["u_num_grid"] = u_num
    results["u_exact_grid"] = u_exact
    results["diff_grid"] = diff
    return nodes, quads, centers, u_num.flatten(), u_exact.flatten(), diff.flatten(), results

def _save_curvilinear_case(case, output_path, config):
    points, quads, _centers, u_num, u_exact, diff, results = _run_curvilinear_case(case, config)
    X = results.pop("X")
    Y = results.pop("Y")
    u_num_grid = results.pop("u_num_grid")
    u_exact_grid = results.pop("u_exact_grid")
    diff_grid = results.pop("diff_grid")
    fig = create_curvilinear_figure(X, Y, u_num_grid, u_exact_grid, diff_grid, results["case"])
    fig.suptitle(f"Curvilinear Verification: {results['case']}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return results


def iter_test_jobs():
    for case, settings in CASE_SETTINGS.items():
        case_info = get_analytical_case(case, settings["alpha"], settings["t_end"])
        if case_info.get("polygonal_only", False):
            continue
        if case_info.get("bc_type", "dirichlet") != "dirichlet":
            continue
        if case_info.get("phase_change_model") is not None:
            continue
        for level in RESOLUTION_LEVELS:
            yield {
                "case": case,
                "settings": settings,
                "level": level,
                "mesh_name": "curvilinear",
                "config": _curvilinear_config(case, settings, level),
            }


def _write_case_summary(case_dir, rows):
    csv_path = case_dir / "convergence_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mesh",
                "resolution",
                "h",
                "N",
                "M",
                "L2_rel",
                "Linf_rel",
                "L2_rate",
                "Linf_rate",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    txt_path = case_dir / "convergence_summary.txt"
    mesh_groups = {mesh: [row for row in rows if row["mesh"] == mesh] for mesh in MESH_ORDER if any(row["mesh"] == mesh for row in rows)}
    with txt_path.open("w", encoding="utf-8") as handle:
        handle.write("Empirical convergence summary.\n")
        handle.write("This checks observed error trends; it does not constitute a mathematical guarantee of convergence.\n\n")
        for mesh_name, mesh_rows in mesh_groups.items():
            l2_values = [row["L2_rel"] for row in mesh_rows]
            linf_values = [row["Linf_rel"] for row in mesh_rows]
            l2_monotone = all(curr < prev for prev, curr in zip(l2_values, l2_values[1:]))
            linf_monotone = all(curr < prev for prev, curr in zip(linf_values, linf_values[1:]))
            handle.write(f"{mesh_name}\n")
            handle.write(f"  L2 monotone decrease: {l2_monotone}\n")
            handle.write(f"  Linf monotone decrease: {linf_monotone}\n")
            for row in mesh_rows:
                handle.write(
                    f"  {row['resolution']}: h={row['h']:.6e}, "
                    f"L2_rel={row['L2_rel']:.6e}, Linf_rel={row['Linf_rel']:.6e}, "
                    f"L2_rate={row['L2_rate']}, Linf_rate={row['Linf_rate']}\n"
                )
            handle.write("\n")


def _mesh_jobs(case, settings, level, level_dir):
    jobs = []
    
    case_info = get_analytical_case(case, settings["alpha"], settings["t_end"])
    if case == "stefan_apparent_capacity":
        jobs.append(
            (
                "square_polygonal",
                _square_config(case, settings, level),
                _save_square_case,
                level_dir / "square_polygonal.png",
            )
        )
        return tuple(jobs)
    if case in {"temperature_dependent_diffusivity", "radiative_manufactured"}:
        jobs.append(
            (
                "square_polygonal",
                _square_config(case, settings, level),
                _save_square_case,
                level_dir / "square_polygonal.png",
            )
        )
        jobs.append(
            (
                "nonorthogonal_tiled_polygonal",
                _nonorthogonal_tiled_config(case, settings, level),
                _save_nonorthogonal_tiled_case,
                level_dir / "nonorthogonal_tiled_polygonal.png",
            )
        )
        return tuple(jobs)
    bc_type = case_info.get("bc_type", "dirichlet")
    if bc_type == "dirichlet":
        jobs.append((
            "curvilinear",
            _curvilinear_config(case, settings, level),
            _save_curvilinear_case,
            level_dir / "curvilinear.png",
        ))
        if case == "anisotropic_heat_kernel":
            jobs.extend([
                (
                    "polygonal",
                    _polygonal_config(case, settings, level),
                    _save_polygonal_case,
                    level_dir / "polygonal.png",
                ),
                (
                    "square_polygonal",
                    _square_config(case, settings, level),
                    _save_square_case,
                    level_dir / "square_polygonal.png",
                ),
                (
                    "mixed_polygonal",
                    _mixed_config(case, settings, level),
                    _save_mixed_case,
                    level_dir / "mixed_polygonal.png",
                ),
                (
                    "nonorthogonal_polygonal",
                    _nonorthogonal_config(case, settings, level),
                    _save_nonorthogonal_case,
                    level_dir / "nonorthogonal_polygonal.png",
                ),
                (
                    "nonorthogonal_tiled_polygonal",
                    _nonorthogonal_tiled_config(case, settings, level),
                    _save_nonorthogonal_tiled_case,
                    level_dir / "nonorthogonal_tiled_polygonal.png",
                ),
                (
                    "delaunay",
                    _delaunay_config(case, settings, level),
                    _save_delaunay_case,
                    level_dir / "delaunay.png",
                ),
            ])
    return tuple(jobs)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Generating empirical convergence plots. This can verify observed trends but cannot guarantee convergence from finite tests alone.")

    for case, settings in CASE_SETTINGS.items():
        case_dir = OUTPUT_DIR / case
        case_dir.mkdir(exist_ok=True)
        summary_rows = []

        previous = {}
        for level in RESOLUTION_LEVELS:
            level_dir = case_dir / level["name"]
            level_dir.mkdir(exist_ok=True)

            mesh_jobs = _mesh_jobs(case, settings, level, level_dir)

            for mesh_name, config, runner, output_path in mesh_jobs:
                results = runner(case, output_path, config)
                h = _resolution_metric(mesh_name, config)
                l2_rate = ""
                linf_rate = ""
                if mesh_name in previous:
                    prev_h, prev_l2, prev_linf = previous[mesh_name]
                    if h != prev_h:
                        curr_l2 = float(results['L2_rel'])
                        curr_linf = float(results['Linf_rel'])
                        if curr_l2 > 0 and prev_l2 > 0:
                            l2_rate = f"{log(prev_l2 / curr_l2) / log(prev_h / h):.6f}"
                        else:
                            l2_rate = "N/A"
                        if curr_linf > 0 and prev_linf > 0:
                            linf_rate = f"{log(prev_linf / curr_linf) / log(prev_h / h):.6f}"
                        else:
                            linf_rate = "N/A"
                previous[mesh_name] = (h, float(results["L2_rel"]), float(results["Linf_rel"]))
                summary_rows.append(
                    {
                        "mesh": mesh_name,
                        "resolution": level["name"],
                        "h": h,
                        "N": int(results["N"]),
                        "M": int(results["M"]),
                        "L2_rel": float(results["L2_rel"]),
                        "Linf_rel": float(results["Linf_rel"]),
                        "L2_rate": l2_rate,
                        "Linf_rate": linf_rate,
                    }
                )
                print(
                    f"{case}/{level['name']}/{mesh_name}: saved {output_path} | "
                    f"h={h:.6e}, L2_rel={float(results['L2_rel']):.6e}, Linf_rel={float(results['Linf_rel']):.6e}"
                )

        _write_case_summary(case_dir, summary_rows)
        print(f"{case}: wrote {case_dir / 'convergence_summary.csv'} and {case_dir / 'convergence_summary.txt'}")


if __name__ == "__main__":
    main()
