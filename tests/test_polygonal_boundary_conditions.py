import numpy as np

from heat_solver import get_analytical_case, run_square_polygonal_test


def _run_boundary_case(case):
    alpha = 0.1
    dt = 0.02
    t_end = 0.1
    case_info = get_analytical_case(case, alpha=alpha, t_end=t_end)
    return run_square_polygonal_test(
        case=case,
        alpha=alpha,
        dt=dt,
        t_init=0.0,
        t_end=t_end,
        nx=12,
        ny=12,
        bbox=case_info["bbox"],
        nonorthogonal_correction=True,
    )


def test_polygonal_solver_preserves_steady_linear_solution_with_neumann_bc():
    _vertices, _polygons, _centers, _u_num, _u_exact, diff, results = _run_boundary_case("steady_linear_neumann")

    assert results["L2"] < 1e-10
    assert results["Linf"] < 1e-10
    assert np.max(np.abs(diff)) < 1e-10


def test_polygonal_solver_preserves_steady_linear_solution_with_robin_bc():
    _vertices, _polygons, _centers, _u_num, _u_exact, diff, results = _run_boundary_case("steady_linear_robin")

    assert results["L2"] < 1e-10
    assert results["Linf"] < 1e-10
    assert np.max(np.abs(diff)) < 1e-10


def main():
    test_polygonal_solver_preserves_steady_linear_solution_with_neumann_bc()
    test_polygonal_solver_preserves_steady_linear_solution_with_robin_bc()
    print("Polygonal boundary-condition regression tests passed.")


if __name__ == "__main__":
    main()
