"""Linear-solver choice (performance item 4): preconditioned Krylov solves and linear_solver="auto"."""

import numpy as np
import pytest

from heat_solver.bc_test_problems import heat_problem
from heat_solver.linear_solvers import (
    amg_available,
    choose_linear_solver,
    eliminate_identity_rows,
    estimated_costs,
    is_symmetric,
    jacobi_condition_bound,
)
from heat_solver.mesh_library import generate
from heat_solver.polygonal import PolygonalHeatSolver

needs_amg = pytest.mark.skipif(not amg_available(), reason="pyamg not installed (requirements-external.txt)")


def _run(bc, linear_solver, preconditioner=None, skew=0.0, n=20, **opts):
    p = heat_problem(bc, manufactured_boundary=True)
    m = generate("nonorthogonal", nx=n, ny=n, bbox=p["bbox"], skew=skew) if skew else generate(
        "square", nx=n, ny=n, bbox=p["bbox"])
    s = PolygonalHeatSolver(*m, p["alpha"], 2e-3, bc_type=bc, bc_func=p["bc_func"], source_func=p["source"],
                            time_scheme="bdf2", nonorthogonal_correction=bool(skew), linear_solver=linear_solver,
                            linear_solver_options={"preconditioner": preconditioner, **opts})
    c = s.cell_centers
    _, u = s.solve(p["exact"](c[:, 0], c[:, 1], 0.0), 0.0, 0.04)
    return u, s


SOLVERS = [("cg", "jacobi"), ("bicgstab", "jacobi"), ("gmres", "jacobi"),
           pytest.param("cg", "amg", marks=needs_amg), pytest.param("bicgstab", "amg", marks=needs_amg)]


@pytest.mark.parametrize("method, pc", SOLVERS)
@pytest.mark.parametrize("bc", ["dirichlet", "neumann", "robin", "flux", "radiative"])
def test_preconditioned_krylov_matches_direct(bc, method, pc):
    u_direct, _ = _run(bc, "direct")
    u, s = _run(bc, method, pc)
    assert np.max(np.abs(u - u_direct)) <= 1e-8 * np.max(np.abs(u_direct))
    assert s.last_solve_stats["krylov_solves"] > 0


@pytest.mark.parametrize("bc", ["dirichlet", "robin"])
def test_bicgstab_on_skewed_mesh_with_correction(bc):
    u_direct, _ = _run(bc, "direct", skew=0.3)
    u, _ = _run(bc, "bicgstab", "jacobi", skew=0.3)
    assert np.max(np.abs(u - u_direct)) <= 1e-8 * np.max(np.abs(u_direct))


def test_dirichlet_elimination_gives_symmetric_system_with_same_solution():
    s = PolygonalHeatSolver(*generate("square", nx=12, ny=12), 0.1, 1e-3)
    s._assemble_system()
    K = s._build_linear_lhs_csr(1e-3)
    assert not is_symmetric(K)  # identity rows break symmetry
    K_el, K_ib = eliminate_identity_rows(K, s.is_boundary)
    assert is_symmetric(K_el)
    b = np.random.default_rng(0).random(s.M)
    from scipy.sparse.linalg import spsolve

    x = spsolve(K.tocsc(), b)
    x_el = spsolve(K_el.tocsc(), b - K_ib @ (b * s.is_boundary))
    np.testing.assert_allclose(x_el, x, rtol=1e-12, atol=1e-14)


def test_preconditioner_is_built_once_and_reused_across_steps_and_calls():
    u, s = _run("dirichlet", "cg", "jacobi")
    assert s.last_solve_stats["krylov_preconditioner_builds"] == 2  # backward-Euler start + BDF2 steps
    p = heat_problem("dirichlet", manufactured_boundary=True)
    c = s.cell_centers
    s.solve(p["exact"](c[:, 0], c[:, 1], 0.0), 0.0, 0.04)
    assert s.last_solve_stats["krylov_preconditioner_builds"] == 0


def test_jacobi_condition_bound_tracks_stiffness():
    s = PolygonalHeatSolver(*generate("square", nx=32, ny=32), 0.1, 1.0)
    s._assemble_system()
    h = 2.0 / 32
    bounds = [jacobi_condition_bound(s.M_diag + dt * s.A) for dt in (0.5 * h * h / 0.1, 1.0)]
    assert bounds[0] < 10 and bounds[1] > 100  # 1 + 2 rho with rho = 4 alpha dt / h^2: 3 and ~206


def test_auto_choice_follows_the_regimes():
    s = PolygonalHeatSolver(*generate("square", nx=64, ny=64), 0.1, 1.0)
    s._assemble_system()
    h = 2.0 / 64
    soft = s.M_diag + (0.5 * h * h / 0.1) * s.A
    stiff = s.M_diag + 1.0 * s.A
    assert choose_linear_solver(soft)[0] == "direct"  # small problem: always direct
    force = {"direct_min_unknowns": 0}
    method, pc, reason = choose_linear_solver(soft, force, n_solves=100)
    assert (method, pc) == ("cg", "jacobi"), reason  # mass-dominated: Jacobi-CG
    assert choose_linear_solver(stiff, force, n_solves=100)[0] == "direct"  # stiff, factor once
    method, pc, _ = choose_linear_solver(stiff, {**force, "direct_max_unknowns": 1000}, n_solves=100)
    assert pc == ("amg" if amg_available() else "jacobi")  # too big for direct
    costs = estimated_costs(1_048_576, 257.0, 20)
    assert min(costs, key=costs.get) == "amg"  # measured winner at 1M cells, stiff
    costs = estimated_costs(1_048_576, 17.0, 20)
    assert min(costs, key=costs.get) == "jacobi"  # measured winner at 1M cells, mass-dominated


def test_auto_plan_is_reported_and_small_runs_stay_direct():
    u_direct, _ = _run("dirichlet", "direct")
    u, s = _run("dirichlet", "auto")
    assert s.last_solve_stats["linear_solver"] == "direct"
    assert np.array_equal(u, u_direct)
    u, s = _run("dirichlet", "auto", direct_min_unknowns=0)
    assert s.last_solve_stats["linear_solver"] in ("cg", "bicgstab")
    assert np.max(np.abs(u - u_direct)) <= 1e-8 * np.max(np.abs(u_direct))


def test_invalid_solver_options_are_rejected():
    m = generate("square", nx=4, ny=4)
    with pytest.raises(ValueError, match="linear_solver"):
        PolygonalHeatSolver(*m, 0.1, 1e-2, linear_solver="multigrid")
    with pytest.raises(ValueError, match="preconditioner"):
        PolygonalHeatSolver(*m, 0.1, 1e-2, linear_solver="cg", linear_solver_options={"preconditioner": "ilu"})
