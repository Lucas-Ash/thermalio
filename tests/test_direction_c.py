import numpy as np
import pytest

from heat_solver.cases import (
    fractional_stefan_apparent_capacity_case,
    hyperbolic_stefan_apparent_capacity_case,
)
from heat_solver.geometry import polygon_area_and_centroid
from heat_solver.meshes import generate_square_polygonal_mesh
from heat_solver.transport import FractionalStefanSolver, HyperbolicStefanSolver


def _square(n, bbox):
    v, p, c = generate_square_polygonal_mesh(nx=n, ny=n, bbox=bbox)
    a = np.array([polygon_area_and_centroid(v[poly])[0] for poly in p])
    return v, p, c, a


def _rel_l2(u, e, a):
    return float(np.sqrt(np.sum(a * (u - e) ** 2)) / max(np.sqrt(np.sum(a * e**2)), 1e-16))


# --------------------------------------------------------------------------- #
# Short-memory window (memory compression) for the fractional Stefan solver
# --------------------------------------------------------------------------- #
def _fractional_error(window, n=14, nt=32, t_end=0.16):
    case = fractional_stefan_apparent_capacity_case(alpha=0.08, beta=0.6)
    v, p, c, a = _square(n, case["bbox"])
    solver = FractionalStefanSolver(
        v, p, case["alpha"], t_end / nt, case["beta"], case["phase_change_model"],
        bc_func=case["boundary"], source_func=case["source"],
        phase_change_options=case["phase_change_options"], memory_window=window,
    )
    u0 = case["solution"](c[:, 0], c[:, 1], 0.0)
    _, u = solver.solve(u0, 0.0, t_end)
    return _rel_l2(u, case["solution"](c[:, 0], c[:, 1], t_end), a)


def test_memory_window_validation():
    v, p, _ = generate_square_polygonal_mesh(nx=4, ny=4, bbox=(0, 1, 0, 1))
    pcm = fractional_stefan_apparent_capacity_case()["phase_change_model"]
    with pytest.raises(ValueError):
        FractionalStefanSolver(v, p, 0.1, 0.01, 0.6, pcm, memory_window=0)


def test_memory_window_large_matches_full_small_degrades():
    err_full = _fractional_error(None)
    err_large = _fractional_error(32)   # >= number of steps -> identical to full
    err_tiny = _fractional_error(2)
    assert err_large == pytest.approx(err_full, rel=1e-6)   # window >= nsteps is a no-op
    assert err_tiny > 1.5 * err_full                        # short memory loses accuracy
    # monotone: shorter window is never more accurate
    assert _fractional_error(8) >= err_full - 1e-12


# --------------------------------------------------------------------------- #
# Convergence metadata + non-raising mode
# --------------------------------------------------------------------------- #
def test_solve_report_metadata_populated():
    case = hyperbolic_stefan_apparent_capacity_case(alpha=0.08, tau=0.05)
    v, p, c, a = _square(14, case["bbox"])
    solver = HyperbolicStefanSolver(
        v, p, case["alpha"], 0.03 / 24, case["relaxation_time"], case["phase_change_model"],
        bc_func=case["boundary"], source_func=case["source"],
        phase_change_options=case["phase_change_options"],
    )
    u0 = case["solution"](c[:, 0], c[:, 1], 0.0)
    du0 = case["initial_rate"](c[:, 0], c[:, 1])
    solver.solve(u0, 0.0, 0.03, du0=du0)
    rep = solver.solve_report
    assert rep["converged"] is True
    assert rep["n_steps"] == 24
    assert len(rep["iterations"]) == 24
    assert rep["max_iterations"] >= 1
    assert rep["failed_steps"] == 0


def test_non_raising_mode_records_failures():
    case = hyperbolic_stefan_apparent_capacity_case(alpha=0.08, tau=0.05)
    v, p, c, a = _square(14, case["bbox"])
    # Starve the Picard iteration so it cannot converge, but do not raise.
    opts = {**case["phase_change_options"], "max_iters": 1, "raise_on_nonconvergence": False}
    solver = HyperbolicStefanSolver(
        v, p, case["alpha"], 0.03 / 24, case["relaxation_time"], case["phase_change_model"],
        bc_func=case["boundary"], source_func=case["source"], phase_change_options=opts,
    )
    u0 = case["solution"](c[:, 0], c[:, 1], 0.0)
    du0 = case["initial_rate"](c[:, 0], c[:, 1])
    _, u = solver.solve(u0, 0.0, 0.03, du0=du0)
    assert solver.solve_report["converged"] is False
    assert solver.solve_report["failed_steps"] > 0
    assert np.all(np.isfinite(u))


def test_non_raising_mode_default_still_raises():
    case = hyperbolic_stefan_apparent_capacity_case(alpha=0.08, tau=0.05)
    v, p, c, a = _square(14, case["bbox"])
    opts = {**case["phase_change_options"], "max_iters": 1}  # default: raise
    solver = HyperbolicStefanSolver(
        v, p, case["alpha"], 0.03 / 24, case["relaxation_time"], case["phase_change_model"],
        bc_func=case["boundary"], source_func=case["source"], phase_change_options=opts,
    )
    u0 = case["solution"](c[:, 0], c[:, 1], 0.0)
    du0 = case["initial_rate"](c[:, 0], c[:, 1])
    with pytest.raises(RuntimeError):
        solver.solve(u0, 0.0, 0.03, du0=du0)


# --------------------------------------------------------------------------- #
# Parameterized latent-heat / transition-width manufactured case
# --------------------------------------------------------------------------- #
def test_parameterized_hyperbolic_case_is_manufactured_exact():
    # The source is rebuilt from the supplied capacity model, so the swept case
    # remains a manufactured solution the solver reproduces accurately.
    case = hyperbolic_stefan_apparent_capacity_case(
        alpha=0.08, tau=0.05, latent_heat=10.0, transition_half_width=0.15)
    pcm = case["phase_change_model"]
    assert pcm.latent_heat == 10.0
    assert pcm.liquidus_temperature == 0.15 and pcm.solidus_temperature == -0.15
    v, p, c, a = _square(20, case["bbox"])
    solver = HyperbolicStefanSolver(
        v, p, case["alpha"], 0.03 / 60, case["relaxation_time"], case["phase_change_model"],
        bc_func=case["boundary"], source_func=case["source"],
        phase_change_options=case["phase_change_options"],
    )
    u0 = case["solution"](c[:, 0], c[:, 1], 0.0)
    du0 = case["initial_rate"](c[:, 0], c[:, 1])
    _, u = solver.solve(u0, 0.0, 0.03, du0=du0)
    err = _rel_l2(u, case["solution"](c[:, 0], c[:, 1], 0.03), a)
    assert err < 5e-3


# --------------------------------------------------------------------------- #
# Study-runner smoke test
# --------------------------------------------------------------------------- #
def test_mushy_stiffness_map_runner_smoke():
    import direction_c_studies as dcs

    rows = dcs.mushy_stiffness_map(latents=(4.0, 24.0), half_widths=(0.3, 0.08), n=12, nt=24)
    assert len(rows) == 4
    for r in rows:
        assert {"latent_heat", "transition_half_width", "analytic_peak_capacity",
                "max_iters", "converged"} <= set(r)
    # Analytic capacity spike grows as the transition narrows / latent rises.
    by_key = {(r["latent_heat"], r["transition_half_width"]): r["analytic_peak_capacity"] for r in rows}
    assert by_key[(24.0, 0.08)] > by_key[(4.0, 0.3)]
