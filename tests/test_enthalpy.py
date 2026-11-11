import numpy as np
import pytest

from heat_solver.enthalpy import (
    AdaptiveStepOptions,
    ApparentCapacityPicardSolver,
    EnthalpyStefanSolver,
    SmoothPhaseChangeModel,
    local_error_estimate,
    temperature_sensitivity_to_latent_heat,
    three_level_coefficients,
)
from heat_solver.meshes import generate_square_polygonal_mesh
from heat_solver.phase_change import ApparentHeatCapacityModel
from heat_solver.transport import HyperbolicHeatSolver, ReactionDiffusionHeatSolver


@pytest.fixture(scope="module")
def mesh():
    vertices, polygons, _ = generate_square_polygonal_mesh(nx=14, ny=14, bbox=(0.0, 1.0, 0.0, 1.0))
    return vertices, polygons


def _mode(solver):
    x, y = solver.cell_centers[:, 0], solver.cell_centers[:, 1]
    return np.sin(np.pi * x) * np.sin(np.pi * y)


@pytest.mark.parametrize("half_width", [0.5, 0.05, 1e-3])
def test_smooth_model_inverts_enthalpy(half_width):
    model = SmoothPhaseChangeModel(melt_temperature=0.1, half_width=half_width, latent_heat=20.0)
    h = np.linspace(-5.0, 30.0, 4001)
    t = model.temperature_from_enthalpy(h)
    assert np.max(np.abs(model.enthalpy(t) - h)) < 1e-10


def test_latent_heat_sensitivity_matches_finite_difference():
    model = SmoothPhaseChangeModel(half_width=0.2, latent_heat=3.0)
    h = np.linspace(-1.0, 5.0, 50)
    t = model.temperature_from_enthalpy(h)
    eps = 1e-6
    plus = SmoothPhaseChangeModel(half_width=0.2, latent_heat=3.0 + eps).temperature_from_enthalpy(h)
    minus = SmoothPhaseChangeModel(half_width=0.2, latent_heat=3.0 - eps).temperature_from_enthalpy(h)
    np.testing.assert_allclose(temperature_sensitivity_to_latent_heat(model, t), (plus - minus) / (2 * eps), atol=1e-7)


def test_without_latent_heat_reduces_to_backward_euler(mesh):
    model = SmoothPhaseChangeModel(latent_heat=0.0)
    solver = EnthalpyStefanSolver(*mesh, 0.1, 0.01, model)
    _, u = solver.solve(_mode(solver), 0.0, 0.2)
    ref = ReactionDiffusionHeatSolver(*mesh, 0.1, 0.01, reaction_rate=0.0)
    _, u_ref = ref.solve(_mode(ref), 0.0, 0.2)
    assert np.max(np.abs(u - u_ref)) < 1e-12


def test_without_latent_heat_reduces_to_cattaneo_scheme(mesh):
    model = SmoothPhaseChangeModel(latent_heat=0.0)
    solver = EnthalpyStefanSolver(*mesh, 0.1, 0.01, model, relaxation_time=0.05)
    _, u = solver.solve(_mode(solver), 0.0, 0.2)
    ref = HyperbolicHeatSolver(*mesh, 0.1, 0.01, 0.05)
    _, u_ref = ref.solve(_mode(ref), 0.0, 0.2)
    assert np.max(np.abs(u - u_ref)) < 1e-12


def test_three_level_coefficients_uniform_limit():
    gamma, p, s = three_level_coefficients(0.1, 0.1, 0.0)
    assert (gamma, p, s) == pytest.approx((10.0, 10.0, 0.0))
    tau, k = 0.05, 0.1
    gamma, p, s = three_level_coefficients(k, k, tau)
    assert gamma == pytest.approx(tau / k**2 + 1 / (2 * k))
    assert p == pytest.approx(2 * tau / k**2)
    assert s == pytest.approx(1 / (2 * k) - tau / k**2)


def test_local_error_estimate_is_exact_for_quadratics():
    k_prev, k = 0.2, 0.1
    h = lambda t: 3.0 * t**2  # H'' = 6
    t_n = 1.0
    # A backward-Euler step on H' = f(t) gives H^{n+1} = H^n + k H'(t_{n+1}).
    h_be = h(t_n) + k * 6.0 * (t_n + k)
    lte = local_error_estimate(h_be, h(t_n), h(t_n - k_prev), k, k_prev)
    assert lte == pytest.approx(h_be - h(t_n + k), rel=1e-12)


@pytest.mark.parametrize("model", [
    ApparentHeatCapacityModel(-0.01, 0.01, latent_heat=5.0),
    SmoothPhaseChangeModel(0.0, 0.005, latent_heat=5.0),
])
def test_enthalpy_scheme_conserves_energy_for_stiff_mushy_zone(mesh, model):
    hot_wall = lambda x, y, t: np.where(x < 0.08, 1.0, -0.5)
    solver = EnthalpyStefanSolver(*mesh, 0.05, 0.05, model, bc_type="dirichlet", bc_func=hot_wall)
    _, temp = solver.solve(-0.5 * np.ones(solver.M), 0.0, 0.5)
    balance = np.asarray(solver.solve_report["energy_balance_residual"])
    assert np.max(np.abs(balance)) < 1e-9
    assert solver.solve_report["converged"]
    assert np.max(model.liquid_fraction(temp)[~solver.is_boundary]) > 0.5  # melting happened


def test_apparent_capacity_baseline_loses_energy_on_narrow_mushy_zone(mesh):
    model = SmoothPhaseChangeModel(0.0, 0.005, latent_heat=5.0)
    hot_wall = lambda x, y, t: np.where(x < 0.08, 1.0, -0.5)
    baseline = ApparentCapacityPicardSolver(
        *mesh, 0.05, 0.05, model, bc_type="dirichlet", bc_func=hot_wall,
        picard_options={"max_iters": 60, "anderson_depth": 5, "relaxation": 0.7},
    )
    baseline.solve(-0.5 * np.ones(baseline.M), 0.0, 0.5)
    report = baseline.solve_report
    assert (not report["converged"]) or abs(np.sum(report["energy_defect"])) > 1e-3


def test_adaptive_stepping_reaches_end_and_hits_outputs(mesh):
    model = ApparentHeatCapacityModel(-0.02, 0.02, latent_heat=4.0)
    pulse = lambda x, y, t: np.where(x < 1e-9, 6.0 * (t < 0.15), 0.0)
    solver = EnthalpyStefanSolver(
        *mesh, 0.05, 0.02, model, bc_type="flux", bc_func=pulse,
        adaptive=AdaptiveStepOptions(rtol=5e-3, atol=1e-3),
    )
    t, _ = solver.solve(-0.3 * np.ones(solver.M), 0.0, 0.6, output_times=[0.15, 0.3])
    rep = solver.solve_report
    assert t == pytest.approx(0.6)
    assert set(solver.outputs) == {0.15, 0.3}
    assert min(rep["dts"]) < max(rep["dts"])  # the step size actually adapted
    gained = rep["energy"][-1] - rep["energy"][0]
    # Exact discrete conservation: enthalpy gain equals the (quadrature of the) injected flux.
    assert gained == pytest.approx(rep["energy_input"][-1], rel=1e-10)
    assert gained == pytest.approx(6.0 * 1.0 * 0.15, rel=0.05)


def _pulse_cooling_problem():
    vertices, polygons, _ = generate_square_polygonal_mesh(nx=24, ny=4, bbox=(0.0, 1.0, 0.0, 0.2))
    model = ApparentHeatCapacityModel(-0.01, 0.01, latent_heat=4.0)
    pulse = lambda x, y, t: np.where(x < 1e-9, 25.0 * (t < 0.02), 0.0)
    outputs = sorted(set(np.round(np.linspace(0.01, 1.0, 12), 6)) | {0.02})
    return vertices, polygons, model, pulse, outputs


def test_adaptive_beats_uniform_on_pulse_then_cooling():
    vertices, polygons, model, pulse, outputs = _pulse_cooling_problem()
    u0 = -0.3 * np.ones(len(polygons))

    def run(dt, adaptive=None):
        solver = EnthalpyStefanSolver(vertices, polygons, 0.05, dt, model, bc_type="flux",
                                      bc_func=pulse, adaptive=adaptive)
        solver.solve(u0, 0.0, 1.0, output_times=outputs)
        return solver

    ref = run(1e-4)
    error = lambda s: max(np.max(np.abs(s.outputs[t] - ref.outputs[t])) for t in outputs)
    ada = run(1e-4, AdaptiveStepOptions(rtol=1e-2, atol=1e-3))
    work = ada.solve_report["total_newton_iterations"]
    uni = run(1.0 / work)  # uniform steps with at least as many Newton solves
    assert uni.solve_report["total_newton_iterations"] >= work
    assert error(ada) < 0.2 * error(uni)


def test_newton_failure_triggers_step_reduction(mesh, monkeypatch):
    model = SmoothPhaseChangeModel(0.0, 0.05, latent_heat=2.0)
    solver = EnthalpyStefanSolver(*mesh, 0.05, 0.1, model, adaptive=AdaptiveStepOptions(rtol=1.0, atol=1.0))
    original = solver._newton
    calls = {"n": 0}

    def flaky(*args):
        calls["n"] += 1
        if calls["n"] == 1:
            from heat_solver.enthalpy import NewtonFailure
            raise NewtonFailure("forced")
        return original(*args)

    monkeypatch.setattr(solver, "_newton", flaky)
    solver.solve(_mode(solver), 0.0, 0.3)
    assert solver.solve_report["rejected_newton"] == 1
    assert solver.solve_report["dts"][0] == pytest.approx(0.05)
