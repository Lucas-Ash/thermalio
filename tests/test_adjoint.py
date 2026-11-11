import numpy as np
import pytest

from heat_solver.adjoint import (
    LinearHeatAdjoint,
    enthalpy_stefan_adjoint,
    fractional_heat_adjoint,
    taylor_test,
)
from heat_solver.enthalpy import AdaptiveStepOptions, EnthalpyStefanSolver, SmoothPhaseChangeModel
from heat_solver.meshes import generate_square_polygonal_mesh
from heat_solver.transport import FractionalHeatSolver, ReactionDiffusionHeatSolver


@pytest.fixture(scope="module")
def mesh():
    vertices, polygons, _ = generate_square_polygonal_mesh(nx=10, ny=10, bbox=(0.0, 1.0, 0.0, 1.0))
    return vertices, polygons


def _gauss(x0, y0, s=0.25):
    return lambda x, y: np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * s * s))


def _ones(x, y):
    return np.ones_like(x)


def _mode(centers):
    return np.sin(np.pi * centers[:, 0]) * np.sin(np.pi * centers[:, 1])


def _assert_second_order(result):
    assert np.all(result["rate1"][-3:] > 1.85)


def test_linear_forward_matches_pennes_solver(mesh):
    adj = LinearHeatAdjoint(*mesh, 0.02, [_ones], perfusion_basis=[_ones])
    u0 = _mode(adj.cell_centers)
    states, _, _, _ = adj.forward([0.07, 1.5], u0, 0.2)
    _, ref = ReactionDiffusionHeatSolver(*mesh, 0.07, 0.02, reaction_rate=1.5).solve(u0, 0.0, 0.2)
    assert np.max(np.abs(states[-1] - ref)) < 1e-12


@pytest.mark.parametrize("bc_type", ["dirichlet", "robin", "flux"])
def test_linear_field_adjoint_passes_taylor_test(mesh, bc_type):
    adj = LinearHeatAdjoint(
        *mesh, 0.025, [_ones, _gauss(0.3, 0.3), _gauss(0.7, 0.6)],
        perfusion_basis=[_gauss(0.5, 0.5)], source_basis=[_gauss(0.4, 0.6, 0.15)],
        source_time_profile=lambda t: np.exp(-t), bc_type=bc_type,
        bc_func=None if bc_type == "robin" else (lambda x, y, t: 0.1 * np.sin(np.pi * x)),
        robin_ambient=0.2,
    )
    truth = np.array([0.05, 0.03, 0.02, 0.5, 2.0] + ([3.0] if bc_type == "robin" else []))
    u0 = _mode(adj.cell_centers)
    times, sensors = [0.1, 0.2], np.arange(0, adj.M, 6)
    states, dt, _, _ = adj.forward(truth, u0, 0.2)
    observed = np.vstack([states[round(t / dt)][sensors] for t in times])
    theta = 1.3 * truth
    grad = adj.gradient(theta, u0, 0.2, times, observed, sensors).gradient
    objective = lambda q: adj.objective(q, u0, 0.2, times, observed, sensors)
    _assert_second_order(taylor_test(objective, grad, theta, hs=0.02 * 0.5 ** np.arange(5)))


def test_linear_adjoint_gradient_vanishes_at_truth(mesh):
    adj = LinearHeatAdjoint(*mesh, 0.025, [_ones, _gauss(0.5, 0.5)])
    u0 = _mode(adj.cell_centers)
    states, dt, _, _ = adj.forward([0.05, 0.02], u0, 0.1)
    result = adj.gradient([0.05, 0.02], u0, 0.1, [0.1], states[-1:])
    assert result.objective < 1e-25
    assert np.max(np.abs(result.gradient)) < 1e-12


def test_fractional_adjoint_matches_solver_and_finite_differences(mesh):
    vertices, polygons = mesh
    probe = FractionalHeatSolver(vertices, polygons, 0.1, 0.02, 0.6)
    u0 = _mode(probe.cell_centers)
    _, u_ref = probe.solve(u0, 0.0, 0.2)
    times, sensors = [0.1, 0.2], np.arange(0, probe.M, 4)
    truth = fractional_heat_adjoint(vertices, polygons, 0.1, 0.02, 0.6, u0, 0.2, times,
                                    np.zeros((2, sensors.size)), sensors)
    assert np.max(np.abs(truth.forward_states[-1] - u_ref)) < 1e-12
    observed = truth.predicted

    def objective(q):
        return fractional_heat_adjoint(vertices, polygons, q[1], 0.02, q[0], u0, 0.2, times,
                                       observed, sensors).objective

    theta = np.array([0.5, 0.13])
    grad = fractional_heat_adjoint(vertices, polygons, theta[1], 0.02, theta[0], u0, 0.2, times,
                                   observed, sensors).gradient
    _assert_second_order(taylor_test(objective, grad, theta, hs=0.02 * 0.5 ** np.arange(5)))


@pytest.mark.parametrize("tau,bc_type,du0", [
    (0.0, "dirichlet", None),
    (0.0, "flux", None),
    (0.03, "dirichlet", None),
    (0.03, "flux", 0.4),
])
def test_enthalpy_stefan_adjoint_matches_central_differences(mesh, tau, bc_type, du0):
    params = ("latent_heat", "alpha") + (("tau",) if tau > 0 else ())

    def make(q):
        model = SmoothPhaseChangeModel(0.0, 0.15, latent_heat=q[0])
        if bc_type == "dirichlet":
            bc = lambda x, y, t: np.where(x < 0.1, 1.0, -0.3)
        else:
            bc = lambda x, y, t: np.where(x < 1e-9, 2.0 * np.exp(-t / 0.1), 0.0)
        return EnthalpyStefanSolver(
            *mesh, q[1], 0.02, model, relaxation_time=q[2] if tau > 0 else 0.0,
            bc_type=bc_type, bc_func=bc, newton_options={"tol": 1e-14, "rtol_residual": 1e-14},
        )

    temp0 = -0.3 * np.ones(len(mesh[1]))
    times, sensors = [0.1, 0.2], np.arange(0, len(mesh[1]), 3)
    truth = np.array([2.0, 0.05] + ([tau] if tau > 0 else []))
    observed = enthalpy_stefan_adjoint(make(truth), temp0, 0.2, times, np.zeros((2, sensors.size)),
                                       sensors, du0=du0, parameters=params).predicted
    theta = truth * np.array([1.2, 0.8] + ([1.3] if tau > 0 else []))
    grad = enthalpy_stefan_adjoint(make(theta), temp0, 0.2, times, observed, sensors,
                                   du0=du0, parameters=params).gradient

    def objective(q):
        return enthalpy_stefan_adjoint(make(q), temp0, 0.2, times, observed, sensors,
                                       du0=du0, parameters=params).objective

    for i in range(theta.size):
        step = np.zeros_like(theta)
        step[i] = 1e-6 * abs(theta[i])
        fd = (objective(theta + step) - objective(theta - step)) / (2 * step[i])
        assert grad[i] == pytest.approx(fd, rel=1e-5, abs=1e-10)


def test_enthalpy_adjoint_runs_on_adaptive_steps(mesh):
    model = SmoothPhaseChangeModel(0.0, 0.05, latent_heat=3.0)
    pulse = lambda x, y, t: np.where(x < 1e-9, 4.0 * (t < 0.1), 0.0)
    solver = EnthalpyStefanSolver(*mesh, 0.05, 0.005, model, bc_type="flux", bc_func=pulse,
                                  adaptive=AdaptiveStepOptions(rtol=1e-2, atol=1e-3))
    temp0 = -0.2 * np.ones(len(mesh[1]))
    sensors = np.arange(0, len(mesh[1]), 5)
    result = enthalpy_stefan_adjoint(solver, temp0, 0.3, [0.1, 0.3], np.zeros((2, sensors.size)), sensors)
    assert len(set(np.round(result.extras["dts"], 12))) > 1
    assert np.all(np.isfinite(result.gradient))


def test_enthalpy_adjoint_rejects_unsupported_requests(mesh):
    model = SmoothPhaseChangeModel(latent_heat=1.0)
    solver = EnthalpyStefanSolver(*mesh, 0.05, 0.02, model)
    with pytest.raises(ValueError):
        enthalpy_stefan_adjoint(solver, np.zeros(len(mesh[1])), 0.1, [0.1], np.zeros((1, 100)),
                                parameters=("tau",))
    with pytest.raises(ValueError):
        enthalpy_stefan_adjoint(solver, np.zeros(len(mesh[1])), 0.1, [0.1], np.zeros((1, 100)),
                                parameters=("bogus",))
