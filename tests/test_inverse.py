import numpy as np
import pytest

from heat_solver.cases import pennes_bioheat_case
from heat_solver.geometry import polygon_area_and_centroid
from heat_solver.inverse import (
    add_observation_noise,
    collect_observations,
    estimate_parameters,
    estimate_scalar_parameter,
    finite_difference_jacobian,
    gauss_newton_covariance,
    identifiability_grid_scan,
    identifiability_scan,
    least_squares_gradient,
    make_synthetic_observations,
    nearest_sensor_indices,
    residual_jacobian,
    regularization_residual,
    residual_vector,
)
from heat_solver.meshes import generate_square_polygonal_mesh
from heat_solver.transport import ReactionDiffusionHeatSolver


def _pennes_perfusion_forward_map(
    nx=12,
    true_perfusion=4.0,
    alpha=0.1,
    t_end=0.08,
    dt=0.002,
):
    case = pennes_bioheat_case(alpha=alpha, perfusion=true_perfusion)
    vertices, polygons, centers = generate_square_polygonal_mesh(nx=nx, ny=nx, bbox=case["bbox"])
    areas = np.array([polygon_area_and_centroid(vertices[p])[0] for p in polygons], dtype=float)
    u0 = case["solution"](centers[:, 0], centers[:, 1], 0.0)

    def forward(perfusion):
        solver = ReactionDiffusionHeatSolver(
            vertices,
            polygons,
            alpha,
            dt,
            reaction_rate=float(perfusion),
            bc_func=case["boundary"],
            source_func=lambda x, y, t: np.zeros_like(np.asarray(x, dtype=float)),
        )
        _, u = solver.solve(u0, 0.0, t_end)
        return u

    return forward, areas


def _pennes_alpha_perfusion_forward_map(
    nx=14,
    t_end=0.07,
    dt=0.002,
):
    vertices, polygons, centers = generate_square_polygonal_mesh(nx=nx, ny=nx, bbox=(0.0, 1.0, 0.0, 1.0))
    areas = np.array([polygon_area_and_centroid(vertices[p])[0] for p in polygons], dtype=float)
    x = centers[:, 0]
    y = centers[:, 1]
    u0 = (
        np.sin(np.pi * x) * np.sin(np.pi * y)
        + 0.45 * np.sin(2.0 * np.pi * x) * np.sin(np.pi * y)
    )

    def forward(theta):
        alpha, perfusion = np.asarray(theta, dtype=float)
        solver = ReactionDiffusionHeatSolver(
            vertices,
            polygons,
            alpha,
            dt,
            reaction_rate=perfusion,
            bc_func=lambda x, y, t: np.zeros_like(np.asarray(x, dtype=float)),
            source_func=lambda x, y, t: np.zeros_like(np.asarray(x, dtype=float)),
        )
        _, u = solver.solve(u0, 0.0, t_end)
        return u

    return forward, areas


def _pennes_multitime_forward_map(
    nx=12,
    true_perfusion=4.0,
    alpha=0.1,
    times=(0.02, 0.05, 0.08),
    dt=0.002,
):
    case = pennes_bioheat_case(alpha=alpha, perfusion=true_perfusion)
    vertices, polygons, centers = generate_square_polygonal_mesh(nx=nx, ny=nx, bbox=case["bbox"])
    u0 = case["solution"](centers[:, 0], centers[:, 1], 0.0)

    def forward(perfusion):
        solver = ReactionDiffusionHeatSolver(
            vertices,
            polygons,
            alpha,
            dt,
            reaction_rate=float(perfusion),
            bc_func=case["boundary"],
            source_func=lambda x, y, t: np.zeros_like(np.asarray(x, dtype=float)),
        )
        snapshots = []
        t = 0.0
        u = u0.copy()
        for t_next in times:
            t, u = solver.solve(u, t, t_next)
            snapshots.append(u.copy())
        return np.asarray(snapshots)

    return forward, centers, np.asarray(times, dtype=float)


def test_residual_vector_weighting_and_validation():
    predicted = np.array([2.0, 4.0])
    observed = np.array([1.0, 1.0])
    weights = np.array([4.0, 9.0])
    assert np.allclose(residual_vector(predicted, observed, weights), [2.0, 9.0])
    with pytest.raises(ValueError):
        residual_vector(predicted, observed, weights=np.array([1.0]))
    with pytest.raises(ValueError):
        residual_vector(predicted, observed, weights=np.array([1.0, -1.0]))


def test_collect_observations_sparse_multitime_and_nearest_sensors():
    points = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [0.0, 1.0]])
    sensor_locations = np.array([[0.45, 0.05], [0.1, 0.9]])
    sensors = nearest_sensor_indices(points, sensor_locations)
    assert np.array_equal(sensors, [1, 3])

    snapshots = np.arange(12, dtype=float).reshape(3, 4)
    obs = collect_observations(
        snapshots,
        sensor_indices=sensors,
        time_indices=np.array([0, 2]),
        weights=np.array([1.0, 2.0, 3.0, 4.0]),
        time_values=np.array([0.0, 0.5, 1.0]),
    )
    assert np.allclose(obs.values, [1.0, 3.0, 9.0, 11.0])
    assert np.allclose(obs.weights, [2.0, 4.0, 2.0, 4.0])
    assert np.allclose(obs.time_values, [0.0, 1.0])
    with pytest.raises(IndexError):
        collect_observations(snapshots, sensor_indices=np.array([10]))


def test_add_observation_noise_is_reproducible():
    observations = np.linspace(-1.0, 1.0, 11)
    noisy_a = add_observation_noise(observations, relative_level=0.01, seed=12)
    noisy_b = add_observation_noise(observations, relative_level=0.01, seed=12)
    assert np.allclose(noisy_a, noisy_b)
    assert not np.allclose(noisy_a, observations)
    assert np.allclose(add_observation_noise(observations), observations)


def test_estimate_scalar_parameter_recovers_pennes_perfusion_noise_free():
    true_perfusion = 4.0
    forward, weights = _pennes_perfusion_forward_map(true_perfusion=true_perfusion)
    clean, observed = make_synthetic_observations(forward, true_perfusion)
    assert np.allclose(clean, observed)

    result = estimate_scalar_parameter(
        forward,
        observed,
        initial_guess=1.5,
        bounds=(0.0, 8.0),
        weights=weights,
        parameter_name="perfusion",
        optimizer_options={"xtol": 1e-12, "ftol": 1e-12, "gtol": 1e-12},
    )

    assert result.success
    assert result.parameter_name == "perfusion"
    assert abs(result.value - true_perfusion) < 2e-4
    assert result.relative_residual < 1e-8


def test_estimate_scalar_parameter_is_robust_to_small_noise():
    true_perfusion = 4.0
    forward, weights = _pennes_perfusion_forward_map(true_perfusion=true_perfusion)
    _, observed = make_synthetic_observations(
        forward,
        true_perfusion,
        relative_noise=1e-3,
        seed=7,
    )
    result = estimate_scalar_parameter(
        forward,
        observed,
        initial_guess=6.5,
        bounds=(0.0, 8.0),
        weights=weights,
        parameter_name="perfusion",
    )

    assert result.success
    assert abs(result.value - true_perfusion) < 0.08
    assert result.relative_residual < 3e-3


def test_identifiability_scan_has_minimum_at_true_perfusion():
    true_perfusion = 4.0
    forward, weights = _pennes_perfusion_forward_map(true_perfusion=true_perfusion)
    observed = forward(true_perfusion)
    scan = identifiability_scan(
        forward,
        observed,
        parameter_values=np.array([2.0, 3.0, 4.0, 5.0, 6.0]),
        weights=weights,
    )

    costs = np.array([row["cost"] for row in scan])
    assert scan[int(np.argmin(costs))]["parameter"] == true_perfusion
    assert costs[2] < 1e-16
    assert costs[1] < costs[0]
    assert costs[3] < costs[4]


def test_estimate_scalar_parameter_validates_bounds():
    forward = lambda theta: np.array([theta])
    with pytest.raises(ValueError):
        estimate_scalar_parameter(forward, np.array([1.0]), initial_guess=2.0, bounds=(0.0, 1.0))


def test_finite_difference_jacobian_and_adjoint_gradient_match_analytic():
    theta = np.array([1.5, 0.3])

    def forward(candidate):
        a, b = candidate
        return np.array([a**2 + b, np.sin(b), a * b])

    jac = finite_difference_jacobian(forward, theta, step=1e-6)
    expected = np.array([
        [2.0 * theta[0], 1.0],
        [0.0, np.cos(theta[1])],
        [theta[1], theta[0]],
    ])
    assert np.allclose(jac, expected, atol=1e-6)

    residual = np.array([0.2, -0.1, 0.4])
    assert np.allclose(least_squares_gradient(jac, residual), expected.T @ residual)
    cov = gauss_newton_covariance(jac, residual_variance=0.25)
    assert cov.shape == (2, 2)
    assert np.all(np.linalg.eigvalsh(cov) > 0.0)


def test_estimate_parameters_recovers_alpha_and_perfusion():
    true_theta = np.array([0.1, 4.0])
    forward, weights = _pennes_alpha_perfusion_forward_map()
    clean, observed = make_synthetic_observations(forward, true_theta)
    assert np.allclose(clean, observed)

    result = estimate_parameters(
        forward,
        observed,
        initial_guess=np.array([0.07, 2.5]),
        bounds=(np.array([0.03, 0.0]), np.array([0.2, 8.0])),
        weights=weights,
        parameter_names=("alpha", "perfusion"),
        optimizer_options={"xtol": 1e-12, "ftol": 1e-12, "gtol": 1e-12},
    )

    assert result.success
    assert result.parameter_names == ("alpha", "perfusion")
    assert np.allclose(result.values, true_theta, atol=np.array([2e-4, 2e-3]))
    assert result.relative_residual < 1e-8


def test_sparse_multitime_observations_recover_perfusion_and_sensitivity():
    true_perfusion = 4.0
    forward_snapshots, centers, times = _pennes_multitime_forward_map(true_perfusion=true_perfusion)
    sensor_locations = np.array([
        [0.25, 0.25],
        [0.50, 0.50],
        [0.75, 0.50],
        [0.50, 0.75],
    ])
    sensor_indices = nearest_sensor_indices(centers, sensor_locations)

    def observe(perfusion):
        snapshots = forward_snapshots(float(perfusion))
        return collect_observations(
            snapshots,
            sensor_indices=sensor_indices,
            time_indices=np.arange(times.size),
            time_values=times,
        ).values

    observed = observe(true_perfusion)
    result = estimate_scalar_parameter(
        observe,
        observed,
        initial_guess=6.0,
        bounds=(0.0, 8.0),
        parameter_name="perfusion",
    )
    assert result.success
    assert abs(result.value - true_perfusion) < 2e-4

    jac = finite_difference_jacobian(lambda theta: observe(theta[0]), np.array([true_perfusion]), step=1e-4)
    assert jac.shape == (observed.size, 1)
    assert np.all(jac < 0.0)
    assert np.allclose(least_squares_gradient(jac, observed - observed), [0.0])

    noisy = add_observation_noise(observed, relative_level=1e-3, seed=42)
    res = residual_vector(observe(true_perfusion), noisy)
    res_jac = residual_jacobian(lambda theta: observe(theta[0]), np.array([true_perfusion]), noisy, step=1e-4)
    gradient = least_squares_gradient(res_jac, res)
    assert gradient.shape == (1,)
    assert np.isfinite(gradient[0])


def test_identifiability_grid_scan_for_alpha_and_perfusion():
    true_theta = np.array([0.1, 4.0])
    forward, weights = _pennes_alpha_perfusion_forward_map(nx=10)
    observed = forward(true_theta)
    scan = identifiability_grid_scan(
        forward,
        observed,
        parameter_grids=(np.array([0.08, 0.1, 0.12]), np.array([3.0, 4.0, 5.0])),
        parameter_names=("alpha", "perfusion"),
        weights=weights,
    )
    costs = np.array([row["cost"] for row in scan])
    best = scan[int(np.argmin(costs))]
    assert best["parameters"] == pytest.approx(tuple(true_theta))
    assert best["alpha"] == pytest.approx(true_theta[0])
    assert best["perfusion"] == pytest.approx(true_theta[1])


def test_regularization_selects_prior_in_underdetermined_problem():
    observed = np.array([2.0])
    prior = np.array([0.25, 1.75])
    result = estimate_parameters(
        lambda theta: np.array([theta[0] + theta[1]]),
        observed,
        initial_guess=np.array([1.2, 0.8]),
        bounds=(np.array([-5.0, -5.0]), np.array([5.0, 5.0])),
        parameter_names=("a", "b"),
        regularization={"prior": prior, "scale": np.ones(2), "strength": 1.0},
    )

    assert result.success
    assert np.allclose(result.values, prior, atol=1e-8)
    assert result.data_residual_norm < 1e-10
    assert result.regularization_residual_norm < 1e-8


def test_regularization_residual_validation():
    assert regularization_residual(np.array([1.0, 2.0]), None).size == 0
    assert np.allclose(
        regularization_residual(
            np.array([2.0, 4.0]),
            {"prior": np.array([1.0, 1.0]), "scale": np.array([1.0, 3.0]), "strength": 4.0},
        ),
        np.array([2.0, 2.0]),
    )
    with pytest.raises(ValueError):
        regularization_residual(np.array([1.0]), {"strength": -1.0})
    with pytest.raises(ValueError):
        regularization_residual(np.array([1.0]), {"scale": np.array([0.0])})


def test_estimate_parameters_validation():
    with pytest.raises(ValueError):
        estimate_parameters(lambda theta: theta, np.array([1.0]), initial_guess=np.array([1.0]), bounds=(2.0, 3.0))
    with pytest.raises(ValueError):
        estimate_parameters(
            lambda theta: theta,
            np.array([1.0]),
            initial_guess=np.array([1.0, 2.0]),
            parameter_names=("only_one",),
        )
    with pytest.raises(ValueError):
        identifiability_grid_scan(lambda theta: theta, np.array([1.0]), parameter_grids=())
