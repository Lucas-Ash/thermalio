import numpy as np
import pytest

from heat_solver.cases import pennes_bioheat_case
from heat_solver.geometry import polygon_area_and_centroid
from heat_solver.inverse import (
    add_observation_noise,
    estimate_parameters,
    estimate_scalar_parameter,
    identifiability_grid_scan,
    identifiability_scan,
    make_synthetic_observations,
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


def test_residual_vector_weighting_and_validation():
    predicted = np.array([2.0, 4.0])
    observed = np.array([1.0, 1.0])
    weights = np.array([4.0, 9.0])
    assert np.allclose(residual_vector(predicted, observed, weights), [2.0, 9.0])
    with pytest.raises(ValueError):
        residual_vector(predicted, observed, weights=np.array([1.0]))
    with pytest.raises(ValueError):
        residual_vector(predicted, observed, weights=np.array([1.0, -1.0]))


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
