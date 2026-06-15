"""Small inverse-problem utilities for parameter-identification studies.

The functions in this module intentionally keep the inverse layer thin: callers
provide a deterministic ``forward_map(theta) -> temperature`` built from any of
Thermalio's verified solvers, and this module handles observation noise,
weighted residuals, scalar/vector optimization, regularization, and
identifiability scans.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class ScalarParameterEstimate:
    """Result of a scalar least-squares parameter-estimation run."""

    parameter_name: str
    value: float
    initial_guess: float
    bounds: tuple[float, float]
    cost: float
    residual_norm: float
    relative_residual: float
    success: bool
    nfev: int
    message: str


@dataclass(frozen=True)
class ParameterEstimate:
    """Result of a vector least-squares parameter-estimation run."""

    parameter_names: tuple[str, ...]
    values: np.ndarray
    initial_guess: np.ndarray
    bounds: tuple[np.ndarray, np.ndarray]
    cost: float
    data_residual_norm: float
    regularization_residual_norm: float
    relative_residual: float
    success: bool
    nfev: int
    message: str


def _as_vector(values, name):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one observation.")
    return arr.reshape(-1)


def residual_vector(predicted, observed, weights=None, normalize=False):
    """Return a flat weighted residual vector ``predicted - observed``.

    ``weights`` are interpreted as quadrature or sensor confidence weights and
    are applied as ``sqrt(weights)`` to preserve the usual least-squares norm.
    When ``normalize=True``, the residual is divided by the weighted observation
    norm, making objective values comparable across data scales.
    """
    predicted_vec = _as_vector(predicted, "predicted")
    observed_vec = _as_vector(observed, "observed")
    if predicted_vec.shape != observed_vec.shape:
        raise ValueError(
            "predicted and observed must have the same flattened shape; "
            f"got {predicted_vec.shape} and {observed_vec.shape}."
        )

    residual = predicted_vec - observed_vec
    if weights is not None:
        weight_vec = _as_vector(weights, "weights")
        if weight_vec.shape != observed_vec.shape:
            raise ValueError(
                "weights must have the same flattened shape as observed; "
                f"got {weight_vec.shape} and {observed_vec.shape}."
            )
        if np.any(weight_vec < 0.0):
            raise ValueError("weights must be non-negative.")
        residual = np.sqrt(weight_vec) * residual
        observed_norm_vec = np.sqrt(weight_vec) * observed_vec
    else:
        observed_norm_vec = observed_vec

    if normalize:
        scale = np.linalg.norm(observed_norm_vec)
        residual = residual / max(float(scale), 1e-16)
    return residual


def add_observation_noise(observations, relative_level=0.0, absolute_level=0.0, seed=None):
    """Return observations with reproducible additive Gaussian noise.

    The noise standard deviation is ``absolute_level + relative_level * max(|u|)``.
    A zero-noise request returns a copy, never a view.
    """
    observations = np.asarray(observations, dtype=float)
    relative_level = float(relative_level)
    absolute_level = float(absolute_level)
    if relative_level < 0.0 or absolute_level < 0.0:
        raise ValueError("relative_level and absolute_level must be non-negative.")
    sigma = absolute_level + relative_level * float(np.max(np.abs(observations)))
    if sigma == 0.0:
        return observations.copy()
    rng = np.random.default_rng(seed)
    return observations + rng.normal(loc=0.0, scale=sigma, size=observations.shape)


def make_synthetic_observations(
    forward_map,
    true_parameter,
    relative_noise=0.0,
    absolute_noise=0.0,
    seed=None,
):
    """Evaluate a forward map at the truth and optionally perturb the data."""
    true_parameter = np.asarray(true_parameter, dtype=float)
    theta = float(true_parameter) if true_parameter.ndim == 0 else true_parameter
    clean = np.asarray(forward_map(theta), dtype=float)
    noisy = add_observation_noise(
        clean,
        relative_level=relative_noise,
        absolute_level=absolute_noise,
        seed=seed,
    )
    return clean, noisy


def _validate_bounds(bounds, initial_guess):
    if bounds is None:
        return (-np.inf, np.inf)
    if len(bounds) != 2:
        raise ValueError("bounds must be a (lower, upper) pair.")
    lower, upper = float(bounds[0]), float(bounds[1])
    if not lower < upper:
        raise ValueError("bounds must satisfy lower < upper.")
    initial_guess = float(initial_guess)
    if not (lower <= initial_guess <= upper):
        raise ValueError("initial_guess must lie within bounds.")
    return lower, upper


def _as_parameter_vector(values, name):
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one parameter.")
    return arr


def _validate_vector_bounds(bounds, initial_guess):
    n_params = initial_guess.size
    if bounds is None:
        return np.full(n_params, -np.inf), np.full(n_params, np.inf)
    if len(bounds) != 2:
        raise ValueError("bounds must be a (lower, upper) pair.")
    lower = np.broadcast_to(np.asarray(bounds[0], dtype=float), (n_params,)).copy()
    upper = np.broadcast_to(np.asarray(bounds[1], dtype=float), (n_params,)).copy()
    if np.any(lower >= upper):
        raise ValueError("bounds must satisfy lower < upper for every parameter.")
    if np.any((initial_guess < lower) | (initial_guess > upper)):
        raise ValueError("initial_guess must lie within bounds.")
    return lower, upper


def regularization_residual(theta, regularization=None):
    """Return prior/scale regularization residuals for a parameter vector.

    ``regularization`` may contain:

    - ``prior``: parameter prior vector.
    - ``scale``: component-wise parameter scale; defaults to ones.
    - ``strength``: non-negative Tikhonov weight; defaults to 1.

    The returned residual is ``sqrt(strength) * (theta - prior) / scale``.
    ``None`` or zero strength returns an empty vector.
    """
    if regularization is None:
        return np.zeros(0, dtype=float)
    theta = _as_parameter_vector(theta, "theta")
    strength = float(regularization.get("strength", 1.0))
    if strength < 0.0:
        raise ValueError("regularization strength must be non-negative.")
    if strength == 0.0:
        return np.zeros(0, dtype=float)
    prior = np.broadcast_to(
        np.asarray(regularization.get("prior", np.zeros_like(theta)), dtype=float),
        theta.shape,
    )
    scale = np.broadcast_to(
        np.asarray(regularization.get("scale", np.ones_like(theta)), dtype=float),
        theta.shape,
    )
    if np.any(scale <= 0.0):
        raise ValueError("regularization scale entries must be positive.")
    return np.sqrt(strength) * (theta - prior) / scale


def estimate_parameters(
    forward_map,
    observed,
    initial_guess,
    bounds=None,
    weights=None,
    parameter_names=None,
    normalize_residual=False,
    regularization=None,
    optimizer_options=None,
):
    """Estimate one or more parameters with SciPy least squares.

    ``forward_map`` receives the full parameter vector.  Use
    :func:`estimate_scalar_parameter` for the legacy scalar convenience wrapper.
    """
    observed_vec = _as_vector(observed, "observed")
    initial = _as_parameter_vector(initial_guess, "initial_guess")
    lower, upper = _validate_vector_bounds(bounds, initial)
    if parameter_names is None:
        parameter_names = tuple(f"theta_{i}" for i in range(initial.size))
    else:
        parameter_names = tuple(str(name) for name in parameter_names)
    if len(parameter_names) != initial.size:
        raise ValueError("parameter_names length must match initial_guess length.")
    options = dict(optimizer_options or {})

    def data_residual(theta):
        return residual_vector(
            forward_map(np.asarray(theta, dtype=float)),
            observed_vec,
            weights=weights,
            normalize=normalize_residual,
        )

    def objective(theta):
        return np.concatenate([data_residual(theta), regularization_residual(theta, regularization)])

    result = least_squares(
        objective,
        x0=initial,
        bounds=(lower, upper),
        **options,
    )
    values = np.asarray(result.x, dtype=float)
    data_res = residual_vector(
        forward_map(values),
        observed_vec,
        weights=weights,
        normalize=False,
    )
    reg_res = regularization_residual(values, regularization)
    observed_norm = residual_vector(
        np.zeros_like(observed_vec),
        observed_vec,
        weights=weights,
        normalize=False,
    )
    data_residual_norm = float(np.linalg.norm(data_res))
    return ParameterEstimate(
        parameter_names=parameter_names,
        values=values,
        initial_guess=initial,
        bounds=(lower, upper),
        cost=float(result.cost),
        data_residual_norm=data_residual_norm,
        regularization_residual_norm=float(np.linalg.norm(reg_res)),
        relative_residual=data_residual_norm / max(float(np.linalg.norm(observed_norm)), 1e-16),
        success=bool(result.success),
        nfev=int(result.nfev),
        message=str(result.message),
    )


def estimate_scalar_parameter(
    forward_map,
    observed,
    initial_guess,
    bounds=None,
    weights=None,
    parameter_name="theta",
    normalize_residual=False,
    optimizer_options=None,
):
    """Estimate one scalar parameter with SciPy least squares.

    Parameters
    ----------
    forward_map : callable
        Function ``forward_map(theta) -> predicted_observations``.
    observed : array_like
        Synthetic or measured temperature data.
    initial_guess : float
        Starting value for the optimizer.
    bounds : tuple, optional
        ``(lower, upper)`` bounds for the scalar parameter.
    weights : array_like, optional
        Quadrature/sensor weights for the residual norm.
    normalize_residual : bool
        If true, optimize relative rather than absolute residuals.
    """
    observed_vec = _as_vector(observed, "observed")
    initial_guess = float(initial_guess)
    lower, upper = _validate_bounds(bounds, initial_guess)
    options = dict(optimizer_options or {})

    def objective(theta_array):
        theta = float(theta_array[0])
        predicted = forward_map(theta)
        return residual_vector(
            predicted,
            observed_vec,
            weights=weights,
            normalize=normalize_residual,
        )

    result = least_squares(
        objective,
        x0=np.array([initial_guess], dtype=float),
        bounds=(np.array([lower]), np.array([upper])),
        **options,
    )
    value = float(result.x[0])
    final_residual = residual_vector(
        forward_map(value),
        observed_vec,
        weights=weights,
        normalize=False,
    )
    observed_norm = residual_vector(
        np.zeros_like(observed_vec),
        observed_vec,
        weights=weights,
        normalize=False,
    )
    residual_norm = float(np.linalg.norm(final_residual))
    relative_residual = residual_norm / max(float(np.linalg.norm(observed_norm)), 1e-16)
    return ScalarParameterEstimate(
        parameter_name=str(parameter_name),
        value=value,
        initial_guess=initial_guess,
        bounds=(lower, upper),
        cost=float(result.cost),
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        success=bool(result.success),
        nfev=int(result.nfev),
        message=str(result.message),
    )


def identifiability_scan(
    forward_map,
    observed,
    parameter_values,
    weights=None,
    normalize_residual=False,
):
    """Evaluate least-squares cost over candidate parameter values."""
    observed_vec = _as_vector(observed, "observed")
    values = _as_vector(parameter_values, "parameter_values")
    rows = []
    for value in values:
        residual = residual_vector(
            forward_map(float(value)),
            observed_vec,
            weights=weights,
            normalize=normalize_residual,
        )
        residual_norm = float(np.linalg.norm(residual))
        rows.append(
            {
                "parameter": float(value),
                "cost": 0.5 * residual_norm**2,
                "residual_norm": residual_norm,
            }
        )
    return rows


def identifiability_grid_scan(
    forward_map,
    observed,
    parameter_grids,
    parameter_names=None,
    weights=None,
    normalize_residual=False,
    regularization=None,
):
    """Evaluate least-squares cost on a Cartesian product of parameter grids."""
    observed_vec = _as_vector(observed, "observed")
    grids = [_as_parameter_vector(grid, "parameter_grid") for grid in parameter_grids]
    if not grids:
        raise ValueError("parameter_grids must contain at least one grid.")
    if parameter_names is None:
        parameter_names = tuple(f"theta_{i}" for i in range(len(grids)))
    else:
        parameter_names = tuple(str(name) for name in parameter_names)
    if len(parameter_names) != len(grids):
        raise ValueError("parameter_names length must match parameter_grids length.")

    rows = []
    for candidate in np.array(np.meshgrid(*grids, indexing="ij")).reshape(len(grids), -1).T:
        data_res = residual_vector(
            forward_map(candidate),
            observed_vec,
            weights=weights,
            normalize=normalize_residual,
        )
        reg_res = regularization_residual(candidate, regularization)
        residual_norm = float(np.linalg.norm(data_res))
        reg_norm = float(np.linalg.norm(reg_res))
        row = {name: float(value) for name, value in zip(parameter_names, candidate)}
        row.update(
            {
                "parameters": tuple(float(value) for value in candidate),
                "cost": 0.5 * (residual_norm**2 + reg_norm**2),
                "data_residual_norm": residual_norm,
                "regularization_residual_norm": reg_norm,
            }
        )
        rows.append(row)
    return rows
