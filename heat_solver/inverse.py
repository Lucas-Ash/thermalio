"""Small inverse-problem utilities for parameter-identification studies.

The functions in this module intentionally keep the inverse layer thin: callers
provide a deterministic ``forward_map(theta) -> temperature`` built from any of
Thermalio's verified solvers, and this module handles observation noise,
weighted residuals, scalar/vector optimization, regularization, and
identifiability scans.  It also provides sparse/multi-time observation helpers
and finite-difference sensitivity utilities for least-squares diagnostics.
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


@dataclass(frozen=True)
class ObservationSet:
    """Flattened observations plus metadata describing the sampling operator."""

    values: np.ndarray
    weights: np.ndarray | None = None
    sensor_indices: np.ndarray | None = None
    time_indices: np.ndarray | None = None
    time_values: np.ndarray | None = None


def _as_vector(values, name):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one observation.")
    return arr.reshape(-1)


def nearest_sensor_indices(points, sensor_locations):
    """Return nearest state-index for each physical sensor location."""
    points = np.asarray(points, dtype=float)
    sensor_locations = np.asarray(sensor_locations, dtype=float)
    if points.ndim != 2 or sensor_locations.ndim != 2:
        raise ValueError("points and sensor_locations must be 2D arrays.")
    if points.shape[1] != sensor_locations.shape[1]:
        raise ValueError("points and sensor_locations must have the same coordinate dimension.")
    if points.shape[0] == 0 or sensor_locations.shape[0] == 0:
        raise ValueError("points and sensor_locations must be non-empty.")
    distances2 = np.sum((sensor_locations[:, None, :] - points[None, :, :]) ** 2, axis=2)
    return np.argmin(distances2, axis=1).astype(int)


def collect_observations(
    snapshots,
    sensor_indices=None,
    time_indices=None,
    weights=None,
    time_values=None,
):
    """Collect sparse sensor data from one or more time snapshots.

    ``snapshots`` may be a single field ``(n_state,)`` or a time stack
    ``(n_times, n_state)``.  Selected values are returned in time-major order:
    all requested sensors at the first requested time, then the next time, etc.
    """
    data = np.asarray(snapshots, dtype=float)
    if data.ndim == 1:
        data = data[None, :]
    if data.ndim != 2:
        raise ValueError("snapshots must have shape (n_state,) or (n_times, n_state).")

    if time_indices is None:
        time_idx = np.arange(data.shape[0], dtype=int)
    else:
        time_idx = np.asarray(time_indices, dtype=int).reshape(-1)
    if sensor_indices is None:
        sensor_idx = np.arange(data.shape[1], dtype=int)
    else:
        sensor_idx = np.asarray(sensor_indices, dtype=int).reshape(-1)
    if time_idx.size == 0 or sensor_idx.size == 0:
        raise ValueError("time_indices and sensor_indices must select at least one value.")
    if np.any((time_idx < 0) | (time_idx >= data.shape[0])):
        raise IndexError("time_indices contain an out-of-range entry.")
    if np.any((sensor_idx < 0) | (sensor_idx >= data.shape[1])):
        raise IndexError("sensor_indices contain an out-of-range entry.")

    selected = data[np.ix_(time_idx, sensor_idx)].reshape(-1)
    if weights is None:
        obs_weights = None
    else:
        weight_array = np.asarray(weights, dtype=float)
        if weight_array.ndim == 1 and weight_array.size == data.shape[1]:
            obs_weights = weight_array[sensor_idx]
            obs_weights = np.tile(obs_weights, time_idx.size)
        elif weight_array.ndim == 1 and weight_array.size == selected.size:
            obs_weights = weight_array.copy()
        elif weight_array.shape == data.shape:
            obs_weights = weight_array[np.ix_(time_idx, sensor_idx)].reshape(-1)
        else:
            raise ValueError(
                "weights must be a state vector, selected-observation vector, or full snapshot array."
            )
        if np.any(obs_weights < 0.0):
            raise ValueError("weights must be non-negative.")

    if time_values is None:
        selected_times = None
    else:
        time_values = np.asarray(time_values, dtype=float).reshape(-1)
        if time_values.size != data.shape[0]:
            raise ValueError("time_values length must match the snapshot time dimension.")
        selected_times = time_values[time_idx]

    return ObservationSet(
        values=selected,
        weights=obs_weights,
        sensor_indices=sensor_idx.copy(),
        time_indices=time_idx.copy(),
        time_values=selected_times,
    )


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


def finite_difference_jacobian(
    forward_map,
    theta,
    step=None,
    method="central",
    bounds=None,
):
    """Approximate ``d forward_map(theta) / d theta`` by finite differences."""
    theta = _as_parameter_vector(theta, "theta")
    method = str(method).lower().strip()
    if method not in {"central", "forward"}:
        raise ValueError("method must be 'central' or 'forward'.")
    if step is None:
        steps = np.sqrt(np.finfo(float).eps) * np.maximum(1.0, np.abs(theta))
    else:
        steps = np.broadcast_to(np.asarray(step, dtype=float), theta.shape).copy()
    if np.any(steps <= 0.0):
        raise ValueError("finite-difference steps must be positive.")

    lower, upper = _validate_vector_bounds(bounds, np.clip(theta, -np.inf, np.inf)) if bounds is not None else (
        np.full(theta.size, -np.inf),
        np.full(theta.size, np.inf),
    )
    f0 = _as_vector(forward_map(theta.copy()), "forward_map(theta)")
    jacobian = np.empty((f0.size, theta.size), dtype=float)

    for j in range(theta.size):
        h = float(steps[j])
        if method == "central" and theta[j] - h >= lower[j] and theta[j] + h <= upper[j]:
            theta_minus = theta.copy()
            theta_plus = theta.copy()
            theta_minus[j] -= h
            theta_plus[j] += h
            f_minus = _as_vector(forward_map(theta_minus), "forward_map(theta-h)")
            f_plus = _as_vector(forward_map(theta_plus), "forward_map(theta+h)")
            jacobian[:, j] = (f_plus - f_minus) / (2.0 * h)
        else:
            theta_plus = theta.copy()
            if theta[j] + h <= upper[j]:
                theta_plus[j] += h
                f_plus = _as_vector(forward_map(theta_plus), "forward_map(theta+h)")
                jacobian[:, j] = (f_plus - f0) / h
            elif theta[j] - h >= lower[j]:
                theta_minus = theta.copy()
                theta_minus[j] -= h
                f_minus = _as_vector(forward_map(theta_minus), "forward_map(theta-h)")
                jacobian[:, j] = (f0 - f_minus) / h
            else:
                raise ValueError("finite-difference step cannot fit inside bounds.")
    return jacobian


def residual_jacobian(
    forward_map,
    theta,
    observed,
    weights=None,
    normalize_residual=False,
    regularization=None,
    step=None,
    method="central",
    bounds=None,
):
    """Finite-difference Jacobian of the full least-squares residual vector."""
    observed_vec = _as_vector(observed, "observed")

    def residual_map(candidate):
        data_res = residual_vector(
            forward_map(np.asarray(candidate, dtype=float)),
            observed_vec,
            weights=weights,
            normalize=normalize_residual,
        )
        reg_res = regularization_residual(candidate, regularization)
        return np.concatenate([data_res, reg_res])

    return finite_difference_jacobian(
        residual_map,
        theta,
        step=step,
        method=method,
        bounds=bounds,
    )


def least_squares_gradient(jacobian, residual):
    """Return the adjoint product ``J.T @ residual`` for least-squares problems."""
    jacobian = np.asarray(jacobian, dtype=float)
    residual = _as_vector(residual, "residual")
    if jacobian.ndim != 2:
        raise ValueError("jacobian must be a 2D array.")
    if jacobian.shape[0] != residual.size:
        raise ValueError("jacobian row count must match residual size.")
    return jacobian.T @ residual


def gauss_newton_covariance(jacobian, residual_variance=1.0, rcond=1e-12):
    """Approximate parameter covariance from ``(J.T J)^+``."""
    jacobian = np.asarray(jacobian, dtype=float)
    if jacobian.ndim != 2:
        raise ValueError("jacobian must be a 2D array.")
    residual_variance = float(residual_variance)
    if residual_variance < 0.0:
        raise ValueError("residual_variance must be non-negative.")
    normal_matrix = jacobian.T @ jacobian
    return residual_variance * np.linalg.pinv(normal_matrix, rcond=float(rcond))


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
