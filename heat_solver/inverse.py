"""Small inverse-problem utilities for parameter-identification studies.

The functions in this module intentionally keep the inverse layer thin: callers
provide a deterministic ``forward_map(theta) -> temperature`` built from any of
Thermalio's verified solvers, and this module handles observation noise,
weighted residuals, scalar/vector optimization, regularization, and
identifiability scans.  It also provides sparse/multi-time observation helpers
and finite-difference sensitivity utilities for least-squares diagnostics.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse.linalg import factorized
from scipy.stats import norm


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


@dataclass(frozen=True)
class ReactionDiffusionAdjointResult:
    """Discrete-adjoint result for scalar reaction/perfusion identification."""

    objective: float
    gradient: float
    predicted: np.ndarray
    residual: np.ndarray
    snapshots: np.ndarray
    adjoint_states: np.ndarray
    observation_steps: np.ndarray


@dataclass(frozen=True)
class GaussianFieldBasis:
    """Normalized Gaussian radial basis for low-dimensional scalar fields."""

    centers: np.ndarray
    radius: float
    normalize: bool = True

    def __post_init__(self):
        centers = np.asarray(self.centers, dtype=float)
        if centers.ndim != 2 or centers.shape[1] != 2 or centers.shape[0] == 0:
            raise ValueError("centers must have shape (n_basis, 2).")
        if float(self.radius) <= 0.0:
            raise ValueError("radius must be positive.")
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "radius", float(self.radius))

    @property
    def n_basis(self):
        return int(self.centers.shape[0])

    def design_matrix(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        points = np.column_stack([x.reshape(-1), y.reshape(-1)])
        diff = points[:, None, :] - self.centers[None, :, :]
        phi = np.exp(-0.5 * np.sum(diff**2, axis=2) / (self.radius**2))
        if self.normalize:
            row_sum = np.sum(phi, axis=1, keepdims=True)
            phi = phi / np.maximum(row_sum, 1e-15)
        return phi

    def evaluate(self, coefficients, x, y):
        coeffs = np.broadcast_to(np.asarray(coefficients, dtype=float), (self.n_basis,))
        values = self.design_matrix(x, y) @ coeffs
        return values.reshape(np.asarray(x, dtype=float).shape)

    def as_callable(self, coefficients):
        coeffs = np.asarray(coefficients, dtype=float).copy()

        def field(x, y):
            return self.evaluate(coeffs, x, y)

        return field


@dataclass(frozen=True)
class InverseStudyResult:
    """Files and metrics written by an inverse-study runner."""

    summary_path: str
    coefficients_path: str
    plot_path: str | None
    summary: dict


@dataclass(frozen=True)
class RbfRidgeSurrogate1D:
    """Dependency-light scalar-parameter RBF ridge surrogate."""

    centers: np.ndarray
    length_scale: float
    coefficients: np.ndarray

    def predict(self, parameter):
        theta = np.asarray(parameter, dtype=float).reshape(-1)
        features = _rbf_1d_features(theta, self.centers, self.length_scale)
        predicted = features @ self.coefficients
        return predicted[0] if np.asarray(parameter).ndim == 0 else predicted


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
    - ``matrix``: optional regularization operator applied to ``theta``.
    - ``target``: target for ``matrix @ theta``; defaults to zeros.
    - ``scale``: component-wise parameter scale; defaults to ones.
    - ``strength``: non-negative Tikhonov weight; defaults to 1.

    Without ``matrix``, the returned residual is
    ``sqrt(strength) * (theta - prior) / scale``.  With ``matrix``, it is
    ``sqrt(strength) * (matrix @ theta - target) / scale``.
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
    matrix = regularization.get("matrix")
    if matrix is None:
        raw = theta - np.broadcast_to(
            np.asarray(regularization.get("prior", np.zeros_like(theta)), dtype=float),
            theta.shape,
        )
    else:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != theta.size:
            raise ValueError("regularization matrix must have shape (n_residuals, n_parameters).")
        target = np.broadcast_to(
            np.asarray(regularization.get("target", np.zeros(matrix.shape[0])), dtype=float),
            (matrix.shape[0],),
        )
        raw = matrix @ theta - target
    scale = np.broadcast_to(
        np.asarray(regularization.get("scale", np.ones_like(raw)), dtype=float),
        raw.shape,
    )
    if np.any(scale <= 0.0):
        raise ValueError("regularization scale entries must be positive.")
    return np.sqrt(strength) * raw / scale


def pairwise_difference_matrix(n_parameters):
    """Return rows ``e_i - e_j`` for all pairwise coefficient differences."""
    n_parameters = int(n_parameters)
    if n_parameters < 2:
        return np.zeros((0, n_parameters), dtype=float)
    rows = []
    for i in range(n_parameters):
        for j in range(i + 1, n_parameters):
            row = np.zeros(n_parameters, dtype=float)
            row[i] = 1.0
            row[j] = -1.0
            rows.append(row)
    return np.vstack(rows)


def _rbf_1d_features(parameter_values, centers, length_scale):
    values = np.asarray(parameter_values, dtype=float).reshape(-1)
    centers = np.asarray(centers, dtype=float).reshape(-1)
    length_scale = float(length_scale)
    if length_scale <= 0.0:
        raise ValueError("length_scale must be positive.")
    rbf = np.exp(-0.5 * ((values[:, None] - centers[None, :]) / length_scale) ** 2)
    return np.column_stack([np.ones(values.size), rbf])


def fit_rbf_ridge_surrogate_1d(parameter_values, observations, length_scale=None, ridge=1e-8):
    """Fit a scalar-parameter RBF ridge surrogate for observation vectors."""
    values = _as_vector(parameter_values, "parameter_values")
    observations = np.asarray(observations, dtype=float)
    if observations.ndim == 1:
        observations = observations[:, None]
    if observations.ndim != 2 or observations.shape[0] != values.size:
        raise ValueError("observations must have shape (n_parameter_values, n_outputs).")
    if length_scale is None:
        unique = np.unique(np.sort(values))
        if unique.size > 1:
            length_scale = 2.0 * float(np.median(np.diff(unique)))
        else:
            length_scale = 1.0
    ridge = float(ridge)
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative.")
    features = _rbf_1d_features(values, values, length_scale)
    reg = ridge * np.eye(features.shape[1])
    reg[0, 0] = 0.0
    coefficients = np.linalg.solve(features.T @ features + reg, features.T @ observations)
    return RbfRidgeSurrogate1D(centers=values.copy(), length_scale=float(length_scale), coefficients=coefficients)


def compare_scalar_inverse_baselines(
    observed,
    true_parameter,
    baselines,
    weights=None,
):
    """Create common metrics for inverse/PINN/ML baseline comparisons."""
    observed_vec = _as_vector(observed, "observed")
    rows = []
    for baseline in baselines:
        name = str(baseline["name"])
        estimate = float(baseline["estimate"])
        predicted = _as_vector(baseline["predicted"], f"{name}.predicted")
        residual = residual_vector(predicted, observed_vec, weights=weights)
        residual_norm = float(np.linalg.norm(residual))
        observed_norm = residual_vector(np.zeros_like(observed_vec), observed_vec, weights=weights)
        rows.append(
            {
                "name": name,
                "estimate": estimate,
                "absolute_error": abs(estimate - float(true_parameter)),
                "relative_error": abs(estimate - float(true_parameter)) / max(abs(float(true_parameter)), 1e-16),
                "residual_norm": residual_norm,
                "relative_residual": residual_norm / max(float(np.linalg.norm(observed_norm)), 1e-16),
                "kind": str(baseline.get("kind", "unspecified")),
            }
        )
    return rows


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


def reaction_diffusion_perfusion_adjoint(
    vertices,
    polygons,
    alpha,
    dt,
    reaction_rate,
    u0,
    observation_times,
    observed,
    sensor_indices=None,
    weights=None,
    bc_func=None,
    source_func=None,
):
    """Discrete adjoint gradient for backward-Euler Pennes/reaction diffusion.

    The differentiated parameter is the scalar ``reaction_rate`` in
    ``u_t - div(alpha grad u) + reaction_rate * u = Q``.  Observations may be
    full-field or sparse sensors at multiple times.  Observation times must land
    on the uniform ``dt`` grid.
    """
    from .transport import ReactionDiffusionHeatSolver

    dt = float(dt)
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    reaction_rate = float(reaction_rate)
    times = np.asarray(observation_times, dtype=float).reshape(-1)
    if times.size == 0:
        raise ValueError("observation_times must be non-empty.")
    if np.any(times <= 0.0) or np.any(np.diff(times) <= 0.0):
        raise ValueError("observation_times must be positive and strictly increasing.")
    steps_float = times / dt
    steps = np.rint(steps_float).astype(int)
    if np.max(np.abs(steps_float - steps)) > 1e-10:
        raise ValueError("observation_times must align with the uniform dt grid.")

    solver = ReactionDiffusionHeatSolver(
        vertices,
        polygons,
        alpha,
        dt,
        reaction_rate=reaction_rate,
        bc_func=bc_func,
        source_func=source_func,
    )
    if solver.bc_type != "dirichlet":
        raise ValueError("reaction_diffusion_perfusion_adjoint currently supports Dirichlet boundaries.")

    u0 = np.broadcast_to(np.asarray(u0, dtype=float), (solver.M,)).copy()
    if sensor_indices is None:
        sensor_idx = np.arange(solver.M, dtype=int)
    else:
        sensor_idx = np.asarray(sensor_indices, dtype=int).reshape(-1)
    if sensor_idx.size == 0:
        raise ValueError("sensor_indices must select at least one sensor.")
    if np.any((sensor_idx < 0) | (sensor_idx >= solver.M)):
        raise IndexError("sensor_indices contain an out-of-range entry.")

    observed_matrix = np.asarray(observed, dtype=float)
    if observed_matrix.ndim == 1:
        observed_matrix = observed_matrix.reshape(times.size, sensor_idx.size)
    if observed_matrix.shape != (times.size, sensor_idx.size):
        raise ValueError(
            "observed must have shape (n_observation_times, n_sensors) or the matching flat size."
        )
    if weights is None:
        weight_matrix = np.ones_like(observed_matrix)
    else:
        weight_array = np.asarray(weights, dtype=float)
        if weight_array.ndim == 1 and weight_array.size == sensor_idx.size:
            weight_matrix = np.tile(weight_array, (times.size, 1))
        elif weight_array.ndim == 1 and weight_array.size == observed_matrix.size:
            weight_matrix = weight_array.reshape(observed_matrix.shape)
        elif weight_array.shape == observed_matrix.shape:
            weight_matrix = weight_array
        else:
            raise ValueError("weights must match sensors, flattened observations, or observation matrix.")
        if np.any(weight_matrix < 0.0):
            raise ValueError("weights must be non-negative.")

    K = (solver.A + solver.R).tocsr()
    lhs = (solver.area_diag + dt * K).tocsr()
    lhs = solver._apply_dirichlet(lhs)
    solve_forward = factorized(lhs.tocsc())
    solve_adjoint = factorized(lhs.T.tocsc())
    boundary_idx = solver._boundary_idx
    is_interior = np.ones(solver.M, dtype=bool)
    is_interior[boundary_idx] = False

    nsteps = int(steps[-1])
    snapshots_all = np.empty((nsteps + 1, solver.M), dtype=float)
    snapshots_all[0] = u0
    u = u0.copy()
    obs_lookup = {int(step): i for i, step in enumerate(steps)}

    for step in range(1, nsteps + 1):
        t_next = step * dt
        rhs = solver.area_diag @ u + dt * (solver.cell_areas * solver._source(t_next))
        bc = solver._bc_values(t_next)
        rhs[boundary_idx] = bc[boundary_idx]
        u = solve_forward(rhs)
        snapshots_all[step] = u

    predicted_matrix = np.vstack([snapshots_all[step, sensor_idx] for step in steps])
    raw_residual = predicted_matrix - observed_matrix
    objective = 0.5 * float(np.sum(weight_matrix * raw_residual**2))

    adjoints = np.zeros((nsteps + 1, solver.M), dtype=float)
    lam_next = np.zeros(solver.M, dtype=float)
    gradient = 0.0
    for step in range(nsteps, 0, -1):
        obs_grad = np.zeros(solver.M, dtype=float)
        obs_pos = obs_lookup.get(step)
        if obs_pos is not None:
            np.add.at(
                obs_grad,
                sensor_idx,
                weight_matrix[obs_pos] * raw_residual[obs_pos],
            )
        rhs_adj = obs_grad + solver.cell_areas * lam_next
        lam = solve_adjoint(rhs_adj)
        adjoints[step] = lam
        gradient -= dt * float(np.dot(lam[is_interior], solver.cell_areas[is_interior] * snapshots_all[step, is_interior]))
        lam_next = lam

    return ReactionDiffusionAdjointResult(
        objective=objective,
        gradient=gradient,
        predicted=predicted_matrix.reshape(-1),
        residual=raw_residual.reshape(-1),
        snapshots=snapshots_all[steps],
        adjoint_states=adjoints,
        observation_steps=steps,
    )


def confidence_intervals(values, covariance, confidence=0.95, parameter_names=None):
    """Return normal-approximation confidence intervals from a covariance matrix."""
    values = _as_parameter_vector(values, "values")
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (values.size, values.size):
        raise ValueError("covariance shape must be (n_parameters, n_parameters).")
    confidence = float(confidence)
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must satisfy 0 < confidence < 1.")
    if parameter_names is None:
        parameter_names = tuple(f"theta_{i}" for i in range(values.size))
    else:
        parameter_names = tuple(str(name) for name in parameter_names)
    if len(parameter_names) != values.size:
        raise ValueError("parameter_names length must match values length.")
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    z_value = float(norm.ppf(0.5 + 0.5 * confidence))
    intervals = []
    for name, value, stderr in zip(parameter_names, values, standard_errors):
        half_width = z_value * float(stderr)
        intervals.append(
            {
                "parameter": name,
                "estimate": float(value),
                "standard_error": float(stderr),
                "lower": float(value - half_width),
                "upper": float(value + half_width),
                "confidence": confidence,
            }
        )
    return intervals


def bootstrap_parameter_estimates(
    estimator,
    observations,
    n_samples,
    relative_noise=0.0,
    absolute_noise=0.0,
    seed=None,
):
    """Generate bootstrap/noise-ensemble parameter estimates."""
    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_samples):
        noisy = add_observation_noise(
            observations,
            relative_level=relative_noise,
            absolute_level=absolute_noise,
            seed=int(rng.integers(0, np.iinfo(np.int32).max)),
        )
        estimate = np.asarray(estimator(noisy), dtype=float).reshape(-1)
        estimates.append(estimate)
    return np.vstack(estimates)


def bootstrap_summary(samples, confidence=0.95, parameter_names=None):
    """Summarize bootstrap samples with means, standard deviations, and quantiles."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.ndim != 2 or samples.shape[0] == 0:
        raise ValueError("samples must have shape (n_samples, n_parameters).")
    confidence = float(confidence)
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must satisfy 0 < confidence < 1.")
    if parameter_names is None:
        parameter_names = tuple(f"theta_{i}" for i in range(samples.shape[1]))
    else:
        parameter_names = tuple(str(name) for name in parameter_names)
    if len(parameter_names) != samples.shape[1]:
        raise ValueError("parameter_names length must match sample dimension.")
    alpha_tail = 0.5 * (1.0 - confidence)
    lower_q = 100.0 * alpha_tail
    upper_q = 100.0 * (1.0 - alpha_tail)
    rows = []
    for j, name in enumerate(parameter_names):
        column = samples[:, j]
        rows.append(
            {
                "parameter": name,
                "mean": float(np.mean(column)),
                "std": float(np.std(column, ddof=1)) if column.size > 1 else 0.0,
                "lower": float(np.percentile(column, lower_q)),
                "upper": float(np.percentile(column, upper_q)),
                "confidence": confidence,
            }
        )
    return rows


def run_pennes_field_inverse_study(
    output_dir,
    nx=14,
    alpha=0.1,
    dt=0.002,
    times=(0.02, 0.05, 0.08),
    basis_centers=None,
    basis_radius=0.45,
    true_coefficients=None,
    initial_guess=None,
    regularization_strength=1e-4,
    relative_noise=0.0,
    seed=0,
    make_plot=True,
):
    """Run a reproducible regularized perfusion-field inversion study.

    The unknown perfusion field is represented by a normalized Gaussian basis.
    The study writes:

    - ``pennes_field_inverse_summary.json``
    - ``pennes_field_inverse_coefficients.csv``
    - ``pennes_field_inverse_fields.png`` when ``make_plot`` is true
    """
    from .transport import ReactionDiffusionHeatSolver

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if basis_centers is None:
        basis_centers = np.array(
            [
                [0.25, 0.25],
                [0.75, 0.25],
                [0.25, 0.75],
                [0.75, 0.75],
            ],
            dtype=float,
        )
    basis = GaussianFieldBasis(np.asarray(basis_centers, dtype=float), radius=basis_radius)
    if true_coefficients is None:
        true_coefficients = np.array([2.8, 4.8, 3.4, 6.0], dtype=float)
    true_coefficients = np.broadcast_to(np.asarray(true_coefficients, dtype=float), (basis.n_basis,)).copy()
    if initial_guess is None:
        initial_guess = np.full(basis.n_basis, float(np.mean(true_coefficients)))
    initial_guess = np.broadcast_to(np.asarray(initial_guess, dtype=float), (basis.n_basis,)).copy()

    from .meshes import generate_square_polygonal_mesh

    vertices, polygons, centers = generate_square_polygonal_mesh(nx=nx, ny=nx, bbox=(0.0, 1.0, 0.0, 1.0))
    x = centers[:, 0]
    y = centers[:, 1]
    u0 = (
        np.sin(np.pi * x) * np.sin(np.pi * y)
        + 0.35 * np.sin(2.0 * np.pi * x) * np.sin(np.pi * y)
        + 0.25 * np.sin(np.pi * x) * np.sin(2.0 * np.pi * y)
    )
    times = np.asarray(times, dtype=float).reshape(-1)
    if times.size == 0 or np.any(times <= 0.0) or np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be positive and strictly increasing.")

    def snapshots_for(coefficients):
        solver = ReactionDiffusionHeatSolver(
            vertices,
            polygons,
            alpha,
            dt,
            reaction_rate=basis.as_callable(coefficients),
            bc_func=lambda x, y, t: np.zeros_like(np.asarray(x, dtype=float)),
            source_func=lambda x, y, t: np.zeros_like(np.asarray(x, dtype=float)),
        )
        snapshots = []
        t = 0.0
        u = u0.copy()
        for t_next in times:
            t, u = solver.solve(u, t, float(t_next))
            snapshots.append(u.copy())
        return np.asarray(snapshots)

    def forward(coefficients):
        return snapshots_for(coefficients).reshape(-1)

    clean_observed = forward(true_coefficients)
    observed = add_observation_noise(clean_observed, relative_level=relative_noise, seed=seed)
    smoothness = pairwise_difference_matrix(basis.n_basis)
    regularization = {
        "matrix": smoothness,
        "strength": float(regularization_strength),
        "scale": np.ones(smoothness.shape[0]) if smoothness.size else np.ones(0),
    }
    result = estimate_parameters(
        forward,
        observed,
        initial_guess=initial_guess,
        bounds=(np.zeros(basis.n_basis), np.full(basis.n_basis, 12.0)),
        parameter_names=tuple(f"k_{i}" for i in range(basis.n_basis)),
        regularization=regularization,
        optimizer_options={"max_nfev": 80},
    )

    recovered_field = basis.evaluate(result.values, x, y)
    true_field = basis.evaluate(true_coefficients, x, y)
    field_rmse = float(np.sqrt(np.mean((recovered_field - true_field) ** 2)))
    coeff_rmse = float(np.sqrt(np.mean((result.values - true_coefficients) ** 2)))
    summary = {
        "case": "Pennes perfusion field inverse",
        "nx": int(nx),
        "alpha": float(alpha),
        "dt": float(dt),
        "times": [float(t) for t in times],
        "basis_radius": float(basis_radius),
        "regularization_strength": float(regularization_strength),
        "relative_noise": float(relative_noise),
        "success": bool(result.success),
        "nfev": int(result.nfev),
        "data_residual_norm": float(result.data_residual_norm),
        "regularization_residual_norm": float(result.regularization_residual_norm),
        "relative_residual": float(result.relative_residual),
        "coefficient_rmse": coeff_rmse,
        "field_rmse": field_rmse,
        "true_coefficients": [float(v) for v in true_coefficients],
        "recovered_coefficients": [float(v) for v in result.values],
    }

    summary_path = output_dir / "pennes_field_inverse_summary.json"
    coefficients_path = output_dir / "pennes_field_inverse_coefficients.csv"
    plot_path = output_dir / "pennes_field_inverse_fields.png" if make_plot else None

    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    with coefficients_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["index", "x", "y", "true", "recovered", "error"],
        )
        writer.writeheader()
        for idx, (center, truth, recovered) in enumerate(zip(basis.centers, true_coefficients, result.values)):
            writer.writerow(
                {
                    "index": idx,
                    "x": float(center[0]),
                    "y": float(center[1]),
                    "true": float(truth),
                    "recovered": float(recovered),
                    "error": float(recovered - truth),
                }
            )

    if make_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5), constrained_layout=True)
        fields = [
            (true_field, "true perfusion"),
            (recovered_field, "recovered perfusion"),
            (recovered_field - true_field, "recovery error"),
        ]
        vmin = min(float(np.min(true_field)), float(np.min(recovered_field)))
        vmax = max(float(np.max(true_field)), float(np.max(recovered_field)))
        for ax, (field, title) in zip(axes, fields):
            if "error" in title:
                limit = max(float(np.max(np.abs(field))), 1e-12)
                contour = ax.tricontourf(x, y, field, levels=24, cmap="coolwarm", vmin=-limit, vmax=limit)
            else:
                contour = ax.tricontourf(x, y, field, levels=24, cmap="viridis", vmin=vmin, vmax=vmax)
            ax.scatter(basis.centers[:, 0], basis.centers[:, 1], c="white", edgecolors="black", s=35)
            ax.set_aspect("equal")
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(contour, ax=ax)
        fig.suptitle("Direction D: regularized Pennes perfusion-field inversion")
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return InverseStudyResult(
        summary_path=str(summary_path),
        coefficients_path=str(coefficients_path),
        plot_path=None if plot_path is None else str(plot_path),
        summary=summary,
    )


def run_pennes_ml_baseline_comparison(
    output_dir,
    nx=12,
    alpha=0.1,
    dt=0.002,
    true_perfusion=4.0,
    times=(0.02, 0.05, 0.08),
    sensor_locations=None,
    training_parameters=None,
    relative_noise=1e-3,
    seed=0,
    make_plot=True,
):
    """Compare trusted FV inversion against simple ML surrogate baselines.

    This is intentionally dependency-light.  The ML baseline is an RBF ridge
    surrogate trained on trusted FV forward snapshots.  The report format is
    designed so a future PINN implementation can add rows with the same fields.
    """
    from .cases import pennes_bioheat_case
    from .meshes import generate_square_polygonal_mesh
    from .transport import ReactionDiffusionHeatSolver

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if sensor_locations is None:
        sensor_locations = np.array(
            [[0.25, 0.25], [0.5, 0.5], [0.75, 0.5], [0.35, 0.7], [0.7, 0.75]],
            dtype=float,
        )
    if training_parameters is None:
        training_parameters = np.linspace(2.0, 6.0, 9)
    training_parameters = _as_vector(training_parameters, "training_parameters")
    times = np.asarray(times, dtype=float).reshape(-1)
    if times.size == 0 or np.any(times <= 0.0) or np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be positive and strictly increasing.")

    case = pennes_bioheat_case(alpha=alpha, perfusion=true_perfusion)
    vertices, polygons, centers = generate_square_polygonal_mesh(nx=nx, ny=nx, bbox=case["bbox"])
    sensor_indices = nearest_sensor_indices(centers, np.asarray(sensor_locations, dtype=float))
    u0 = case["solution"](centers[:, 0], centers[:, 1], 0.0)

    def snapshots_for(perfusion):
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
            t, u = solver.solve(u, t, float(t_next))
            snapshots.append(u.copy())
        return np.asarray(snapshots)

    def observe(perfusion):
        return collect_observations(
            snapshots_for(perfusion),
            sensor_indices=sensor_indices,
            time_indices=np.arange(times.size),
            time_values=times,
        ).values

    clean_observed = observe(float(true_perfusion))
    observed = add_observation_noise(clean_observed, relative_level=relative_noise, seed=seed)
    trusted = estimate_scalar_parameter(
        observe,
        observed,
        initial_guess=float(np.mean(training_parameters)),
        bounds=(float(np.min(training_parameters)), float(np.max(training_parameters))),
        parameter_name="perfusion",
    )

    train_outputs = np.vstack([observe(value) for value in training_parameters])
    surrogate = fit_rbf_ridge_surrogate_1d(training_parameters, train_outputs, ridge=1e-8)
    surrogate_estimate = estimate_scalar_parameter(
        lambda value: surrogate.predict(float(value)),
        observed,
        initial_guess=float(np.mean(training_parameters)),
        bounds=(float(np.min(training_parameters)), float(np.max(training_parameters))),
        parameter_name="perfusion",
    )
    lookup_residuals = [
        float(np.linalg.norm(residual_vector(train_output, observed)))
        for train_output in train_outputs
    ]
    lookup_idx = int(np.argmin(lookup_residuals))
    lookup_estimate = float(training_parameters[lookup_idx])

    baseline_rows = compare_scalar_inverse_baselines(
        observed,
        true_perfusion,
        [
            {
                "name": "trusted_fv_inverse",
                "kind": "trusted_fv",
                "estimate": trusted.value,
                "predicted": observe(trusted.value),
            },
            {
                "name": "rbf_ridge_surrogate",
                "kind": "ml_surrogate",
                "estimate": surrogate_estimate.value,
                "predicted": surrogate.predict(surrogate_estimate.value),
            },
            {
                "name": "training_grid_lookup",
                "kind": "ml_lookup",
                "estimate": lookup_estimate,
                "predicted": train_outputs[lookup_idx],
            },
        ],
    )
    summary = {
        "case": "Pennes PINN/ML baseline comparison",
        "true_perfusion": float(true_perfusion),
        "alpha": float(alpha),
        "dt": float(dt),
        "times": [float(t) for t in times],
        "training_parameters": [float(v) for v in training_parameters],
        "relative_noise": float(relative_noise),
        "sensor_indices": [int(i) for i in sensor_indices],
        "baselines": baseline_rows,
        "pinn_plugin_schema": {
            "required_fields": ["name", "kind", "estimate", "predicted"],
            "metric_function": "compare_scalar_inverse_baselines",
        },
    }

    summary_path = output_dir / "pennes_ml_baseline_summary.json"
    csv_path = output_dir / "pennes_ml_baseline_metrics.csv"
    plot_path = output_dir / "pennes_ml_baseline_comparison.png" if make_plot else None
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["name", "kind", "estimate", "absolute_error", "relative_error", "residual_norm", "relative_residual"],
        )
        writer.writeheader()
        writer.writerows(baseline_rows)

    if make_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [row["name"] for row in baseline_rows]
        estimates = [row["estimate"] for row in baseline_rows]
        errors = [row["absolute_error"] for row in baseline_rows]
        residuals = [row["relative_residual"] for row in baseline_rows]
        fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)
        axes[0].plot(training_parameters, train_outputs[:, 0], "o-", color="tab:gray", label="training output[0]")
        axes[0].axvline(true_perfusion, color="tab:red", ls="--", label="truth")
        axes[0].axvline(surrogate_estimate.value, color="tab:blue", ls=":", label="surrogate estimate")
        axes[0].set_xlabel("perfusion k")
        axes[0].set_ylabel("first observation")
        axes[0].set_title("FV training sweep")
        axes[0].legend(fontsize=8)
        axes[1].bar(names, estimates, color=["tab:green", "tab:blue", "tab:orange"])
        axes[1].axhline(true_perfusion, color="tab:red", ls="--")
        axes[1].tick_params(axis="x", rotation=25)
        axes[1].set_ylabel("estimated k")
        axes[1].set_title("Recovered parameter")
        axes[2].bar(names, residuals, color=["tab:green", "tab:blue", "tab:orange"], label="relative residual")
        axes[2].plot(names, errors, "ko-", label="absolute error")
        axes[2].tick_params(axis="x", rotation=25)
        axes[2].set_title("Error metrics")
        axes[2].legend(fontsize=8)
        fig.suptitle("Direction D: trusted FV vs ML surrogate inverse baselines")
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return InverseStudyResult(
        summary_path=str(summary_path),
        coefficients_path=str(csv_path),
        plot_path=None if plot_path is None else str(plot_path),
        summary=summary,
    )


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
