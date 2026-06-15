"""Optional JAX PINN baselines for inverse heat problems.

This module is intentionally isolated from the core package imports.  JAX is an
optional dependency; functions import it lazily and raise a clear error when the
backend is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json

import numpy as np


def _require_jax():
    try:
        import jax
        import jax.numpy as jnp
    except ModuleNotFoundError as exc:
        raise ImportError(
            "The JAX PINN backend requires optional dependencies 'jax' and 'jaxlib'. "
            "Install them to run heat_solver.pinn_jax training routines."
        ) from exc
    return jax, jnp


@dataclass(frozen=True)
class JaxPinnConfig:
    """Training configuration for the source-free Pennes inverse PINN."""

    hidden_layers: tuple[int, ...] = (32, 32)
    learning_rate: float = 1e-3
    iterations: int = 2000
    pde_weight: float = 1.0
    data_weight: float = 1.0
    initial_weight: float = 1.0
    boundary_weight: float = 1.0
    seed: int = 0
    perfusion_bounds: tuple[float, float] = (0.0, 12.0)
    log_every: int = 100


@dataclass(frozen=True)
class JaxPinnResult:
    """Result of JAX PINN inverse training."""

    perfusion: float
    final_loss: float
    data_loss: float
    pde_loss: float
    initial_loss: float
    boundary_loss: float
    history: list[dict]


def jax_available():
    """Return whether the optional JAX backend can be imported."""
    try:
        _require_jax()
    except ImportError:
        return False
    return True


def _validate_points(points, name):
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n_points, 3) for x, y, t.")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one point.")
    return arr


def _validate_values(values, n_points, name):
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size != n_points:
        raise ValueError(f"{name} must contain {n_points} values.")
    return arr


def _init_mlp(layer_sizes, key, scale=1.0):
    jax, jnp = _require_jax()
    keys = jax.random.split(key, len(layer_sizes) - 1)
    params = []
    for key_i, fan_in, fan_out in zip(keys, layer_sizes[:-1], layer_sizes[1:]):
        limit = scale * np.sqrt(6.0 / (fan_in + fan_out))
        w = jax.random.uniform(key_i, (fan_in, fan_out), minval=-limit, maxval=limit)
        b = jnp.zeros((fan_out,))
        params.append((w, b))
    return params


def _mlp_apply(params, inputs):
    _, jnp = _require_jax()
    z = inputs
    for w, b in params[:-1]:
        z = jnp.tanh(z @ w + b)
    w, b = params[-1]
    return (z @ w + b).reshape(-1)


def _adam_init(train_state):
    jax, jnp = _require_jax()
    zeros = jax.tree_util.tree_map(jnp.zeros_like, train_state)
    return {"m": zeros, "v": zeros, "t": 0}


def _adam_step(train_state, grads, opt_state, learning_rate):
    jax, jnp = _require_jax()
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = opt_state["t"] + 1
    m = jax.tree_util.tree_map(lambda m_i, g_i: beta1 * m_i + (1.0 - beta1) * g_i, opt_state["m"], grads)
    v = jax.tree_util.tree_map(lambda v_i, g_i: beta2 * v_i + (1.0 - beta2) * (g_i * g_i), opt_state["v"], grads)

    def update(param, m_i, v_i):
        m_hat = m_i / (1.0 - beta1**t)
        v_hat = v_i / (1.0 - beta2**t)
        return param - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)

    new_state = jax.tree_util.tree_map(update, train_state, m, v)
    return new_state, {"m": m, "v": v, "t": t}


def _perfusion_from_raw(raw, bounds):
    _, jnp = _require_jax()
    lower, upper = bounds
    return lower + (upper - lower) * jax_sigmoid(raw, jnp)


def jax_sigmoid(x, jnp):
    return 1.0 / (1.0 + jnp.exp(-x))


def train_pennes_inverse_pinn_jax(
    observation_points,
    observation_values,
    collocation_points,
    alpha=0.1,
    initial_points=None,
    initial_values=None,
    boundary_points=None,
    boundary_values=None,
    config=None,
):
    """Train a JAX PINN for source-free Pennes perfusion identification.

    The neural network represents ``u(x, y, t)`` and the scalar perfusion
    coefficient is trained jointly.  The PDE residual is
    ``u_t - alpha * (u_xx + u_yy) + k * u``.
    """
    jax, jnp = _require_jax()
    cfg = JaxPinnConfig() if config is None else config
    obs_points = _validate_points(observation_points, "observation_points")
    obs_values = _validate_values(observation_values, obs_points.shape[0], "observation_values")
    collocation = _validate_points(collocation_points, "collocation_points")

    if initial_points is None:
        init_points = np.zeros((0, 3), dtype=float)
        init_values = np.zeros(0, dtype=float)
    else:
        init_points = _validate_points(initial_points, "initial_points")
        init_values = _validate_values(initial_values, init_points.shape[0], "initial_values")
    if boundary_points is None:
        bnd_points = np.zeros((0, 3), dtype=float)
        bnd_values = np.zeros(0, dtype=float)
    else:
        bnd_points = _validate_points(boundary_points, "boundary_points")
        bnd_values = _validate_values(boundary_values, bnd_points.shape[0], "boundary_values")

    layer_sizes = (3, *tuple(int(width) for width in cfg.hidden_layers), 1)
    key = jax.random.PRNGKey(int(cfg.seed))
    nn_params = _init_mlp(layer_sizes, key)
    lower, upper = map(float, cfg.perfusion_bounds)
    if not lower < upper:
        raise ValueError("perfusion_bounds must satisfy lower < upper.")
    midpoint = 0.5 * (lower + upper)
    raw_k = jnp.asarray(np.log((midpoint - lower) / (upper - midpoint)), dtype=jnp.float32)
    train_state = {"params": nn_params, "raw_k": raw_k}
    opt_state = _adam_init(train_state)

    obs_x = jnp.asarray(obs_points, dtype=jnp.float32)
    obs_y = jnp.asarray(obs_values, dtype=jnp.float32)
    col_x = jnp.asarray(collocation, dtype=jnp.float32)
    init_x = jnp.asarray(init_points, dtype=jnp.float32)
    init_y = jnp.asarray(init_values, dtype=jnp.float32)
    bnd_x = jnp.asarray(bnd_points, dtype=jnp.float32)
    bnd_y = jnp.asarray(bnd_values, dtype=jnp.float32)
    alpha = float(alpha)
    bounds = (lower, upper)

    def scalar_u(params, point):
        return _mlp_apply(params, point[None, :])[0]

    def pde_residual(params, raw_k_value, point):
        grad_u = jax.grad(lambda p: scalar_u(params, p))(point)
        hess_u = jax.hessian(lambda p: scalar_u(params, p))(point)
        k_value = _perfusion_from_raw(raw_k_value, bounds)
        return grad_u[2] - alpha * (hess_u[0, 0] + hess_u[1, 1]) + k_value * scalar_u(params, point)

    vectorized_residual = jax.vmap(pde_residual, in_axes=(None, None, 0))

    def component_losses(state):
        params = state["params"]
        raw_k_value = state["raw_k"]
        data_pred = _mlp_apply(params, obs_x)
        data_loss = jnp.mean((data_pred - obs_y) ** 2)
        residual = vectorized_residual(params, raw_k_value, col_x)
        pde_loss = jnp.mean(residual**2)
        if init_x.shape[0] > 0:
            initial_loss = jnp.mean((_mlp_apply(params, init_x) - init_y) ** 2)
        else:
            initial_loss = jnp.asarray(0.0)
        if bnd_x.shape[0] > 0:
            boundary_loss = jnp.mean((_mlp_apply(params, bnd_x) - bnd_y) ** 2)
        else:
            boundary_loss = jnp.asarray(0.0)
        total = (
            cfg.data_weight * data_loss
            + cfg.pde_weight * pde_loss
            + cfg.initial_weight * initial_loss
            + cfg.boundary_weight * boundary_loss
        )
        return total, (data_loss, pde_loss, initial_loss, boundary_loss)

    value_and_grad = jax.value_and_grad(lambda state: component_losses(state)[0])
    history = []
    final_components = None
    for iteration in range(int(cfg.iterations)):
        loss, grads = value_and_grad(train_state)
        train_state, opt_state = _adam_step(train_state, grads, opt_state, float(cfg.learning_rate))
        if iteration % max(int(cfg.log_every), 1) == 0 or iteration == int(cfg.iterations) - 1:
            total, components = component_losses(train_state)
            final_components = components
            history.append(
                {
                    "iteration": int(iteration),
                    "loss": float(total),
                    "perfusion": float(_perfusion_from_raw(train_state["raw_k"], bounds)),
                    "data_loss": float(components[0]),
                    "pde_loss": float(components[1]),
                    "initial_loss": float(components[2]),
                    "boundary_loss": float(components[3]),
                }
            )

    final_loss, components = component_losses(train_state)
    if final_components is None:
        final_components = components
    return JaxPinnResult(
        perfusion=float(_perfusion_from_raw(train_state["raw_k"], bounds)),
        final_loss=float(final_loss),
        data_loss=float(final_components[0]),
        pde_loss=float(final_components[1]),
        initial_loss=float(final_components[2]),
        boundary_loss=float(final_components[3]),
        history=history,
    )


def run_pennes_jax_pinn_baseline(
    output_dir,
    observation_points,
    observation_values,
    collocation_points,
    alpha=0.1,
    true_perfusion=None,
    initial_points=None,
    initial_values=None,
    boundary_points=None,
    boundary_values=None,
    config=None,
):
    """Run the optional JAX PINN baseline and write JSON/CSV reports."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = train_pennes_inverse_pinn_jax(
        observation_points,
        observation_values,
        collocation_points,
        alpha=alpha,
        initial_points=initial_points,
        initial_values=initial_values,
        boundary_points=boundary_points,
        boundary_values=boundary_values,
        config=config,
    )
    summary = {
        "case": "Pennes JAX PINN inverse baseline",
        "backend": "jax",
        "perfusion": float(result.perfusion),
        "true_perfusion": None if true_perfusion is None else float(true_perfusion),
        "absolute_error": None if true_perfusion is None else abs(float(result.perfusion) - float(true_perfusion)),
        "final_loss": float(result.final_loss),
        "data_loss": float(result.data_loss),
        "pde_loss": float(result.pde_loss),
        "initial_loss": float(result.initial_loss),
        "boundary_loss": float(result.boundary_loss),
    }
    summary_path = output_dir / "pennes_jax_pinn_summary.json"
    history_path = output_dir / "pennes_jax_pinn_history.csv"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    with history_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = ["iteration", "loss", "perfusion", "data_loss", "pde_loss", "initial_loss", "boundary_loss"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result.history)
    return InverseStudyResult(
        summary_path=str(summary_path),
        coefficients_path=str(history_path),
        plot_path=None,
        summary=summary,
    )
