import numpy as np
import pytest

from heat_solver.pinn_jax import (
    JaxPinnConfig,
    jax_available,
    run_pennes_jax_pinn_baseline,
    train_pennes_inverse_pinn_jax,
)


def test_jax_pinn_backend_is_lazy_or_available():
    # Importing the module must not require JAX.  In this environment JAX may be
    # absent; training should then fail with a clear optional-dependency message.
    if not jax_available():
        with pytest.raises(ImportError, match="jax"):
            train_pennes_inverse_pinn_jax(
                observation_points=np.array([[0.5, 0.5, 0.1]]),
                observation_values=np.array([0.1]),
                collocation_points=np.array([[0.5, 0.5, 0.1]]),
                config=JaxPinnConfig(iterations=1),
            )


@pytest.mark.skipif(not jax_available(), reason="JAX optional backend is not installed")
def test_jax_pinn_trains_tiny_pennes_inverse_problem():
    # Source-free Pennes with spatially constant field u(t)=exp(-k t) gives a
    # tiny analytic inverse problem that exercises data + PDE residual training.
    true_k = 2.0
    times = np.linspace(0.0, 0.2, 8)
    observation_points = np.column_stack([
        np.full(times.size, 0.5),
        np.full(times.size, 0.5),
        times,
    ])
    observation_values = np.exp(-true_k * times)
    collocation_points = np.column_stack([
        np.full(24, 0.5),
        np.full(24, 0.5),
        np.linspace(0.0, 0.2, 24),
    ])

    result = train_pennes_inverse_pinn_jax(
        observation_points,
        observation_values,
        collocation_points,
        alpha=0.1,
        config=JaxPinnConfig(
            hidden_layers=(16, 16),
            learning_rate=2e-3,
            iterations=200,
            pde_weight=0.1,
            data_weight=5.0,
            perfusion_bounds=(0.0, 5.0),
            seed=3,
            log_every=50,
        ),
    )
    assert np.isfinite(result.final_loss)
    assert abs(result.perfusion - true_k) < 1.0
    assert len(result.history) >= 2


@pytest.mark.skipif(not jax_available(), reason="JAX optional backend is not installed")
def test_run_pennes_jax_pinn_baseline_writes_reports(tmp_path):
    true_k = 1.5
    times = np.linspace(0.0, 0.15, 6)
    observation_points = np.column_stack([0.5 * np.ones_like(times), 0.5 * np.ones_like(times), times])
    observation_values = np.exp(-true_k * times)
    collocation_points = np.column_stack([
        0.5 * np.ones(12),
        0.5 * np.ones(12),
        np.linspace(0.0, 0.15, 12),
    ])
    result = run_pennes_jax_pinn_baseline(
        tmp_path,
        observation_points,
        observation_values,
        collocation_points,
        true_perfusion=true_k,
        config=JaxPinnConfig(
            hidden_layers=(12,),
            iterations=30,
            learning_rate=1e-3,
            log_every=10,
            perfusion_bounds=(0.0, 4.0),
        ),
    )
    assert result.summary["backend"] == "jax"
    assert result.summary["true_perfusion"] == pytest.approx(true_k)
    assert "pennes_jax_pinn_summary.json" in result.summary_path
    assert "pennes_jax_pinn_history.csv" in result.coefficients_path
    assert result.plot_path is not None
    assert "pennes_jax_pinn_training.png" in result.plot_path
    assert tmp_path.joinpath("pennes_jax_pinn_summary.json").is_file()
    assert tmp_path.joinpath("pennes_jax_pinn_history.csv").is_file()
    assert tmp_path.joinpath("pennes_jax_pinn_training.png").is_file()
