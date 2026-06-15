"""Generate paper-style Direction D figures and caption snippets.

Run from the repository root:

    MPLCONFIGDIR=/tmp/matplotlib python report_images/generate_direction_d_report_images.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from heat_solver.inverse import (
    GaussianFieldBasis,
    add_observation_noise,
    bootstrap_parameter_estimates,
    bootstrap_summary,
    confidence_intervals,
    estimate_parameters,
    finite_difference_jacobian,
    gauss_newton_covariance,
    identifiability_grid_scan,
    identifiability_scan,
    least_squares_gradient,
    residual_jacobian,
    residual_vector,
)


OUT = ROOT / "report_images"
DIRECTION_D = ROOT / "test_plots" / "direction_D_inverse_problems"


def _save_caption(name: str, text: str, label: str) -> None:
    path = OUT / f"{name}.caption.tex"
    path.write_text(f"\\caption{{{text}}}\\label{{fig:{label}}}\n", encoding="utf-8")


def _save(fig, name: str, caption: str, label: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    _save_caption(name, caption, label)


def _pennes_signal(perfusion: float, times: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    mode_decay = 2.0 * np.pi**2 * alpha
    return np.exp(-(mode_decay + float(perfusion)) * times)


def _two_mode_signal(theta: np.ndarray, times: np.ndarray) -> np.ndarray:
    alpha, perfusion = map(float, theta)
    lambdas = np.array([2.0 * np.pi**2, 5.0 * np.pi**2])
    amplitudes = np.array([1.0, 0.42])
    response = [amp * np.exp(-(lam * alpha + perfusion) * times) for amp, lam in zip(amplitudes, lambdas)]
    return np.concatenate(response)


def workflow_tree() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    ax.axis("off")
    nodes = {
        "truth": (0.08, 0.72, "Synthetic truth\n$\\theta_\\star$, forward PDE"),
        "obs": (0.30, 0.72, "Observation operator\n$y = H u + \\eta$"),
        "res": (0.52, 0.72, "Weighted residual\n$r(\\theta)$"),
        "opt": (0.75, 0.72, "Least-squares solve\nbounds + regularization"),
        "report": (0.92, 0.72, "Reports\nJSON / CSV / PNG"),
        "sens": (0.30, 0.34, "Sensitivities\nfinite differences / $J^T r$"),
        "adj": (0.52, 0.34, "Discrete adjoint\nPennes gradient"),
        "uq": (0.75, 0.34, "Uncertainty\nGN covariance + bootstrap"),
        "compare": (0.92, 0.34, "Baselines\nFV / RBF / lookup / JAX PINN"),
    }
    for _, (x, y, text) in nodes.items():
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f7f7f7", edgecolor="#333333", linewidth=1.0),
        )
    edges = [
        ("truth", "obs"),
        ("obs", "res"),
        ("res", "opt"),
        ("opt", "report"),
        ("obs", "sens"),
        ("res", "sens"),
        ("sens", "adj"),
        ("adj", "uq"),
        ("uq", "compare"),
        ("compare", "report"),
        ("opt", "uq"),
    ]
    for a, b in edges:
        x0, y0, _ = nodes[a]
        x1, y1, _ = nodes[b]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.5, color="#356a9a"))
    ax.set_title("Direction D inverse-problem workflow", fontsize=15)
    _save(
        fig,
        "direction_d_inverse_workflow",
        "Functional workflow of the Direction D inverse-problem layer. The verified forward model generates synthetic data; sparse observations define the residual; optimization, sensitivities, adjoints, uncertainty estimates, and baseline comparisons all operate on the same residual contract.",
        "direction-d-workflow",
    )


def scalar_identifiability_noise() -> None:
    true_k = 3.2
    alpha = 0.1
    times = np.linspace(0.01, 0.16, 10)
    clean = _pennes_signal(true_k, times, alpha)
    grid = np.linspace(1.0, 5.5, 160)
    noise_levels = [0.0, 0.002, 0.01, 0.03]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)
    for level in noise_levels:
        observed = add_observation_noise(clean, relative_level=level, seed=42)
        rows = identifiability_scan(lambda k: _pennes_signal(k, times, alpha), observed, grid)
        costs = np.array([row["cost"] for row in rows])
        axes[0].plot(grid, costs / max(np.min(costs[costs > 0]) if np.any(costs > 0) else 1.0, 1e-16), label=f"{100*level:.1f}% noise")
    axes[0].axvline(true_k, color="black", ls="--", lw=1.2, label="truth")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("candidate perfusion $k$")
    axes[0].set_ylabel("relative scan cost")
    axes[0].set_title("Identifiability scans")
    axes[0].legend(fontsize=8)

    rng = np.random.default_rng(4)
    ensemble_levels = np.linspace(0.0, 0.04, 9)
    means = []
    lo = []
    hi = []
    for level in ensemble_levels:
        estimates = []
        for _ in range(24):
            observed = add_observation_noise(clean, relative_level=float(level), seed=int(rng.integers(0, 2**31 - 1)))
            result = estimate_parameters(
                lambda theta: _pennes_signal(theta[0], times, alpha),
                observed,
                initial_guess=[2.8],
                bounds=([0.2], [7.0]),
                parameter_names=("k",),
            )
            estimates.append(result.values[0])
        estimates = np.asarray(estimates)
        means.append(float(np.mean(estimates)))
        lo.append(float(np.percentile(estimates, 5)))
        hi.append(float(np.percentile(estimates, 95)))
    axes[1].plot(100.0 * ensemble_levels, means, "o-", color="tab:blue", label="ensemble mean")
    axes[1].fill_between(100.0 * ensemble_levels, lo, hi, color="tab:blue", alpha=0.18, label="5-95% range")
    axes[1].axhline(true_k, color="black", ls="--", lw=1.2, label="truth")
    axes[1].set_xlabel("relative noise level (%)")
    axes[1].set_ylabel("estimated $k$")
    axes[1].set_title("Noise robustness")
    axes[1].legend(fontsize=8)
    _save(
        fig,
        "scalar_identifiability_noise",
        "Scalar Pennes perfusion recovery under increasing observation noise. Left: scan costs retain a clear minimum near the truth for low-to-moderate noise. Right: repeated noisy inversions show the estimator remains centered near the truth while the 5--95\\% uncertainty band expands with noise.",
        "scalar-identifiability-noise",
    )


def multiparameter_landscape() -> None:
    true_theta = np.array([0.12, 2.1])
    times = np.linspace(0.01, 0.14, 12)
    observed = add_observation_noise(_two_mode_signal(true_theta, times), relative_level=0.002, seed=8)
    alpha_grid = np.linspace(0.07, 0.17, 80)
    k_grid = np.linspace(1.1, 3.2, 88)
    rows = identifiability_grid_scan(
        lambda theta: _two_mode_signal(theta, times),
        observed,
        (alpha_grid, k_grid),
        parameter_names=("alpha", "k"),
    )
    cost = np.array([row["cost"] for row in rows]).reshape(alpha_grid.size, k_grid.size)
    result = estimate_parameters(
        lambda theta: _two_mode_signal(theta, times),
        observed,
        initial_guess=[0.10, 1.8],
        bounds=([0.05, 0.5], [0.25, 5.0]),
        parameter_names=("alpha", "k"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)
    levels = np.geomspace(max(float(np.min(cost[cost > 0])), 1e-14), float(np.max(cost)), 24)
    contour = axes[0].contourf(k_grid, alpha_grid, cost, levels=levels, norm="log", cmap="magma")
    axes[0].plot(true_theta[1], true_theta[0], "wo", mec="black", label="truth")
    axes[0].plot(result.values[1], result.values[0], "c*", ms=12, mec="black", label="estimate")
    axes[0].set_xlabel("perfusion $k$")
    axes[0].set_ylabel("diffusivity $\\alpha$")
    axes[0].set_title("Two-parameter cost surface")
    axes[0].legend(fontsize=8)
    fig.colorbar(contour, ax=axes[0], label="least-squares cost")

    residual = residual_vector(_two_mode_signal(result.values, times), observed)
    jac = residual_jacobian(lambda theta: _two_mode_signal(theta, times), result.values, observed)
    cov = gauss_newton_covariance(jac, residual_variance=float(np.var(residual, ddof=1)))
    eigvals, eigvecs = np.linalg.eigh(cov)
    angles = np.linspace(0.0, 2.0 * np.pi, 200)
    ellipse = eigvecs @ (np.sqrt(np.maximum(eigvals, 0.0))[:, None] * np.vstack([np.cos(angles), np.sin(angles)]))
    axes[1].plot(result.values[1] + 1.96 * ellipse[1], result.values[0] + 1.96 * ellipse[0], color="tab:blue")
    axes[1].plot(true_theta[1], true_theta[0], "ko", label="truth")
    axes[1].plot(result.values[1], result.values[0], "r*", ms=12, label="estimate")
    axes[1].set_xlabel("perfusion $k$")
    axes[1].set_ylabel("diffusivity $\\alpha$")
    axes[1].set_title("Local Gauss-Newton uncertainty ellipse")
    axes[1].legend(fontsize=8)
    _save(
        fig,
        "multi_parameter_identifiability",
        "Joint diffusivity-perfusion identification for a two-mode synthetic Pennes response. The cost surface shows the local basin used by the optimizer, while the Gauss-Newton covariance ellipse summarizes parameter coupling around the recovered point.",
        "multi-parameter-identifiability",
    )


def sensor_time_information() -> None:
    alpha = 0.1
    k = 3.0
    x = np.linspace(0.0, 1.0, 90)
    y = np.linspace(0.0, 1.0, 90)
    xx, yy = np.meshgrid(x, y)
    times = np.linspace(0.0, 0.35, 160)
    spatial_amplitude = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    decay = 2.0 * np.pi**2 * alpha + k
    t_late = 0.18
    sensitivity_field = -t_late * spatial_amplitude * np.exp(-decay * t_late)

    center_amp = 1.0
    edge_amp = np.sin(np.pi * 0.25) * np.sin(np.pi * 0.25)
    line_amp = np.sin(np.pi * 0.5) * np.sin(np.pi * 0.25)
    amplitudes = {
        "center sensor": center_amp,
        "quarter sensor": edge_amp,
        "mid-edge interior": line_amp,
    }
    info_curves = {}
    for name, amp in amplitudes.items():
        sensitivity = -times * amp * np.exp(-decay * times)
        info_curves[name] = np.cumsum(sensitivity**2) / max(np.sum(sensitivity**2), 1e-16)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), constrained_layout=True)
    im = axes[0].pcolormesh(xx, yy, sensitivity_field**2, shading="auto", cmap="viridis")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    axes[0].set_title(f"Spatial information density at $t={t_late:.2f}$")
    fig.colorbar(im, ax=axes[0], label="$|\\partial u/\\partial k|^2$")
    for name, curve in info_curves.items():
        axes[1].plot(times, curve, label=name)
    axes[1].set_xlabel("latest observation time")
    axes[1].set_ylabel("fraction of cumulative information")
    axes[1].set_title("Timing and sensor placement effect")
    axes[1].legend(fontsize=8)
    _save(
        fig,
        "sensor_time_information",
        "Sensor/time information diagnostic for scalar Pennes perfusion. Information is proportional to $|\\partial u/\\partial k|^2$: central sensors see the largest modal amplitude, while later observations accumulate more perfusion information until the thermal signal has decayed.",
        "sensor-time-information",
    )


def profile_likelihood_diagnostic() -> None:
    true_k = 2.75
    alpha = 0.1
    times = np.linspace(0.01, 0.22, 18)
    observed = add_observation_noise(_pennes_signal(true_k, times, alpha), relative_level=0.006, seed=99)
    forward = lambda theta: _pennes_signal(theta[0], times, alpha)
    result = estimate_parameters(forward, observed, initial_guess=[2.4], bounds=([0.2], [6.0]), parameter_names=("k",))
    k_grid = np.linspace(1.4, 4.4, 220)
    rows = identifiability_scan(lambda k: _pennes_signal(k, times, alpha), observed, k_grid)
    costs = np.array([row["cost"] for row in rows])
    min_cost = float(np.min(costs))
    jac = residual_jacobian(forward, result.values, observed)
    curvature = float((jac.T @ jac)[0, 0])
    quadratic = min_cost + 0.5 * curvature * (k_grid - result.values[0]) ** 2

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)
    axes[0].plot(k_grid, costs - min_cost, color="tab:blue", label="profile scan")
    axes[0].plot(k_grid, quadratic - min_cost, color="tab:orange", ls="--", label="GN quadratic")
    axes[0].axvline(true_k, color="black", ls=":", label="truth")
    axes[0].axvline(result.values[0], color="tab:red", ls="-.", label="estimate")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("perfusion $k$")
    axes[0].set_ylabel("$\\Phi(k)-\\Phi(\\widehat{k})$")
    axes[0].set_title("Profile likelihood curvature")
    axes[0].legend(fontsize=8)

    residuals = observed - _pennes_signal(result.values[0], times, alpha)
    axes[1].plot(times, observed, "o", label="noisy observations", color="tab:gray")
    axes[1].plot(times, _pennes_signal(true_k, times, alpha), "-", label="truth", color="black")
    axes[1].plot(times, _pennes_signal(result.values[0], times, alpha), "--", label="fit", color="tab:red")
    ax2 = axes[1].twinx()
    ax2.bar(times, residuals, width=0.006, color="tab:purple", alpha=0.25, label="fit residual")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("temperature signal")
    ax2.set_ylabel("residual")
    axes[1].set_title("Fit and residual structure")
    axes[1].legend(fontsize=8, loc="upper right")
    ax2.legend(fontsize=8, loc="center right")
    _save(
        fig,
        "profile_likelihood_diagnostic",
        "Profile-likelihood diagnostic for scalar perfusion. The left panel compares the scanned objective to the local Gauss-Newton quadratic model, showing where the normal approximation is reliable; the right panel shows the fitted signal and residual structure.",
        "profile-likelihood-diagnostic",
    )


def sensitivity_adjoint_uq() -> None:
    true_k = 2.6
    candidate = 2.25
    alpha = 0.1
    times = np.linspace(0.01, 0.18, 14)
    observed = add_observation_noise(_pennes_signal(true_k, times, alpha), relative_level=0.004, seed=17)
    forward = lambda theta: _pennes_signal(theta[0], times, alpha)
    residual = residual_vector(forward(np.array([candidate])), observed)
    steps = np.logspace(-7, -2, 18)
    gradients = []
    for h in steps:
        jac = finite_difference_jacobian(forward, np.array([candidate]), step=np.array([h]), method="central")
        gradients.append(float(least_squares_gradient(jac, residual)[0]))
    analytic_sens = -times * np.exp(-(2.0 * np.pi**2 * alpha + candidate) * times)
    analytic_grad = float(analytic_sens @ residual)

    result = estimate_parameters(forward, observed, initial_guess=[2.0], bounds=([0.1], [6.0]), parameter_names=("k",))
    jac = residual_jacobian(forward, result.values, observed)
    fit_residual = residual_vector(forward(result.values), observed)
    sigma2 = float(np.var(fit_residual, ddof=1))
    cov = gauss_newton_covariance(jac, residual_variance=sigma2)
    intervals = confidence_intervals(result.values, cov, parameter_names=("k",))
    rng = np.random.default_rng(10)
    samples = bootstrap_parameter_estimates(
        lambda obs: estimate_parameters(forward, obs, initial_guess=[2.0], bounds=([0.1], [6.0])).values,
        observed,
        n_samples=96,
        relative_noise=0.004,
        seed=int(rng.integers(0, 2**31 - 1)),
    )
    summary = bootstrap_summary(samples, parameter_names=("k",))[0]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)
    axes[0].loglog(steps, np.abs(np.asarray(gradients) - analytic_grad), "o-", color="tab:purple")
    axes[0].set_xlabel("finite-difference step")
    axes[0].set_ylabel("$|J^T r - (J^T r)_{analytic}|$")
    axes[0].set_title("Sensitivity-gradient step study")

    axes[1].hist(samples[:, 0], bins=18, color="tab:green", alpha=0.65, density=True, label="bootstrap")
    axes[1].axvline(result.values[0], color="black", lw=1.4, label="estimate")
    axes[1].axvspan(intervals[0]["lower"], intervals[0]["upper"], color="tab:blue", alpha=0.18, label="GN 95% CI")
    axes[1].axvspan(summary["lower"], summary["upper"], color="tab:orange", alpha=0.18, label="bootstrap 95%")
    axes[1].axvline(true_k, color="red", ls="--", lw=1.2, label="truth")
    axes[1].set_xlabel("perfusion $k$")
    axes[1].set_ylabel("density")
    axes[1].set_title("Uncertainty diagnostics")
    axes[1].legend(fontsize=8)
    _save(
        fig,
        "sensitivity_uq_diagnostics",
        "Sensitivity and uncertainty diagnostics. The finite-difference gradient shows the expected step-size tradeoff, and the fitted perfusion uncertainty is summarized with both a local Gauss-Newton 95\\% confidence interval and a bootstrap/noise-ensemble 95\\% interval.",
        "sensitivity-uq-diagnostics",
    )


def regularization_path() -> None:
    true_theta = np.array([2.0, 4.8, 3.2, 6.2])
    sensing = np.array(
        [
            [0.9, 0.7, 0.1, 0.0],
            [0.1, 0.8, 0.7, 0.2],
            [0.2, 0.0, 0.7, 0.9],
        ],
        dtype=float,
    )
    observed = add_observation_noise(sensing @ true_theta, relative_level=0.01, seed=12)
    prior = np.full(true_theta.size, np.mean(true_theta))
    lambdas = np.logspace(-6, 1, 40)
    data_norm = []
    reg_norm = []
    solution_norm = []
    estimates = []
    for lam in lambdas:
        result = estimate_parameters(
            lambda theta: sensing @ theta,
            observed,
            initial_guess=prior,
            bounds=(np.zeros(true_theta.size), np.full(true_theta.size, 10.0)),
            regularization={"prior": prior, "strength": float(lam), "scale": np.ones(true_theta.size)},
            optimizer_options={"max_nfev": 100},
        )
        estimates.append(result.values)
        data_norm.append(result.data_residual_norm)
        reg_norm.append(result.regularization_residual_norm)
        solution_norm.append(float(np.linalg.norm(result.values - true_theta)))
    estimates = np.asarray(estimates)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)
    axes[0].loglog(data_norm, reg_norm, "o-", color="tab:blue")
    idx = int(np.argmin(solution_norm))
    axes[0].plot(data_norm[idx], reg_norm[idx], "r*", ms=12, label="lowest coefficient error")
    axes[0].set_xlabel("data residual norm")
    axes[0].set_ylabel("regularization residual norm")
    axes[0].set_title("Regularization L-curve")
    axes[0].legend(fontsize=8)
    for j in range(estimates.shape[1]):
        axes[1].semilogx(lambdas, estimates[:, j], label=f"$\\theta_{j}$")
        axes[1].axhline(true_theta[j], color=f"C{j}", ls=":", lw=1.0)
    axes[1].set_xlabel("regularization strength $\\lambda$")
    axes[1].set_ylabel("coefficient value")
    axes[1].set_title("Recovered coefficients along path")
    axes[1].legend(fontsize=8, ncol=2)
    _save(
        fig,
        "regularization_path_expanded",
        "Regularization-path diagnostic for an underdetermined coefficient inverse problem. The L-curve exposes the data-fit versus prior-penalty tradeoff, while the coefficient traces show how stronger Tikhonov regularization damps non-identifiable directions.",
        "regularization-path-expanded",
    )


def field_basis_coverage() -> None:
    centers = np.array([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]], dtype=float)
    basis = GaussianFieldBasis(centers=centers, radius=0.45, normalize=True)
    x = np.linspace(0.0, 1.0, 95)
    y = np.linspace(0.0, 1.0, 95)
    xx, yy = np.meshgrid(x, y)
    design = basis.design_matrix(xx.ravel(), yy.ravel())
    fields = [design[:, j].reshape(xx.shape) for j in range(basis.n_basis)]
    gram = design.T @ design / design.shape[0]
    normalized_gram = gram / np.sqrt(np.outer(np.diag(gram), np.diag(gram)))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for j, field in enumerate(fields):
        axes[0].contour(xx, yy, field, levels=[0.18, 0.32, 0.46], colors=[colors[j]], linewidths=1.4)
        axes[0].plot(centers[j, 0], centers[j, 1], "o", color=colors[j], mec="black")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    axes[0].set_title("Normalized Gaussian basis support")
    im = axes[1].imshow(normalized_gram, vmin=0.0, vmax=1.0, cmap="magma")
    axes[1].set_xticks(np.arange(basis.n_basis))
    axes[1].set_yticks(np.arange(basis.n_basis))
    axes[1].set_xlabel("basis index")
    axes[1].set_ylabel("basis index")
    axes[1].set_title("Basis overlap / parameter coupling")
    for i in range(basis.n_basis):
        for j in range(basis.n_basis):
            axes[1].text(j, i, f"{normalized_gram[i, j]:.2f}", ha="center", va="center", color="white" if normalized_gram[i, j] > 0.55 else "black", fontsize=8)
    fig.colorbar(im, ax=axes[1], label="normalized overlap")
    _save(
        fig,
        "field_basis_coverage",
        "Gaussian perfusion-field basis diagnostic. The left panel shows the spatial support of each normalized radial basis function; the right panel reports basis overlap, which acts as a practical proxy for coefficient coupling in the field inverse problem.",
        "field-basis-coverage",
    )


def field_inversion_coefficients() -> None:
    csv_path = DIRECTION_D / "pennes_field_inverse_study" / "pennes_field_inverse_coefficients.csv"
    summary_path = DIRECTION_D / "pennes_field_inverse_study" / "pennes_field_inverse_summary.json"
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    with summary_path.open(encoding="utf-8") as fh:
        summary = json.load(fh)
    idx = np.array([int(row["index"]) for row in rows])
    truth = np.array([float(row["true"]) for row in rows])
    recovered = np.array([float(row["recovered"]) for row in rows])
    error = recovered - truth

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), constrained_layout=True)
    width = 0.38
    axes[0].bar(idx - width / 2, truth, width=width, label="truth", color="tab:gray")
    axes[0].bar(idx + width / 2, recovered, width=width, label="recovered", color="tab:blue")
    axes[0].set_xlabel("Gaussian basis coefficient")
    axes[0].set_ylabel("perfusion value")
    axes[0].set_title("Field-basis coefficient recovery")
    axes[0].legend(fontsize=8)
    axes[1].bar(idx, error, color=np.where(error >= 0.0, "tab:red", "tab:green"))
    axes[1].axhline(0.0, color="black", lw=1.0)
    axes[1].set_xlabel("Gaussian basis coefficient")
    axes[1].set_ylabel("recovered - truth")
    axes[1].set_title(f"Coefficient error; field RMSE={summary['field_rmse']:.3f}")
    _save(
        fig,
        "field_inversion_coefficients",
        "Regularized Pennes perfusion-field inversion summarized in coefficient space. The normalized Gaussian basis recovers the main spatial trend while the error panel identifies which basis locations dominate the remaining field RMSE.",
        "field-inversion-coefficients",
    )


def baseline_comparison() -> None:
    ml_csv = DIRECTION_D / "pinn_ml_baseline_comparison" / "pennes_ml_baseline_metrics.csv"
    jax_summary = DIRECTION_D / "jax_pinn_baseline" / "pennes_jax_pinn_summary.json"
    jax_history = DIRECTION_D / "jax_pinn_baseline" / "pennes_jax_pinn_history.csv"
    rows = []
    with ml_csv.open(newline="", encoding="utf-8") as fh:
        rows.extend(csv.DictReader(fh))
    with jax_summary.open(encoding="utf-8") as fh:
        jax = json.load(fh)
    rows.append(
        {
            "name": "jax_pinn",
            "kind": "physics_informed_nn",
            "estimate": str(jax["perfusion"]),
            "absolute_error": str(jax["absolute_error"]),
            "relative_error": str(jax["absolute_error"] / max(abs(jax["true_perfusion"]), 1e-16)),
            "residual_norm": str(jax["final_loss"]),
            "relative_residual": str(jax["final_loss"]),
        }
    )
    names = [row["name"].replace("_", "\n") for row in rows]
    estimates = np.array([float(row["estimate"]) for row in rows])
    errors = np.array([float(row["absolute_error"]) for row in rows])
    relres = np.array([float(row["relative_residual"]) for row in rows])
    truth = float(jax["true_perfusion"])

    hist_rows = []
    with jax_history.open(newline="", encoding="utf-8") as fh:
        hist_rows.extend(csv.DictReader(fh))
    iterations = np.array([int(row["iteration"]) for row in hist_rows])
    loss = np.array([float(row["loss"]) for row in hist_rows])
    k_hist = np.array([float(row["perfusion"]) for row in hist_rows])

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), constrained_layout=True)
    axes[0].bar(names, estimates, color=["tab:green", "tab:blue", "tab:orange", "tab:purple"])
    axes[0].axhline(truth, color="black", ls="--", label="JAX truth")
    axes[0].set_ylabel("estimated perfusion")
    axes[0].set_title("Baseline parameter estimates")
    axes[0].legend(fontsize=8)
    axes[1].bar(names, errors, color=["tab:green", "tab:blue", "tab:orange", "tab:purple"], label="absolute error")
    axes[1].plot(names, relres, "ko-", label="relative residual / final loss")
    axes[1].set_yscale("log")
    axes[1].set_title("Error and residual metrics")
    axes[1].legend(fontsize=8)
    axes[2].semilogy(iterations, loss, "o-", color="tab:purple", label="PINN loss")
    axes[2].set_xlabel("JAX PINN iteration")
    axes[2].set_ylabel("loss")
    ax2 = axes[2].twinx()
    ax2.plot(iterations, k_hist, "s-", color="tab:red", label="PINN $k$")
    ax2.axhline(truth, color="black", ls="--", lw=1.0)
    ax2.set_ylabel("perfusion")
    axes[2].set_title("JAX PINN convergence")
    axes[2].legend(fontsize=8, loc="upper right")
    ax2.legend(fontsize=8, loc="center right")
    _save(
        fig,
        "baseline_comparison_expanded",
        "Expanded baseline comparison. Trusted finite-volume inversion, an RBF-ridge surrogate, a training-grid lookup, and the optional JAX PINN are placed on a common estimate/error schema; the right panel shows the JAX PINN loss and perfusion trajectory.",
        "baseline-comparison-expanded",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    workflow_tree()
    scalar_identifiability_noise()
    sensor_time_information()
    multiparameter_landscape()
    profile_likelihood_diagnostic()
    sensitivity_adjoint_uq()
    regularization_path()
    field_basis_coverage()
    field_inversion_coefficients()
    baseline_comparison()


if __name__ == "__main__":
    main()
