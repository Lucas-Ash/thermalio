#!/usr/bin/env python3
"""Generate expanded Direction C non-Fourier/fractional Stefan study plots.

Run from the repository root:

    MPLCONFIGDIR=/tmp/matplotlib /home/user/thermalio/.venv/bin/python direction_c_expansion_studies.py

Writes figures to:

    test_plots/direction_C_nonfourier_phase_change/expansion_studies/
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heat_solver.cases import (
    fractional_stefan_apparent_capacity_case,
    hyperbolic_stefan_apparent_capacity_case,
)
from heat_solver.geometry import polygon_area_and_centroid
from heat_solver.meshes import generate_square_polygonal_mesh
from heat_solver.phase_change import ApparentHeatCapacityModel
from heat_solver.transport import FractionalStefanSolver, HyperbolicHeatSolver, HyperbolicStefanSolver


OUT = ROOT / "test_plots" / "direction_C_nonfourier_phase_change" / "expansion_studies"


def _square(n, bbox):
    vertices, polygons, centers = generate_square_polygonal_mesh(nx=n, ny=n, bbox=bbox)
    areas = np.array([polygon_area_and_centroid(vertices[poly])[0] for poly in polygons])
    return vertices, polygons, centers, areas


def _rel_l2(u, exact, areas):
    return float(np.sqrt(np.sum(areas * (u - exact) ** 2)) / max(np.sqrt(np.sum(areas * exact**2)), 1e-16))


def _centerline(centers, values, y_target):
    row_score = np.abs(centers[:, 1] - y_target)
    row = row_score <= np.min(row_score) + 1e-12
    order = np.argsort(centers[row, 0])
    return centers[row, 0][order], values[row][order]


def hyperbolic_relaxation_study():
    """Show Cattaneo relaxation term size, wave-speed scaling, and solver accuracy."""
    alpha = 0.08
    taus = np.array([0.05, 0.09, 0.15])
    t = 0.04
    x = np.linspace(-1.0, 1.0, 600)
    amplitude = 0.8
    width = 0.18
    speed = 0.45
    x0 = -0.35
    pcm = ApparentHeatCapacityModel(
        solidus_temperature=-0.25,
        liquidus_temperature=0.25,
        latent_heat=6.0,
        specific_heat=1.0,
    )
    z = (x - x0 - speed * t) / width
    tanh_z = np.tanh(z)
    sech2 = 1.0 - tanh_z**2
    temp = amplitude * tanh_z
    dTdt = -(amplitude * speed / width) * sech2
    dTtt = -(2.0 * amplitude * speed**2 / width**2) * tanh_z * sech2
    dTxx = -(2.0 * amplitude / width**2) * tanh_z * sech2
    cp = pcm.effective_heat_capacity(temp)

    errors = []
    for tau in taus:
        case = hyperbolic_stefan_apparent_capacity_case(alpha=alpha, tau=float(tau))
        vertices, polygons, centers, areas = _square(18, case["bbox"])
        solver = HyperbolicStefanSolver(
            vertices,
            polygons,
            case["alpha"],
            t / 72,
            case["relaxation_time"],
            case["phase_change_model"],
            bc_func=case["boundary"],
            source_func=case["source"],
            phase_change_options={
                **case["phase_change_options"],
                "max_iters": 160,
                "relaxation": 0.7,
            },
        )
        u0 = case["solution"](centers[:, 0], centers[:, 1], 0.0)
        du0 = case["initial_rate"](centers[:, 0], centers[:, 1])
        _, u = solver.solve(u0, 0.0, t, du0=du0)
        exact = case["solution"](centers[:, 0], centers[:, 1], t)
        errors.append(_rel_l2(u, exact, areas))

    tau_ref = 0.05
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0), constrained_layout=True)
    axes[0].plot(x, tau_ref * dTtt, label=r"$\tau T_{tt}$", color="tab:purple")
    axes[0].plot(x, cp * dTdt, label=r"$c(T)T_t$", color="tab:blue")
    axes[0].plot(x, -alpha * dTxx, label=r"$-\alpha T_{xx}$", color="tab:orange")
    axes[0].axvline(x0 + speed * t, color="black", ls="--", lw=1.0, label="interface")
    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel("operator contribution")
    axes[0].set_title("Hyperbolic Stefan balance near front")
    axes[0].legend(fontsize=8)

    wave_speeds = np.sqrt(alpha / taus)
    relaxation_ratio = [np.max(np.abs(tau * dTtt)) / max(np.max(np.abs(cp * dTdt)), 1e-16) for tau in taus]
    axes[1].plot(taus, wave_speeds, "o-", label=r"$\sqrt{\alpha/\tau}$", color="tab:red")
    axes[1].set_xlabel("relaxation time $\\tau$")
    axes[1].set_ylabel("thermal wave speed")
    axb = axes[1].twinx()
    axb.plot(taus, relaxation_ratio, "s--", label=r"$\max|\tau T_{tt}|/\max|cT_t|$", color="tab:purple")
    axb.set_ylabel("relaxation-term ratio")
    axes[1].set_title("Relaxation controls finite-speed response")
    lines, labels = axes[1].get_legend_handles_labels()
    lines_b, labels_b = axb.get_legend_handles_labels()
    axes[1].legend(lines + lines_b, labels + labels_b, fontsize=8)

    axes[2].loglog(taus, errors, "o-", color="tab:green")
    axes[2].set_xlabel("relaxation time $\\tau$")
    axes[2].set_ylabel("relative $L^2$ error")
    axes[2].set_title("Solver-backed manufactured error")
    fig.suptitle("Direction C expansion 1: relaxation-aware hyperbolic Stefan physics")
    fig.savefig(OUT / "hyperbolic_relaxation_balance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fractional_memory_study():
    """Show Caputo memory kernels and FractionalStefanSolver temporal behavior."""
    betas = np.array([0.35, 0.6, 0.85])
    k = np.arange(80)
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.0), constrained_layout=True)
    for beta in betas:
        weights = (k + 1.0) ** (1.0 - beta) - k**(1.0 - beta)
        axes[0].loglog(k + 1, weights / weights[0], "o-", ms=3, label=fr"$\beta={beta:.2f}$")
        axes[1].plot(k + 1, np.cumsum(weights) / np.sum(weights), label=fr"$\beta={beta:.2f}$")
    axes[0].set_xlabel("history lag $k+1$")
    axes[0].set_ylabel("normalized L1 weight")
    axes[0].set_title("Caputo memory tail")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("number of retained lags")
    axes[1].set_ylabel("fraction of finite-window memory")
    axes[1].set_title("Cumulative memory")
    axes[1].legend(fontsize=8)

    t_end = 0.16
    beta_values = [0.4, 0.6, 0.8]
    expected_orders = []
    fine_errors = []
    coarse_errors = []
    for beta in beta_values:
        case = fractional_stefan_apparent_capacity_case(alpha=0.08, beta=beta)
        errs = []
        for nt in (16, 32):
            vertices, polygons, centers, areas = _square(18, case["bbox"])
            solver = FractionalStefanSolver(
                vertices,
                polygons,
                case["alpha"],
                t_end / nt,
                case["beta"],
                case["phase_change_model"],
                bc_func=case["boundary"],
                source_func=case["source"],
                phase_change_options=case["phase_change_options"],
            )
            u0 = case["solution"](centers[:, 0], centers[:, 1], 0.0)
            _, u = solver.solve(u0, 0.0, t_end)
            exact = case["solution"](centers[:, 0], centers[:, 1], t_end)
            errs.append(_rel_l2(u, exact, areas))
        expected_orders.append(2.0 - beta)
        coarse_errors.append(errs[0])
        fine_errors.append(errs[1])
    width = 0.33
    x = np.arange(len(beta_values))
    axes[2].bar(x - width / 2, expected_orders, width=width, color="tab:gray", label=r"expected L1 order $2-\beta$")
    ax_err = axes[2].twinx()
    ax_err.semilogy(x, coarse_errors, "o--", color="tab:orange", label="16 steps")
    ax_err.semilogy(x, fine_errors, "s-", color="tab:blue", label="32 steps")
    axes[2].set_xticks(x, [fr"$\beta={beta:.1f}$" for beta in beta_values])
    axes[2].set_ylabel("expected temporal order")
    ax_err.set_ylabel("solver-backed relative $L^2$ error")
    axes[2].set_title("Fractional Stefan accuracy by beta")
    lines, labels = axes[2].get_legend_handles_labels()
    lines_b, labels_b = ax_err.get_legend_handles_labels()
    axes[2].legend(lines + lines_b, labels + labels_b, fontsize=8)
    fig.suptitle("Direction C expansion 2: fractional memory in Stefan phase change")
    fig.savefig(OUT / "fractional_memory_convergence.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def mushy_zone_capacity_study():
    """Show apparent-capacity stiffness and phase-coupling necessity."""
    temp = np.linspace(-0.8, 0.8, 700)
    widths = [0.12, 0.25, 0.5]
    latent = 6.0
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.0), constrained_layout=True)
    for width in widths:
        pcm = ApparentHeatCapacityModel(
            solidus_temperature=-0.5 * width,
            liquidus_temperature=0.5 * width,
            latent_heat=latent,
            specific_heat=1.0,
        )
        axes[0].plot(temp, pcm.effective_heat_capacity(temp), label=fr"$\Delta T_m={width:.2f}$")
        axes[1].plot(temp, pcm.enthalpy(temp), label=fr"$\Delta T_m={width:.2f}$")
    axes[0].set_xlabel("temperature")
    axes[0].set_ylabel("effective heat capacity")
    axes[0].set_title("Latent heat creates a stiff capacity spike")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("temperature")
    axes[1].set_ylabel("enthalpy")
    axes[1].set_title("Enthalpy smooths the phase transition")
    axes[1].legend(fontsize=8)

    case = hyperbolic_stefan_apparent_capacity_case(alpha=0.08, tau=0.05)
    t_end = 0.04
    vertices, polygons, centers, areas = _square(22, case["bbox"])
    u0 = case["solution"](centers[:, 0], centers[:, 1], 0.0)
    du0 = case["initial_rate"](centers[:, 0], centers[:, 1])
    phase_solver = HyperbolicStefanSolver(
        vertices,
        polygons,
        case["alpha"],
        t_end / 88,
        case["relaxation_time"],
        case["phase_change_model"],
        bc_func=case["boundary"],
        source_func=case["source"],
        phase_change_options=case["phase_change_options"],
    )
    _, u_phase = phase_solver.solve(u0, 0.0, t_end, du0=du0)
    no_phase_solver = HyperbolicHeatSolver(
        vertices,
        polygons,
        case["alpha"],
        t_end / 88,
        case["relaxation_time"],
        bc_func=case["boundary"],
        source_func=case["source"],
    )
    _, u_no_phase = no_phase_solver.solve(u0, 0.0, t_end, du0=du0)
    exact = case["solution"](centers[:, 0], centers[:, 1], t_end)
    y_mid = 0.0
    x_line, exact_line = _centerline(centers, exact, y_mid)
    _, phase_line = _centerline(centers, u_phase, y_mid)
    _, no_phase_line = _centerline(centers, u_no_phase, y_mid)
    axes[2].plot(x_line, exact_line, "k-", label="manufactured truth")
    axes[2].plot(x_line, phase_line, "o", ms=3, label="with apparent capacity")
    axes[2].plot(x_line, no_phase_line, "--", label="capacity omitted")
    axes[2].set_xlabel("$x$ along centerline")
    axes[2].set_ylabel("temperature")
    axes[2].set_title(
        "Capacity coupling is not optional\n"
        f"rel L2: phase={_rel_l2(u_phase, exact, areas):.1e}, omitted={_rel_l2(u_no_phase, exact, areas):.1e}"
    )
    axes[2].legend(fontsize=8)
    fig.suptitle("Direction C expansion 3: latent-heat and mushy-zone sensitivity")
    fig.savefig(OUT / "mushy_zone_capacity_coupling.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hyperbolic_relaxation_study()
    fractional_memory_study()
    mushy_zone_capacity_study()
    print(f"Wrote Direction C expansion plots to {OUT}")


if __name__ == "__main__":
    main()
