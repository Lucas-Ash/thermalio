#!/usr/bin/env python3
"""Direction C application-study runners (computing-capability step 5).

Reproducible 2D non-Fourier phase-change scenarios beyond manufactured
solutions, each reporting front position, peak temperature, liquid fraction,
injected/extracted energy, and the sensible/latent enthalpy budget via the
sharp-interface diagnostics (step 4).  Six scenarios:

* ``pulsed_laser_melting``   -- a boundary heat-flux pulse melts a cold solid
  slab (Cattaneo/hyperbolic Stefan; the classic finite-speed laser-heating
  regime), with an energy-closure audit (injected boundary energy vs enthalpy
  rise).
* ``cryosurgery_freezing``   -- a cold cryoprobe boundary freezes a warm domain,
  tracking the freezing-front margin and frozen volume fraction.
* ``moving_scan_melt_pool``  -- a moving volumetric heat source creates a
  melt-pool track, mimicking a small additive-manufacturing scan.
* ``dual_pulse_remelting``   -- two separated boundary pulses show remelting
  and latent-heat retention between pulses.
* ``rapid_solidification_quench`` -- a hot liquid slab cools against cold
  boundaries, tracking liquid-fraction collapse.
* ``buried_hot_inclusion_relaxation`` -- a localized hot inclusion melts and
  then refreezes inside a colder matrix.

Outputs go to ``test_plots/direction_C_nonfourier_phase_change/applications/``.

Usage:
    MPLCONFIGDIR=/tmp/matplotlib python direction_c_applications.py
    python direction_c_applications.py --quick
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heat_solver import interface_diagnostics as idiag
from heat_solver.geometry import polygon_area_and_centroid
from heat_solver.meshes import generate_square_polygonal_mesh
from heat_solver.phase_change import ApparentHeatCapacityModel
from heat_solver.transport import HyperbolicStefanSolver

OUT = ROOT / "test_plots" / "direction_C_nonfourier_phase_change" / "applications"

CONVERGENCE_FIELDS = [
    "solve_converged", "solve_steps", "failed_steps", "max_iterations",
    "mean_iterations", "final_residual", "max_residual", "min_capacity",
    "max_capacity", "tolerance", "relaxation", "anderson_depth",
]
ENERGY_AUDIT_FIELDS = [
    "energy_in", "energy_out", "initial_total_enthalpy", "sensible_enthalpy",
    "latent_enthalpy", "total_enthalpy", "enthalpy_change",
    "expected_enthalpy_change", "energy_closure_residual",
    "relative_energy_closure_residual",
]


def _mesh(nx, ny, bbox):
    vertices, polygons, centers = generate_square_polygonal_mesh(nx=nx, ny=ny, bbox=bbox)
    areas = np.array([polygon_area_and_centroid(vertices[poly])[0] for poly in polygons])
    return vertices, polygons, centers, areas


def _write(name, summary, rows, fieldnames):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (OUT / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fieldnames(names):
    ordered = []
    for name in [*names, *ENERGY_AUDIT_FIELDS, *CONVERGENCE_FIELDS]:
        if name not in ordered:
            ordered.append(name)
    return ordered


def _attach_audit_and_report(row, solver, audit):
    row.update(audit)
    row.update(idiag.summarize_solve_report(getattr(solver, "solve_report", None)))
    return row


def _final_diagnostics(rows):
    final = rows[-1]
    return {
        "nonconverged_steps": int(final["failed_steps"]),
        "final_max_iterations": int(final["max_iterations"]),
        "final_mean_iterations": float(final["mean_iterations"]),
        "final_max_residual": float(final["max_residual"]),
        "final_max_capacity": float(final["max_capacity"]),
        "final_relative_energy_closure_residual": float(final["relative_energy_closure_residual"]),
    }


def _source_energy(centers, areas, source_func, t_end, n_steps):
    times = np.linspace(0.0, float(t_end), max(int(n_steps), 2) + 1)
    values = []
    for t in times:
        q = np.asarray(source_func(centers[:, 0], centers[:, 1], float(t)), dtype=float)
        values.append(float(np.sum(areas * q)))
    return float(np.trapezoid(values, times))


def _plot_generic_application(name, rows, final_field, title, metric_specs, cmap="inferno"):
    times = [r["time"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0), constrained_layout=True)
    for key, label, color, style in metric_specs[:2]:
        axes[0].plot(times, [r[key] for r in rows], style, color=color, label=label)
    axes[0].set_xlabel("time")
    axes[0].set_title(metric_specs[0][1] if len(metric_specs) == 1 else "Front / phase metrics")
    axes[0].legend(fontsize=8)

    for key, label, color, style in metric_specs[2:]:
        axes[1].plot(times, [r[key] for r in rows], style, color=color, label=label)
    if len(metric_specs) > 2:
        axes[1].legend(fontsize=8)
    axes[1].set_xlabel("time")
    axes[1].set_title("Thermal / enthalpy metrics")

    centers, u = final_field
    tcf = axes[2].tripcolor(centers[:, 0], centers[:, 1], u, shading="gouraud", cmap=cmap)
    axes[2].set_aspect("equal")
    axes[2].set_title("Final temperature field")
    fig.colorbar(tcf, ax=axes[2], shrink=0.8)
    fig.suptitle(title)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_diagnostics_dashboard(name, rows, title):
    times = [r["time"] for r in rows]
    max_residual = np.maximum([r["max_residual"] for r in rows], 1e-16)
    final_residual = np.maximum([r["final_residual"] for r in rows], 1e-16)
    tolerance = [r["tolerance"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.0), constrained_layout=True)
    axes[0, 0].plot(times, [r["max_iterations"] for r in rows], "o-", label="max iterations")
    axes[0, 0].plot(times, [r["mean_iterations"] for r in rows], "s--", label="mean iterations")
    axes[0, 0].set_xlabel("time")
    axes[0, 0].set_title("Nonlinear Picard/Anderson work")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].semilogy(times, max_residual, "o-", label="max step residual")
    axes[0, 1].semilogy(times, final_residual, "s--", label="final step residual")
    axes[0, 1].semilogy(times, tolerance, "k:", label="tolerance")
    axes[0, 1].set_xlabel("time")
    axes[0, 1].set_title("Normalized nonlinear residual")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(times, [r["energy_closure_residual"] for r in rows], "o-", label="absolute")
    axes[1, 0].plot(times, [r["relative_energy_closure_residual"] for r in rows], "s--", label="relative")
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set_xlabel("time")
    axes[1, 0].set_title("Energy closure residual")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(times, [r["sensible_enthalpy"] for r in rows], "o-", label="sensible")
    axes[1, 1].plot(times, [r["latent_enthalpy"] for r in rows], "s-", label="latent")
    axes[1, 1].plot(times, [r["energy_in"] for r in rows], "^-", label="energy in")
    axes[1, 1].plot(times, [r["energy_out"] for r in rows], "v-", label="energy out")
    axes[1, 1].set_xlabel("time")
    axes[1, 1].set_title("Enthalpy and boundary/source energy")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle(f"{title}: convergence and energy audit")
    fig.savefig(OUT / f"{name}_diagnostics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Scenario 1: pulsed-laser melting of a cold slab (flux-driven hyperbolic Stefan)
# --------------------------------------------------------------------------- #
def pulsed_laser_melting(nx=48, ny=12, bbox=(0.0, 1.5, 0.0, 0.3), alpha=0.06, tau=0.04,
                         q0=2.2, t_pulse=0.25, t_end=0.5, nt=320, n_snapshots=6,
                         latent_heat=3.0, half_width=0.45, T0=-0.6):
    pcm = ApparentHeatCapacityModel(
        solidus_temperature=-half_width, liquidus_temperature=half_width,
        latent_heat=latent_heat, specific_heat=1.0,
    )
    vertices, polygons, centers, areas = _mesh(nx, ny, bbox)
    x_left, y_mid = bbox[0], 0.5 * (bbox[2] + bbox[3])
    left_len = bbox[3] - bbox[2]

    def pulse_flux(x, y, t, nx_, ny_):
        on_left = np.isclose(x, x_left)
        active = q0 if t <= t_pulse else 0.0
        return np.where(on_left, active, 0.0)

    u0 = np.full(centers.shape[0], float(T0))
    H0 = idiag.enthalpy_budget(pcm, u0, areas)["total"]
    opts = {"max_iters": 200, "tol": 1e-8, "relaxation": 0.4, "anderson_depth": 6,
            "raise_on_nonconvergence": False}

    rows = []
    snaps = np.linspace(t_end / n_snapshots, t_end, n_snapshots)
    final_field = None
    for t_k in snaps:
        steps = max(2, int(round(nt * t_k / t_end)))
        solver = HyperbolicStefanSolver(
            vertices, polygons, alpha, t_k / steps, tau, pcm,
            bc_type="flux", bc_func=pulse_flux, phase_change_options=opts,
        )
        _, u = solver.solve(u0, 0.0, float(t_k), du0=np.zeros_like(u0))
        budget = idiag.enthalpy_budget(pcm, u, areas)
        front = idiag.interface_position(centers, u, pcm, axis="x", coord=y_mid, pick="last")
        injected = q0 * left_len * min(float(t_k), t_pulse)
        audit = idiag.enthalpy_audit(pcm, u0, u, areas, energy_in=injected)
        rows.append(_attach_audit_and_report({
            "time": float(t_k),
            "front_position": float("nan") if front is None else float(front),
            "peak_temperature": float(np.max(u)),
            "liquid_fraction": idiag.liquid_volume_fraction(pcm, u, areas),
            "mushy_thickness": idiag.mushy_zone_thickness(centers, u, pcm, axis="x", coord=y_mid),
            "enthalpy_rise": budget["total"] - H0,
            "injected_energy": injected,
        }, solver, audit))
        final_field = (centers, u)

    summary = {
        "scenario": "pulsed_laser_melting", "alpha": alpha, "tau": tau, "q0": q0,
        "t_pulse": t_pulse, "t_end": t_end, "latent_heat": latent_heat,
        "wave_speed": float(np.sqrt(alpha / tau)),
        "final_front_position": rows[-1]["front_position"],
        "final_liquid_fraction": rows[-1]["liquid_fraction"],
        "final_energy_closure_residual": rows[-1]["energy_closure_residual"],
        "final_injected_energy": rows[-1]["injected_energy"],
    }
    summary.update(_final_diagnostics(rows))
    _write("pulsed_laser_melting", summary, rows,
           _fieldnames(["time", "front_position", "peak_temperature", "liquid_fraction",
                        "mushy_thickness", "enthalpy_rise", "injected_energy"]))
    _plot_melting(rows, final_field, summary)
    _plot_diagnostics_dashboard("pulsed_laser_melting", rows, "Pulsed-laser melting")
    return summary, rows


def _plot_melting(rows, final_field, summary):
    times = [r["time"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0), constrained_layout=True)
    axes[0].plot(times, [r["front_position"] for r in rows], "o-", color="tab:red", label="melt front")
    axes[0].plot(times, [r["liquid_fraction"] for r in rows], "s--", color="tab:blue", label="liquid fraction")
    axes[0].set_xlabel("time"); axes[0].set_title("Melt-front advance & liquid fraction"); axes[0].legend(fontsize=8)
    axes[1].plot(times, [r["enthalpy_rise"] for r in rows], "o-", label="enthalpy rise")
    axes[1].plot(times, [r["injected_energy"] for r in rows], "k--", label="injected energy")
    axes[1].plot(times, [r["latent_enthalpy"] for r in rows], "^-", color="tab:purple", label="latent enthalpy")
    axes[1].set_xlabel("time"); axes[1].set_ylabel("energy"); axes[1].set_title("Energy closure & latent budget")
    axes[1].legend(fontsize=8)
    centers, u = final_field
    tcf = axes[2].tripcolor(centers[:, 0], centers[:, 1], u, shading="gouraud", cmap="inferno")
    axes[2].set_aspect("equal"); axes[2].set_title("Final temperature field")
    fig.colorbar(tcf, ax=axes[2], shrink=0.8)
    fig.suptitle(f"Direction C application: pulsed-laser melting (wave speed {summary['wave_speed']:.2f})")
    fig.savefig(OUT / "pulsed_laser_melting.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Scenario 2: cryosurgery freezing front from a cold probe boundary
# --------------------------------------------------------------------------- #
def cryosurgery_freezing(nx=40, ny=40, bbox=(0.0, 1.0, 0.0, 1.0), alpha=0.08, tau=0.005,
                         T_probe=-0.8, T_far=0.6, t_end=0.5, nt=100, n_snapshots=6,
                         latent_heat=4.0, half_width=0.15):
    pcm = ApparentHeatCapacityModel(
        solidus_temperature=-half_width, liquidus_temperature=half_width,
        latent_heat=latent_heat, specific_heat=1.0,
    )
    vertices, polygons, centers, areas = _mesh(nx, ny, bbox)
    x_left, y_mid = bbox[0], 0.5 * (bbox[2] + bbox[3])
    # Dirichlet data is sampled at cell centers, so the cold-probe boundary
    # column is selected by a one-cell-wide band (centres sit at x ~ dx/2).
    dx = (bbox[1] - bbox[0]) / nx

    def probe_dirichlet(x, y, t):
        return np.where(x < x_left + 0.75 * dx, T_probe, T_far)

    u0 = np.full(centers.shape[0], float(T_far))
    H0 = idiag.enthalpy_budget(pcm, u0, areas)["total"]
    opts = {"max_iters": 200, "tol": 1e-8, "relaxation": 0.5, "anderson_depth": 6,
            "raise_on_nonconvergence": False}

    rows = []
    snaps = np.linspace(t_end / n_snapshots, t_end, n_snapshots)
    final_field = None
    for t_k in snaps:
        steps = max(2, int(round(nt * t_k / t_end)))
        solver = HyperbolicStefanSolver(
            vertices, polygons, alpha, t_k / steps, tau, pcm,
            bc_type="dirichlet", bc_func=probe_dirichlet, phase_change_options=opts,
        )
        _, u = solver.solve(u0, 0.0, float(t_k), du0=np.zeros_like(u0))
        budget = idiag.enthalpy_budget(pcm, u, areas)
        energy_out = max(H0 - budget["total"], 0.0)
        audit = idiag.enthalpy_audit(pcm, u0, u, areas, energy_out=energy_out)
        # Freezing front = solidus isotherm distance from the cold probe.
        front = idiag.interface_position(centers, u, pcm, axis="x", coord=y_mid,
                                         level=pcm.solidus_temperature, pick="first")
        frozen = float(np.sum(areas[u < pcm.solidus_temperature]) / np.sum(areas))
        rows.append(_attach_audit_and_report({
            "time": float(t_k),
            "freezing_margin": float("nan") if front is None else float(front - x_left),
            "min_temperature": float(np.min(u)),
            "frozen_fraction": frozen,
            "solid_liquid_fraction": idiag.liquid_volume_fraction(pcm, u, areas),
            "enthalpy_removed": energy_out,
        }, solver, audit))
        final_field = (centers, u)

    summary = {
        "scenario": "cryosurgery_freezing", "alpha": alpha, "T_probe": T_probe, "T_far": T_far,
        "t_end": t_end, "latent_heat": latent_heat,
        "final_freezing_margin": rows[-1]["freezing_margin"],
        "final_frozen_fraction": rows[-1]["frozen_fraction"],
        "enthalpy_removed": -rows[-1]["enthalpy_change"],
    }
    summary.update(_final_diagnostics(rows))
    _write("cryosurgery_freezing", summary, rows,
           _fieldnames(["time", "freezing_margin", "min_temperature", "frozen_fraction",
                        "solid_liquid_fraction", "enthalpy_removed"]))
    _plot_freezing(rows, final_field, summary, pcm)
    _plot_diagnostics_dashboard("cryosurgery_freezing", rows, "Cryosurgery freezing")
    return summary, rows


def _plot_freezing(rows, final_field, summary, pcm):
    times = [r["time"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0), constrained_layout=True)
    axes[0].plot(times, [r["freezing_margin"] for r in rows], "o-", color="tab:blue")
    axes[0].set_xlabel("time"); axes[0].set_ylabel("freezing margin"); axes[0].set_title("Freezing-front margin")
    axes[1].plot(times, [r["frozen_fraction"] for r in rows], "o-", color="tab:cyan", label="frozen fraction")
    axes[1].plot(times, [-r["enthalpy_change"] for r in rows], "s--", color="tab:purple", label="enthalpy removed")
    axes[1].set_xlabel("time"); axes[1].set_title("Frozen fraction & energy removed"); axes[1].legend(fontsize=8)
    centers, u = final_field
    tcf = axes[2].tripcolor(centers[:, 0], centers[:, 1], u, shading="gouraud", cmap="coolwarm")
    axes[2].tricontour(centers[:, 0], centers[:, 1], u, levels=[pcm.solidus_temperature],
                       colors="k", linewidths=1.2)
    axes[2].set_aspect("equal"); axes[2].set_title("Final field (black = freezing front)")
    fig.colorbar(tcf, ax=axes[2], shrink=0.8)
    fig.suptitle("Direction C application: cryosurgery freezing front")
    fig.savefig(OUT / "cryosurgery_freezing.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Scenario 3: additive-manufacturing-like moving scan melt pool
# --------------------------------------------------------------------------- #
def moving_scan_melt_pool(nx=36, ny=18, bbox=(0.0, 1.6, 0.0, 0.8), alpha=0.055, tau=0.025,
                          source_power=18.0, source_sigma=0.10, scan_y=0.4,
                          scan_start=0.2, scan_end=1.35, T0=-0.45, t_end=0.45,
                          nt=180, n_snapshots=5, latent_heat=3.0, half_width=0.28):
    pcm = ApparentHeatCapacityModel(
        solidus_temperature=-half_width, liquidus_temperature=half_width,
        latent_heat=latent_heat, specific_heat=1.0,
    )
    vertices, polygons, centers, areas = _mesh(nx, ny, bbox)

    def ambient_boundary(x, y, t):
        return T0 * np.ones_like(np.asarray(x, dtype=float))

    def moving_source(x, y, t):
        x_laser = scan_start + (scan_end - scan_start) * min(max(float(t) / t_end, 0.0), 1.0)
        r2 = (np.asarray(x) - x_laser) ** 2 + (np.asarray(y) - scan_y) ** 2
        return source_power * np.exp(-0.5 * r2 / (source_sigma**2))

    u0 = np.full(centers.shape[0], float(T0))
    H0 = idiag.enthalpy_budget(pcm, u0, areas)["total"]
    opts = {"max_iters": 180, "tol": 1e-8, "relaxation": 0.45, "anderson_depth": 6,
            "raise_on_nonconvergence": False}
    rows = []
    final_field = None
    snaps = np.linspace(t_end / n_snapshots, t_end, n_snapshots)
    for t_k in snaps:
        steps = max(2, int(round(nt * t_k / t_end)))
        solver = HyperbolicStefanSolver(
            vertices, polygons, alpha, t_k / steps, tau, pcm,
            bc_type="dirichlet", bc_func=ambient_boundary, source_func=moving_source,
            phase_change_options=opts,
        )
        _, u = solver.solve(u0, 0.0, float(t_k), du0=np.zeros_like(u0))
        budget = idiag.enthalpy_budget(pcm, u, areas)
        source_energy = _source_energy(centers, areas, moving_source, t_k, steps)
        audit = idiag.enthalpy_audit(pcm, u0, u, areas, energy_in=source_energy)
        liquid = pcm.liquid_fraction(u) > 0.05
        if np.any(liquid):
            xs = centers[liquid, 0]
            ys = centers[liquid, 1]
            pool_length = float(np.max(xs) - np.min(xs))
            pool_width = float(np.max(ys) - np.min(ys))
        else:
            pool_length = 0.0
            pool_width = 0.0
        rows.append(_attach_audit_and_report({
            "time": float(t_k),
            "laser_x": scan_start + (scan_end - scan_start) * float(t_k) / t_end,
            "peak_temperature": float(np.max(u)),
            "liquid_fraction": idiag.liquid_volume_fraction(pcm, u, areas),
            "melt_pool_length": pool_length,
            "melt_pool_width": pool_width,
            "mushy_thickness": idiag.mushy_zone_thickness(centers, u, pcm, axis="x", coord=scan_y),
            "enthalpy_rise": budget["total"] - H0,
            "source_energy": source_energy,
        }, solver, audit))
        final_field = (centers, u)
    summary = {
        "scenario": "moving_scan_melt_pool", "alpha": alpha, "tau": tau,
        "source_power": source_power, "source_sigma": source_sigma, "t_end": t_end,
        "final_peak_temperature": rows[-1]["peak_temperature"],
        "final_liquid_fraction": rows[-1]["liquid_fraction"],
        "final_melt_pool_length": rows[-1]["melt_pool_length"],
        "final_melt_pool_width": rows[-1]["melt_pool_width"],
        "final_source_energy": rows[-1]["source_energy"],
    }
    summary.update(_final_diagnostics(rows))
    _write("moving_scan_melt_pool", summary, rows,
           _fieldnames(["time", "laser_x", "peak_temperature", "liquid_fraction",
                        "melt_pool_length", "melt_pool_width", "mushy_thickness",
                        "enthalpy_rise", "source_energy"]))
    _plot_generic_application(
        "moving_scan_melt_pool", rows, final_field,
        "Direction C application: moving scan melt pool",
        [
            ("laser_x", "laser x", "tab:gray", "o-"),
            ("melt_pool_length", "melt-pool length", "tab:red", "s-"),
            ("peak_temperature", "peak temperature", "tab:orange", "o-"),
            ("liquid_fraction", "liquid fraction", "tab:blue", "s--"),
            ("source_energy", "source energy", "black", "^-"),
        ],
    )
    _plot_diagnostics_dashboard("moving_scan_melt_pool", rows, "Moving scan melt pool")
    return summary, rows


# --------------------------------------------------------------------------- #
# Scenario 4: dual-pulse remelting from separated boundary heat pulses
# --------------------------------------------------------------------------- #
def dual_pulse_remelting(nx=40, ny=10, bbox=(0.0, 1.4, 0.0, 0.28), alpha=0.06, tau=0.035,
                         q0=2.6, pulses=((0.0, 0.14), (0.28, 0.40)), T0=-0.58,
                         t_end=0.55, nt=220, n_snapshots=7, latent_heat=3.2,
                         half_width=0.34):
    pcm = ApparentHeatCapacityModel(
        solidus_temperature=-half_width, liquidus_temperature=half_width,
        latent_heat=latent_heat, specific_heat=1.0,
    )
    vertices, polygons, centers, areas = _mesh(nx, ny, bbox)
    x_left, y_mid = bbox[0], 0.5 * (bbox[2] + bbox[3])
    left_len = bbox[3] - bbox[2]

    def pulse_flux(x, y, t, nx_, ny_):
        active = any(start <= float(t) <= stop for start, stop in pulses)
        return np.where(np.isclose(x, x_left), q0 if active else 0.0, 0.0)

    def injected_energy(t):
        active_time = sum(max(0.0, min(float(t), stop) - start) for start, stop in pulses)
        return q0 * left_len * active_time

    u0 = np.full(centers.shape[0], float(T0))
    H0 = idiag.enthalpy_budget(pcm, u0, areas)["total"]
    opts = {"max_iters": 200, "tol": 1e-8, "relaxation": 0.4, "anderson_depth": 6,
            "raise_on_nonconvergence": False}
    rows = []
    final_field = None
    snaps = np.linspace(t_end / n_snapshots, t_end, n_snapshots)
    for t_k in snaps:
        steps = max(2, int(round(nt * t_k / t_end)))
        solver = HyperbolicStefanSolver(
            vertices, polygons, alpha, t_k / steps, tau, pcm,
            bc_type="flux", bc_func=pulse_flux, phase_change_options=opts,
        )
        _, u = solver.solve(u0, 0.0, float(t_k), du0=np.zeros_like(u0))
        budget = idiag.enthalpy_budget(pcm, u, areas)
        injected = injected_energy(t_k)
        audit = idiag.enthalpy_audit(pcm, u0, u, areas, energy_in=injected)
        front = idiag.interface_position(centers, u, pcm, axis="x", coord=y_mid, pick="last")
        rows.append(_attach_audit_and_report({
            "time": float(t_k),
            "front_position": float("nan") if front is None else float(front),
            "peak_temperature": float(np.max(u)),
            "liquid_fraction": idiag.liquid_volume_fraction(pcm, u, areas),
            "enthalpy_rise": budget["total"] - H0,
            "injected_energy": injected,
        }, solver, audit))
        final_field = (centers, u)
    summary = {
        "scenario": "dual_pulse_remelting", "alpha": alpha, "tau": tau,
        "q0": q0, "pulses": [[float(a), float(b)] for a, b in pulses], "t_end": t_end,
        "peak_liquid_fraction": float(max(r["liquid_fraction"] for r in rows)),
        "final_liquid_fraction": rows[-1]["liquid_fraction"],
        "final_front_position": rows[-1]["front_position"],
        "final_injected_energy": rows[-1]["injected_energy"],
    }
    summary.update(_final_diagnostics(rows))
    _write("dual_pulse_remelting", summary, rows,
           _fieldnames(["time", "front_position", "peak_temperature", "liquid_fraction",
                        "enthalpy_rise", "injected_energy"]))
    _plot_generic_application(
        "dual_pulse_remelting", rows, final_field,
        "Direction C application: dual-pulse remelting",
        [
            ("front_position", "melt front", "tab:red", "o-"),
            ("liquid_fraction", "liquid fraction", "tab:blue", "s--"),
            ("peak_temperature", "peak temperature", "tab:orange", "o-"),
            ("latent_enthalpy", "latent enthalpy", "tab:purple", "s-"),
            ("injected_energy", "injected energy", "black", "^-"),
        ],
    )
    _plot_diagnostics_dashboard("dual_pulse_remelting", rows, "Dual-pulse remelting")
    return summary, rows


# --------------------------------------------------------------------------- #
# Scenario 5: rapid solidification quench of a hot liquid slab
# --------------------------------------------------------------------------- #
def rapid_solidification_quench(nx=36, ny=18, bbox=(0.0, 1.2, 0.0, 0.6), alpha=0.075,
                                tau=0.006, T_initial=0.68, T_wall=-0.65, t_end=0.36,
                                nt=120, n_snapshots=6, latent_heat=3.6,
                                half_width=0.18):
    pcm = ApparentHeatCapacityModel(
        solidus_temperature=-half_width, liquidus_temperature=half_width,
        latent_heat=latent_heat, specific_heat=1.0,
    )
    vertices, polygons, centers, areas = _mesh(nx, ny, bbox)

    def cold_wall(x, y, t):
        return T_wall * np.ones_like(np.asarray(x, dtype=float))

    u0 = np.full(centers.shape[0], float(T_initial))
    H0 = idiag.enthalpy_budget(pcm, u0, areas)["total"]
    opts = {"max_iters": 200, "tol": 1e-8, "relaxation": 0.45, "anderson_depth": 6,
            "raise_on_nonconvergence": False}
    rows = []
    final_field = None
    snaps = np.linspace(t_end / n_snapshots, t_end, n_snapshots)
    for t_k in snaps:
        steps = max(2, int(round(nt * t_k / t_end)))
        solver = HyperbolicStefanSolver(
            vertices, polygons, alpha, t_k / steps, tau, pcm,
            bc_type="dirichlet", bc_func=cold_wall, phase_change_options=opts,
        )
        _, u = solver.solve(u0, 0.0, float(t_k), du0=np.zeros_like(u0))
        budget = idiag.enthalpy_budget(pcm, u, areas)
        energy_out = max(H0 - budget["total"], 0.0)
        audit = idiag.enthalpy_audit(pcm, u0, u, areas, energy_out=energy_out)
        phases = idiag.phase_fractions(pcm, u, areas)
        rows.append(_attach_audit_and_report({
            "time": float(t_k),
            "peak_temperature": float(np.max(u)),
            "min_temperature": float(np.min(u)),
            "liquid_fraction": idiag.liquid_volume_fraction(pcm, u, areas),
            "solid_fraction": phases["solid"],
            "mushy_fraction": phases["mushy"],
            "enthalpy_removed": energy_out,
        }, solver, audit))
        final_field = (centers, u)
    summary = {
        "scenario": "rapid_solidification_quench", "alpha": alpha, "tau": tau,
        "T_initial": T_initial, "T_wall": T_wall, "t_end": t_end,
        "initial_liquid_fraction": idiag.liquid_volume_fraction(pcm, u0, areas),
        "final_liquid_fraction": rows[-1]["liquid_fraction"],
        "final_solid_fraction": rows[-1]["solid_fraction"],
        "enthalpy_removed": rows[-1]["enthalpy_removed"],
    }
    summary.update(_final_diagnostics(rows))
    _write("rapid_solidification_quench", summary, rows,
           _fieldnames(["time", "peak_temperature", "min_temperature", "liquid_fraction",
                        "solid_fraction", "mushy_fraction", "enthalpy_removed"]))
    _plot_generic_application(
        "rapid_solidification_quench", rows, final_field,
        "Direction C application: rapid solidification quench",
        [
            ("liquid_fraction", "liquid fraction", "tab:blue", "o-"),
            ("solid_fraction", "solid fraction", "tab:cyan", "s-"),
            ("peak_temperature", "peak temperature", "tab:orange", "o-"),
            ("enthalpy_removed", "enthalpy removed", "tab:purple", "s--"),
        ],
        cmap="coolwarm",
    )
    _plot_diagnostics_dashboard("rapid_solidification_quench", rows, "Rapid solidification quench")
    return summary, rows


# --------------------------------------------------------------------------- #
# Scenario 6: buried hot inclusion relaxation in a cold matrix
# --------------------------------------------------------------------------- #
def buried_hot_inclusion_relaxation(nx=42, ny=42, bbox=(0.0, 1.0, 0.0, 1.0), alpha=0.065,
                                    tau=0.012, inclusion_center=(0.5, 0.5),
                                    inclusion_radius=0.18, T_matrix=-0.55,
                                    T_inclusion=0.82, t_end=0.42, nt=150,
                                    n_snapshots=6, latent_heat=3.3, half_width=0.22):
    pcm = ApparentHeatCapacityModel(
        solidus_temperature=-half_width, liquidus_temperature=half_width,
        latent_heat=latent_heat, specific_heat=1.0,
    )
    vertices, polygons, centers, areas = _mesh(nx, ny, bbox)

    def cold_boundary(x, y, t):
        return T_matrix * np.ones_like(np.asarray(x, dtype=float))

    r = np.hypot(centers[:, 0] - inclusion_center[0], centers[:, 1] - inclusion_center[1])
    transition = 0.03
    inclusion_profile = 0.5 * (1.0 - np.tanh((r - inclusion_radius) / transition))
    u0 = T_matrix + (T_inclusion - T_matrix) * inclusion_profile
    H0 = idiag.enthalpy_budget(pcm, u0, areas)["total"]
    opts = {"max_iters": 200, "tol": 1e-8, "relaxation": 0.45, "anderson_depth": 6,
            "raise_on_nonconvergence": False}
    rows = []
    final_field = None
    snaps = np.linspace(t_end / n_snapshots, t_end, n_snapshots)
    for t_k in snaps:
        steps = max(2, int(round(nt * t_k / t_end)))
        solver = HyperbolicStefanSolver(
            vertices, polygons, alpha, t_k / steps, tau, pcm,
            bc_type="dirichlet", bc_func=cold_boundary, phase_change_options=opts,
        )
        _, u = solver.solve(u0, 0.0, float(t_k), du0=np.zeros_like(u0))
        budget = idiag.enthalpy_budget(pcm, u, areas)
        energy_out = max(H0 - budget["total"], 0.0)
        audit = idiag.enthalpy_audit(pcm, u0, u, areas, energy_out=energy_out)
        liquid = pcm.liquid_fraction(u) > 0.5
        melted_area = float(np.sum(areas[liquid]))
        rows.append(_attach_audit_and_report({
            "time": float(t_k),
            "peak_temperature": float(np.max(u)),
            "liquid_fraction": idiag.liquid_volume_fraction(pcm, u, areas),
            "melted_area": melted_area,
            "enthalpy_removed": energy_out,
        }, solver, audit))
        final_field = (centers, u)
    summary = {
        "scenario": "buried_hot_inclusion_relaxation", "alpha": alpha, "tau": tau,
        "inclusion_radius": inclusion_radius, "t_end": t_end,
        "initial_liquid_fraction": idiag.liquid_volume_fraction(pcm, u0, areas),
        "final_liquid_fraction": rows[-1]["liquid_fraction"],
        "final_melted_area": rows[-1]["melted_area"],
        "enthalpy_removed": rows[-1]["enthalpy_removed"],
    }
    summary.update(_final_diagnostics(rows))
    _write("buried_hot_inclusion_relaxation", summary, rows,
           _fieldnames(["time", "peak_temperature", "liquid_fraction", "melted_area",
                        "enthalpy_removed"]))
    _plot_generic_application(
        "buried_hot_inclusion_relaxation", rows, final_field,
        "Direction C application: buried hot inclusion relaxation",
        [
            ("melted_area", "melted area", "tab:red", "o-"),
            ("liquid_fraction", "liquid fraction", "tab:blue", "s--"),
            ("peak_temperature", "peak temperature", "tab:orange", "o-"),
            ("enthalpy_removed", "enthalpy removed", "tab:purple", "s-"),
        ],
        cmap="coolwarm",
    )
    _plot_diagnostics_dashboard(
        "buried_hot_inclusion_relaxation", rows, "Buried hot inclusion relaxation")
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description="Direction C application studies.")
    parser.add_argument("--quick", action="store_true", help="smaller/faster runs")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.quick:
        laser, _ = pulsed_laser_melting(nx=32, ny=8, nt=48, n_snapshots=4, t_end=0.5)
        cryo, _ = cryosurgery_freezing(nx=24, ny=24, nt=40, n_snapshots=4, t_end=0.4)
        scan, _ = moving_scan_melt_pool(nx=24, ny=12, nt=48, n_snapshots=3, t_end=0.32)
        remelt, _ = dual_pulse_remelting(nx=24, ny=6, nt=54, n_snapshots=4, t_end=0.45)
        quench, _ = rapid_solidification_quench(nx=22, ny=12, nt=42, n_snapshots=3, t_end=0.28)
        inclusion, _ = buried_hot_inclusion_relaxation(nx=24, ny=24, nt=48, n_snapshots=3, t_end=0.3)
    else:
        laser, _ = pulsed_laser_melting()
        cryo, _ = cryosurgery_freezing()
        scan, _ = moving_scan_melt_pool()
        remelt, _ = dual_pulse_remelting()
        quench, _ = rapid_solidification_quench()
        inclusion, _ = buried_hot_inclusion_relaxation()

    print("=== Direction C application studies ===")
    print(f"[pulsed-laser melting] front={laser['final_front_position']:.3f} "
          f"liquid_frac={laser['final_liquid_fraction']:.3f} "
          f"energy closure residual={laser['final_energy_closure_residual']:.3e} "
          f"(injected {laser['final_injected_energy']:.3f})")
    print(f"[cryosurgery freezing] margin={cryo['final_freezing_margin']:.3f} "
          f"frozen_frac={cryo['final_frozen_fraction']:.3f} "
          f"enthalpy_removed={cryo['enthalpy_removed']:.3f}")
    print(f"[moving scan melt pool] length={scan['final_melt_pool_length']:.3f} "
          f"width={scan['final_melt_pool_width']:.3f} liquid_frac={scan['final_liquid_fraction']:.3f}")
    print(f"[dual-pulse remelting] peak_liquid_frac={remelt['peak_liquid_fraction']:.3f} "
          f"final_front={remelt['final_front_position']:.3f}")
    print(f"[rapid solidification quench] liquid {quench['initial_liquid_fraction']:.3f}->"
          f"{quench['final_liquid_fraction']:.3f} enthalpy_removed={quench['enthalpy_removed']:.3f}")
    print(f"[buried hot inclusion] liquid {inclusion['initial_liquid_fraction']:.3f}->"
          f"{inclusion['final_liquid_fraction']:.3f} melted_area={inclusion['final_melted_area']:.3f}")
    print(f"wrote artifacts to {OUT}")


if __name__ == "__main__":
    main()
