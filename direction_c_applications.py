#!/usr/bin/env python3
"""Direction C application-study runners (computing-capability step 5).

Reproducible 2D non-Fourier phase-change scenarios beyond manufactured
solutions, each reporting front position, peak temperature, liquid fraction,
injected/extracted energy, and the sensible/latent enthalpy budget via the
sharp-interface diagnostics (step 4).  Two scenarios:

* ``pulsed_laser_melting``   -- a boundary heat-flux pulse melts a cold solid
  slab (Cattaneo/hyperbolic Stefan; the classic finite-speed laser-heating
  regime), with an energy-closure audit (injected boundary energy vs enthalpy
  rise).
* ``cryosurgery_freezing``   -- a cold cryoprobe boundary freezes a warm domain,
  tracking the freezing-front margin and frozen volume fraction.

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


def _mesh(nx, ny, bbox):
    vertices, polygons, centers = generate_square_polygonal_mesh(nx=nx, ny=ny, bbox=bbox)
    areas = np.array([polygon_area_and_centroid(vertices[poly])[0] for poly in polygons])
    return vertices, polygons, centers, areas


def _write(name, summary, rows, fieldnames):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (OUT / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    failed_steps_total = 0
    for t_k in snaps:
        steps = max(2, int(round(nt * t_k / t_end)))
        solver = HyperbolicStefanSolver(
            vertices, polygons, alpha, t_k / steps, tau, pcm,
            bc_type="flux", bc_func=pulse_flux, phase_change_options=opts,
        )
        _, u = solver.solve(u0, 0.0, float(t_k), du0=np.zeros_like(u0))
        failed_steps_total = max(failed_steps_total, solver.solve_report["failed_steps"])
        budget = idiag.enthalpy_budget(pcm, u, areas)
        front = idiag.interface_position(centers, u, pcm, axis="x", coord=y_mid, pick="last")
        injected = q0 * left_len * min(float(t_k), t_pulse)
        rows.append({
            "time": float(t_k),
            "front_position": float("nan") if front is None else float(front),
            "peak_temperature": float(np.max(u)),
            "liquid_fraction": idiag.liquid_volume_fraction(pcm, u, areas),
            "mushy_thickness": idiag.mushy_zone_thickness(centers, u, pcm, axis="x", coord=y_mid),
            "sensible_enthalpy": budget["sensible"], "latent_enthalpy": budget["latent"],
            "enthalpy_rise": budget["total"] - H0, "injected_energy": injected,
            "energy_closure_residual": (budget["total"] - H0) - injected,
        })
        final_field = (centers, u)

    summary = {
        "scenario": "pulsed_laser_melting", "alpha": alpha, "tau": tau, "q0": q0,
        "t_pulse": t_pulse, "t_end": t_end, "latent_heat": latent_heat,
        "wave_speed": float(np.sqrt(alpha / tau)),
        "final_front_position": rows[-1]["front_position"],
        "final_liquid_fraction": rows[-1]["liquid_fraction"],
        "final_energy_closure_residual": rows[-1]["energy_closure_residual"],
        "final_injected_energy": rows[-1]["injected_energy"],
        "nonconverged_steps": int(failed_steps_total),
    }
    _write("pulsed_laser_melting", summary, rows,
           ["time", "front_position", "peak_temperature", "liquid_fraction", "mushy_thickness",
            "sensible_enthalpy", "latent_enthalpy", "enthalpy_rise", "injected_energy",
            "energy_closure_residual"])
    _plot_melting(rows, final_field, summary)
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
        # Freezing front = solidus isotherm distance from the cold probe.
        front = idiag.interface_position(centers, u, pcm, axis="x", coord=y_mid,
                                         level=pcm.solidus_temperature, pick="first")
        frozen = float(np.sum(areas[u < pcm.solidus_temperature]) / np.sum(areas))
        rows.append({
            "time": float(t_k),
            "freezing_margin": float("nan") if front is None else float(front - x_left),
            "min_temperature": float(np.min(u)),
            "frozen_fraction": frozen,
            "solid_liquid_fraction": idiag.liquid_volume_fraction(pcm, u, areas),
            "latent_enthalpy": budget["latent"], "enthalpy_change": budget["total"] - H0,
        })
        final_field = (centers, u)

    summary = {
        "scenario": "cryosurgery_freezing", "alpha": alpha, "T_probe": T_probe, "T_far": T_far,
        "t_end": t_end, "latent_heat": latent_heat,
        "final_freezing_margin": rows[-1]["freezing_margin"],
        "final_frozen_fraction": rows[-1]["frozen_fraction"],
        "enthalpy_removed": -rows[-1]["enthalpy_change"],
    }
    _write("cryosurgery_freezing", summary, rows,
           ["time", "freezing_margin", "min_temperature", "frozen_fraction",
            "solid_liquid_fraction", "latent_enthalpy", "enthalpy_change"])
    _plot_freezing(rows, final_field, summary, pcm)
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


def main():
    parser = argparse.ArgumentParser(description="Direction C application studies.")
    parser.add_argument("--quick", action="store_true", help="smaller/faster runs")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.quick:
        laser, _ = pulsed_laser_melting(nx=32, ny=8, nt=48, n_snapshots=4, t_end=0.5)
        cryo, _ = cryosurgery_freezing(nx=24, ny=24, nt=40, n_snapshots=4, t_end=0.4)
    else:
        laser, _ = pulsed_laser_melting()
        cryo, _ = cryosurgery_freezing()

    print("=== Direction C application studies ===")
    print(f"[pulsed-laser melting] front={laser['final_front_position']:.3f} "
          f"liquid_frac={laser['final_liquid_fraction']:.3f} "
          f"energy closure residual={laser['final_energy_closure_residual']:.3e} "
          f"(injected {laser['final_injected_energy']:.3f})")
    print(f"[cryosurgery freezing] margin={cryo['final_freezing_margin']:.3f} "
          f"frozen_frac={cryo['final_frozen_fraction']:.3f} "
          f"enthalpy_removed={cryo['enthalpy_removed']:.3f}")
    print(f"wrote artifacts to {OUT}")


if __name__ == "__main__":
    main()
