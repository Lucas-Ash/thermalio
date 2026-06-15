#!/usr/bin/env python3
"""Direction C parameter-sweep study runners (expansion targets 1-3).

Turns the Direction C expansion diagnostics into reproducible, solver-backed
parameter sweeps that write JSON + CSV + PNG artifacts:

1. ``relaxation_sweep``      -- relaxation-aware hyperbolic Stefan: sweep the
   relaxation time ``tau`` and time resolution, recording manufactured error,
   the finite thermal-wave speed ``sqrt(alpha/tau)``, and nonlinear-solver
   stability/convergence metadata.
2. ``fractional_memory_sweep`` -- fractional-memory Stefan: a beta-calibration
   study (observed L1 order ~ 2-beta) and a memory-compression study (error vs
   short-memory window, using the new ``memory_window`` option).
3. ``mushy_stiffness_map``   -- latent-heat / mushy-zone stiffness: a 2D failure
   map over latent heat x transition half-width recording nonlinear convergence
   and iteration counts.

Outputs go to ``test_plots/direction_C_nonfourier_phase_change/sweeps/``.

Usage:
    MPLCONFIGDIR=/tmp/matplotlib python direction_c_studies.py
    python direction_c_studies.py --quick
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

from heat_solver.cases import (
    fractional_stefan_apparent_capacity_case,
    hyperbolic_stefan_apparent_capacity_case,
)
from heat_solver.geometry import polygon_area_and_centroid
from heat_solver.meshes import generate_square_polygonal_mesh
from heat_solver.transport import FractionalStefanSolver, HyperbolicStefanSolver

OUT = ROOT / "test_plots" / "direction_C_nonfourier_phase_change" / "sweeps"


def _square(n, bbox):
    vertices, polygons, centers = generate_square_polygonal_mesh(nx=n, ny=n, bbox=bbox)
    areas = np.array([polygon_area_and_centroid(vertices[poly])[0] for poly in polygons])
    return vertices, polygons, centers, areas


def _rel_l2(u, exact, areas):
    return float(np.sqrt(np.sum(areas * (u - exact) ** 2)) / max(np.sqrt(np.sum(areas * exact**2)), 1e-16))


def _write_artifacts(name, rows, fieldnames):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (OUT / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# Step 1: relaxation-aware hyperbolic Stefan sweep
# --------------------------------------------------------------------------- #
def relaxation_sweep(taus=(0.04, 0.07, 0.12, 0.2), nts=(48, 72), alpha=0.08, t_end=0.04, n=18):
    rows = []
    for tau in taus:
        case = hyperbolic_stefan_apparent_capacity_case(alpha=alpha, tau=float(tau))
        v, p, c, a = _square(n, case["bbox"])
        u0 = case["solution"](c[:, 0], c[:, 1], 0.0)
        du0 = case["initial_rate"](c[:, 0], c[:, 1])
        exact = case["solution"](c[:, 0], c[:, 1], t_end)
        for nt in nts:
            opts = {**case["phase_change_options"], "max_iters": 160, "relaxation": 0.7,
                    "raise_on_nonconvergence": False}
            solver = HyperbolicStefanSolver(
                v, p, case["alpha"], t_end / nt, case["relaxation_time"], case["phase_change_model"],
                bc_func=case["boundary"], source_func=case["source"], phase_change_options=opts,
            )
            _, u = solver.solve(u0, 0.0, t_end, du0=du0)
            rep = solver.solve_report
            rows.append({
                "tau": float(tau), "nt": int(nt),
                "wave_speed": float(np.sqrt(alpha / tau)),
                "rel_l2": _rel_l2(u, exact, a),
                "max_iters": rep["max_iterations"], "failed_steps": rep["failed_steps"],
                "converged": rep["converged"],
            })
    _write_artifacts("relaxation_sweep", rows,
                     ["tau", "nt", "wave_speed", "rel_l2", "max_iters", "failed_steps", "converged"])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0), constrained_layout=True)
    for nt in nts:
        sub = [r for r in rows if r["nt"] == nt]
        axes[0].loglog([r["tau"] for r in sub], [r["rel_l2"] for r in sub], "o-", label=f"{nt} steps")
    axes[0].set_xlabel(r"relaxation time $\tau$"); axes[0].set_ylabel("relative $L^2$ error")
    axes[0].set_title("Manufactured error vs relaxation time"); axes[0].legend(fontsize=8)
    taus_arr = np.array(sorted(set(r["tau"] for r in rows)))
    axes[1].plot(taus_arr, np.sqrt(alpha / taus_arr), "s-", color="tab:red")
    axes[1].set_xlabel(r"$\tau$"); axes[1].set_ylabel(r"wave speed $\sqrt{\alpha/\tau}$")
    axes[1].set_title("Finite thermal-wave speed")
    for nt in nts:
        sub = [r for r in rows if r["nt"] == nt]
        axes[2].plot([r["tau"] for r in sub], [r["max_iters"] for r in sub], "o-", label=f"{nt} steps")
    axes[2].set_xlabel(r"$\tau$"); axes[2].set_ylabel("max Picard iterations / step")
    axes[2].set_title("Nonlinear stability metadata"); axes[2].legend(fontsize=8)
    fig.suptitle("Direction C step 1: relaxation-aware hyperbolic Stefan sweep")
    fig.savefig(OUT / "relaxation_sweep.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return rows


# --------------------------------------------------------------------------- #
# Step 2: fractional-memory Stefan (beta calibration + memory compression)
# --------------------------------------------------------------------------- #
def fractional_memory_sweep(betas=(0.4, 0.6, 0.8), windows=(None, 32, 16, 8, 4, 2),
                            alpha=0.08, t_end=0.16, n=18):
    # Temporal self-convergence (against a fine-dt reference on the same mesh) so
    # the observed L1 order ~ 2-beta is not masked by the fixed spatial error.
    nt_ref = 160
    order_rows = []
    for beta in betas:
        case = fractional_stefan_apparent_capacity_case(alpha=alpha, beta=float(beta))
        v, p, c, a = _square(n, case["bbox"])
        u0 = case["solution"](c[:, 0], c[:, 1], 0.0)
        ref = FractionalStefanSolver(
            v, p, case["alpha"], t_end / nt_ref, case["beta"], case["phase_change_model"],
            bc_func=case["boundary"], source_func=case["source"],
            phase_change_options=case["phase_change_options"],
        )
        _, u_ref = ref.solve(u0, 0.0, t_end)
        errs = []
        for nt in (16, 32):
            solver = FractionalStefanSolver(
                v, p, case["alpha"], t_end / nt, case["beta"], case["phase_change_model"],
                bc_func=case["boundary"], source_func=case["source"],
                phase_change_options=case["phase_change_options"],
            )
            _, u = solver.solve(u0, 0.0, t_end)
            errs.append(_rel_l2(u, u_ref, a))  # temporal self-error
        order_rows.append({
            "beta": float(beta), "err_coarse": errs[0], "err_fine": errs[1],
            "observed_order": float(np.log2(errs[0] / errs[1])), "expected_order": 2.0 - float(beta),
        })

    mem_rows = []
    beta_fix, nt = 0.6, 40
    case = fractional_stefan_apparent_capacity_case(alpha=alpha, beta=beta_fix)
    v, p, c, a = _square(n, case["bbox"])
    u0 = case["solution"](c[:, 0], c[:, 1], 0.0)
    exact = case["solution"](c[:, 0], c[:, 1], t_end)
    for w in windows:
        solver = FractionalStefanSolver(
            v, p, case["alpha"], t_end / nt, case["beta"], case["phase_change_model"],
            bc_func=case["boundary"], source_func=case["source"],
            phase_change_options=case["phase_change_options"], memory_window=w,
        )
        _, u = solver.solve(u0, 0.0, t_end)
        mem_rows.append({"beta": beta_fix, "window": ("full" if w is None else int(w)),
                         "retained_lags": (nt if w is None else min(nt, int(w))),
                         "rel_l2": _rel_l2(u, exact, a)})

    _write_artifacts("fractional_beta_calibration", order_rows,
                     ["beta", "err_coarse", "err_fine", "observed_order", "expected_order"])
    _write_artifacts("fractional_memory_window", mem_rows, ["beta", "window", "retained_lags", "rel_l2"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    bx = np.arange(len(order_rows))
    axes[0].bar(bx - 0.2, [r["expected_order"] for r in order_rows], width=0.4, label=r"expected $2-\beta$", color="tab:gray")
    axes[0].bar(bx + 0.2, [r["observed_order"] for r in order_rows], width=0.4, label="observed", color="tab:blue")
    axes[0].set_xticks(bx, [fr"$\beta={r['beta']:.1f}$" for r in order_rows])
    axes[0].set_ylabel("temporal order"); axes[0].set_title("Fractional Stefan order calibration"); axes[0].legend(fontsize=8)
    lags = [r["retained_lags"] for r in mem_rows]
    axes[1].semilogx(lags, [r["rel_l2"] for r in mem_rows], "o-", color="tab:green")
    axes[1].set_xlabel("retained memory lags"); axes[1].set_ylabel("relative $L^2$ error")
    axes[1].set_title(fr"Memory compression ($\beta={beta_fix}$): accuracy vs window")
    fig.suptitle("Direction C step 2: fractional memory calibration & compression")
    fig.savefig(OUT / "fractional_memory_sweep.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return order_rows, mem_rows


# --------------------------------------------------------------------------- #
# Step 3: latent-heat / mushy-zone stiffness failure map
# --------------------------------------------------------------------------- #
def mushy_stiffness_map(latents=(4.0, 12.0, 24.0), half_widths=(0.3, 0.12, 0.05),
                        alpha=0.08, tau=0.05, t_end=0.03, n=14, nt=48, max_iters=25):
    rows = []
    for L in latents:
        for hw in half_widths:
            case = hyperbolic_stefan_apparent_capacity_case(
                alpha=alpha, tau=tau, latent_heat=float(L), transition_half_width=float(hw))
            v, p, c, a = _square(n, case["bbox"])
            u0 = case["solution"](c[:, 0], c[:, 1], 0.0)
            du0 = case["initial_rate"](c[:, 0], c[:, 1])
            # Strict (no Anderson, no under-relaxation) so the latent-heat /
            # mushy-zone stiffness shows up as iteration growth and failures.
            opts = {"max_iters": max_iters, "tol": 1e-9, "relaxation": 1.0,
                    "anderson_depth": 0, "raise_on_nonconvergence": False}
            solver = HyperbolicStefanSolver(
                v, p, case["alpha"], t_end / nt, case["relaxation_time"], case["phase_change_model"],
                bc_func=case["boundary"], source_func=case["source"], phase_change_options=opts,
            )
            _, u = solver.solve(u0, 0.0, t_end, du0=du0)
            rep = solver.solve_report
            pcm = case["phase_change_model"]
            analytic_cap = float(np.asarray(pcm.effective_heat_capacity(np.array([0.0])))[0])
            rows.append({
                "latent_heat": float(L), "transition_half_width": float(hw),
                # analytic spike height (mesh-independent stiffness) vs the value
                # the discrete solve actually sampled (a resolution indicator).
                "analytic_peak_capacity": analytic_cap,
                "sampled_peak_capacity": rep["max_capacity"],
                "resolved": rep["max_capacity"] > 1.5,
                "max_iters": rep["max_iterations"],
                "failed_steps": rep["failed_steps"], "converged": rep["converged"],
            })
    _write_artifacts("mushy_stiffness_map", rows,
                     ["latent_heat", "transition_half_width", "analytic_peak_capacity",
                      "sampled_peak_capacity", "resolved", "max_iters", "failed_steps", "converged"])

    # Analytic peak capacity (= specific + latent/(2*half_width)) is the true,
    # mesh-independent stiffness measure; annotate iterations, convergence and
    # whether the discrete mesh actually resolved the mushy zone.
    capacity = np.array([r["analytic_peak_capacity"] for r in rows]).reshape(len(latents), len(half_widths))
    fig, ax = plt.subplots(figsize=(7, 5.2))
    im = ax.imshow(np.log10(capacity), origin="lower", aspect="auto", cmap="magma")
    ax.set_xticks(range(len(half_widths)), [f"{hw:.2f}" for hw in half_widths])
    ax.set_yticks(range(len(latents)), [f"{L:.0f}" for L in latents])
    ax.set_xlabel("transition half-width (narrower -> stiffer)")
    ax.set_ylabel("latent heat")
    for i in range(len(latents)):
        for j in range(len(half_widths)):
            r = rows[i * len(half_widths) + j]
            status = "ok" if r["converged"] else "FAIL"
            res = "" if r["resolved"] else "\nunder-res"
            ax.text(j, i, f"cap={r['analytic_peak_capacity']:.0f}\n{r['max_iters']} it {status}{res}",
                    ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(im, ax=ax, label=r"$\log_{10}$ peak apparent capacity")
    ax.set_title("Direction C step 3: mushy-zone stiffness map\n"
                 "(peak capacity spike; strict Picard iterations & convergence)")
    fig.savefig(OUT / "mushy_stiffness_map.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Direction C parameter-sweep studies.")
    parser.add_argument("--quick", action="store_true", help="smaller sweeps for speed")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.quick:
        rel = relaxation_sweep(taus=(0.07, 0.15), nts=(48,), n=14)
        order_rows, mem_rows = fractional_memory_sweep(betas=(0.4, 0.8), windows=(None, 8, 2), n=14)
        mushy = mushy_stiffness_map(latents=(2.0, 12.0), half_widths=(0.3, 0.1), n=12, nt=32)
    else:
        rel = relaxation_sweep()
        order_rows, mem_rows = fractional_memory_sweep()
        mushy = mushy_stiffness_map()

    print("=== Direction C studies ===")
    print(f"[1] relaxation sweep: {len(rel)} runs; "
          f"error range {min(r['rel_l2'] for r in rel):.2e}-{max(r['rel_l2'] for r in rel):.2e}")
    print(f"[2] beta calibration: " + ", ".join(
        f"b={r['beta']:.1f} order={r['observed_order']:.2f}(exp {r['expected_order']:.2f})" for r in order_rows))
    print("    memory window error: " + ", ".join(
        f"{r['window']}:{r['rel_l2']:.2e}" for r in mem_rows))
    print(f"[3] mushy map: {sum(1 for r in mushy if r['converged'])}/{len(mushy)} converged; "
          f"max iters {max(r['max_iters'] for r in mushy)}")
    print(f"wrote artifacts to {OUT}")


if __name__ == "__main__":
    main()
