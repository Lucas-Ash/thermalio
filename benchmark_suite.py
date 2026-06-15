#!/usr/bin/env python3
"""Consolidated V&V benchmark runner for Thermalio (direction A capstone).

Ties together the verification tooling into one reproducible report:

  * observed order of accuracy via Richardson analysis (heat_solver.verification),
  * SymPy-auto-derived manufactured sources (heat_solver.mms),
  * cross-scheme + cross-mesh N-version agreement (heat_solver.nversion),
  * cross-implementation validation against an independent finite-difference
    reference solver (heat_solver.reference_fd).

For a panel of manufactured cases it runs a 3-level convergence study on the
square polygonal mesh, estimates the observed order, checks N-version agreement,
and (for scalar-diffusivity Dirichlet cases) checks agreement with the FD
reference.  Results are written to ``test_plots/benchmark/`` as JSON + CSV with a
PASS/FAIL summary.

Usage:
    MPLCONFIGDIR=/tmp/matplotlib python benchmark_suite.py
    python benchmark_suite.py --quick      # coarser, faster
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heat_solver.drivers import run_square_polygonal_test
from heat_solver.geometry import polygon_area_and_centroid
from heat_solver.meshes import generate_square_polygonal_mesh
from heat_solver.mms import manufactured_case
from heat_solver.nversion import run_nversion
from heat_solver.polygonal import PolygonalHeatSolver
from heat_solver.reference_fd import relative_l2, solve_fd_reference
from heat_solver.verification import triplet_report

OUTPUT_DIR = ROOT / "test_plots" / "benchmark"


def _rel_l2(u, u_exact, areas):
    return float(np.sqrt(np.sum(areas * (u - u_exact) ** 2)) / (np.sqrt(np.sum(areas * u_exact**2)) + 1e-300))


def _solve_square_mms(case, alpha, dt, t_end, n, bbox, time_scheme="crank_nicolson"):
    v, p, c = generate_square_polygonal_mesh(nx=n, ny=n, bbox=bbox)
    areas = np.array([polygon_area_and_centroid(v[poly])[0] for poly in p])
    solver = PolygonalHeatSolver(
        v, p, alpha, dt, bc_type="dirichlet", time_scheme=time_scheme,
        bc_func=case["boundary"], source_func=case["source"],
    )
    u0 = case["solution"](c[:, 0], c[:, 1], 0.0)
    _, u = solver.solve(u0, 0.0, t_end)
    ue = case["solution"](c[:, 0], c[:, 1], t_end)
    return _rel_l2(u, ue, areas)


def _convergence_string_case(case_name, alpha, dt, t_init, t_end, bbox, resolutions,
                             time_scheme="crank_nicolson"):
    # Crank-Nicolson keeps the temporal error second order so the convergence
    # study reflects the spatial order rather than a backward-Euler time floor.
    hs, errs = [], []
    for n in resolutions:
        *_, results = run_square_polygonal_test(
            case=case_name, alpha=alpha, dt=dt, t_init=t_init, t_end=t_end,
            nx=n, ny=n, bbox=bbox, time_scheme=time_scheme,
        )
        hs.append((bbox[1] - bbox[0]) / n)
        errs.append(float(results["L2_rel"]))
    return hs, errs


def _convergence_mms_case(case, alpha, dt, t_end, bbox, resolutions):
    hs, errs = [], []
    for n in resolutions:
        hs.append((bbox[1] - bbox[0]) / n)
        errs.append(_solve_square_mms(case, alpha, dt, t_end, n, bbox))
    return hs, errs


def run_benchmark(quick=False):
    resolutions = (16, 32, 48) if quick else (16, 32, 64)
    report = {"resolutions": list(resolutions), "entries": []}

    bbox_sym = (-1.0, 1.0, -1.0, 1.0)

    # --- Panel entry 1: source-driven diffusion (scalar alpha, Dirichlet). ---
    hs, errs = _convergence_string_case(
        "source_driven_sine", alpha=0.1, dt=5e-4, t_init=0.0, t_end=0.02,
        bbox=bbox_sym, resolutions=resolutions,
    )
    rep = triplet_report(hs, errs)
    nv = run_nversion("source_driven_sine", alpha=0.1, dt=1e-4, t_end=0.02,
                      bbox=bbox_sym, n=resolutions[1], tol=2e-2, cross_tol=5e-2)
    # Cross-code: at the finest resolution, the independent backward-Euler FD
    # reference and the FV solver (also backward Euler, same dt) must achieve the
    # same accuracy against the exact solution.
    n_fine = resolutions[-1]
    mms_sds = manufactured_case("exp(-t)*sin(pi*x)*sin(pi*y)", alpha=0.1, model="diffusion")
    X, Y, u_fd = solve_fd_reference(bbox_sym, n_fine, 0.1, 1e-4, 0.02,
                                    mms_sds["solution"], mms_sds["source"], mms_sds["boundary"])
    fd_err = relative_l2(u_fd, mms_sds["solution"](X, Y, 0.02))
    fv_be_err = _solve_square_mms(mms_sds, 0.1, 1e-4, 0.02, n_fine, bbox_sym,
                                  time_scheme="backward_euler")
    report["entries"].append({
        "name": "source_driven_sine (scalar diffusion)",
        "observed_order": rep["p_obs"], "order_3grid": rep["p_obs_3grid"],
        "extrap_error": rep["extrap_error"], "asymptotic": rep["asymptotic"],
        "L2_rel": errs, "nversion_within_mesh": nv["within_mesh_max_spread"],
        "nversion_cross_mesh": nv["cross_mesh_max_spread"],
        "fd_reference_error": fd_err, "fv_be_error": fv_be_err,
        "checks": {
            "order_near_2": rep["p_obs"] is not None and rep["p_obs"] > 1.6,
            "nversion_agree": nv["all_agree"] and nv["cross_mesh_ok"],
            "fd_cross_code": abs(fd_err - fv_be_err) < 0.1 * max(fd_err, fv_be_err),
        },
    })

    # --- Panel entry 2: pure diffusion eigenmode (no source). ---
    hs, errs = _convergence_string_case(
        "sine_mode", alpha=0.1, dt=5e-4, t_init=0.0, t_end=0.02,
        bbox=bbox_sym, resolutions=resolutions,
    )
    rep = triplet_report(hs, errs)
    nv = run_nversion("sine_mode", alpha=0.1, dt=1e-4, t_end=0.02,
                      bbox=bbox_sym, n=resolutions[1], tol=2e-2, cross_tol=5e-2)
    report["entries"].append({
        "name": "sine_mode (pure diffusion)",
        "observed_order": rep["p_obs"], "order_3grid": rep["p_obs_3grid"],
        "extrap_error": rep["extrap_error"], "asymptotic": rep["asymptotic"],
        "L2_rel": errs, "nversion_within_mesh": nv["within_mesh_max_spread"],
        "nversion_cross_mesh": nv["cross_mesh_max_spread"],
        "checks": {
            "order_near_2": rep["p_obs"] is not None and rep["p_obs"] > 1.6,
            "nversion_agree": nv["all_agree"] and nv["cross_mesh_ok"],
        },
    })

    # --- Panel entry 3: anisotropic (tensor alpha) via auto-derived MMS source. ---
    bbox01 = (0.0, 1.0, 0.0, 1.0)
    alpha_tensor = [[0.2, 0.05], [0.05, 0.1]]
    mms_aniso = manufactured_case(
        "exp(-t)*sin(pi*x)*sin(pi*y)", alpha=alpha_tensor, model="diffusion",
    )
    hs, errs = _convergence_mms_case(mms_aniso, alpha_tensor, dt=1e-4, t_end=0.02,
                                     bbox=bbox01, resolutions=resolutions)
    rep = triplet_report(hs, errs)
    report["entries"].append({
        "name": "anisotropic diffusion (MMS tensor alpha)",
        "observed_order": rep["p_obs"], "order_3grid": rep["p_obs_3grid"],
        "extrap_error": rep["extrap_error"], "asymptotic": rep["asymptotic"],
        "L2_rel": errs,
        "checks": {"order_near_2": rep["p_obs"] is not None and rep["p_obs"] > 1.6},
    })

    # Overall pass/fail.
    all_checks = [v for e in report["entries"] for v in e["checks"].values()]
    report["all_pass"] = bool(all(all_checks))
    return report


def _write_report(report):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "benchmark_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = OUTPUT_DIR / "benchmark_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = _csv.writer(handle)
        writer.writerow(["entry", "observed_order", "order_3grid", "asymptotic", "finest_L2_rel", "checks_pass"])
        for e in report["entries"]:
            writer.writerow([
                e["name"],
                "" if e["observed_order"] is None else f"{e['observed_order']:.4f}",
                "" if e.get("order_3grid") is None else f"{e['order_3grid']:.4f}",
                e["asymptotic"],
                f"{e['L2_rel'][-1]:.3e}",
                all(e["checks"].values()),
            ])
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Thermalio V&V benchmark suite.")
    parser.add_argument("--quick", action="store_true", help="coarser resolutions for speed")
    args = parser.parse_args()

    report = run_benchmark(quick=args.quick)
    csv_path = _write_report(report)

    print("=== Thermalio V&V benchmark ===")
    print(f"resolutions: {report['resolutions']}")
    for e in report["entries"]:
        p = e["observed_order"]
        pstr = "n/a" if p is None else f"{p:.3f}"
        checks = ", ".join(f"{k}={v}" for k, v in e["checks"].items())
        print(f"  {e['name']}")
        print(f"      observed order = {pstr}, finest L2_rel = {e['L2_rel'][-1]:.3e} | {checks}")
    print(f"OVERALL: {'PASS' if report['all_pass'] else 'FAIL'}")
    print(f"wrote {OUTPUT_DIR / 'benchmark_report.json'} and {csv_path}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
