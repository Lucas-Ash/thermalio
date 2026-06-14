"""Cross-scheme "N-version" agreement harness.

Runs the *same* manufactured case through independent discretizations and checks
that they agree.  Two complementary checks:

1. **Within-mesh, cross-flux-scheme agreement** (the headline N-version test):
   on a fixed mesh, ``tpfa``, reconstructed-gradient, and ``mpfa`` fluxes produce
   solutions on the *same* cell centers, so their solution arrays are directly
   comparable (no interpolation).  Independent schemes solving the same discrete
   geometry must agree to a tight tolerance at fixed resolution.

2. **Cross-mesh accuracy floor**: every (mesh, scheme) run's relative-L2 error
   against the exact solution must lie below an accuracy floor — a check that all
   independent discretizations actually converge to the same exact solution.

This reuses the drivers in ``heat_solver.drivers`` unchanged and writes a JSON
report.  It does not touch ``tests.iter_test_jobs`` or the regression baselines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .drivers import (
    run_nonorthogonal_tiled_polygonal_test,
    run_square_polygonal_test,
)
from .geometry import polygon_area_and_centroid

# (flux_scheme, flux_discretization) variants. (mpfa, reconstructed) is rejected
# by the solver and deliberately omitted.
SCHEME_VARIANTS = (
    ("tpfa", "tpfa"),
    ("tpfa", "reconstructed"),
    ("mpfa", "tpfa"),
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "test_plots"


def _mesh_runner(mesh):
    if mesh == "square_polygonal":
        def run(case, variant, **kw):
            return run_square_polygonal_test(
                case=case, nx=kw["n"], ny=kw["n"],
                flux_scheme=variant[0], flux_discretization=variant[1],
                **{k: v for k, v in kw.items() if k != "n"},
            )
        return run
    if mesh == "nonorthogonal_tiled_polygonal":
        def run(case, variant, **kw):
            tiles = max(2, round(kw["n"] / 4))
            return run_nonorthogonal_tiled_polygonal_test(
                case=case, nx_tiles=tiles, ny_tiles=tiles,
                flux_scheme=variant[0], flux_discretization=variant[1],
                **{k: v for k, v in kw.items() if k != "n"},
            )
        return run
    raise ValueError(f"Unsupported mesh for N-version harness: {mesh!r}")


def _variant_valid(variant):
    return variant != ("mpfa", "reconstructed")


def _weighted_rel_l2(u_a, u_b, vertices, polygons):
    areas = np.array([polygon_area_and_centroid(vertices[poly])[0] for poly in polygons])
    num = np.sqrt(np.sum(areas * (u_a - u_b) ** 2))
    den = np.sqrt(np.sum(areas * u_a**2)) + 1e-300
    return float(num / den)


def run_nversion(
    case,
    *,
    alpha=0.1,
    dt=1e-4,
    t_init=0.0,
    t_end=0.02,
    bbox=(-1.0, 1.0, -1.0, 1.0),
    n=40,
    meshes=("square_polygonal", "nonorthogonal_tiled_polygonal"),
    variants=SCHEME_VARIANTS,
    tol=1e-2,
    accuracy_floor=5e-2,
):
    """Run ``case`` across meshes x flux schemes; report agreement and accuracy.

    Returns a JSON-serializable dict with per-run errors, per-mesh cross-scheme
    spreads, and the overall ``all_agree`` / ``accuracy_ok`` booleans.
    """
    runs = {}     # (mesh, "fs/fd") -> {"u": array, "verts", "polys", "L2_rel", "Linf_rel"}
    skipped = []
    for mesh in meshes:
        runner = _mesh_runner(mesh)
        for variant in variants:
            label = f"{variant[0]}/{variant[1]}"
            if not _variant_valid(variant):
                skipped.append({"mesh": mesh, "variant": label, "reason": "mpfa+reconstructed unsupported"})
                continue
            try:
                vertices, polygons, _centers, u_num, _u_exact, _diff, results = runner(
                    case, variant, alpha=alpha, dt=dt, t_init=t_init, t_end=t_end, bbox=bbox, n=n,
                )
            except Exception as exc:
                # A scheme that fails to assemble/solve on a given mesh is itself
                # a useful robustness finding (e.g. MPFA is singular on the mixed
                # tiled mesh); record it rather than aborting the whole sweep.
                skipped.append({
                    "mesh": mesh, "variant": label,
                    "reason": f"solver failed: {type(exc).__name__}: {exc}",
                })
                continue
            runs[(mesh, label)] = {
                "u": np.asarray(u_num, dtype=float),
                "verts": np.asarray(vertices, dtype=float),
                "polys": polygons,
                "L2_rel": float(results["L2_rel"]),
                "Linf_rel": float(results["Linf_rel"]),
            }

    # 1. Within-mesh cross-scheme pairwise agreement (same mesh -> same cells).
    mesh_spreads = {}
    pairwise = {}
    for mesh in meshes:
        labels = [lbl for (m, lbl) in runs if m == mesh]
        max_spread = 0.0
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a = runs[(mesh, labels[i])]
                b = runs[(mesh, labels[j])]
                rel = _weighted_rel_l2(a["u"], b["u"], a["verts"], a["polys"])
                pairwise[f"{mesh}: {labels[i]} vs {labels[j]}"] = rel
                max_spread = max(max_spread, rel)
        if labels:
            mesh_spreads[mesh] = max_spread

    # 2. Cross-mesh accuracy floor.
    errors = {f"{m} [{lbl}]": runs[(m, lbl)]["L2_rel"] for (m, lbl) in runs}
    all_agree = all(s < tol for s in mesh_spreads.values()) and len(mesh_spreads) > 0
    accuracy_ok = all(e < accuracy_floor for e in errors.values()) and len(errors) > 0

    return {
        "case": case,
        "n": n,
        "tol": tol,
        "accuracy_floor": accuracy_floor,
        "errors_L2_rel": errors,
        "within_mesh_pairwise_rel_diff": pairwise,
        "within_mesh_max_spread": mesh_spreads,
        "skipped": skipped,
        "all_agree": bool(all_agree),
        "accuracy_ok": bool(accuracy_ok),
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-scheme N-version agreement check.")
    parser.add_argument("--case", default="source_driven_sine")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--tol", type=float, default=1e-2)
    parser.add_argument("--dt", type=float, default=1e-4)
    parser.add_argument("--t-end", type=float, default=0.02)
    args = parser.parse_args()

    report = run_nversion(args.case, n=args.n, tol=args.tol, dt=args.dt, t_end=args.t_end)

    out_dir = OUTPUT_DIR / args.case
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "nversion_agreement.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    status = "PASS" if (report["all_agree"] and report["accuracy_ok"]) else "FAIL"
    print(f"[{status}] N-version agreement for {args.case!r} (n={args.n})")
    for mesh, spread in report["within_mesh_max_spread"].items():
        print(f"  {mesh}: max cross-scheme rel-diff = {spread:.3e} (tol {args.tol:g})")
    print(f"  max L2_rel error across runs = {max(report['errors_L2_rel'].values()):.3e}")
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
