import argparse
import json
import math
from pathlib import Path

import numpy as np

import tests


ROOT_DIR = Path(__file__).resolve().parent
QA_DIR = ROOT_DIR / "qa_regression"
BASELINE_DIR = QA_DIR / "baseline"
LATEST_DIR = QA_DIR / "latest"

ARRAY_NAMES = ("vertices", "centers", "u_num", "u_exact", "diff")
RUNNERS = {
    "polygonal": tests._run_polygonal_case,
    "square_polygonal": tests._run_square_polygonal_case,
    "mixed_polygonal": tests._run_mixed_polygonal_case,
    "nonorthogonal_polygonal": tests._run_nonorthogonal_polygonal_case,
    "nonorthogonal_tiled_polygonal": tests._run_nonorthogonal_tiled_case,
    "delaunay": tests._run_delaunay_case,
    "curvilinear": tests._run_curvilinear_case,
}


def _to_builtin(value):
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _job_relpath(job):
    return Path(job["case"]) / job["level"]["name"] / job["mesh_name"]


def _run_job(job):
    runner = RUNNERS[job["mesh_name"]]
    vertices, polygons, centers, u_num, u_exact, diff, results = runner(job["case"], job["config"])
    return {
        "arrays": {
            "vertices": np.asarray(vertices, dtype=float),
            "centers": np.asarray(centers, dtype=float),
            "u_num": np.asarray(u_num, dtype=float),
            "u_exact": np.asarray(u_exact, dtype=float),
            "diff": np.asarray(diff, dtype=float),
        },
        "polygons": [list(map(int, poly)) for poly in polygons],
        "results": _to_builtin(results),
        "config": _to_builtin(job["config"]),
    }


def _write_dataset(root, job, data):
    target_dir = root / _job_relpath(job)
    target_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(target_dir / "arrays.npz", **data["arrays"])
    metadata = {
        "case": job["case"],
        "level": job["level"]["name"],
        "mesh_name": job["mesh_name"],
        "config": _to_builtin(job["config"]),
        "polygons": data["polygons"],
        "results": data["results"],
    }
    with (target_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def _load_dataset(root, job):
    target_dir = root / _job_relpath(job)
    with np.load(target_dir / "arrays.npz") as arrays_file:
        arrays = {name: arrays_file[name] for name in ARRAY_NAMES}
    with (target_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return {
        "arrays": arrays,
        "polygons": metadata["polygons"],
        "results": metadata["results"],
        "config": metadata["config"],
    }


def _compare_arrays(name, current, baseline, atol, rtol):
    if current.shape != baseline.shape:
        return {
            "name": name,
            "passed": False,
            "reason": f"shape mismatch: current={current.shape}, baseline={baseline.shape}",
            "max_abs_diff": math.inf,
        }

    diff = current - baseline
    max_abs_diff = float(np.max(np.abs(diff))) if diff.size else 0.0
    passed = bool(np.allclose(current, baseline, atol=atol, rtol=rtol))
    return {
        "name": name,
        "passed": passed,
        "reason": "" if passed else f"max_abs_diff={max_abs_diff:.6e}",
        "max_abs_diff": max_abs_diff,
    }


def _compare_numeric(name, current, baseline, atol, rtol):
    diff = abs(float(current) - float(baseline))
    limit = atol + rtol * abs(float(baseline))
    return {
        "name": name,
        "passed": diff <= limit,
        "reason": "" if diff <= limit else f"abs_diff={diff:.6e}, limit={limit:.6e}",
        "abs_diff": diff,
    }


def _compare_dataset(current, baseline, atol, rtol):
    checks = []

    if current["polygons"] != baseline["polygons"]:
        checks.append({"name": "polygons", "passed": False, "reason": "polygon connectivity changed"})
    else:
        checks.append({"name": "polygons", "passed": True, "reason": ""})

    if current["config"] != baseline["config"]:
        checks.append({"name": "config", "passed": False, "reason": "test configuration changed"})
    else:
        checks.append({"name": "config", "passed": True, "reason": ""})

    for name in ARRAY_NAMES:
        checks.append(_compare_arrays(name, current["arrays"][name], baseline["arrays"][name], atol=atol, rtol=rtol))

    result_keys = sorted(set(current["results"]) | set(baseline["results"]))
    for key in result_keys:
        if key not in current["results"] or key not in baseline["results"]:
            checks.append({"name": f"results.{key}", "passed": False, "reason": "result key mismatch"})
            continue
        current_value = current["results"][key]
        baseline_value = baseline["results"][key]
        if isinstance(current_value, (int, float)) and isinstance(baseline_value, (int, float)):
            checks.append(_compare_numeric(f"results.{key}", current_value, baseline_value, atol=atol, rtol=rtol))
        elif current_value != baseline_value:
            checks.append({"name": f"results.{key}", "passed": False, "reason": "value changed"})
        else:
            checks.append({"name": f"results.{key}", "passed": True, "reason": ""})

    passed = all(check["passed"] for check in checks)
    return {
        "passed": passed,
        "checks": checks,
    }


def _print_failures(relpath, comparison):
    print(f"FAIL {relpath}")
    for check in comparison["checks"]:
        if not check["passed"]:
            print(f"  {check['name']}: {check['reason']}")


def _write_report(report, target_path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate QA regression data for the configured tests. Missing baselines are created "
            "automatically; existing baselines are compared against the freshly generated data."
        )
    )
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_DIR)
    parser.add_argument("--latest-root", type=Path, default=LATEST_DIR)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--rtol", type=float, default=1e-8)
    args = parser.parse_args()

    args.baseline_root.mkdir(parents=True, exist_ok=True)
    args.latest_root.mkdir(parents=True, exist_ok=True)

    report = {
        "baseline_root": str(args.baseline_root),
        "latest_root": str(args.latest_root),
        "atol": args.atol,
        "rtol": args.rtol,
        "jobs": [],
    }
    failures = 0
    created = 0
    passed = 0

    for job in tests.iter_test_jobs():
        relpath = _job_relpath(job)
        current = _run_job(job)
        _write_dataset(args.latest_root, job, current)

        baseline_dir = args.baseline_root / relpath
        if not (baseline_dir / "arrays.npz").exists() or not (baseline_dir / "metadata.json").exists():
            _write_dataset(args.baseline_root, job, current)
            created += 1
            print(f"CREATE {relpath}")
            report["jobs"].append(
                {
                    "job": str(relpath),
                    "status": "created",
                }
            )
            continue

        baseline = _load_dataset(args.baseline_root, job)
        comparison = _compare_dataset(current, baseline, atol=args.atol, rtol=args.rtol)
        status = "passed" if comparison["passed"] else "failed"
        report["jobs"].append(
            {
                "job": str(relpath),
                "status": status,
                "comparison": comparison,
            }
        )
        if comparison["passed"]:
            passed += 1
            print(f"PASS   {relpath}")
        else:
            failures += 1
            _print_failures(relpath, comparison)

    report["summary"] = {
        "created": created,
        "passed": passed,
        "failed": failures,
    }
    _write_report(report, args.latest_root / "regression_report.json")

    print(
        "Summary: "
        f"created={created}, passed={passed}, failed={failures}, "
        f"report={args.latest_root / 'regression_report.json'}"
    )

    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
