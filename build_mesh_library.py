"""Pre-generate and save every mesh configuration used by the verification workflow.

Instead of building meshes on the fly, the drivers load named configurations
from the mesh library (``heat_solver/mesh_library.py``).  This script walks
the exact job lists of the main verification sweep (``tests.py``) and of the
QA regression (``tests.iter_test_jobs()``) through the real drivers,
intercepting each driver right after its mesh has been resolved (and saved),
so no solve is performed and the library matches what the workflows request.
It also stores the mesh families used by the research studies.

Usage::

    MPLCONFIGDIR=/tmp/matplotlib python build_mesh_library.py            # sweep + QA + studies
    python build_mesh_library.py --only sweep --library /path/to/lib
    python build_mesh_library.py --list                                    # show the manifest
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


class _MeshResolved(Exception):
    pass


def _collect_workflow_meshes(which):
    import heat_solver.drivers as drivers
    import tests as sweep

    original = drivers._driver_mesh
    seen = []

    def intercept(mesh, use_library, generator, **params):
        m = original(mesh, use_library, generator, **params)
        seen.append(m.metadata.get("configuration"))
        raise _MeshResolved

    drivers._driver_mesh = intercept
    try:
        jobs = []
        if "sweep" in which:
            for case, settings in sweep.CASE_SETTINGS_MINI.items():
                for level in sweep.RESOLUTION_LEVELS:
                    for _name, config, runner, _path in sweep._mesh_jobs(case, settings, level, Path(os.devnull)):
                        jobs.append((runner, case, config))
        if "qa" in which:
            run_by_mesh = {
                "curvilinear": sweep._run_curvilinear_case,
                "polygonal": sweep._run_polygonal_case,
                "square_polygonal": sweep._run_square_polygonal_case,
                "mixed_polygonal": sweep._run_mixed_polygonal_case,
                "nonorthogonal_polygonal": sweep._run_nonorthogonal_polygonal_case,
                "nonorthogonal_tiled_polygonal": sweep._run_nonorthogonal_tiled_case,
                "delaunay": sweep._run_delaunay_case,
            }
            for job in sweep.iter_test_jobs():
                run = run_by_mesh[job["mesh_name"]]
                jobs.append((lambda case, _out, config, run=run: run(case, config), job["case"], job["config"]))
        for runner, case, config in jobs:
            try:
                runner(case, None, config)
            except _MeshResolved:
                pass
    finally:
        drivers._driver_mesh = original
    return seen


STUDY_CONFIGS = [
    # quadratic-flux / cross-code studies: anisotropic problems on four mesh families
    *[("nonorthogonal", dict(nx=n, ny=n, bbox=(-1, 1, -1, 1), skew=0.3)) for n in (16, 32, 64, 128)],
    *[("hexagonal", dict(nx=n, ny=n, spacing=2.0 / n, bbox=(-1, 1, -1, 1))) for n in (16, 32, 64, 128)],
    *[("perturbed", dict(nx=n, ny=n, bbox=(-1, 1, -1, 1), amplitude=0.2, seed=0)) for n in (16, 32, 64, 128)],
    *[("nonorthogonal_tiled", dict(nx_tiles=n // 3, ny_tiles=n // 3, bbox=(-1, 1, -1, 1), skew=0.3))
      for n in (12, 24, 48, 96)],
    *[("square", dict(nx=n, ny=n, bbox=(-1, 1, -1, 1))) for n in (8, 16, 32, 64)],
    *[("nonorthogonal", dict(nx=n, ny=n, bbox=(-1, 1, -1, 1), skew=0.3)) for n in (8,)],
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=("sweep", "qa", "studies"), action="append",
                        help="restrict to one or more groups (default: all)")
    parser.add_argument("--library", help="library directory (default: $THERMALIO_MESH_LIBRARY or ./mesh_library)")
    parser.add_argument("--list", action="store_true", help="print the manifest and exit")
    args = parser.parse_args()
    if args.library:
        os.environ["THERMALIO_MESH_LIBRARY"] = args.library
    from heat_solver import mesh_library

    lib = mesh_library.library_dir()
    if args.list:
        for name, entry in mesh_library.read_manifest().items():
            print(f"{name:32s} {entry['kind']:10s} cells={entry['n_cells']:7d} {entry['parameters']}")
        return
    which = set(args.only or ("sweep", "qa", "studies"))
    t0 = time.perf_counter()
    names = []
    if which & {"sweep", "qa"}:
        names += _collect_workflow_meshes(which)
    if "studies" in which:
        for generator, params in STUDY_CONFIGS:
            names.append(mesh_library.get_mesh(generator, **params).metadata["configuration"])
    unique = sorted(set(names))
    size = sum((lib / f"{n}.npz").stat().st_size for n in unique if (lib / f"{n}.npz").exists())
    print(f"{len(unique)} configurations ({len(names)} requests) in {lib} "
          f"[{size / 1e6:.1f} MB] in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
