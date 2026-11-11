"""Named, persistent mesh configurations.

Every mesh generator in :mod:`heat_solver.meshes` is registered here under a
short name.  A *configuration* is a generator name plus its full parameter set
(defaults filled in), and it maps to a canonical file name, so the same
configuration is generated **once** and then imported by every problem that
needs it::

    from heat_solver.mesh_library import get_mesh
    mesh = get_mesh("nonorthogonal", nx=64, ny=64, bbox=(-1, 1, -1, 1), skew=0.3)
    solver = PolygonalHeatSolver(*mesh, alpha, dt)   # geometry comes from the file too

Library location: ``$THERMALIO_MESH_LIBRARY`` if set, else ``mesh_library/``
next to the ``heat_solver`` package.  Set ``THERMALIO_MESH_LIBRARY=off`` to
generate in memory without touching the disk.  The library directory holds
one ``.npz`` per configuration and a ``manifest.json`` listing every saved
configuration (generator, parameters, cell count, content hash).

User meshes (imported from Gmsh/VTU or built by hand) can be stored under a
custom name with :func:`save_named` and reloaded with :func:`load_named`.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np

from . import meshes as _gen
from .mesh_io import Mesh2D, load_mesh

# name -> (generator function, mesh kind, adapter turning the raw return value into a Mesh2D)
GENERATORS = {
    "square": (_gen.generate_square_polygonal_mesh, "polygonal"),
    "hexagonal": (_gen.generate_hexagonal_polygonal_mesh, "polygonal"),
    "mixed": (_gen.generate_mixed_polygonal_mesh, "polygonal"),
    "nonorthogonal": (_gen.generate_nonorthogonal_polygonal_mesh, "polygonal"),
    "nonorthogonal_tiled": (_gen.generate_nonorthogonal_tiled_polygonal_mesh, "polygonal"),
    "perturbed": (_gen.generate_perturbed_quad_mesh, "polygonal"),
    "delaunay": (_gen.generate_nonuniform_delaunay, "triangular"),
    "curvilinear": (_gen.generate_curvilinear_grid, "structured"),
}


def _normalize(value):
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_normalize(v) for v in np.asarray(value).tolist()] if isinstance(value, np.ndarray) else [
            _normalize(v) for v in value]
    if isinstance(value, (complex, np.complexfloating)):
        if complex(value).imag != 0.0:
            raise ValueError(f"mesh parameters must be real; got {value!r}.")
        value = complex(value).real  # e.g. bounding boxes computed from np.linalg.eigvals
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return int(f) if f.is_integer() and abs(f) < 2**53 else f
    return value


def canonical_parameters(generator, **params):
    """Full parameter set (defaults filled in, values normalized) for a generator."""
    if generator not in GENERATORS:
        raise KeyError(f"unknown mesh generator {generator!r}; choose from {sorted(GENERATORS)}.")
    func = GENERATORS[generator][0]
    sig = inspect.signature(func)
    unknown = set(params) - set(sig.parameters)
    if unknown:
        raise TypeError(f"{generator!r} does not take {sorted(unknown)}.")
    bound = sig.bind_partial(**params)
    bound.apply_defaults()
    return {k: _normalize(v) for k, v in bound.arguments.items()}


def configuration_name(generator, **params):
    """Canonical file stem ``<generator>-<12-hex digest of the parameters>``."""
    canon = canonical_parameters(generator, **params)
    digest = hashlib.sha1(json.dumps(canon, sort_keys=True).encode()).hexdigest()[:12]
    return f"{generator}-{digest}"


def generate(generator, **params):
    """Run a generator and wrap the result as a :class:`Mesh2D` with its configuration."""
    func, kind = GENERATORS[generator]
    canon = canonical_parameters(generator, **params)
    out = func(**canon)
    meta = {"generator": generator, "parameters": canon, "configuration": configuration_name(generator, **canon)}
    if kind == "structured":
        return Mesh2D.from_structured(*out, **meta)
    if kind == "triangular":
        points, tris = out[:2]
        return Mesh2D.from_triangles(points, tris, **meta)
    vertices, polygons = out[:2]
    return Mesh2D.from_polygonal(vertices, polygons, **meta)


def library_dir(path=None):
    """Resolve the library directory (``None`` when disabled with ``THERMALIO_MESH_LIBRARY=off``)."""
    if path is not None:
        return Path(path)
    env = os.environ.get("THERMALIO_MESH_LIBRARY")
    if env is not None:
        return None if env.strip().lower() in {"", "off", "none", "0"} else Path(env)
    return Path(__file__).resolve().parent.parent / "mesh_library"


def _manifest_path(lib):
    return lib / "manifest.json"


def read_manifest(path=None):
    lib = library_dir(path)
    if lib is None or not _manifest_path(lib).exists():
        return {}
    return json.loads(_manifest_path(lib).read_text())


def _update_manifest(lib, name, mesh, file_name):
    manifest = read_manifest(lib)
    manifest[name] = {
        "file": file_name,
        "generator": mesh.metadata.get("generator"),
        "parameters": mesh.metadata.get("parameters"),
        "kind": mesh.kind,
        "n_cells": mesh.n_cells,
        "n_vertices": mesh.n_vertices,
        "hash": mesh.hash,
    }
    tmp = _manifest_path(lib).with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(sorted(manifest.items())), indent=1))
    tmp.replace(_manifest_path(lib))


def precompute_geometry(mesh):
    """Populate the cached solver geometry (incl. LSQ gradient stencils) so it is saved with the mesh."""
    if mesh.kind == "structured":
        return mesh
    from .polygonal import PolygonalHeatSolver

    PolygonalHeatSolver(mesh.vertices, mesh.polygons, 1.0, 1.0, nonorthogonal_correction=True)
    return mesh


def get_mesh(generator, *, library=None, save=True, include_geometry=True, **params):
    """Load a generator configuration from the library, generating and saving it on first use."""
    name = configuration_name(generator, **params)
    lib = library_dir(library)
    if lib is not None:
        f = lib / f"{name}.npz"
        if f.exists():
            mesh = load_mesh(f)
            if mesh.metadata.get("parameters") != canonical_parameters(generator, **params):
                raise ValueError(f"{f} does not match the requested configuration.")
            return mesh
    mesh = generate(generator, **params)
    if lib is not None and save:
        try:
            lib.mkdir(parents=True, exist_ok=True)
            if include_geometry:
                precompute_geometry(mesh)
            mesh.save(lib / f"{name}.npz", include_geometry=include_geometry)
            _update_manifest(lib, name, mesh, f"{name}.npz")
        except OSError:
            pass  # read-only checkout: fall back to the in-memory mesh
    return mesh


def save_named(name, mesh, library=None, include_geometry=True):
    """Store any :class:`Mesh2D` (e.g. imported from Gmsh) under a user-chosen name."""
    if not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("mesh names may contain letters, digits, '-' and '_' only.")
    lib = library_dir(library)
    if lib is None:
        raise ValueError("the mesh library is disabled (THERMALIO_MESH_LIBRARY=off).")
    lib.mkdir(parents=True, exist_ok=True)
    if include_geometry:
        precompute_geometry(mesh)
    path = mesh.save(lib / f"{name}.npz", include_geometry=include_geometry)
    _update_manifest(lib, name, mesh, path.name)
    return path


def load_named(name, library=None):
    lib = library_dir(library)
    if lib is None:
        raise ValueError("the mesh library is disabled (THERMALIO_MESH_LIBRARY=off).")
    f = lib / f"{name}.npz"
    if not f.exists():
        raise FileNotFoundError(f"no mesh named {name!r} in {lib}.")
    return load_mesh(f)


def resolve_mesh(mesh, library=None):
    """Accept a :class:`Mesh2D`, a file path, or a library name and return a :class:`Mesh2D`."""
    if mesh is None or isinstance(mesh, Mesh2D):
        return mesh
    path = Path(mesh)
    if path.suffix and path.exists():
        return load_mesh(path)
    return load_named(str(mesh), library=library)
