"""2D mesh container, persistent mesh files, and cached derived geometry.

Thermalio is a 2D code.  Meshes used to be produced by the generator
functions in :mod:`heat_solver.meshes` every time a problem was set up, and
every solver then recomputed the same derived geometry (centroids, areas,
edge-cell connectivity, boundary faces, gradient stencils) in Python loops.
This module lets a mesh be generated once, saved with its configuration and
(optionally) its derived geometry, and imported by any problem.

Mesh kinds
----------
``"polygonal"``   arbitrary polygons ``(vertices, polygons)`` -- all
                  cell-centred solvers (``PolygonalHeatSolver`` and everything
                  built on it: transport, enthalpy, AFC, adjoint, quadratic flux).
``"triangular"``  triangles ``(points, tris)`` -- the vertex-centred
                  ``NonUniformHeatSolver``; also usable as a polygonal mesh.
``"structured"``  logically rectangular curvilinear grid ``(X, Y)`` -- the
                  ``CurvilinearHeatSolver``; also usable as a polygonal (quad) mesh.

Using a mesh in a problem::

    mesh = load_mesh("my_mesh.npz")          # or Mesh2D.from_generator(...)
    PolygonalHeatSolver(*mesh, alpha, dt)    # Mesh2D unpacks to (vertices, polygons)
    HyperbolicHeatSolver(*mesh, alpha, dt, tau)
    NonUniformHeatSolver(*mesh.triangulation(), alpha, dt)
    CurvilinearHeatSolver(*mesh.structured(), alpha=alpha, dt=dt)

File formats
------------
``.npz``  native, exact, versioned; stores the mesh, its generator
          configuration, a content hash and (optionally) derived geometry so
          solvers skip recomputation.
``.vtu``  VTK XML unstructured grid (ParaView); export with optional
          cell/point solution fields, and import.
``.msh``  Gmsh 2.2 ASCII (triangles and quads); import and export.
Anything else is routed through ``meshio`` when it is installed.

Derived geometry is cached in memory by mesh content hash, so solvers that
receive plain ``(vertices, polygons)`` arrays still reuse geometry that was
loaded from a file or computed once.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
_KINDS = ("polygonal", "triangular", "structured")


# ---------------------------------------------------------------------------
# Ragged polygon storage
# ---------------------------------------------------------------------------


def _pack_polygons(polygons):
    polys = [np.asarray(p, dtype=np.int64).ravel() for p in polygons]
    offsets = np.zeros(len(polys) + 1, dtype=np.int64)
    if polys:
        offsets[1:] = np.cumsum([p.size for p in polys])
    flat = np.concatenate(polys) if polys else np.zeros(0, dtype=np.int64)
    return offsets, flat


def _unpack_polygons(offsets, flat):
    return [flat[offsets[k]:offsets[k + 1]].tolist() for k in range(len(offsets) - 1)]


def mesh_hash(vertices, polygons):
    """Content hash of a mesh (vertex coordinates + cell connectivity)."""
    offsets, flat = _pack_polygons(polygons)
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(np.asarray(vertices, dtype=np.float64)).tobytes())
    h.update(offsets.tobytes())
    h.update(flat.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Derived geometry (vectorized; matches the legacy per-cell loops)
# ---------------------------------------------------------------------------


@dataclass
class MeshGeometry:
    """Derived geometry used by the cell-centred solvers.

    Field semantics are identical to the attributes ``PolygonalHeatSolver``
    used to compute itself (same ordering, same values up to round-off).
    """

    cell_centers: np.ndarray
    cell_areas: np.ndarray
    edge_keys: np.ndarray        # (E, 2) sorted vertex pairs, first-occurrence order
    edge_cells: np.ndarray       # (E, 2) adjacent cells (second = -1 on the boundary)
    boundary_faces: dict
    neighbors: list
    is_boundary: np.ndarray
    extras: dict = field(default_factory=dict)  # e.g. gradient coefficients

    @property
    def edge_to_cells(self):
        d = self.extras.get("_edge_to_cells")
        if d is None:
            d = {}
            for (a, b), (c0, c1) in zip(self.edge_keys.tolist(), self.edge_cells.tolist()):
                d[(a, b)] = [c0] if c1 < 0 else [c0, c1]
            self.extras["_edge_to_cells"] = d
        return d


def compute_geometry(vertices, polygons):
    """Vectorized computation of :class:`MeshGeometry`."""
    verts = np.asarray(vertices, dtype=float)
    offsets, flat = _pack_polygons(polygons)
    n_cells = len(offsets) - 1
    counts = np.diff(offsets)
    # next vertex within each polygon
    local = np.arange(flat.size) - np.repeat(offsets[:-1], counts)
    nxt = flat[np.repeat(offsets[:-1], counts) + (local + 1) % np.repeat(counts, counts)]
    x, y = verts[flat, 0], verts[flat, 1]
    xn, yn = verts[nxt, 0], verts[nxt, 1]
    cross = x * yn - xn * y
    # Per-polygon sums with the *same* reduction as the legacy per-cell
    # ``np.sum`` (contiguous rows grouped by vertex count), so the geometry is
    # bit-identical to what the solvers computed before.
    signed = np.empty(n_cells)
    cx_num = np.empty(n_cells)
    cy_num = np.empty(n_cells)
    mx = np.empty(n_cells)
    my = np.empty(n_cells)
    for k in np.unique(counts):
        cells_k = np.flatnonzero(counts == k)
        idx = (offsets[cells_k][:, None] + np.arange(k)[None, :])
        cr = cross[idx]
        signed[cells_k] = 0.5 * np.sum(cr, axis=1)
        cx_num[cells_k] = np.sum((x[idx] + xn[idx]) * cr, axis=1)
        cy_num[cells_k] = np.sum((y[idx] + yn[idx]) * cr, axis=1)
        mx[cells_k] = np.mean(x[idx], axis=1)
        my[cells_k] = np.mean(y[idx], axis=1)
    degenerate = np.isclose(signed, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        centers = np.column_stack([cx_num, cy_num]) / (6.0 * signed)[:, None]
    centers[degenerate] = np.column_stack([mx, my])[degenerate]
    areas = np.where(degenerate, 0.0, np.abs(signed))

    # Edges in first-occurrence order (matches the legacy dict construction).
    owner = np.repeat(np.arange(n_cells), counts)
    a = np.minimum(flat, nxt)
    b = np.maximum(flat, nxt)
    keys = a * (verts.shape[0] + 1) + b
    uniq, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    edge_id = rank[inverse]
    n_edges = uniq.size
    edge_keys = np.column_stack([a[first[order]], b[first[order]]])
    edge_cells = np.full((n_edges, 2), -1, dtype=np.int64)
    occ_sorted = np.argsort(edge_id, kind="stable")  # occurrences grouped by edge, in order
    ids_sorted = edge_id[occ_sorted]
    is_first = np.ones(ids_sorted.size, dtype=bool)
    is_first[1:] = ids_sorted[1:] != ids_sorted[:-1]
    edge_cells[ids_sorted[is_first], 0] = owner[occ_sorted[is_first]]
    second = ~is_first
    edge_cells[ids_sorted[second], 1] = owner[occ_sorted[second]]
    occ_count = np.bincount(edge_id, minlength=n_edges)
    if np.any(occ_count > 2):
        raise ValueError("non-manifold mesh: an edge is shared by more than two cells.")

    # Boundary faces in (cell, local edge) order.  There are only O(sqrt(N)) of
    # them, so they use the legacy per-face arithmetic (bit-identical results).
    bmask = np.flatnonzero(occ_count[edge_id] == 1)
    b_cells, b_mid, b_norm, b_len, b_dist = [], [], [], [], []
    for k in bmask.tolist():
        cell = int(owner[k])
        v0 = verts[flat[k]]
        v1 = verts[nxt[k]]
        edge_vec = v1 - v0
        edge_len = np.linalg.norm(edge_vec)
        if edge_len <= 1e-14:
            continue
        tangent = edge_vec / edge_len
        normal = np.array([-tangent[1], tangent[0]])
        midpoint = 0.5 * (v0 + v1)
        center = centers[cell]
        if np.dot(midpoint - center, normal) < 0:
            normal = -normal
        distance = abs(np.dot(midpoint - center, normal))
        if distance <= 1e-12:
            distance = max(np.linalg.norm(midpoint - center), 1e-12)
        b_cells.append(cell)
        b_mid.append(midpoint)
        b_norm.append(normal)
        b_len.append(edge_len)
        b_dist.append(distance)
    bcells = np.asarray(b_cells, dtype=int)
    boundary_faces = {
        "cells": bcells,
        "midpoints": np.asarray(b_mid, dtype=float).reshape(-1, 2),
        "normals": np.asarray(b_norm, dtype=float).reshape(-1, 2),
        "lengths": np.asarray(b_len, dtype=float),
        "distances": np.asarray(b_dist, dtype=float),
    }

    interior = edge_cells[:, 1] >= 0
    nb = [set() for _ in range(n_cells)]
    for i, j in edge_cells[interior].tolist():
        nb[i].add(j)
        nb[j].add(i)
    neighbors = [sorted(s) for s in nb]
    is_boundary = np.zeros(n_cells, dtype=bool)
    is_boundary[bcells] = True
    return MeshGeometry(centers, areas, edge_keys, edge_cells, boundary_faces, neighbors, is_boundary)


def _freeze(geometry):
    """Shared geometry is read-only: solvers on the same mesh must not mutate it."""
    for arr in (geometry.cell_centers, geometry.cell_areas, geometry.edge_keys, geometry.edge_cells,
                geometry.is_boundary, *geometry.boundary_faces.values()):
        if isinstance(arr, np.ndarray):
            arr.setflags(write=False)
    return geometry


class _GeometryCache:
    """Small LRU cache of :class:`MeshGeometry` keyed by mesh content hash."""

    def __init__(self, maxsize=16):
        self.maxsize = maxsize
        self._data = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        g = self._data.get(key)
        if g is not None:
            self._data.move_to_end(key)
            self.hits += 1
        return g

    def put(self, key, geometry):
        self._data[key] = _freeze(geometry)
        self._data.move_to_end(key)
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def clear(self):
        self._data.clear()
        self.hits = self.misses = 0


GEOMETRY_CACHE = _GeometryCache()


def geometry_for(vertices, polygons):
    """Return cached geometry for a mesh, computing (and caching) it on a miss."""
    key = mesh_hash(vertices, polygons)
    g = GEOMETRY_CACHE.get(key)
    if g is None:
        GEOMETRY_CACHE.misses += 1
        g = compute_geometry(vertices, polygons)
        GEOMETRY_CACHE.put(key, g)
    return key, g


# ---------------------------------------------------------------------------
# Mesh container
# ---------------------------------------------------------------------------


@dataclass
class Mesh2D:
    """A 2D mesh plus the configuration that produced it.

    ``vertices``: (N, 2) coordinates.  ``polygons``: list of vertex-index
    lists (counter-clockwise or clockwise; any number of vertices).
    ``grid_shape``: ``(ny, nx)`` for structured grids, whose vertices are
    stored row-major so that ``X = vertices[:, 0].reshape(ny, nx)``.
    ``boundary_tags``: optional ``{name: (k, 2) vertex pairs}`` for labelled
    boundary segments (e.g. imported Gmsh physical groups).
    """

    vertices: np.ndarray
    polygons: list
    kind: str = "polygonal"
    grid_shape: tuple | None = None
    metadata: dict = field(default_factory=dict)
    boundary_tags: dict = field(default_factory=dict)

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=float)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 2:
            raise ValueError("Thermalio meshes are 2D: vertices must have shape (N, 2).")
        if self.kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}.")
        self.polygons = [list(map(int, p)) for p in self.polygons]
        if self.kind == "triangular" and any(len(p) != 3 for p in self.polygons):
            raise ValueError("triangular meshes must contain only triangles.")
        if self.kind == "structured":
            if self.grid_shape is None:
                raise ValueError("structured meshes need grid_shape=(ny, nx).")
            self.grid_shape = tuple(int(s) for s in self.grid_shape)
            if self.grid_shape[0] * self.grid_shape[1] != self.vertices.shape[0]:
                raise ValueError("grid_shape does not match the number of vertices.")
        n = self.vertices.shape[0]
        if self.polygons:
            flat = np.concatenate([np.asarray(p) for p in self.polygons])
            if flat.size and (flat.min() < 0 or flat.max() >= n):
                raise ValueError("polygon vertex index out of range.")

    # -- views for the different solver families ------------------------------
    def __iter__(self):
        """``*mesh`` unpacks to ``(vertices, polygons)`` for every polygonal solver."""
        yield self.vertices
        yield self.polygons

    @property
    def n_cells(self):
        return len(self.polygons)

    @property
    def n_vertices(self):
        return self.vertices.shape[0]

    @property
    def hash(self):
        return mesh_hash(self.vertices, self.polygons)

    def polygonal(self):
        return self.vertices, self.polygons

    def triangulation(self):
        """``(points, tris)`` for the vertex-centred triangular solver."""
        if self.kind == "triangular":
            return self.vertices, np.asarray(self.polygons, dtype=int)
        if all(len(p) == 3 for p in self.polygons):
            return self.vertices, np.asarray(self.polygons, dtype=int)
        raise ValueError("triangulation() needs a triangular mesh (re-triangulating polygons would change the problem).")

    def structured(self):
        """``(X, Y)`` arrays of shape ``(ny, nx)`` for the curvilinear solver."""
        if self.kind != "structured":
            raise ValueError("structured() needs a structured (curvilinear) mesh.")
        ny, nx = self.grid_shape
        return self.vertices[:, 0].reshape(ny, nx), self.vertices[:, 1].reshape(ny, nx)

    @property
    def geometry(self):
        """Cached :class:`MeshGeometry` (shared with every solver built on this mesh)."""
        return geometry_for(self.vertices, self.polygons)[1]

    @property
    def bbox(self):
        lo = self.vertices.min(axis=0)
        hi = self.vertices.max(axis=0)
        return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])

    def summary(self):
        lens = np.array([len(p) for p in self.polygons])
        return {
            "kind": self.kind, "n_vertices": self.n_vertices, "n_cells": self.n_cells,
            "vertices_per_cell": {int(k): int(v) for k, v in zip(*np.unique(lens, return_counts=True))},
            "bbox": self.bbox, "hash": self.hash[:16],
            "generator": self.metadata.get("generator"), "parameters": self.metadata.get("parameters"),
        }

    # -- constructors -----------------------------------------------------------
    @classmethod
    def from_polygonal(cls, vertices, polygons, **metadata):
        return cls(vertices, polygons, kind="polygonal", metadata=dict(metadata))

    @classmethod
    def from_triangles(cls, points, tris, **metadata):
        return cls(points, np.asarray(tris, dtype=int).tolist(), kind="triangular", metadata=dict(metadata))

    @classmethod
    def from_structured(cls, X, Y, **metadata):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if X.shape != Y.shape or X.ndim != 2:
            raise ValueError("X and Y must be 2D arrays of equal shape (ny, nx).")
        ny, nx = X.shape
        verts = np.column_stack([X.ravel(), Y.ravel()])
        vid = lambda i, j: j * nx + i
        polys = [[vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)]
                 for j in range(ny - 1) for i in range(nx - 1)]
        return cls(verts, polys, kind="structured", grid_shape=(ny, nx), metadata=dict(metadata))

    @classmethod
    def from_generator(cls, generator, **params):
        """Generate with a :mod:`heat_solver.mesh_library` generator and record the configuration."""
        from .mesh_library import generate

        return generate(generator, **params)

    # -- persistence --------------------------------------------------------------
    def save(self, path, include_geometry=True, **extra_metadata):
        """Write the mesh (and by default its derived geometry) to ``path``.

        The format is chosen from the suffix: ``.npz`` (native), ``.vtu``,
        ``.msh`` (Gmsh 2.2), or any meshio-supported format.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        path.parent.mkdir(parents=True, exist_ok=True)
        if suffix == ".npz":
            return _save_npz(self, path, include_geometry, extra_metadata)
        if suffix == ".vtu":
            return export_vtu(self, path)
        if suffix == ".msh":
            return export_gmsh(self, path)
        return _meshio_write(self, path)


def load_mesh(path, kind=None):
    """Load a mesh from ``.npz``, ``.vtu``, ``.msh`` (Gmsh 2.2) or a meshio format."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        mesh = _load_npz(path)
    elif suffix == ".vtu":
        mesh = import_vtu(path)
    elif suffix == ".msh":
        mesh = import_gmsh(path)
    else:
        mesh = _meshio_read(path)
    if kind is not None and mesh.kind != kind:
        if kind == "polygonal":
            mesh = Mesh2D(mesh.vertices, mesh.polygons, kind="polygonal", metadata=mesh.metadata,
                          boundary_tags=mesh.boundary_tags)
        else:
            raise ValueError(f"mesh in {path} is {mesh.kind!r}, not {kind!r}.")
    return mesh


# ---------------------------------------------------------------------------
# Native .npz
# ---------------------------------------------------------------------------


def _save_npz(mesh, path, include_geometry, extra_metadata):
    offsets, flat = _pack_polygons(mesh.polygons)
    meta = dict(mesh.metadata)
    meta.update(extra_metadata)
    meta.setdefault("created", _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
    header = {
        "schema_version": SCHEMA_VERSION, "kind": mesh.kind, "grid_shape": mesh.grid_shape,
        "hash": mesh.hash, "metadata": meta, "boundary_tags": sorted(mesh.boundary_tags),
    }
    arrays = {"vertices": mesh.vertices, "cell_offsets": offsets, "cell_vertices": flat,
              "header": np.frombuffer(json.dumps(header, default=_json_default).encode(), dtype=np.uint8)}
    for name, pairs in mesh.boundary_tags.items():
        arrays[f"tag__{name}"] = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if include_geometry:
        g = mesh.geometry
        arrays.update({
            "geom__cell_centers": g.cell_centers, "geom__cell_areas": g.cell_areas,
            "geom__edge_keys": g.edge_keys, "geom__edge_cells": g.edge_cells,
            "geom__is_boundary": g.is_boundary,
        })
        for k, v in g.boundary_faces.items():
            arrays[f"geom__bf__{k}"] = v
        nb_off, nb_flat = _pack_polygons(g.neighbors)
        arrays["geom__nb_offsets"] = nb_off
        arrays["geom__nb_flat"] = nb_flat
        grad = g.extras.get("gradient_coeffs")
        if grad is not None:
            arrays.update(_pack_gradient_coeffs(grad))
    np.savez_compressed(path, **arrays)
    return path


def _load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        header = json.loads(bytes(data["header"]).decode())
        if header.get("schema_version", 0) > SCHEMA_VERSION:
            raise ValueError(f"{path} uses mesh schema {header['schema_version']}, newer than {SCHEMA_VERSION}.")
        polygons = _unpack_polygons(data["cell_offsets"], data["cell_vertices"])
        tags = {name: data[f"tag__{name}"] for name in header.get("boundary_tags", [])}
        mesh = Mesh2D(data["vertices"], polygons, kind=header["kind"],
                      grid_shape=header.get("grid_shape"), metadata=header.get("metadata", {}),
                      boundary_tags=tags)
        if mesh.hash != header["hash"]:
            raise ValueError(f"{path}: content hash mismatch (file corrupted or edited).")
        if "geom__cell_centers" in data.files:
            bf = {k: data[f"geom__bf__{k}"] for k in ("cells", "midpoints", "normals", "lengths", "distances")}
            nbs = _unpack_polygons(data["geom__nb_offsets"], data["geom__nb_flat"])
            g = MeshGeometry(data["geom__cell_centers"], data["geom__cell_areas"], data["geom__edge_keys"],
                             data["geom__edge_cells"], bf, nbs, data["geom__is_boundary"])
            if "grad__offsets" in data.files:
                g.extras["gradient_coeffs"] = _unpack_gradient_coeffs(data)
            GEOMETRY_CACHE.put(header["hash"], g)
    return mesh


def _pack_gradient_coeffs(coeffs):
    offsets = np.zeros(len(coeffs) + 1, dtype=np.int64)
    idx, vec = [], []
    for c, d in enumerate(coeffs):
        offsets[c + 1] = offsets[c] + len(d)
        for k, v in d.items():
            idx.append(int(k))
            vec.append(np.asarray(v, dtype=float))
    return {"grad__offsets": offsets, "grad__idx": np.asarray(idx, dtype=np.int64),
            "grad__vec": np.asarray(vec, dtype=float).reshape(-1, 2)}


def _unpack_gradient_coeffs(data):
    off, idx, vec = data["grad__offsets"], data["grad__idx"], data["grad__vec"]
    return [{int(idx[k]): vec[k].copy() for k in range(off[c], off[c + 1])} for c in range(len(off) - 1)]


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"not JSON serializable: {type(obj)}")


# ---------------------------------------------------------------------------
# VTK XML (.vtu)
# ---------------------------------------------------------------------------

_VTK_TRIANGLE, _VTK_QUAD, _VTK_POLYGON = 5, 9, 7


def export_vtu(mesh, path, cell_data=None, point_data=None):
    """Write a VTK XML unstructured grid (ASCII) with optional solution fields.

    ``cell_data`` / ``point_data``: ``{name: array}`` with one value per cell /
    vertex (e.g. a cell-centred temperature field).  Open in ParaView.
    """
    path = Path(path)
    offsets, flat = _pack_polygons(mesh.polygons)
    counts = np.diff(offsets)
    types = np.where(counts == 3, _VTK_TRIANGLE, np.where(counts == 4, _VTK_QUAD, _VTK_POLYGON))
    pts = np.column_stack([mesh.vertices, np.zeros(mesh.n_vertices)])

    def arr(name, values, ncomp=1, dtype="Float64"):
        values = np.asarray(values)
        body = " ".join(map(repr, values.ravel().tolist())) if dtype.startswith("Float") else " ".join(
            map(str, values.ravel().tolist()))
        return (f'<DataArray type="{dtype}" Name="{name}" NumberOfComponents="{ncomp}" format="ascii">'
                f"{body}</DataArray>")

    meta = json.dumps({"kind": mesh.kind, "grid_shape": mesh.grid_shape, "metadata": mesh.metadata},
                      default=_json_default).replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")
    parts = ['<?xml version="1.0"?>',
             '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">',
             f'<UnstructuredGrid><!-- thermalio:{meta} -->',
             f'<Piece NumberOfPoints="{mesh.n_vertices}" NumberOfCells="{mesh.n_cells}">',
             "<Points>", arr("Points", pts, 3), "</Points>",
             "<Cells>", arr("connectivity", flat, dtype="Int64"), arr("offsets", offsets[1:], dtype="Int64"),
             arr("types", types, dtype="UInt8"), "</Cells>"]
    if cell_data:
        parts.append("<CellData>")
        for k, v in cell_data.items():
            v = np.asarray(v, dtype=float)
            if v.shape[0] != mesh.n_cells:
                raise ValueError(f"cell_data[{k!r}] needs {mesh.n_cells} values.")
            parts.append(arr(k, v, 1 if v.ndim == 1 else v.shape[1]))
        parts.append("</CellData>")
    if point_data:
        parts.append("<PointData>")
        for k, v in point_data.items():
            v = np.asarray(v, dtype=float)
            if v.shape[0] != mesh.n_vertices:
                raise ValueError(f"point_data[{k!r}] needs {mesh.n_vertices} values.")
            parts.append(arr(k, v, 1 if v.ndim == 1 else v.shape[1]))
        parts.append("</PointData>")
    parts += ["</Piece>", "</UnstructuredGrid>", "</VTKFile>"]
    path.write_text("\n".join(parts))
    return path


def import_vtu(path):
    """Read an ASCII ``.vtu`` (as written by :func:`export_vtu`, or by ParaView in ASCII mode).

    Returns the mesh; any cell/point data arrays are attached as
    ``mesh.metadata["cell_data"]`` / ``["point_data"]`` (lists).
    """
    import re
    import xml.etree.ElementTree as ET

    text = Path(path).read_text()
    root = ET.fromstring(text)
    piece = root.find(".//Piece")
    if piece is None:
        raise ValueError(f"{path}: no <Piece> found.")

    def read(el):
        if el.get("format", "ascii") != "ascii":
            raise ValueError(f"{path}: only ASCII .vtu is supported natively; install meshio for binary files.")
        dtype = float if el.get("type", "").startswith("Float") else np.int64
        return np.array((el.text or "").split(), dtype=dtype)

    pts = read(piece.find("Points/DataArray")).reshape(-1, 3)
    if np.any(np.abs(pts[:, 2]) > 0):
        raise ValueError("Thermalio is 2D: the .vtu has non-zero z coordinates.")
    cells = {el.get("Name"): read(el) for el in piece.find("Cells")}
    conn, off = cells["connectivity"], cells["offsets"]
    starts = np.concatenate([[0], off[:-1]])
    polygons = [conn[s:e].tolist() for s, e in zip(starts, off)]
    meta, kind, grid_shape = {}, "polygonal", None
    m = re.search(r"<!-- thermalio:(.*?) -->", text, re.S)
    if m:
        info = json.loads(m.group(1).replace("&quot;", '"').replace("&lt;", "<").replace("&amp;", "&"))
        kind, grid_shape, meta = info.get("kind", "polygonal"), info.get("grid_shape"), info.get("metadata", {})
    elif polygons and all(len(p) == 3 for p in polygons):
        kind = "triangular"
    for section, key in (("CellData", "cell_data"), ("PointData", "point_data")):
        sec = piece.find(section)
        if sec is not None:
            meta[key] = {el.get("Name"): read(el).tolist() for el in sec}
    return Mesh2D(pts[:, :2], polygons, kind=kind, grid_shape=grid_shape, metadata=meta)


# ---------------------------------------------------------------------------
# Gmsh 2.2 ASCII (.msh)
# ---------------------------------------------------------------------------

_GMSH_LINE, _GMSH_TRI, _GMSH_QUAD = 1, 2, 3


def export_gmsh(mesh, path):
    """Write Gmsh 2.2 ASCII (triangles and quads only; tagged boundary lines included)."""
    lens = {len(p) for p in mesh.polygons}
    if not lens <= {3, 4}:
        raise ValueError("Gmsh 2.2 export supports triangles and quads only; use .vtu or .npz for polygons.")
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$Nodes", str(mesh.n_vertices)]
    lines += [f"{i + 1} {x!r} {y!r} 0" for i, (x, y) in enumerate(mesh.vertices.tolist())]
    lines.append("$EndNodes")
    elements = []
    phys_names = []
    for tag_id, (name, pairs) in enumerate(sorted(mesh.boundary_tags.items()), start=1):
        phys_names.append(f'1 {tag_id} "{name}"')
        for a, b in np.asarray(pairs).tolist():
            elements.append(f"{_GMSH_LINE} 2 {tag_id} {tag_id} {a + 1} {b + 1}")
    for p in mesh.polygons:
        etype = _GMSH_TRI if len(p) == 3 else _GMSH_QUAD
        elements.append(f"{etype} 2 0 0 " + " ".join(str(v + 1) for v in p))
    if phys_names:
        lines += ["$PhysicalNames", str(len(phys_names)), *phys_names, "$EndPhysicalNames"]
    lines += ["$Elements", str(len(elements))]
    lines += [f"{k + 1} {e}" for k, e in enumerate(elements)]
    lines.append("$EndElements")
    Path(path).write_text("\n".join(lines) + "\n")
    return Path(path)


def import_gmsh(path):
    """Read Gmsh 2.2 ASCII: triangles/quads become cells, tagged lines become boundary tags.

    Gmsh 4.x files are routed through meshio when it is installed.
    """
    text = Path(path).read_text().splitlines()
    it = iter(text)
    nodes, elements, names = {}, [], {}
    for line in it:
        tok = line.strip()
        if tok == "$MeshFormat":
            version = next(it).split()[0]
            if not version.startswith("2"):
                return _meshio_read(Path(path))
        elif tok == "$PhysicalNames":
            for _ in range(int(next(it))):
                dim, tag, name = next(it).split(maxsplit=2)
                names[int(tag)] = name.strip('"')
        elif tok == "$Nodes":
            for _ in range(int(next(it))):
                i, x, y, z = next(it).split()[:4]
                if abs(float(z)) > 0:
                    raise ValueError("Thermalio is 2D: the .msh has non-zero z coordinates.")
                nodes[int(i)] = (float(x), float(y))
        elif tok == "$Elements":
            for _ in range(int(next(it))):
                elements.append([int(v) for v in next(it).split()])
    ids = sorted(nodes)
    index = {nid: k for k, nid in enumerate(ids)}
    verts = np.array([nodes[n] for n in ids])
    polygons, tags = [], {}
    for e in elements:
        etype, ntags = e[1], e[2]
        conn = [index[v] for v in e[3 + ntags:]]
        if etype in (_GMSH_TRI, _GMSH_QUAD):
            polygons.append(conn)
        elif etype == _GMSH_LINE and ntags >= 1:
            phys = e[3]
            tags.setdefault(names.get(phys, f"physical_{phys}"), []).append(conn)
    kind = "triangular" if polygons and all(len(p) == 3 for p in polygons) else "polygonal"
    return Mesh2D(verts, polygons, kind=kind, metadata={"source": str(path), "format": "gmsh-2.2"},
                  boundary_tags={k: np.asarray(v) for k, v in tags.items()})


# ---------------------------------------------------------------------------
# meshio bridge (optional)
# ---------------------------------------------------------------------------


def _meshio_read(path):
    try:
        import meshio
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ValueError(f"Unsupported mesh format {path.suffix!r}; install meshio for more formats.") from exc
    m = meshio.read(path)
    if m.points.shape[1] == 3 and np.any(np.abs(m.points[:, 2]) > 0):
        raise ValueError("Thermalio is 2D: the mesh has non-zero z coordinates.")
    polygons = []
    for block in m.cells:
        if block.type in ("triangle", "quad", "polygon"):
            polygons += block.data.tolist()
    kind = "triangular" if polygons and all(len(p) == 3 for p in polygons) else "polygonal"
    return Mesh2D(m.points[:, :2], polygons, kind=kind, metadata={"source": str(path)})


def _meshio_write(mesh, path):
    try:
        import meshio
    except ImportError as exc:  # pragma: no cover
        raise ValueError(f"Unsupported mesh format {path.suffix!r}; use .npz/.vtu/.msh or install meshio.") from exc
    blocks = {}
    for p in mesh.polygons:
        key = {3: "triangle", 4: "quad"}.get(len(p), "polygon")
        blocks.setdefault(key, []).append(p)
    meshio.write(path, meshio.Mesh(mesh.vertices, [(k, np.asarray(v)) for k, v in blocks.items()]))
    return path
