# Meshes: generate once, save, import anywhere

Thermalio is a **2D** code.  Meshes are no longer rebuilt every time a
problem is set up: each mesh configuration is generated once by its generator
script, saved (with its derived geometry), and imported by any problem.

## Pieces

| Module / file | Role |
|---|---|
| `heat_solver/mesh_io.py` | `Mesh2D` container (polygonal / triangular / structured), native `.npz` format, VTU export/import, Gmsh 2.2 import/export, meshio bridge, vectorized geometry + content-hash geometry cache |
| `heat_solver/mesh_library.py` | Registry of generators; `get_mesh(generator, **params)` loads a configuration from the library (generating and saving it on first use); `save_named` / `load_named` for user meshes |
| `mesh_library/` | The library: one `.npz` per configuration plus `manifest.json` (generator, parameters, cell count, hash) |
| `build_mesh_library.py` | Pre-generates every configuration used by the verification sweep (`tests.py`), the QA regression and the research studies |
| `mesh_io_demo.py` | Imports an L-shaped mesh from Gmsh, solves on it, exports VTU for ParaView |

## Using a mesh in any problem

```python
from heat_solver import get_mesh, load_mesh, PolygonalHeatSolver
from heat_solver.transport import HyperbolicHeatSolver
from heat_solver.triangular import NonUniformHeatSolver
from heat_solver.curvilinear import CurvilinearHeatSolver

mesh = get_mesh("nonorthogonal", nx=64, ny=64, skew=0.3)    # library configuration
mesh = load_mesh("my_domain.msh")                           # or a Gmsh / VTU / npz file

PolygonalHeatSolver(*mesh, alpha, dt)            # every cell-centred solver: unpack the mesh
HyperbolicHeatSolver(*mesh, alpha, dt, tau)      # transport, enthalpy, AFC, adjoints alike
NonUniformHeatSolver(*tri_mesh.triangulation(), alpha, dt)    # vertex-centred triangles
CurvilinearHeatSolver(*grid.structured(), alpha=alpha, dt=dt)  # structured curvilinear grids
```

Every verification driver (`run_square_polygonal_test`, `run_test`,
`run_curvilinear_test`, ...) also accepts `mesh=` (a `Mesh2D`, a file path, or
a library name); without it the driver loads its generator configuration from
the library (`use_library=False` restores on-the-fly generation).

## Generators and configurations

| Generator name | Function | Kind |
|---|---|---|
| `square` | `generate_square_polygonal_mesh` | polygonal |
| `hexagonal` | `generate_hexagonal_polygonal_mesh` | polygonal |
| `mixed` | `generate_mixed_polygonal_mesh` | polygonal |
| `nonorthogonal` | `generate_nonorthogonal_polygonal_mesh` | polygonal |
| `nonorthogonal_tiled` | `generate_nonorthogonal_tiled_polygonal_mesh` | polygonal |
| `perturbed` | `generate_perturbed_quad_mesh` | polygonal |
| `delaunay` | `generate_nonuniform_delaunay` | triangular |
| `curvilinear` | `generate_curvilinear_grid` | structured |

A configuration is the generator name plus its full parameter set, with
defaults filled in and values normalized.  Equivalent calls therefore share
one file, named `<generator>-<12-hex digest>.npz`.  Library meshes are
bit-identical to calling the generator directly; this is checked for every
request of the verification sweep.

```bash
MPLCONFIGDIR=/tmp/matplotlib python build_mesh_library.py        # sweep + QA + studies (~30 s first time)
python build_mesh_library.py --list                               # show saved configurations
THERMALIO_MESH_LIBRARY=/elsewhere python build_mesh_library.py    # another location
THERMALIO_MESH_LIBRARY=off python tests.py                        # disable the library
```

## What is stored in a `.npz` mesh file

- Vertices, ragged cell connectivity, kind, grid shape, and boundary tags.
- A header recording the generator, its parameters, the creation time and the
  schema version.
- A **content hash**.  A file whose content no longer matches it is refused
  on load.
- **Derived geometry** (default): centroids, areas, edge-cell connectivity,
  boundary faces, neighbours and least-squares gradient stencils.  Loading a
  file registers this geometry in the in-memory cache, so solvers skip all of
  that set-up.  Geometry computed for an unsaved mesh is cached in memory by
  content hash and shared by every solver on that mesh, including those that
  receive plain `(vertices, polygons)` arrays.  Shared geometry arrays are
  read-only.

The vectorized geometry is **bit-identical** to the previous per-cell loops
(unit test), so no existing numerical result changes.  Set-up of a 65,536-cell
mesh went from about 18 s to under 1 s.

## Exchange formats

| Format | Import | Export | Notes |
|---|---|---|---|
| `.npz` | yes | yes | native, exact, with geometry and configuration |
| `.vtu` | ASCII | yes | ParaView; `export_vtu(mesh, path, cell_data={...}, point_data={...})` writes solution fields; Thermalio metadata round-trips |
| `.msh` (Gmsh 2.2 ASCII) | yes | tri/quad | tagged boundary lines (physical groups) become `mesh.boundary_tags` |
| other (Gmsh 4, Exodus, ...) | via meshio | via meshio | install `meshio` (optional) |

Files with non-zero z coordinates are rejected: Thermalio is 2D.
