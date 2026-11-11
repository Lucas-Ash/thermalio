import functools
import json

import numpy as np
import pytest

from heat_solver import mesh_io, mesh_library
from heat_solver.afc import AFCMonotoneSolver
from heat_solver.curvilinear import CurvilinearHeatSolver
from heat_solver.drivers import run_curvilinear_test, run_square_polygonal_test, run_test
from heat_solver.enthalpy import EnthalpyStefanSolver, SmoothPhaseChangeModel
from heat_solver.mesh_io import Mesh2D, export_vtu, import_gmsh, load_mesh
from heat_solver.meshes import generate_nonorthogonal_tiled_polygonal_mesh
from heat_solver.polygonal import PolygonalHeatSolver
from heat_solver.transport import HyperbolicHeatSolver
from heat_solver.triangular import NonUniformHeatSolver


@pytest.fixture
def lib(tmp_path, monkeypatch):
    monkeypatch.setenv("THERMALIO_MESH_LIBRARY", str(tmp_path / "lib"))
    mesh_io.GEOMETRY_CACHE.clear()
    return tmp_path / "lib"


def _zero(x, y, t):
    return np.zeros_like(np.asarray(x, dtype=float))


def _mode(x, y, t=0.0):
    return np.sin(np.pi * x) * np.sin(np.pi * y)


KINDS = {
    "polygonal": dict(generator="nonorthogonal_tiled", nx_tiles=3, ny_tiles=3, skew=0.3),
    "triangular": dict(generator="delaunay", nx=10, ny=10, bbox=(-1, 1, -1, 1)),
    "structured": dict(generator="curvilinear", nx=9, ny=7, warp=0.1, twist=0.05),
}


# --- container and native format ---------------------------------------------------

@pytest.mark.parametrize("kind", list(KINDS))
def test_npz_round_trip_preserves_mesh_configuration_and_geometry(kind, tmp_path):
    params = dict(KINDS[kind])
    mesh = Mesh2D.from_generator(params.pop("generator"), **params)
    assert mesh.kind == kind
    path = mesh.save(tmp_path / "m.npz")
    mesh_io.GEOMETRY_CACHE.clear()
    back = load_mesh(path)
    assert back.kind == kind and back.hash == mesh.hash and back.grid_shape == mesh.grid_shape
    assert back.polygons == mesh.polygons
    np.testing.assert_array_equal(back.vertices, mesh.vertices)
    assert back.metadata["parameters"] == mesh.metadata["parameters"]
    # loading registered the stored geometry: no recomputation needed
    assert mesh_io.GEOMETRY_CACHE.get(back.hash) is not None


def test_loaded_geometry_is_used_without_recomputation(tmp_path, monkeypatch):
    mesh = Mesh2D.from_generator("hexagonal", nx=8, ny=8, jitter=0.05, seed=2)
    mesh_library.precompute_geometry(mesh)  # includes LSQ gradient stencils
    path = mesh.save(tmp_path / "hex.npz")
    reference = PolygonalHeatSolver(*mesh, 0.1, 1e-3)
    mesh_io.GEOMETRY_CACHE.clear()
    loaded = load_mesh(path)
    monkeypatch.setattr(mesh_io, "compute_geometry", lambda *a: pytest.fail("geometry recomputed"))
    monkeypatch.setattr(PolygonalHeatSolver, "_build_gradient_reconstruction",
                        lambda self: pytest.fail("gradients recomputed"))
    s = PolygonalHeatSolver(*loaded, 0.1, 1e-3)
    np.testing.assert_array_equal(s.cell_centers, reference.cell_centers)
    assert s.edge_to_cells == reference.edge_to_cells
    for k in reference.boundary_faces:
        np.testing.assert_array_equal(s.boundary_faces[k], reference.boundary_faces[k])
    assert s.gradient_coeffs is not None


def test_vectorized_geometry_is_bit_identical_to_legacy_builders():
    v, p, _ = generate_nonorthogonal_tiled_polygonal_mesh(nx_tiles=4, ny_tiles=4, skew=0.35)
    s = PolygonalHeatSolver(v, p, 0.1, 1e-3)
    np.testing.assert_array_equal(s.cell_centers, s._compute_cell_centers())
    np.testing.assert_array_equal(s.cell_areas, s._compute_cell_areas())
    assert s.edge_to_cells == s._build_edge_to_cells()
    legacy = s._build_boundary_faces()
    for k in legacy:
        np.testing.assert_array_equal(s.boundary_faces[k], legacy[k])
    assert s.neighbors == s._build_neighbors()


def test_tampered_file_is_rejected(tmp_path):
    mesh = Mesh2D.from_generator("square", nx=4, ny=4)
    path = mesh.save(tmp_path / "sq.npz", include_geometry=False)
    data = dict(np.load(path))
    data["vertices"] = data["vertices"] + 1e-3
    np.savez(path, **data)
    with pytest.raises(ValueError, match="hash"):
        load_mesh(path)


def test_meshes_are_strictly_2d():
    with pytest.raises(ValueError, match="2D"):
        Mesh2D(np.zeros((4, 3)), [[0, 1, 2, 3]])


def test_shared_geometry_is_read_only():
    mesh = Mesh2D.from_generator("square", nx=4, ny=4)
    s = PolygonalHeatSolver(*mesh, 0.1, 1e-3)
    with pytest.raises(ValueError):
        s.cell_centers[0, 0] = 99.0


def test_mesh_object_must_be_unpacked():
    mesh = Mesh2D.from_generator("square", nx=4, ny=4)
    with pytest.raises(TypeError, match="unpacked"):
        PolygonalHeatSolver(mesh, mesh.polygons, 0.1, 1e-3)


# --- exchange formats ----------------------------------------------------------------

def test_vtu_round_trip_with_solution_fields(tmp_path):
    mesh = Mesh2D.from_generator("mixed", nx_tiles=2, ny_tiles=2)
    s = PolygonalHeatSolver(*mesh, 0.1, 1e-3)
    u = _mode(s.cell_centers[:, 0], s.cell_centers[:, 1])
    path = export_vtu(mesh, tmp_path / "field.vtu", cell_data={"temperature": u},
                      point_data={"x": mesh.vertices[:, 0]})
    back = load_mesh(path)
    assert back.polygons == mesh.polygons and back.kind == mesh.kind
    np.testing.assert_allclose(back.vertices, mesh.vertices, rtol=0, atol=0)
    np.testing.assert_allclose(back.metadata["cell_data"]["temperature"], u)
    assert back.metadata["generator"] == "mixed"


def test_gmsh_round_trip_with_boundary_tags(tmp_path):
    mesh = Mesh2D.from_generator("square", nx=3, ny=2, bbox=(0, 3, 0, 2))
    left = [[0, 4], [4, 8]]
    mesh.boundary_tags = {"left_wall": np.asarray(left)}
    path = mesh.save(tmp_path / "sq.msh")
    back = load_mesh(path)
    assert back.polygons == mesh.polygons
    np.testing.assert_allclose(back.vertices, mesh.vertices)
    np.testing.assert_array_equal(back.boundary_tags["left_wall"], left)


def test_import_handwritten_gmsh_triangle_mesh(tmp_path):
    msh = """$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
1
1 7 "cold"
$EndPhysicalNames
$Nodes
4
1 0 0 0
2 1 0 0
3 1 1 0
4 0 1 0
$EndNodes
$Elements
3
1 1 2 7 1 1 4
2 2 2 0 0 1 2 3
3 2 2 0 0 1 3 4
$EndElements
"""
    path = tmp_path / "tri.msh"
    path.write_text(msh)
    mesh = import_gmsh(path)
    assert mesh.kind == "triangular" and mesh.n_cells == 2
    np.testing.assert_array_equal(mesh.boundary_tags["cold"], [[0, 3]])
    points, tris = mesh.triangulation()
    assert tris.shape == (2, 3)


def test_3d_files_are_rejected(tmp_path):
    path = tmp_path / "bad.msh"
    path.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n1\n1 0 0 1\n$EndNodes\n")
    with pytest.raises(ValueError, match="2D"):
        import_gmsh(path)


# --- mesh library ----------------------------------------------------------------------

def test_library_generates_once_then_loads(lib, monkeypatch):
    calls = {"n": 0}
    func, kind = mesh_library.GENERATORS["square"]

    @functools.wraps(func)
    def counting(**kw):
        calls["n"] += 1
        return func(**kw)

    monkeypatch.setitem(mesh_library.GENERATORS, "square", (counting, kind))
    a = mesh_library.get_mesh("square", nx=6, ny=5)
    b = mesh_library.get_mesh("square", nx=6, ny=5, bbox=(-1.0, 1.0, -1.0, 1.0))  # same configuration
    assert calls["n"] == 1
    assert a.hash == b.hash
    manifest = json.loads((lib / "manifest.json").read_text())
    name = mesh_library.configuration_name("square", nx=6, ny=5)
    assert manifest[name]["parameters"]["nx"] == 6 and manifest[name]["n_cells"] == 30
    assert (lib / manifest[name]["file"]).exists()
    assert mesh_library.configuration_name("square", nx=6, ny=6) != name


def test_library_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("THERMALIO_MESH_LIBRARY", "off")
    mesh = mesh_library.get_mesh("square", nx=3, ny=3)
    assert mesh.n_cells == 9
    assert not any(tmp_path.iterdir())


def test_named_user_meshes(lib, tmp_path):
    imported = Mesh2D.from_generator("perturbed", nx=5, ny=5, amplitude=0.2, seed=3)
    mesh_library.save_named("plate_with_holes", imported)
    assert mesh_library.load_named("plate_with_holes").hash == imported.hash
    assert mesh_library.resolve_mesh("plate_with_holes").hash == imported.hash
    p = imported.save(tmp_path / "x.npz")
    assert mesh_library.resolve_mesh(p).hash == imported.hash
    with pytest.raises(ValueError):
        mesh_library.save_named("bad name!", imported)


def test_unknown_generator_parameters_are_rejected():
    with pytest.raises(TypeError):
        mesh_library.configuration_name("square", nx=4, colour="red")
    with pytest.raises(KeyError):
        mesh_library.configuration_name("voronoi", n=4)


# --- every problem family accepts imported meshes ----------------------------------

def test_all_solver_families_run_on_imported_meshes(tmp_path):
    poly = load_mesh(Mesh2D.from_generator("nonorthogonal", nx=8, ny=8, skew=0.3).save(tmp_path / "p.npz"))
    tri = load_mesh(Mesh2D.from_generator("delaunay", nx=10, ny=10, bbox=(-1, 1, -1, 1)).save(tmp_path / "t.npz"))
    grid = load_mesh(Mesh2D.from_generator("curvilinear", nx=9, ny=9).save(tmp_path / "c.npz"))
    c = PolygonalHeatSolver(*poly, 0.1, 1e-3).cell_centers
    u0 = _mode(c[:, 0], c[:, 1])

    PolygonalHeatSolver(*poly, 0.1, 1e-3, bc_type="dirichlet", bc_func=_zero).solve(u0, 0.0, 2e-3)
    PolygonalHeatSolver(*poly, 0.1, 1e-3, flux_discretization="quadratic").solve(u0, 0.0, 2e-3)
    HyperbolicHeatSolver(*poly, 0.1, 1e-3, 0.05).solve(u0, 0.0, 2e-3)
    EnthalpyStefanSolver(*poly, 0.1, 1e-3, SmoothPhaseChangeModel(latent_heat=1.0)).solve(u0, 0.0, 2e-3)
    AFCMonotoneSolver(*poly, 0.1, 1e-3, bc_func=_zero, source_func=_zero).solve(u0, 0.0, 2e-3)
    PolygonalHeatSolver(*grid, 0.1, 1e-3).solve(np.zeros(grid.n_cells), 0.0, 2e-3)  # quads from a grid

    points, tris = tri.triangulation()
    NonUniformHeatSolver(points, tris, 0.1, 1e-3, bc_func=lambda x, y, t: _mode(x, y)).solve(
        _mode(points[:, 0], points[:, 1]), 0.0, 2e-3)
    X, Y = grid.structured()
    CurvilinearHeatSolver(X, Y, alpha=0.1, dt=1e-3, bc_func=lambda x, y, t: _mode(x, y)).solve(
        _mode(X, Y), 0.0, 2e-3)


def test_drivers_give_identical_results_from_library_file_or_generator(lib, tmp_path):
    kw = dict(case="source_driven_sine", alpha=0.1, dt=2e-3, t_init=0.0, t_end=0.01, nx=10, ny=10,
              bbox=(0.0, 1.0, 0.0, 1.0))
    *_, u_gen, _, _, _ = run_square_polygonal_test(use_library=False, **kw)
    *_, u_lib, _, _, _ = run_square_polygonal_test(**kw)            # generates + saves
    *_, u_lib2, _, _, _ = run_square_polygonal_test(**kw)           # loads
    mesh = Mesh2D.from_generator("square", nx=10, ny=10, bbox=(0.0, 1.0, 0.0, 1.0))
    *_, u_file, _, _, _ = run_square_polygonal_test(mesh=mesh.save(tmp_path / "sq.npz"), **kw)
    for u in (u_lib, u_lib2, u_file):
        np.testing.assert_array_equal(u, u_gen)
    assert any(lib.glob("square-*.npz"))


def test_triangular_and_curvilinear_drivers_use_the_library(lib):
    run_test(case="sine_mode", nx=12, ny=12, t_end=0.01, dt=5e-3)
    run_curvilinear_test(case="sine_mode", nx=10, ny=10, t_end=0.01, dt=5e-3)
    gens = {v["generator"] for v in json.loads((lib / "manifest.json").read_text()).values()}
    assert {"delaunay", "curvilinear"} <= gens
