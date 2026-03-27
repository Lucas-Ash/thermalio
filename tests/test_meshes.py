import numpy as np
from heat_solver.meshes import (
    generate_hexagonal_polygonal_mesh,
    generate_square_polygonal_mesh,
    generate_nonorthogonal_polygonal_mesh,
    generate_nonorthogonal_tiled_polygonal_mesh,
    generate_mixed_polygonal_mesh,
    generate_nonuniform_delaunay
)

def test_generate_square_polygonal_mesh():
    nx, ny = 4, 4
    vertices, polygons, centers = generate_square_polygonal_mesh(nx, ny)
    
    assert len(vertices) == (nx + 1) * (ny + 1)
    assert len(polygons) == nx * ny
    assert len(centers) == nx * ny
    # Each polygon should have 4 vertices
    assert all(len(p) == 4 for p in polygons)

def test_generate_hexagonal_polygonal_mesh():
    vertices, polygons, centers = generate_hexagonal_polygonal_mesh(nx=5, ny=5)
    assert len(polygons) == 25
    assert len(centers) == 25
    assert len(vertices) > 0
    # Hexagons should have 6 vertices
    assert all(len(p) == 6 for p in polygons)

def test_generate_nonorthogonal_polygonal_mesh():
    nx, ny = 4, 4
    vertices, polygons, centers = generate_nonorthogonal_polygonal_mesh(nx, ny, skew=0.5)
    assert len(polygons) == nx * ny
    assert len(centers) == nx * ny
    assert all(len(p) == 4 for p in polygons)

def test_generate_mixed_polygonal_mesh():
    nx_tiles, ny_tiles = 2, 2
    vertices, polygons, centers = generate_mixed_polygonal_mesh(nx_tiles, ny_tiles)
    # 8 polygons per tile
    expected_polygons = 8 * nx_tiles * ny_tiles
    assert len(polygons) == expected_polygons
    assert len(centers) == expected_polygons

def test_generate_nonorthogonal_tiled_polygonal_mesh():
    nx_tiles, ny_tiles = 2, 2
    vertices, polygons, centers = generate_nonorthogonal_tiled_polygonal_mesh(nx_tiles, ny_tiles)
    expected_polygons = 8 * nx_tiles * ny_tiles
    assert len(polygons) == expected_polygons
    assert len(centers) == expected_polygons

def test_generate_nonuniform_delaunay():
    nx, ny = 5, 5
    points, simplices = generate_nonuniform_delaunay(nx=nx, ny=ny, jitter=0.0)
    assert len(points) == nx * ny
    assert len(simplices) > 0
    # Simplices are triangles, 3 vertices each
    assert simplices.shape[1] == 3
