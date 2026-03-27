from collections import Counter

import numpy as np
from scipy.spatial import Delaunay

from .geometry import polygon_area_and_centroid


def generate_hexagonal_polygonal_mesh(nx=10, ny=10, spacing=0.2, jitter=0.0, bbox=(-1.5, 1.5, -1.5, 1.5), seed=1):
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = bbox
    hex_radius = spacing
    dx = np.sqrt(3.0) * hex_radius
    dy = 1.5 * hex_radius
    centers = []
    for iy in range(ny):
        y = ymin + iy * dy
        x_offset = 0.5 * dx if iy % 2 else 0.0
        for ix in range(nx):
            x = xmin + ix * dx + x_offset
            if xmin <= x <= xmax and ymin <= y <= ymax:
                centers.append([x, y])
    centers = np.array(centers, dtype=float)
    vertex_dict = {}
    polygons = []
    vertices = []

    def vertex_hash(x, y, tol=1e-10):
        return (int(np.round(x / tol)), int(np.round(y / tol)))

    base_angles = np.pi / 6.0 + (np.pi / 3.0) * np.arange(6)
    for cx, cy in centers:
        poly = []
        for angle in base_angles:
            vx = cx + hex_radius * np.cos(angle)
            vy = cy + hex_radius * np.sin(angle)
            key = vertex_hash(vx, vy)
            if key not in vertex_dict:
                vertex_dict[key] = len(vertices)
                vertices.append([vx, vy])
            poly.append(vertex_dict[key])
        polygons.append(poly)
    vertices = np.array(vertices, dtype=float)
    if jitter > 0:
        edge_counts = {}
        for poly in polygons:
            for i in range(len(poly)):
                edge = tuple(sorted((poly[i], poly[(i + 1) % len(poly)])))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
        boundary_vertices = np.zeros(len(vertices), dtype=bool)
        for (i, j), count in edge_counts.items():
            if count == 1:
                boundary_vertices[i] = True
                boundary_vertices[j] = True
        amplitude = jitter * hex_radius
        interior = ~boundary_vertices
        vertices[interior, 0] += rng.uniform(-amplitude, amplitude, size=interior.sum())
        vertices[interior, 1] += rng.uniform(-amplitude, amplitude, size=interior.sum())
    centers = np.array([polygon_area_and_centroid(vertices[poly])[1] for poly in polygons])
    return vertices, polygons, centers


def generate_square_polygonal_mesh(nx=20, ny=20, bbox=(-1.0, 1.0, -1.0, 1.0)):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx + 1)
    ys = np.linspace(ymin, ymax, ny + 1)
    vertices = np.array([(x, y) for y in ys for x in xs], dtype=float)

    def point_id(i, j):
        return j * (nx + 1) + i

    polygons = []
    for j in range(ny):
        for i in range(nx):
            polygons.append([
                point_id(i, j),
                point_id(i + 1, j),
                point_id(i + 1, j + 1),
                point_id(i, j + 1),
            ])
    centers = np.array([polygon_area_and_centroid(vertices[poly])[1] for poly in polygons])
    return vertices, polygons, centers


def generate_nonorthogonal_polygonal_mesh(nx=20, ny=20, bbox=(-1.0, 1.0, -1.0, 1.0), skew=0.35):
    xmin, xmax, ymin, ymax = bbox
    y_mid = 0.5 * (ymin + ymax)
    shift_extent = 0.5 * abs(skew) * (ymax - ymin)
    xs = np.linspace(xmin + shift_extent, xmax - shift_extent, nx + 1)
    ys = np.linspace(ymin, ymax, ny + 1)
    vertices = []
    for y in ys:
        x_shift = skew * (y - y_mid)
        for x in xs:
            vertices.append((x + x_shift, y))
    vertices = np.asarray(vertices, dtype=float)

    def point_id(i, j):
        return j * (nx + 1) + i

    polygons = []
    for j in range(ny):
        for i in range(nx):
            polygons.append([
                point_id(i, j),
                point_id(i + 1, j),
                point_id(i + 1, j + 1),
                point_id(i, j + 1),
            ])
    centers = np.array([polygon_area_and_centroid(vertices[poly])[1] for poly in polygons])
    return vertices, polygons, centers


def generate_nonorthogonal_tiled_polygonal_mesh(nx_tiles=4, ny_tiles=4, bbox=(-1.0, 1.0, -1.0, 1.0), skew=0.35):
    xmin, xmax, ymin, ymax = bbox
    y_mid = 0.5 * (ymin + ymax)
    shift_extent = 0.5 * abs(skew) * (ymax - ymin)
    base_bbox = (xmin + shift_extent, xmax - shift_extent, ymin, ymax)

    vertices, polygons, _ = generate_mixed_polygonal_mesh(nx_tiles=nx_tiles, ny_tiles=ny_tiles, bbox=base_bbox)
    vertices = np.asarray(vertices, dtype=float).copy()
    vertices[:, 0] += skew * (vertices[:, 1] - y_mid)
    centers = np.array([polygon_area_and_centroid(vertices[poly])[1] for poly in polygons])
    return vertices, polygons, centers


def generate_mixed_polygonal_mesh(nx_tiles=4, ny_tiles=4, bbox=(-1.0, 1.0, -1.0, 1.0)):
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, 3 * nx_tiles + 1)
    ys = np.linspace(ymin, ymax, 3 * ny_tiles + 1)
    vertices = np.array([(x, y) for y in ys for x in xs], dtype=float)

    def point_id(i, j):
        return j * (3 * nx_tiles + 1) + i

    local_triangle_groups = [
        [0, 1],
        [2],
        [3, 6, 8, 9],
        [4, 5],
        [7, 12, 13],
        [10, 11, 16, 17],
        [14],
        [15],
    ]

    def polygon_from_triangles(local_triangles, group):
        edge_counts = Counter()
        for tri_idx in group:
            tri = local_triangles[tri_idx]
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                edge_counts[tuple(sorted((a, b)))] += 1
        directed_boundary_edges = []
        for tri_idx in group:
            tri = local_triangles[tri_idx]
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                if edge_counts[tuple(sorted((a, b)))] == 1:
                    directed_boundary_edges.append((a, b))
        next_map = {a: b for a, b in directed_boundary_edges}
        start = min(next_map)
        polygon = [start]
        current = start
        while True:
            nxt = next_map[current]
            if nxt == start:
                break
            polygon.append(nxt)
            current = nxt
            if len(polygon) > len(directed_boundary_edges):
                raise RuntimeError("Failed to order polygon boundary for mixed mesh.")
        return polygon

    polygons = []
    for ty in range(ny_tiles):
        for tx in range(nx_tiles):
            i0 = 3 * tx
            j0 = 3 * ty
            local_triangles = []
            for j in range(3):
                for i in range(3):
                    bl = point_id(i0 + i, j0 + j)
                    br = point_id(i0 + i + 1, j0 + j)
                    tl = point_id(i0 + i, j0 + j + 1)
                    tr = point_id(i0 + i + 1, j0 + j + 1)
                    local_triangles.append((bl, br, tr))
                    local_triangles.append((bl, tr, tl))
            for group in local_triangle_groups:
                polygons.append(polygon_from_triangles(local_triangles, group))
    centers = np.array([polygon_area_and_centroid(vertices[poly])[1] for poly in polygons])
    return vertices, polygons, centers


def generate_nonuniform_delaunay(nx=30, ny=30, jitter=0.15, bbox=(-1.5, 1.5, -1.5, 1.5), seed=1):
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = bbox
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel()])
    hx = (xmax - xmin) / max(nx - 1, 1)
    hy = (ymax - ymin) / max(ny - 1, 1)
    jitter_xy = np.zeros_like(pts)
    is_boundary = (
        np.isclose(pts[:, 0], xmin)
        | np.isclose(pts[:, 0], xmax)
        | np.isclose(pts[:, 1], ymin)
        | np.isclose(pts[:, 1], ymax)
    )
    jitter_xy[~is_boundary, 0] = rng.uniform(-jitter * hx, jitter * hx, size=(~is_boundary).sum())
    jitter_xy[~is_boundary, 1] = rng.uniform(-jitter * hy, jitter * hy, size=(~is_boundary).sum())
    tri = Delaunay(pts + jitter_xy)
    return tri.points, tri.simplices.copy()
