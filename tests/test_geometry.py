import numpy as np
from heat_solver.geometry import polygon_area_and_centroid

def test_polygon_area_and_centroid_square():
    # Counter-clockwise square
    vertices = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ])
    area, centroid = polygon_area_and_centroid(vertices)
    assert np.isclose(area, 1.0)
    assert np.allclose(centroid, [0.5, 0.5])

def test_polygon_area_and_centroid_triangle():
    # Counter-clockwise triangle
    vertices = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [0.0, 2.0]
    ])
    area, centroid = polygon_area_and_centroid(vertices)
    assert np.isclose(area, 2.0)
    assert np.allclose(centroid, [2.0/3.0, 2.0/3.0])

def test_polygon_area_and_centroid_clockwise():
    # Clockwise square (absolute area should still return incorrectly if it doesn't handle, 
    # but the solver expects counter-clockwise generally. Let's check signed implementation)
    vertices = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, 0.0]
    ])
    area, centroid = polygon_area_and_centroid(vertices)
    assert np.isclose(abs(area), 1.0)
    assert np.allclose(centroid, [0.5, 0.5])


def test_cross2d_matches_scalar_cross_and_broadcasts():
    from heat_solver.geometry import cross2d

    assert cross2d([1.0, 0.0], [0.0, 1.0]) == 1.0
    assert cross2d([0.0, 1.0], [1.0, 0.0]) == -1.0
    a = np.array([[1.0, 2.0], [3.0, -1.0]])
    b = np.array([[4.0, 5.0], [0.5, 2.0]])
    np.testing.assert_allclose(cross2d(a, b), [1.0 * 5.0 - 2.0 * 4.0, 3.0 * 2.0 + 0.5])
