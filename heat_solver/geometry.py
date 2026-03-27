import numpy as np


def polygon_area_and_centroid(points):
    """
    Compute polygon area and centroid from an ordered vertex list.
    """
    pts = np.asarray(points, dtype=float)
    x = pts[:, 0]
    y = pts[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    cross = x * y_next - x_next * y
    signed_area = 0.5 * np.sum(cross)
    if np.isclose(signed_area, 0.0):
        return 0.0, pts.mean(axis=0)
    centroid = np.array([
        np.sum((x + x_next) * cross),
        np.sum((y + y_next) * cross),
    ]) / (6.0 * signed_area)
    return abs(signed_area), centroid
