"""Cross-code validation against FiPy and scikit-fem (skipped if not installed)."""

import numpy as np
import pytest

from heat_solver.meshes import generate_square_polygonal_mesh
from heat_solver.polygonal import PolygonalHeatSolver

K = 0.1


def _exact(x, y, t):
    return np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y)


def _source(x, y, t):
    return (2 * K * np.pi**2 - 1.0) * _exact(x, y, t)


BBOX = (-1.0, 1.0, -1.0, 1.0)
DT, T_END = 1e-3, 0.02


def _thermalio(n):
    v, p, _ = generate_square_polygonal_mesh(nx=n, ny=n, bbox=BBOX)
    s = PolygonalHeatSolver(v, p, K, DT, bc_type="dirichlet", bc_func=_exact, source_func=_source)
    x, y = s.cell_centers[:, 0], s.cell_centers[:, 1]
    return s.cell_centers, s.solve(_exact(x, y, 0.0), 0.0, T_END)[1]


def test_fipy_agrees_with_thermalio_and_converges():
    pytest.importorskip("fipy")
    from heat_solver.external_codes import fipy_heat_solve

    errs, diffs = [], []
    for n in (8, 16):
        c, u_fp = fipy_heat_solve(n, BBOX, K, DT, T_END, lambda x, y: _exact(x, y, 0.0), _source, _exact)
        c_th, u_th = _thermalio(n)
        assert np.allclose(c, c_th)  # co-located cell centres
        ue = _exact(c[:, 0], c[:, 1], T_END)
        errs.append(np.sqrt(np.mean((u_fp - ue) ** 2) / np.mean(ue**2)))
        diffs.append(np.sqrt(np.mean((u_fp - u_th) ** 2) / np.mean(u_th**2)))
    assert errs[1] < 0.35 * errs[0]   # ~second order
    assert diffs[1] < 0.35 * diffs[0]  # codes converge towards each other
    assert diffs[1] < 2e-2


@pytest.mark.parametrize("order", [1, 2])
def test_skfem_agrees_with_thermalio(order):
    pytest.importorskip("skfem")
    from heat_solver.external_codes import skfem_heat_solve

    errs = []
    for n in (8, 16):
        res = skfem_heat_solve(n, BBOX, K, DT, T_END, lambda x, y: _exact(x, y, 0.0), _source,
                               dirichlet=_exact, order=order)
        errs.append(res.l2_error(_exact, T_END))
    assert errs[1] < (0.35 if order == 1 else 0.2) * errs[0]
    c_th, u_th = _thermalio(16)
    at_centres = res.evaluate(c_th)
    # P1 is itself ~3% off the exact solution at n = 16 (P2 ~0.1%).
    tol = 8e-2 if order == 1 else 2e-2
    assert np.sqrt(np.mean((at_centres - u_th) ** 2) / np.mean(u_th**2)) < tol


def test_skfem_robin_pennes_matches_exact():
    pytest.importorskip("skfem")
    from heat_solver.external_codes import skfem_heat_solve

    h, perf = 3.0, 2.0
    ex = lambda x, y, t: np.exp(-t) * (2 + np.sin(np.pi * x) * np.cos(np.pi * y))

    def src(x, y, t):
        lap = -2 * np.pi**2 * np.exp(-t) * np.sin(np.pi * x) * np.cos(np.pi * y)
        return -ex(x, y, t) - K * lap + perf * ex(x, y, t)

    def robin(x, y, t, nx, ny):
        gx = np.exp(-t) * np.pi * np.cos(np.pi * x) * np.cos(np.pi * y)
        gy = -np.exp(-t) * np.pi * np.sin(np.pi * x) * np.sin(np.pi * y)
        return K * (gx * nx + gy * ny) + h * ex(x, y, t)

    res = skfem_heat_solve(16, (0, 1, 0, 1), K, DT, T_END, lambda x, y: ex(x, y, 0.0), src,
                           robin=(h, robin), reaction=perf, order=2)
    assert res.l2_error(ex, T_END) < 1e-4
