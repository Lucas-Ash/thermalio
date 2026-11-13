"""Dirichlet data imposed on boundary faces (``dirichlet_mode="face"``), as FiPy does.

Every cell is an unknown and the boundary-face flux is ``alpha_n (g_f - u_c) / d``
(the Robin closure with ``beta -> infinity``).  On Cartesian meshes this is
FiPy's discretization exactly.
"""

import numpy as np
import pytest

from heat_solver.external_codes import fipy_available
from heat_solver.mesh_library import generate
from heat_solver.polygonal import PolygonalHeatSolver

ALPHA = 0.1


def exact(x, y, t):
    return np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y) + 0.3 * x * y + 0.5


def grad(x, y, t):
    e = np.exp(-t)
    return (e * np.pi * np.cos(np.pi * x) * np.sin(np.pi * y) + 0.3 * y,
            e * np.pi * np.sin(np.pi * x) * np.cos(np.pi * y) + 0.3 * x)


def source(x, y, t):
    return (2 * ALPHA * np.pi**2 - 1) * np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y)


MESHES = {
    "square": lambda n: generate("square", nx=n, ny=n),
    "skewed": lambda n: generate("nonorthogonal", nx=n, ny=n, skew=0.3),
    "hexagonal": lambda n: generate("hexagonal", nx=n, ny=n, spacing=2.4 / n, bbox=(-1.2, 1.2, -1.2, 1.2)),
    "tiled": lambda n: generate("nonorthogonal_tiled", nx_tiles=n // 4, ny_tiles=n // 4, skew=0.3),
}


def _error(mesh, mode="face", t_end=0.1, dt=2e-3, **kw):
    s = PolygonalHeatSolver(*mesh, ALPHA, dt, bc_func=exact, source_func=source, dirichlet_mode=mode,
                            nonorthogonal_correction=True, **kw)
    c = s.cell_centers
    _, u = s.solve(exact(c[:, 0], c[:, 1], 0.0), 0.0, t_end)
    ue = exact(c[:, 0], c[:, 1], t_end)
    return float(np.sqrt(np.sum(s.cell_areas * (u - ue) ** 2) / np.sum(s.cell_areas * ue**2))), u, s


@pytest.mark.parametrize("mesh", list(MESHES))
def test_face_dirichlet_is_second_order_on_every_mesh(mesh):
    errs = [_error(MESHES[mesh](n), time_scheme="bdf2", dt=1e-3)[0] for n in (16, 32)]
    assert errs[1] < 5e-3
    assert np.log2(errs[0] / errs[1]) > 1.6, errs


@pytest.mark.parametrize("scheme", ["backward_euler", "crank_nicolson", "bdf2"])
def test_face_dirichlet_is_the_large_beta_limit_of_robin(scheme):
    m = MESHES["skewed"](16)
    _, u_face, _ = _error(m, time_scheme=scheme)
    beta = 1e10

    def robin(x, y, t, nx, ny):
        gx, gy = grad(x, y, t)
        return beta * np.ones_like(x), ALPHA * (gx * nx + gy * ny) + beta * exact(x, y, t)

    s = PolygonalHeatSolver(*m, ALPHA, 2e-3, bc_type="robin", bc_func=robin, source_func=source,
                            nonorthogonal_correction=True, time_scheme=scheme)
    c = s.cell_centers
    _, u_robin = s.solve(exact(c[:, 0], c[:, 1], 0.0), 0.0, 0.1)
    assert np.max(np.abs(u_robin - u_face)) < 1e-7


def test_face_mode_reuses_factorization_and_matches_iterative_solvers():
    m = MESHES["square"](24)
    err, u, s = _error(m, time_scheme="bdf2")
    assert s.last_solve_stats["boundary_factorizations"] <= 6  # one per distinct (dt, BDF2 coefficient)
    assert s.last_solve_stats["factorizations_reused"] >= 45
    for solver, pc in (("cg", "jacobi"), ("bicgstab", "jacobi")):
        _, u_it, _ = _error(m, time_scheme="bdf2", linear_solver=solver, linear_solver_options={"preconditioner": pc})
        assert np.max(np.abs(u_it - u)) <= 1e-9 * np.max(np.abs(u))


def test_boundary_cells_are_unknowns_in_face_mode():
    m = MESHES["square"](16)
    _, u_band, s = _error(m, mode="band")
    _, u_face, _ = _error(m, mode="face")
    b = s.is_boundary
    c = s.cell_centers
    g = exact(c[b, 0], c[b, 1], 0.1)
    assert np.array_equal(u_band[b], g)  # band: held at the centroid data
    assert np.max(np.abs(u_face[b] - g)) > 1e-6  # face: solved for


def test_face_mode_rejects_cell_averaged_unknowns():
    # The quadratic flux's unknowns are cell averages; the face closure would be first order there.
    with pytest.raises(ValueError, match="dirichlet_mode='face'"):
        PolygonalHeatSolver(*MESHES["skewed"](8), ALPHA, 1e-3, bc_func=exact, flux_discretization="quadratic",
                            dirichlet_mode="face")


def test_face_mode_through_the_nonlinear_path():
    # temperature-dependent conductivity goes through the generic nonlinear path
    s = PolygonalHeatSolver(*MESHES["square"](16), lambda x, y, T: ALPHA + 0.0 * T, 2e-3, bc_func=exact,
                            source_func=source, dirichlet_mode="face")
    c = s.cell_centers
    _, u_nl = s.solve(exact(c[:, 0], c[:, 1], 0.0), 0.0, 0.1)
    _, u_lin, _ = _error(MESHES["square"](16))
    assert np.max(np.abs(u_nl - u_lin)) < 1e-8


def test_invalid_dirichlet_mode_is_rejected():
    with pytest.raises(ValueError, match="dirichlet_mode"):
        PolygonalHeatSolver(*MESHES["square"](4), ALPHA, 1e-2, dirichlet_mode="ghost")


@pytest.mark.skipif(not fipy_available(), reason="FiPy not installed (requirements-external.txt)")
def test_face_mode_reproduces_fipy_on_cartesian_meshes():
    from heat_solver.external_codes import fipy_heat_solve

    bbox, dt, t_end = (-1.0, 1.0, -1.0, 1.0), 2.5e-4, 0.02
    for n in (8, 24):
        m = generate("square", nx=n, ny=n, bbox=bbox)
        s = PolygonalHeatSolver(*m, ALPHA, dt, bc_func=exact, source_func=source, dirichlet_mode="face")
        c = s.cell_centers
        _, u = s.solve(exact(c[:, 0], c[:, 1], 0.0), 0.0, t_end)
        centers, u_fp = fipy_heat_solve(n, bbox, ALPHA, dt, t_end, lambda x, y: exact(x, y, 0.0), source, exact)
        assert np.allclose(centers, c)  # same cell ordering (x fastest)
        assert np.max(np.abs(u - u_fp)) <= 1e-12 * np.max(np.abs(u_fp))
