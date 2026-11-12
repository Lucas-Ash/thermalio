"""BDF2 time stepping (performance item 2): second order, damping, and the driver default."""

import inspect

import numpy as np
import pytest

from heat_solver import drivers
from heat_solver.curvilinear import CurvilinearHeatSolver
from heat_solver.mesh_library import generate
from heat_solver.phase_change import ApparentHeatCapacityModel
from heat_solver.polygonal import PolygonalHeatSolver
from heat_solver.triangular import NonUniformHeatSolver

ALPHA = 0.1
T_END = 0.5


def exact(x, y, t):
    return np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y)


def source(x, y, t):
    return (2 * ALPHA * np.pi**2 - 1) * exact(x, y, t)


def _order(run, ref_scheme="crank_nicolson"):
    """Observed temporal order of BDF2 against a fine-step reference on the same mesh."""
    ref = run(ref_scheme, 1024)
    e = [np.max(np.abs(run("bdf2", k) - ref)) for k in (10, 20, 40)]
    return np.log2(e[1] / e[2]), e


def _polygonal(alpha=ALPHA, **kw):
    m = generate("square", nx=16, ny=16)

    def run(scheme, k):
        s = PolygonalHeatSolver(*m, alpha, T_END / k, time_scheme=scheme, bc_func=exact, source_func=source, **kw)
        c = s.cell_centers
        return s.solve(exact(c[:, 0], c[:, 1], 0.0), 0.0, T_END)[1]

    return run


@pytest.mark.parametrize("variant", ["cached", "uncached", "nonlinear_path"])
def test_polygonal_bdf2_is_second_order(variant):
    kw = {"cached": {}, "uncached": {"reuse_linear_lhs": False},
          "nonlinear_path": {"alpha": lambda x, y, T: ALPHA + 0 * x}}[variant]
    order, errs = _order(_polygonal(**kw))
    assert order > 1.9
    assert errs[-1] < 5e-5


@pytest.mark.parametrize("bc_type", ["neumann", "robin", "flux", "radiative"])
def test_polygonal_bdf2_second_order_with_natural_boundaries(bc_type):
    from heat_solver.bc_test_problems import heat_problem

    prob = heat_problem(bc_type)
    b = prob["bbox"]
    m = generate("square", nx=12, ny=12, bbox=b)

    def run(scheme, k):
        s = PolygonalHeatSolver(*m, prob["alpha"], 0.2 / k, time_scheme=scheme, bc_type=prob["bc_type"],
                                bc_func=prob["bc_func"], source_func=prob["source"])
        c = s.cell_centers
        return s.solve(prob["exact"](c[:, 0], c[:, 1], 0.0), 0.0, 0.2)[1]

    ref = run("crank_nicolson", 1024)
    e = [np.max(np.abs(run("bdf2", k) - ref)) for k in (10, 20, 40)]
    if max(e) < 1e-12:  # flux problem: exactly linear in time, reproduced to round-off
        return
    assert np.log2(e[1] / e[2]) > 1.8


def test_curvilinear_bdf2_is_second_order():
    X, Y = generate("curvilinear", nx=17, ny=17, warp=0.08).structured()

    def run(scheme, k):
        s = CurvilinearHeatSolver(X, Y, alpha=ALPHA, dt=T_END / k, time_scheme=scheme, bc_func=exact, source_func=source)
        return s.solve(exact(X, Y, 0.0), 0.0, T_END)[1]

    assert _order(run)[0] > 1.9


@pytest.mark.parametrize("variant", ["cached", "uncached", "phase_change"])
def test_triangular_bdf2_is_second_order(variant):
    p, tri = generate("delaunay", nx=14, ny=14, bbox=(-1, 1, -1, 1)).triangulation()
    kw = {"cached": {}, "uncached": {"reuse_linear_lhs": False},
          # mushy zone wider than the solution range: smooth (constant) capacity, Picard path
          "phase_change": {"phase_change_model": ApparentHeatCapacityModel(-2.0, 2.0, latent_heat=0.5)}}[variant]

    def run(scheme, k):
        s = NonUniformHeatSolver(p, tri, ALPHA, T_END / k, bc_func=exact, time_scheme=scheme, **kw)
        return s.solve(exact(p[:, 0], p[:, 1], 0.0), 0.0, T_END)[1]

    assert _order(run, ref_scheme="bdf2")[0] > 1.9


def test_bdf2_handles_a_short_final_step():
    run = _polygonal()
    ref = run("crank_nicolson", 1024)
    m = generate("square", nx=16, ny=16)
    s = PolygonalHeatSolver(*m, ALPHA, T_END / 22.5, time_scheme="bdf2", bc_func=exact, source_func=source)
    c = s.cell_centers
    _, u = s.solve(exact(c[:, 0], c[:, 1], 0.0), 0.0, T_END)  # 23 steps, the last one half length
    assert np.max(np.abs(u - ref)) < 2e-4


def test_bdf2_damps_where_crank_nicolson_rings():
    m = generate("square", nx=24, ny=24)
    out = {}
    for scheme in ("crank_nicolson", "bdf2"):
        s = PolygonalHeatSolver(*m, ALPHA, 1.0, time_scheme=scheme, bc_func=lambda x, y, t: 0 * x)
        c = s.cell_centers
        hot = ((np.abs(c[:, 0]) < 0.3) & (np.abs(c[:, 1]) < 0.3)).astype(float)
        _, u = s.solve(hot, 0.0, 2.0)
        out[scheme] = u[~s.is_boundary]
    assert out["crank_nicolson"].min() < -0.1  # high-frequency modes flip sign
    assert out["bdf2"].min() >= 0.0  # damped instead


def test_unknown_time_scheme_is_rejected():
    m = generate("square", nx=4, ny=4)
    p, tri = generate("delaunay", nx=5, ny=5).triangulation()
    X, Y = generate("curvilinear", nx=5, ny=5).structured()
    for make in (lambda: PolygonalHeatSolver(*m, 0.1, 1e-2, time_scheme="bdf3"),
                 lambda: NonUniformHeatSolver(p, tri, 0.1, 1e-2, time_scheme="crank_nicolson"),
                 lambda: CurvilinearHeatSolver(X, Y, alpha=0.1, dt=1e-2, time_scheme="bdf3")):
        with pytest.raises(ValueError, match="time_scheme"):
            make()


def test_verification_drivers_default_to_bdf2():
    assert drivers.DEFAULT_TIME_SCHEME == "bdf2"
    for name in ("run_polygonal_test", "run_square_polygonal_test", "run_mixed_polygonal_test",
                 "run_nonorthogonal_polygonal_test", "run_nonorthogonal_tiled_polygonal_test",
                 "run_polygonal_mesh_test", "run_curvilinear_test", "run_test"):
        assert inspect.signature(getattr(drivers, name)).parameters["time_scheme"].default == "bdf2"
    *_, bdf2 = drivers.run_square_polygonal_test(case="sine_mode", nx=16, ny=16, dt=5e-3, t_end=0.1, use_library=False)
    *_, be = drivers.run_square_polygonal_test(case="sine_mode", nx=16, ny=16, dt=5e-3, t_end=0.1, use_library=False,
                                               time_scheme="backward_euler")
    assert bdf2["L2_rel"] < be["L2_rel"]
