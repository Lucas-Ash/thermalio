"""Every solver supports every boundary type, verified on standard test problems.

Problems (heat_solver/bc_test_problems.py):
- dirichlet  decaying sine mode,
- neumann    insulated plate (cosine mode),
- robin      convective cooling of a square (Carslaw & Jaeger, lam tan lam = Bi),
- flux       uniform heating through all walls (exact quadratic-in-space solution),
- radiative  radiating plate (manufactured, exact Stefan-Boltzmann data),
and manufactured problems for the transport models with each boundary type.
"""

import numpy as np
import pytest

from heat_solver.afc import AFCMonotoneSolver
from heat_solver.bc_test_problems import heat_problem, transport_problem
from heat_solver.boundary_conditions import (
    BC_TYPES,
    inflow_linearization,
    manufactured_bc_func,
    normalize_bc_type,
)
from heat_solver.curvilinear import CurvilinearHeatSolver
from heat_solver.enthalpy import ApparentCapacityPicardSolver, EnthalpyStefanSolver, SmoothPhaseChangeModel
from heat_solver.mesh_library import generate
from heat_solver.polygonal import PolygonalHeatSolver
from heat_solver.transport import (
    AdvectionDiffusionHeatSolver,
    FractionalHeatSolver,
    HyperbolicHeatSolver,
    ReactionDiffusionHeatSolver,
)
from heat_solver.triangular import NonUniformHeatSolver

T_END = 0.05
LIN = SmoothPhaseChangeModel(latent_heat=0.0)


def _rel(u, ue, w):
    return float(np.sqrt(np.sum(w * (u - ue) ** 2) / np.sum(w * ue**2)))


def _dt(n, span):
    return min(0.25 * (span / n) ** 2, T_END / 10)


def _cell_centred(make, prob, n, skew=0.0, averages=False):
    b = prob["bbox"]
    mesh = generate("nonorthogonal" if skew else "square", nx=n, ny=n, bbox=b, **({"skew": skew} if skew else {}))
    s = make(mesh, prob, _dt(n, b[1] - b[0]))
    if averages:
        u0, ue = s.cell_average(prob["exact"], 0.0), s.cell_average(prob["exact"], T_END)
    else:
        c = s.cell_centers
        u0, ue = prob["exact"](c[:, 0], c[:, 1], 0.0), prob["exact"](c[:, 0], c[:, 1], T_END)
    kw = {"du0": prob["initial_rate"](*s.cell_centers.T)} if isinstance(s, HyperbolicHeatSolver) else {}
    _, u = s.solve(u0, 0.0, T_END, **kw)
    return _rel(u, ue, s.cell_areas)


def _triangular(prob, n):
    b = prob["bbox"]
    p, tri = generate("delaunay", nx=n + 1, ny=n + 1, jitter=0.15, bbox=b, seed=2).triangulation()
    s = NonUniformHeatSolver(p, tri, prob["alpha"], _dt(n, b[1] - b[0]), bc_type=prob["bc_type"],
                             bc_func=prob["bc_func"], source_func=prob["source"])
    _, u = s.solve(prob["exact"](p[:, 0], p[:, 1], 0.0), 0.0, T_END)
    return _rel(u, prob["exact"](p[:, 0], p[:, 1], T_END), s.cv_area)


def _curvilinear(prob, n):
    b = prob["bbox"]
    X, Y = generate("curvilinear", nx=n + 1, ny=n + 1, bbox=b, warp=0.08).structured()
    s = CurvilinearHeatSolver(X, Y, alpha=prob["alpha"], dt=_dt(n, b[1] - b[0]), bc_type=prob["bc_type"],
                              bc_func=prob["bc_func"], source_func=prob["source"])
    _, u = s.solve(prob["exact"](X, Y, 0.0), 0.0, T_END)
    return _rel(u, prob["exact"](X, Y, T_END), s.J)


def _kw(p):
    return dict(bc_type=p["bc_type"], bc_func=p["bc_func"], source_func=p["source"])


HEAT_SOLVERS = {
    "polygonal_square": lambda p, n: _cell_centred(lambda m, p, dt: PolygonalHeatSolver(*m, p["alpha"], dt, **_kw(p)), p, n),
    "polygonal_skewed": lambda p, n: _cell_centred(
        lambda m, p, dt: PolygonalHeatSolver(*m, p["alpha"], dt, nonorthogonal_correction=True, **_kw(p)), p, n, 0.3),
    "quadratic_skewed": lambda p, n: _cell_centred(
        lambda m, p, dt: PolygonalHeatSolver(*m, p["alpha"], dt, flux_discretization="quadratic", **_kw(p)),
        p, n, 0.3, True),
    "afc_global": lambda p, n: _cell_centred(
        lambda m, p, dt: AFCMonotoneSolver(*m, p["alpha"], dt, limiter="global", flux_discretization="tpfa",
                                           **_kw(p)), p, n),
    "enthalpy": lambda p, n: _cell_centred(lambda m, p, dt: EnthalpyStefanSolver(*m, p["alpha"], dt, LIN, **_kw(p)), p, n),
    "apparent_capacity": lambda p, n: _cell_centred(
        lambda m, p, dt: ApparentCapacityPicardSolver(*m, p["alpha"], dt, LIN, **_kw(p)), p, n),
    "triangular": _triangular,
    "curvilinear": _curvilinear,
}

TRANSPORT_SOLVERS = {
    "cattaneo": lambda m, p, dt: HyperbolicHeatSolver(*m, p["alpha"], dt, p["tau"], **_kw(p)),
    "fractional": lambda m, p, dt: FractionalHeatSolver(*m, p["alpha"], dt, p["beta"], **_kw(p)),
    "advection": lambda m, p, dt: AdvectionDiffusionHeatSolver(*m, p["alpha"], dt, p["velocity"], scheme="central",
                                                               **_kw(p)),
    "pennes": lambda m, p, dt: ReactionDiffusionHeatSolver(*m, p["alpha"], dt, p["perfusion"], **_kw(p)),
}


def _check_convergence(errs, min_order=1.4, max_error=2e-2):
    if max(errs) < 1e-11:  # reproduced to round-off (e.g. quadratic-in-space solution)
        return
    assert errs[-1] < max_error
    assert np.log2(errs[0] / errs[1]) > min_order, errs


@pytest.mark.parametrize("bc_type", BC_TYPES)
@pytest.mark.parametrize("solver", list(HEAT_SOLVERS))
def test_every_heat_solver_supports_every_boundary_type(solver, bc_type):
    prob = heat_problem(bc_type, manufactured_boundary="skewed" in solver)
    errs = [HEAT_SOLVERS[solver](prob, n) for n in (16, 32)]
    min_order = 1.2 if solver in ("triangular",) else 1.4
    _check_convergence(errs, min_order=min_order, max_error=3e-2 if solver == "curvilinear" else 2e-3)


@pytest.mark.parametrize("bc_type", BC_TYPES)
@pytest.mark.parametrize("model", list(TRANSPORT_SOLVERS))
def test_every_transport_model_supports_every_boundary_type(model, bc_type):
    prob = transport_problem(model, bc_type)
    errs = [_cell_centred(TRANSPORT_SOLVERS[model], prob, n) for n in (16, 32)]
    _check_convergence(errs, min_order=1.6, max_error=1e-4)


# --- physical properties of the standard problems -----------------------------------

@pytest.mark.parametrize("solver", ["polygonal", "afc", "enthalpy", "cattaneo_tau0_limit", "triangular"])
def test_insulated_plate_conserves_heat(solver):
    prob = heat_problem("neumann")
    u0f = lambda x, y: 1.0 + np.exp(-((x - 0.3) ** 2 + (y - 0.6) ** 2) / 0.02)  # off-centre hot spot
    insulated = prob["bc_func"]
    if solver == "triangular":
        p, tri = generate("delaunay", nx=17, ny=17, jitter=0.15, bbox=prob["bbox"], seed=2).triangulation()
        s = NonUniformHeatSolver(p, tri, 0.1, 5e-3, bc_type="neumann", bc_func=insulated)
        u0 = u0f(p[:, 0], p[:, 1])
        _, u = s.solve(u0, 0.0, 0.2)
        w = s.cv_area
    else:
        m = generate("square", nx=16, ny=16, bbox=prob["bbox"])
        makers = {
            "polygonal": lambda: PolygonalHeatSolver(*m, 0.1, 5e-3, bc_type="neumann", bc_func=insulated),
            "afc": lambda: AFCMonotoneSolver(*m, 0.1, 5e-3, bc_type="neumann", bc_func=insulated,
                                             flux_discretization="tpfa"),
            "enthalpy": lambda: EnthalpyStefanSolver(*m, 0.1, 5e-3, LIN, bc_type="neumann", bc_func=insulated),
            "cattaneo_tau0_limit": lambda: HyperbolicHeatSolver(*m, 0.1, 5e-3, 1e-3, bc_type="neumann",
                                                                bc_func=insulated),
        }
        s = makers[solver]()
        u0 = u0f(s.cell_centers[:, 0], s.cell_centers[:, 1])
        _, u = s.solve(u0, 0.0, 0.2)
        w = s.cell_areas
    assert np.sum(w * u) == pytest.approx(np.sum(w * u0), rel=1e-10)
    assert u.max() < u0.max()  # the hot spot spreads


@pytest.mark.parametrize("solver", ["polygonal", "triangular", "pennes_k0"])
def test_uniform_wall_heating_injects_exactly_q_times_perimeter(solver):
    q, dt, t_end = 1.0, 5e-3, 0.1
    prob = heat_problem("flux", q=q)
    if solver == "triangular":
        p, tri = generate("delaunay", nx=13, ny=13, jitter=0.15, bbox=prob["bbox"], seed=2).triangulation()
        s = NonUniformHeatSolver(p, tri, 0.1, dt, bc_type="flux", bc_func=prob["bc_func"])
        u0 = prob["exact"](p[:, 0], p[:, 1], 0.0)
        _, u = s.solve(u0, 0.0, t_end)
        gained = np.sum(s.cv_area * (u - u0))
        np.testing.assert_allclose(u, prob["exact"](p[:, 0], p[:, 1], t_end), atol=5e-2)
    else:
        m = generate("square", nx=12, ny=12, bbox=prob["bbox"])
        s = (PolygonalHeatSolver(*m, 0.1, dt, bc_type="flux", bc_func=prob["bc_func"]) if solver == "polygonal"
             else ReactionDiffusionHeatSolver(*m, 0.1, dt, 0.0, bc_type="flux", bc_func=prob["bc_func"]))
        u0 = prob["exact"](s.cell_centers[:, 0], s.cell_centers[:, 1], 0.0)
        _, u = s.solve(u0, 0.0, t_end)
        gained = np.sum(s.cell_areas * (u - u0))
        np.testing.assert_allclose(u, prob["exact"](s.cell_centers[:, 0], s.cell_centers[:, 1], t_end), atol=1e-10)
    assert gained == pytest.approx(q * 8.0 * t_end, rel=1e-10)  # perimeter of [-1, 1]^2 is 8


@pytest.mark.parametrize("solver", ["polygonal", "triangular", "afc", "fractional"])
def test_convective_and_radiative_cooling_relax_to_ambient(solver):
    for bc_type, bc in (("robin", lambda x, y, t, nx, ny: (2.0 * np.ones_like(x), 2.0 * 0.25 * np.ones_like(x))),
                        ("radiative", lambda x, y, t: (0.8, 0.25, 1.0))):
        if solver == "triangular":
            p, tri = generate("delaunay", nx=11, ny=11, jitter=0.1, bbox=(0, 1, 0, 1), seed=1).triangulation()
            s = NonUniformHeatSolver(p, tri, 0.5, 0.05, bc_type=bc_type, bc_func=bc)
            u0 = np.ones(len(p))
        else:
            m = generate("square", nx=10, ny=10, bbox=(0, 1, 0, 1))
            s = {
                "polygonal": lambda: PolygonalHeatSolver(*m, 0.5, 0.05, bc_type=bc_type, bc_func=bc),
                "afc": lambda: AFCMonotoneSolver(*m, 0.5, 0.05, bc_type=bc_type, bc_func=bc, limiter="global",
                                                 flux_discretization="tpfa"),
                "fractional": lambda: FractionalHeatSolver(*m, 0.5, 0.05, 0.9, bc_type=bc_type, bc_func=bc),
            }[solver]()
            u0 = np.ones(m.n_cells)
        _, u_short = s.solve(u0, 0.0, 0.5)
        _, u_long = s.solve(u0, 0.0, 6.0)
        assert 0.25 - 1e-9 <= u_long.min() and u_long.max() <= 1.0 + 1e-9  # bounded by data and ambient
        assert np.max(np.abs(u_long - 0.25)) < np.max(np.abs(u_short - 0.25))  # approaches ambient
        assert np.max(np.abs(u_long - 0.25)) < 0.2 if bc_type == "robin" else True


def test_robin_cooling_matches_carslaw_jaeger_eigenvalue():
    prob = heat_problem("robin", alpha=0.1, h=0.6)
    lam = prob["eigenvalue"]
    assert lam * np.tan(lam) == pytest.approx(0.6 / 0.1, rel=1e-12)
    x = np.linspace(-1, 1, 5)
    # exact data: alpha du/dn + h u = 0 on x = +-1
    for xb, nx in ((1.0, 1.0), (-1.0, -1.0)):
        gx = -lam * np.sin(lam * xb) * np.cos(lam * x)
        assert np.allclose(0.1 * gx * nx + 0.6 * prob["exact"](xb, x, 0.0), 0.0, atol=1e-12)


def test_tangential_correction_vanishes_on_orthogonal_meshes():
    m = generate("square", nx=6, ny=6, bbox=(0, 1, 0, 1))
    bc = lambda x, y, t, nx, ny: (2.0 * np.ones_like(x), np.ones_like(x))
    s = PolygonalHeatSolver(*m, 0.1, 1e-2, bc_type="robin", bc_func=bc, nonorthogonal_correction=True)
    B, _ = s._assemble_boundary_system(0.0)
    assert np.allclose(B.toarray(), np.diag(B.diagonal()), atol=1e-15)
    sk = PolygonalHeatSolver(*generate("nonorthogonal", nx=6, ny=6, bbox=(0, 1, 0, 1), skew=0.3), 0.1, 1e-2, bc_type="robin", bc_func=bc, nonorthogonal_correction=True)
    Bk, _ = sk._assemble_boundary_system(0.0)
    assert np.count_nonzero(Bk.toarray() - np.diag(Bk.diagonal())) > 0  # skewed: gradient stencil appears
    Bd, _ = sk._assemble_boundary_system(0.0, tangential=False)
    assert np.allclose(Bd.toarray(), np.diag(Bd.diagonal()))


# --- the shared BC module ------------------------------------------------------------

def test_bc_type_validation_is_shared():
    assert normalize_bc_type(" Robin ") == "robin"
    m = generate("square", nx=3, ny=3)
    for make in (lambda: PolygonalHeatSolver(*m, 0.1, 1e-2, bc_type="convective"),
                 lambda: AFCMonotoneSolver(*m, 0.1, 1e-2, bc_type="convective"),
                 lambda: NonUniformHeatSolver(*generate("delaunay", nx=5, ny=5).triangulation(), 0.1, 1e-2,
                                              bc_type="convective")):
        with pytest.raises(ValueError, match="bc_type"):
            make()


def test_inflow_linearization_conventions():
    a, c = inflow_linearization("robin", (2.0, 3.0), 2)
    np.testing.assert_allclose(a, 2.0)
    np.testing.assert_allclose(c, 3.0)
    a, c = inflow_linearization("neumann", 0.5, 2, alpha_n=np.array([2.0, 4.0]))
    np.testing.assert_allclose(c, [1.0, 2.0])
    # radiative tangent linearization is exact at the linearization point
    u = np.array([1.3, 0.7])
    a, c = inflow_linearization("radiative", (0.5, 1.0, 1.0), 2, temperature=u)
    np.testing.assert_allclose(c - a * u, -0.5 * (u**4 - 1.0))


@pytest.mark.parametrize("bc_type", BC_TYPES)
def test_manufactured_boundary_data_are_consistent(bc_type):
    exact = lambda x, y, t: 1.0 + 0.2 * x * y
    grad = lambda x, y, t: (0.2 * y, 0.2 * x)
    f = manufactured_bc_func(bc_type, exact, grad, alpha=0.5, robin_beta=3.0, emissivity=0.5, sigma=1.0)
    x, y, nx, ny = np.array([1.0]), np.array([0.5]), np.array([1.0]), np.array([0.0])
    dudn = 0.2 * 0.5
    if bc_type == "dirichlet":
        assert f(x, y, 0.0) == pytest.approx(exact(x, y, 0.0))
    elif bc_type == "neumann":
        assert f(x, y, 0.0, nx, ny) == pytest.approx(dudn)
    elif bc_type == "flux":
        assert f(x, y, 0.0, nx, ny) == pytest.approx(0.5 * dudn)
    elif bc_type == "robin":
        beta, value = f(x, y, 0.0, nx, ny)
        assert value == pytest.approx(0.5 * dudn + 3.0 * exact(x, y, 0.0))
    else:
        eps, t_inf, sigma = f(x, y, 0.0, nx, ny)
        assert 0.5 * dudn == pytest.approx(-eps * sigma * (exact(x, y, 0.0) ** 4 - t_inf**4))


# --- drivers: the Delaunay runner accepts every boundary type from the case catalogue -----

@pytest.mark.parametrize("case, tol", [("steady_linear_neumann", 1e-12), ("steady_linear_robin", 1e-12),
                                       ("radiative_manufactured", 2e-4)])
def test_delaunay_driver_runs_non_dirichlet_cases(case, tol):
    from heat_solver.drivers import run_test

    # P1 reproduces the linear steady profiles exactly with natural (Neumann/Robin) data;
    # the radiative manufactured case is limited by the backward-Euler time error at this dt.
    *_, report = run_test(case=case, nx=13, ny=13, dt=2e-3, t_end=0.06, use_library=False)
    assert report["L2_rel"] < tol


@pytest.mark.parametrize("bc_type", BC_TYPES)
def test_enthalpy_energy_audit_closes_for_every_boundary_type(bc_type):
    from heat_solver.boundary_flux import enthalpy_solver_flux_audit

    prob = heat_problem(bc_type)
    m = generate("square", nx=10, ny=10, bbox=prob["bbox"])
    s = EnthalpyStefanSolver(*m, 0.1, 5e-3, LIN, **_kw(prob))
    c = s.cell_centers
    s.solve(prob["exact"](c[:, 0], c[:, 1], 0.0), 0.0, 0.1, store_history=True)
    report = enthalpy_solver_flux_audit(s)
    assert abs(report["energy_closure_residual"]) < 1e-11
