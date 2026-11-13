"""``dirichlet_mode="face"`` in the solvers built on ``PolygonalHeatSolver``.

The transport models (Cattaneo, fractional, advection-diffusion, Pennes and the
Stefan variants), the enthalpy / apparent-capacity Stefan solvers and the AFC
bound-preserving solver all impose face Dirichlet data through the base
solver's closure ``alpha_n (g_f - v) / d``.  Band mode (the default) is
unchanged; these tests check face mode.
"""

import numpy as np
import pytest

from heat_solver.afc import AFCMonotoneSolver
from heat_solver.bc_test_problems import heat_problem, transport_problem
from heat_solver.boundary_flux import enthalpy_solver_flux_audit, hyperbolic_stefan_flux_audit
from heat_solver.enthalpy import ApparentCapacityPicardSolver, EnthalpyStefanSolver, SmoothPhaseChangeModel
from heat_solver.mesh_library import generate
from heat_solver.phase_change import ApparentHeatCapacityModel
from heat_solver.polygonal import PolygonalHeatSolver
from heat_solver.transport import (
    AdvectionDiffusionHeatSolver,
    FractionalHeatSolver,
    HyperbolicHeatSolver,
    HyperbolicStefanSolver,
    ReactionDiffusionHeatSolver,
)

T_END = 0.05
LIN = SmoothPhaseChangeModel(latent_heat=0.0)


def _kw(p, mode):
    return dict(bc_type="dirichlet", bc_func=p["bc_func"], source_func=p["source"], dirichlet_mode=mode)


SOLVERS = {
    "cattaneo": ("cattaneo", lambda m, p, dt, k: HyperbolicHeatSolver(*m, p["alpha"], dt, p["tau"], **k)),
    "fractional": ("fractional", lambda m, p, dt, k: FractionalHeatSolver(*m, p["alpha"], dt, p["beta"], **k)),
    "advection_central": ("advection", lambda m, p, dt, k: AdvectionDiffusionHeatSolver(
        *m, p["alpha"], dt, p["velocity"], scheme="central", **k)),
    "pennes": ("pennes", lambda m, p, dt, k: ReactionDiffusionHeatSolver(*m, p["alpha"], dt, p["perfusion"], **k)),
    "enthalpy": (None, lambda m, p, dt, k: EnthalpyStefanSolver(*m, p["alpha"], dt, LIN, **k)),
    "apparent_capacity": (None, lambda m, p, dt, k: ApparentCapacityPicardSolver(*m, p["alpha"], dt, LIN, **k)),
    "afc_global": (None, lambda m, p, dt, k: AFCMonotoneSolver(*m, p["alpha"], dt, limiter="global",
                                                               flux_discretization="tpfa", **k)),
    "afc_local": (None, lambda m, p, dt, k: AFCMonotoneSolver(*m, p["alpha"], dt, limiter="local",
                                                              flux_discretization="tpfa", **k)),
}


def _error(name, n, mode, skew=0.0):
    model, make = SOLVERS[name]
    p = transport_problem(model, "dirichlet") if model else heat_problem("dirichlet", manufactured_boundary=bool(skew))
    b = p["bbox"]
    m = generate("nonorthogonal", nx=n, ny=n, bbox=b, skew=skew) if skew else generate("square", nx=n, ny=n, bbox=b)
    kw = _kw(p, mode)
    if skew and not name.startswith("afc"):
        kw["nonorthogonal_correction"] = True
    s = make(m, p, min(0.25 * ((b[1] - b[0]) / n) ** 2, T_END / 10), kw)
    c = s.cell_centers
    extra = {"du0": p["initial_rate"](c[:, 0], c[:, 1])} if isinstance(s, HyperbolicHeatSolver) else {}
    _, u = s.solve(p["exact"](c[:, 0], c[:, 1], 0.0), 0.0, T_END, **extra)
    ue = p["exact"](c[:, 0], c[:, 1], T_END)
    return float(np.sqrt(np.sum(s.cell_areas * (u - ue) ** 2) / np.sum(s.cell_areas * ue**2))), u, s


@pytest.mark.parametrize("name", list(SOLVERS))
def test_face_mode_is_second_order_and_as_accurate_as_the_band(name):
    face = [_error(name, n, "face")[0] for n in (16, 32)]
    band = _error(name, 32, "band")[0]
    assert np.log2(face[0] / face[1]) > 1.8, face
    assert face[1] < 1.2 * band  # Cartesian mesh: the two modes are about equally accurate


@pytest.mark.parametrize("name", ["pennes", "enthalpy"])
def test_face_mode_on_skewed_meshes(name):
    errs = [_error(name, n, "face", skew=0.3)[0] for n in (16, 32)]
    assert np.log2(errs[0] / errs[1]) > 1.8, errs


@pytest.mark.parametrize("name", ["cattaneo", "enthalpy", "afc_global"])
def test_boundary_cells_are_unknowns(name):
    _, u_face, s = _error(name, 16, "face")
    _, u_band, _ = _error(name, 16, "band")
    b = s.is_boundary
    assert np.max(np.abs(u_face[b] - u_band[b])) > 1e-6  # band cells are held at the centroid data
    assert not np.array_equal(u_face, u_band)


def _sine(x, y, t):
    return np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y) + 0.3 * x * y + 0.5


def _src(alpha):
    return lambda x, y, t: (2 * alpha * np.pi**2 - 1) * np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y)


def test_transport_and_enthalpy_reduce_to_the_polygonal_face_solver():
    # Pennes with zero perfusion and the enthalpy solver with a unit-capacity law are
    # backward-Euler heat equations: the face closure must match PolygonalHeatSolver's.
    m = generate("nonorthogonal", nx=12, ny=12, skew=0.3)
    kw = dict(bc_func=_sine, source_func=_src(0.1), dirichlet_mode="face", nonorthogonal_correction=True)
    ref = PolygonalHeatSolver(*m, 0.1, 2e-3, **kw)
    c = ref.cell_centers
    u0 = _sine(c[:, 0], c[:, 1], 0.0)
    _, u_ref = ref.solve(u0, 0.0, 0.04)
    _, u_pennes = ReactionDiffusionHeatSolver(*m, 0.1, 2e-3, 0.0, **kw).solve(u0, 0.0, 0.04)
    np.testing.assert_allclose(u_pennes, u_ref, rtol=0, atol=1e-12)
    _, u_enth = EnthalpyStefanSolver(*m, 0.1, 2e-3, SmoothPhaseChangeModel(latent_heat=0.0), **kw).solve(
        u0, 0.0, 0.04)
    np.testing.assert_allclose(u_enth, u_ref, rtol=0, atol=1e-9)


def test_face_mode_is_the_large_beta_robin_limit_for_cattaneo():
    m = generate("square", nx=12, ny=12)
    beta = 1e10

    def robin(x, y, t):
        return beta * np.ones_like(x), beta * _sine(x, y, t)

    common = dict(source_func=_src(0.1))
    face = HyperbolicHeatSolver(*m, 0.1, 2e-3, 0.02, bc_func=_sine, dirichlet_mode="face", **common)
    rob = HyperbolicHeatSolver(*m, 0.1, 2e-3, 0.02, bc_type="robin", bc_func=robin, **common)
    c = face.cell_centers
    u0 = _sine(c[:, 0], c[:, 1], 0.0)
    _, u_face = face.solve(u0, 0.0, 0.04)
    _, u_rob = rob.solve(u0, 0.0, 0.04)
    # alpha du/dn = beta (g - u_f): the Robin face value differs from g by O(1/beta)
    assert np.max(np.abs(u_face - u_rob)) < 1e-8


def test_upwind_advection_uses_the_data_on_inflow_faces_only():
    p = transport_problem("advection", "dirichlet")
    m = generate("square", nx=16, ny=16, bbox=p["bbox"])
    s = AdvectionDiffusionHeatSolver(*m, p["alpha"], 1e-3, p["velocity"], **_kw(p, "face"))
    flow = s._boundary_flow
    B, b = s._boundary_system(0.3)
    B0, b0 = super(AdvectionDiffusionHeatSolver, s)._boundary_system(0.3)  # diffusive closure only
    g = s._base._face_dirichlet_values(0.3, s._face_closure)
    conv = np.bincount(s._base.boundary_faces["cells"], weights=np.where(flow < 0, flow * g, 0.0), minlength=s.M)
    np.testing.assert_allclose(b, b0 - conv, rtol=1e-14, atol=1e-14)
    # outflow faces keep the donor cell value (upwind stays monotone)
    donor = np.bincount(s._base.boundary_faces["cells"], weights=np.where(flow < 0, 0.0, flow), minlength=s.M)
    assert np.all(s.C.diagonal() - s._assemble_convection().tocsr().diagonal() == pytest.approx(donor))


@pytest.mark.parametrize("limiter", ["global", "local", "source_envelope"])
def test_afc_face_mode_preserves_bounds(limiter):
    # Discontinuous data on a skewed mesh with the (non-monotone) reconstructed flux:
    # the limited solution must stay within [min, max] of the initial and face data.
    m = generate("nonorthogonal", nx=16, ny=16, skew=0.3)

    def g(x, y, t):
        return np.where(x < 0.5, 1.0, 0.0)

    s = AFCMonotoneSolver(*m, 0.05, 5e-3, bc_func=g, limiter=limiter, dirichlet_mode="face")
    c = s.cell_centers
    u0 = np.where((np.abs(c[:, 0] - 0.6) < 0.2) & (np.abs(c[:, 1] - 0.5) < 0.2), 1.0, 0.0)
    _, u = s.solve(u0, 0.0, 0.1)
    assert u.min() >= -1e-10 and u.max() <= 1.0 + 1e-10
    lo, hi = s._global_bounds(u0, 0.0, 0.1)
    assert (lo, hi) == (0.0, 1.0)


def test_enthalpy_audit_closes_with_face_dirichlet():
    m = generate("nonorthogonal", nx=12, ny=12, skew=0.25)
    model = SmoothPhaseChangeModel(0.0, 0.1, latent_heat=3.0)
    probe = lambda x, y, t: np.where(x < 1e-9, -0.8, 0.6)  # cold left wall (face midpoints at x = 0)
    s = EnthalpyStefanSolver(*m, 0.08, 0.01, model, bc_func=probe, dirichlet_mode="face",
                             source_func=lambda x, y, t: 0.5 * np.sin(np.pi * x))
    s.solve(np.full(s.M, 0.6), 0.0, 0.2, store_history=True)
    audit = enthalpy_solver_flux_audit(s)
    # the face closure is the boundary heat flow: no band reaction to reconstruct
    assert audit["boundary_flux_energy"] == 0.0 and audit["prescribed_flux_energy"] < 0.0
    assert abs(audit["energy_closure_residual"]) < 1e-9 * max(1.0, abs(audit["enthalpy_change"]))
    # every cell is an unknown, so the solver's own energy balance covers the whole domain
    assert np.nanmax(np.abs(s.solve_report["energy_balance_residual"])) < 1e-9


def test_hyperbolic_stefan_audit_with_face_dirichlet():
    m = generate("square", nx=12, ny=12)
    pcm = ApparentHeatCapacityModel(-0.15, 0.15, latent_heat=4.0)
    probe = lambda x, y, t: np.where(x < 1e-9, -0.8, 0.6)
    s = HyperbolicStefanSolver(*m, 0.08, 0.01, 0.005, pcm, bc_func=probe, dirichlet_mode="face",
                               phase_change_options={"max_iters": 200, "tol": 1e-10, "relaxation": 0.5,
                                                     "anderson_depth": 6, "raise_on_nonconvergence": False})
    s.record_history = True
    u0 = np.full(s.M, 0.6)
    s.solve(u0, 0.0, 0.2, du0=np.zeros_like(u0))
    a = hyperbolic_stefan_flux_audit(s, pcm)
    assert a["boundary_flux_energy"] < 0.0  # heat leaves through the cold wall
    assert abs(a["scheme_identity_residual"]) < 1e-10
    # No clamped band, so no boundary cell jumps through the mushy interval in one step
    # (in band mode that latent heat, L x band area, dominates the audit residual).
    assert a["band_capacity_defect"] == pytest.approx(0.0, abs=1e-12)
    assert abs(a["energy_closure_residual"]) < 0.05 * abs(a["enthalpy_change"])


def test_invalid_mode_is_rejected_everywhere():
    m = generate("square", nx=4, ny=4)
    for make in (lambda: ReactionDiffusionHeatSolver(*m, 0.1, 1e-2, 1.0, dirichlet_mode="ghost"),
                 lambda: EnthalpyStefanSolver(*m, 0.1, 1e-2, LIN, dirichlet_mode="ghost"),
                 lambda: AFCMonotoneSolver(*m, 0.1, 1e-2, dirichlet_mode="ghost")):
        with pytest.raises(ValueError, match="dirichlet_mode"):
            make()
