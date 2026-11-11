import numpy as np
import pytest

from heat_solver.boundary_flux import (
    band_interface_flux,
    enthalpy_solver_flux_audit,
    hyperbolic_stefan_flux_audit,
)
from heat_solver.enthalpy import EnthalpyStefanSolver, SmoothPhaseChangeModel
from heat_solver.meshes import generate_nonorthogonal_polygonal_mesh, generate_square_polygonal_mesh
from heat_solver.phase_change import ApparentHeatCapacityModel
from heat_solver.transport import HyperbolicStefanSolver

N = 12


def _cold_probe(x, y, t):
    return np.where(x < 0.75 / N, -0.8, 0.6)


@pytest.mark.parametrize("mesh", ["square", "skewed"])
def test_enthalpy_flux_audit_closes_full_domain_balance(mesh):
    if mesh == "square":
        v, p, _ = generate_square_polygonal_mesh(nx=N, ny=N, bbox=(0, 1, 0, 1))
    else:
        v, p, _ = generate_nonorthogonal_polygonal_mesh(nx=N, ny=N, bbox=(0, 1, 0, 1), skew=0.25)
    model = SmoothPhaseChangeModel(0.0, 0.1, latent_heat=3.0)
    s = EnthalpyStefanSolver(v, p, 0.08, 0.01, model, bc_type="dirichlet", bc_func=_cold_probe,
                             source_func=lambda x, y, t: 0.5 * np.sin(np.pi * x))
    s.solve(np.full(s.M, 0.6), 0.0, 0.2, store_history=True)
    audit = enthalpy_solver_flux_audit(s)
    assert audit["energy_extracted"] > 0.0
    assert abs(audit["energy_closure_residual"]) < 1e-10 * max(1.0, abs(audit["enthalpy_change"]))


def test_hyperbolic_flux_audit_identity_and_band_jump():
    v, p, _ = generate_square_polygonal_mesh(nx=N, ny=N, bbox=(0, 1, 0, 1))
    L = 4.0
    pcm = ApparentHeatCapacityModel(-0.15, 0.15, latent_heat=L)
    s = HyperbolicStefanSolver(v, p, 0.08, 0.01, 0.005, pcm, bc_type="dirichlet", bc_func=_cold_probe,
                               phase_change_options={"max_iters": 200, "tol": 1e-10, "relaxation": 0.5,
                                                     "anderson_depth": 6, "raise_on_nonconvergence": False})
    s.record_history = True
    u0 = np.full(s.M, 0.6)
    s.solve(u0, 0.0, 0.2, du0=np.zeros_like(u0))
    a = hyperbolic_stefan_flux_audit(s, pcm)
    assert abs(a["scheme_identity_residual"]) < 1e-10
    assert a["operator_conservation_defect"] < 1e-12
    assert a["energy_closure_residual"] == pytest.approx(a["capacity_defect"] + a["nonlinear_residual_energy"],
                                                         abs=1e-10)
    # Probe band cells jump 0.6 -> -0.8 through the whole mushy interval in one step:
    # the apparent-capacity time term misses exactly L * (band area) of latent heat.
    probe_area = np.sum(s.cell_areas[s.cell_centers[:, 0] < 0.75 / N])
    assert a["band_capacity_defect"] == pytest.approx(-L * probe_area, rel=1e-6)


def test_band_interface_flux_matches_interior_energy_balance():
    v, p, _ = generate_square_polygonal_mesh(nx=N, ny=N, bbox=(0, 1, 0, 1))
    model = SmoothPhaseChangeModel(0.0, 0.1, latent_heat=0.0)
    s = EnthalpyStefanSolver(v, p, 0.08, 0.01, model, bc_type="dirichlet", bc_func=_cold_probe)
    u0 = np.full(s.M, 0.6)
    s.solve(u0, 0.0, 0.05, store_history=True)
    inter = ~s.is_boundary
    gained = np.sum((s.cell_areas * (s.history["H"][-1] - s.history["H"][0]))[inter])
    inflow = sum(k * band_interface_flux(s.A, s.temperature(h), s._boundary_idx)
                 for k, h in zip(s.history["k"], s.history["H"][1:]))
    assert gained == pytest.approx(inflow, rel=1e-10)


def test_audit_requires_history():
    v, p, _ = generate_square_polygonal_mesh(nx=6, ny=6, bbox=(0, 1, 0, 1))
    pcm = ApparentHeatCapacityModel(-0.1, 0.1, latent_heat=1.0)
    s = HyperbolicStefanSolver(v, p, 0.08, 0.01, 0.005, pcm, bc_type="dirichlet", bc_func=_cold_probe)
    with pytest.raises(ValueError):
        hyperbolic_stefan_flux_audit(s, pcm)
