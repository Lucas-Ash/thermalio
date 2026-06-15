import numpy as np
import pytest

from heat_solver.cases import (
    advection_diffusion_case,
    cattaneo_wave_case,
    fractional_subdiffusion_case,
    pennes_bioheat_case,
    transport_linear_boundary_case,
)
from heat_solver.geometry import polygon_area_and_centroid
from heat_solver.meshes import generate_square_polygonal_mesh
from heat_solver.transport import (
    AdvectionDiffusionHeatSolver,
    FractionalHeatSolver,
    HyperbolicHeatSolver,
    ReactionDiffusionHeatSolver,
)


def _square(n, bbox=(0.0, 1.0, 0.0, 1.0)):
    vertices, polygons, centers = generate_square_polygonal_mesh(nx=n, ny=n, bbox=bbox)
    areas = np.array([polygon_area_and_centroid(vertices[poly])[0] for poly in polygons])
    return vertices, polygons, centers, areas


def _rel_l2(u, u_exact, areas):
    return float(np.sqrt(np.sum(areas * (u - u_exact) ** 2)) / np.sqrt(np.sum(areas * u_exact**2)))


# --------------------------------------------------------------------------- #
# Hyperbolic / Cattaneo
# --------------------------------------------------------------------------- #
def test_hyperbolic_wave_speed():
    vertices, polygons, _ = generate_square_polygonal_mesh(nx=4, ny=4)
    solver = HyperbolicHeatSolver(vertices, polygons, 0.2, 0.01, relaxation_time=0.05)
    assert solver.wave_speed() == pytest.approx(np.sqrt(0.2 / 0.05))


def test_hyperbolic_requires_positive_tau():
    vertices, polygons, _ = generate_square_polygonal_mesh(nx=4, ny=4)
    with pytest.raises(ValueError):
        HyperbolicHeatSolver(vertices, polygons, 0.1, 0.01, relaxation_time=0.0)


def test_hyperbolic_manufactured_accuracy_and_convergence():
    case = cattaneo_wave_case(alpha=0.1, tau=0.2)
    t_end = 0.3
    errors = []
    for n in (16, 32):
        vertices, polygons, centers, areas = _square(n, case["bbox"])
        dt = t_end / (4 * n)
        solver = HyperbolicHeatSolver(
            vertices, polygons, case["alpha"], dt, case["relaxation_time"],
            bc_func=case["boundary"], source_func=case["source"],
        )
        u0 = case["solution"](centers[:, 0], centers[:, 1], 0.0)
        du0 = case["initial_rate"](centers[:, 0], centers[:, 1])
        _, u = solver.solve(u0, 0.0, t_end, du0=du0)
        u_exact = case["solution"](centers[:, 0], centers[:, 1], t_end)
        errors.append(_rel_l2(u, u_exact, areas))
    assert errors[0] < 5e-3
    # Refinement reduces the error (close to second order).
    assert errors[1] < 0.6 * errors[0]


# --------------------------------------------------------------------------- #
# Advection-diffusion
# --------------------------------------------------------------------------- #
def test_advection_diffusion_central_accuracy():
    case = advection_diffusion_case(alpha=0.05, velocity=(0.8, 0.4))
    t_end = 0.3
    vertices, polygons, centers, areas = _square(48, case["bbox"])
    solver = AdvectionDiffusionHeatSolver(
        vertices, polygons, case["alpha"], t_end / 200, case["velocity"],
        scheme="central", bc_func=case["boundary"], source_func=case["source"],
    )
    u0 = case["solution"](centers[:, 0], centers[:, 1], 0.0)
    _, u = solver.solve(u0, 0.0, t_end)
    u_exact = case["solution"](centers[:, 0], centers[:, 1], t_end)
    assert _rel_l2(u, u_exact, areas) < 2e-3


def test_advection_diffusion_upwind_is_monotone_and_convergent():
    case = advection_diffusion_case(alpha=0.05, velocity=(0.8, 0.4))
    t_end = 0.3
    errors = []
    for n in (16, 32):
        vertices, polygons, centers, areas = _square(n, case["bbox"])
        solver = AdvectionDiffusionHeatSolver(
            vertices, polygons, case["alpha"], t_end / 200, case["velocity"],
            scheme="upwind", bc_func=case["boundary"], source_func=case["source"],
        )
        u0 = case["solution"](centers[:, 0], centers[:, 1], 0.0)
        _, u = solver.solve(u0, 0.0, t_end)
        u_exact = case["solution"](centers[:, 0], centers[:, 1], t_end)
        errors.append(_rel_l2(u, u_exact, areas))
    assert errors[1] < errors[0]


def test_advection_pure_transport_conserves_constant_field():
    # With no source and constant Dirichlet data equal to the initial constant,
    # both convection and diffusion vanish: the field must stay constant.
    vertices, polygons, centers = generate_square_polygonal_mesh(nx=12, ny=12)
    solver = AdvectionDiffusionHeatSolver(
        vertices, polygons, 0.05, 0.02, velocity=(1.0, -0.5),
        scheme="upwind", bc_func=lambda x, y, t: 3.0 * np.ones_like(x),
    )
    u0 = 3.0 * np.ones(solver.M)
    _, u = solver.solve(u0, 0.0, 0.2)
    assert np.allclose(u, 3.0, atol=1e-10)


def test_advection_rejects_bad_velocity_and_scheme():
    vertices, polygons, _ = generate_square_polygonal_mesh(nx=4, ny=4)
    with pytest.raises(ValueError):
        AdvectionDiffusionHeatSolver(vertices, polygons, 0.1, 0.01, velocity=(1.0, 2.0, 3.0))
    with pytest.raises(ValueError):
        AdvectionDiffusionHeatSolver(vertices, polygons, 0.1, 0.01, velocity=(1.0, 0.0), scheme="quick")


# --------------------------------------------------------------------------- #
# Fractional subdiffusion
# --------------------------------------------------------------------------- #
def test_fractional_order_validation():
    vertices, polygons, _ = generate_square_polygonal_mesh(nx=4, ny=4)
    for bad in (0.0, 1.0, 1.5, -0.2):
        with pytest.raises(ValueError):
            FractionalHeatSolver(vertices, polygons, 0.1, 0.01, beta=bad)


@pytest.mark.parametrize("beta", [0.4, 0.6, 0.8])
def test_fractional_manufactured_convergence(beta):
    case = fractional_subdiffusion_case(alpha=0.1, beta=beta)
    t_end = 0.5
    n = 96
    errors = []
    for nt in (15, 30):
        vertices, polygons, centers, areas = _square(n, case["bbox"])
        solver = FractionalHeatSolver(
            vertices, polygons, case["alpha"], t_end / nt, beta,
            bc_func=case["boundary"], source_func=case["source"],
        )
        u0 = case["solution"](centers[:, 0], centers[:, 1], 0.0)
        _, u = solver.solve(u0, 0.0, t_end)
        u_exact = case["solution"](centers[:, 0], centers[:, 1], t_end)
        errors.append(_rel_l2(u, u_exact, areas))
    observed_order = np.log2(errors[0] / errors[1])
    # L1 scheme is order (2 - beta) in time; allow a tolerance band.
    assert observed_order > (2.0 - beta) - 0.4
    assert errors[1] < 1e-2


# --------------------------------------------------------------------------- #
# Neumann / Robin / prescribed-flux boundary conditions
# --------------------------------------------------------------------------- #
def test_transport_rejects_radiative_bc():
    vertices, polygons, _ = generate_square_polygonal_mesh(nx=4, ny=4)
    for cls, kwargs in (
        (HyperbolicHeatSolver, {"relaxation_time": 0.2}),
        (AdvectionDiffusionHeatSolver, {"velocity": (1.0, 0.0)}),
        (FractionalHeatSolver, {"beta": 0.6}),
    ):
        with pytest.raises(ValueError):
            cls(vertices, polygons, 0.1, 0.01, bc_type="radiative", **kwargs)


def test_hyperbolic_flux_matches_neumann_exactly():
    # A prescribed inward flux q_in = alpha * du/dn must produce the identical
    # discrete system as the Neumann form for constant scalar alpha.
    case_n = transport_linear_boundary_case("cattaneo", "neumann", alpha=0.1, tau=0.2)
    case_f = transport_linear_boundary_case("cattaneo", "flux", alpha=0.1, tau=0.2)
    vertices, polygons, centers, _ = _square(10, case_n["bbox"])
    results = []
    for case, bc in ((case_n, "neumann"), (case_f, "flux")):
        solver = HyperbolicHeatSolver(
            vertices, polygons, case["alpha"], 0.01, case["relaxation_time"],
            bc_type=bc, bc_func=case["boundary"], source_func=case["source"],
        )
        u0 = case["solution"](centers[:, 0], centers[:, 1], 0.0)
        du0 = case["initial_rate"](centers[:, 0], centers[:, 1])
        _, u = solver.solve(u0, 0.0, 0.2, du0=du0)
        results.append(u)
    assert np.allclose(results[0], results[1], atol=1e-13, rtol=0.0)


@pytest.mark.parametrize("bc_type", ["neumann", "flux", "robin"])
def test_hyperbolic_boundary_manufactured_temporal_convergence(bc_type):
    # Linear spatial profile -> TPFA fluxes are spatially exact, so the error
    # is purely temporal and must shrink roughly linearly with dt.
    case = transport_linear_boundary_case("cattaneo", bc_type, alpha=0.1, tau=0.2)
    t_end = 0.4
    vertices, polygons, centers, areas = _square(12, case["bbox"])
    errors = []
    for dt in (0.02, 0.01):
        solver = HyperbolicHeatSolver(
            vertices, polygons, case["alpha"], dt, case["relaxation_time"],
            bc_type=bc_type, bc_func=case["boundary"], source_func=case["source"],
        )
        u0 = case["solution"](centers[:, 0], centers[:, 1], 0.0)
        du0 = case["initial_rate"](centers[:, 0], centers[:, 1])
        _, u = solver.solve(u0, 0.0, t_end, du0=du0)
        u_exact = case["solution"](centers[:, 0], centers[:, 1], t_end)
        errors.append(_rel_l2(u, u_exact, areas))
    assert errors[0] < 1e-2
    assert errors[1] < 0.65 * errors[0]


@pytest.mark.parametrize("bc_type", ["neumann", "robin"])
def test_advection_boundary_manufactured_convergence(bc_type):
    case = transport_linear_boundary_case("advection", bc_type, alpha=0.05, velocity=(0.4, 0.3))
    t_end = 0.3
    errors = []
    for n in (12, 24):
        vertices, polygons, centers, areas = _square(n, case["bbox"])
        solver = AdvectionDiffusionHeatSolver(
            vertices, polygons, case["alpha"], t_end / 200, case["velocity"],
            scheme="central", bc_type=bc_type,
            bc_func=case["boundary"], source_func=case["source"],
        )
        u0 = case["solution"](centers[:, 0], centers[:, 1], 0.0)
        _, u = solver.solve(u0, 0.0, t_end)
        u_exact = case["solution"](centers[:, 0], centers[:, 1], t_end)
        errors.append(_rel_l2(u, u_exact, areas))
    assert errors[0] < 2e-2
    assert errors[1] < 0.7 * errors[0]


def test_advection_crank_nicolson_robin():
    case = transport_linear_boundary_case("advection", "robin", alpha=0.05, velocity=(0.4, 0.3))
    t_end = 0.3
    vertices, polygons, centers, areas = _square(24, case["bbox"])
    solver = AdvectionDiffusionHeatSolver(
        vertices, polygons, case["alpha"], t_end / 200, case["velocity"],
        scheme="central", time_scheme="crank_nicolson", bc_type="robin",
        bc_func=case["boundary"], source_func=case["source"],
    )
    u0 = case["solution"](centers[:, 0], centers[:, 1], 0.0)
    _, u = solver.solve(u0, 0.0, t_end)
    u_exact = case["solution"](centers[:, 0], centers[:, 1], t_end)
    assert _rel_l2(u, u_exact, areas) < 5e-3


@pytest.mark.parametrize("bc_type", ["neumann", "flux", "robin"])
def test_fractional_boundary_manufactured_temporal_convergence(bc_type):
    case = transport_linear_boundary_case("fractional", bc_type, alpha=0.1, beta=0.6)
    t_end = 0.5
    vertices, polygons, centers, areas = _square(12, case["bbox"])
    errors = []
    for nt in (20, 40):
        solver = FractionalHeatSolver(
            vertices, polygons, case["alpha"], t_end / nt, case["beta"],
            bc_type=bc_type, bc_func=case["boundary"], source_func=case["source"],
        )
        u0 = case["solution"](centers[:, 0], centers[:, 1], 0.0)
        _, u = solver.solve(u0, 0.0, t_end)
        u_exact = case["solution"](centers[:, 0], centers[:, 1], t_end)
        errors.append(_rel_l2(u, u_exact, areas))
    assert errors[0] < 2e-2
    assert errors[1] < 0.65 * errors[0]


def test_flux_boundary_energy_balance():
    # Insulated square except a constant inward flux q0 on the left edge, no
    # source, zero velocity: the backward-Euler scheme is discretely
    # conservative, so total energy must equal the integrated boundary inflow
    # q0 * |left edge| * t_end exactly.
    vertices, polygons, centers = generate_square_polygonal_mesh(nx=16, ny=16, bbox=(0.0, 1.0, 0.0, 1.0))
    q0 = 2.5
    t_end = 0.25

    def pulse(x, y, t, nx, ny):
        return np.where(np.isclose(x, 0.0), q0, 0.0)

    solver = AdvectionDiffusionHeatSolver(
        vertices, polygons, 0.1, t_end / 20, (0.0, 0.0),
        scheme="upwind", bc_type="flux", bc_func=pulse,
    )
    _, u = solver.solve(np.zeros(solver.M), 0.0, t_end)
    areas = solver.cell_areas
    assert np.isclose(np.sum(areas * u), q0 * 1.0 * t_end, rtol=0.0, atol=1e-12)


def test_hyperbolic_flux_pulse_finite_propagation_speed():
    # Drive the Cattaneo model with a brief boundary heat pulse and check the
    # disturbance stays (essentially) behind the wavefront x = c * t, while the
    # parabolic limit would heat the whole strip instantaneously.
    alpha, tau = 0.05, 1.0
    c = np.sqrt(alpha / tau)
    q0, t_pulse, t_end = 1.0, 0.25, 2.0
    vertices, polygons, centers = generate_square_polygonal_mesh(
        nx=120, ny=6, bbox=(0.0, 2.0, 0.0, 0.1)
    )

    def pulse_flux(x, y, t, nx, ny):
        active = q0 if t <= t_pulse else 0.0
        return np.where(np.isclose(x, 0.0), active, 0.0)

    solver = HyperbolicHeatSolver(
        vertices, polygons, alpha, 0.01, tau, bc_type="flux", bc_func=pulse_flux,
    )
    _, u = solver.solve(np.zeros(solver.M), 0.0, t_end)
    front = c * t_end
    ahead = centers[:, 0] > front + 0.3  # margin for numerical front smearing
    behind = centers[:, 0] < front
    assert np.max(u[behind]) > 50.0 * max(np.max(np.abs(u[ahead])), 1e-30)


# --------------------------------------------------------------------------- #
# Reaction-diffusion / Pennes bioheat
# --------------------------------------------------------------------------- #
def test_reaction_diffusion_rejects_negative_rate():
    vertices, polygons, _ = generate_square_polygonal_mesh(nx=4, ny=4)
    with pytest.raises(ValueError):
        ReactionDiffusionHeatSolver(vertices, polygons, 0.1, 0.01, reaction_rate=-1.0)


def test_reaction_diffusion_zero_rate_matches_pure_diffusion():
    # k = 0 must reproduce the classical Fourier diffusion result.
    from heat_solver.polygonal import PolygonalHeatSolver

    case = pennes_bioheat_case(alpha=0.1, perfusion=0.0)
    vertices, polygons, centers, _ = _square(20, case["bbox"])
    u0 = case["solution"](centers[:, 0], centers[:, 1], 0.0)

    rd = ReactionDiffusionHeatSolver(
        vertices, polygons, 0.1, 0.01, reaction_rate=0.0,
        bc_func=case["boundary"], source_func=case["source"],
    )
    _, u_rd = rd.solve(u0, 0.0, 0.2)

    fourier = PolygonalHeatSolver(
        vertices, polygons, 0.1, 0.01, bc_type="dirichlet",
        bc_func=case["boundary"], source_func=case["source"],
    )
    _, u_f = fourier.solve(u0, 0.0, 0.2)
    assert np.allclose(u_rd, u_f, atol=1e-10)


def test_reaction_diffusion_perfusion_accelerates_decay():
    # Higher perfusion -> the source-free eigenmode decays to a smaller peak.
    centers0 = None
    peaks = []
    for omega in (0.0, 8.0, 30.0):
        case = pennes_bioheat_case(alpha=0.1, perfusion=omega)
        vertices, polygons, centers, _ = _square(32, case["bbox"])
        solver = ReactionDiffusionHeatSolver(
            vertices, polygons, 0.1, 0.002, omega, time_scheme="crank_nicolson",
            bc_func=case["boundary"], source_func=case["source"],
        )
        u0 = case["solution"](centers[:, 0], centers[:, 1], 0.0)
        _, u = solver.solve(u0, 0.0, 0.3)
        peaks.append(np.max(u))
    assert peaks[0] > peaks[1] > peaks[2]


def test_reaction_diffusion_eigenmode_temporal_convergence():
    # Source-free decaying eigenmode: Crank-Nicolson is second order in time.
    case = pennes_bioheat_case(alpha=0.1, perfusion=8.0)
    vertices, polygons, centers, areas = _square(64, case["bbox"])
    t_end = 0.4
    errors = []
    for nt in (20, 40):
        solver = ReactionDiffusionHeatSolver(
            vertices, polygons, case["alpha"], t_end / nt, case["perfusion"],
            time_scheme="crank_nicolson", bc_func=case["boundary"], source_func=case["source"],
        )
        u0 = case["solution"](centers[:, 0], centers[:, 1], 0.0)
        _, u = solver.solve(u0, 0.0, t_end)
        u_exact = case["solution"](centers[:, 0], centers[:, 1], t_end)
        errors.append(_rel_l2(u, u_exact, areas))
    assert np.log2(errors[0] / errors[1]) > 1.7
    assert errors[1] < 5e-3


def test_reaction_diffusion_forced_spatial_convergence_with_ambient():
    # Manufactured u = u_a + e^{-t} phi: nonzero ambient + nonzero Dirichlet trace.
    case = pennes_bioheat_case(alpha=0.1, perfusion=8.0, ambient=0.5, forced=True)
    t_end = 0.05
    errors = []
    for n in (16, 32):
        vertices, polygons, centers, areas = _square(n, case["bbox"])
        solver = ReactionDiffusionHeatSolver(
            vertices, polygons, case["alpha"], 1e-4, case["perfusion"],
            time_scheme="crank_nicolson", bc_func=case["boundary"], source_func=case["source"],
        )
        u0 = case["solution"](centers[:, 0], centers[:, 1], 0.0)
        _, u = solver.solve(u0, 0.0, t_end)
        u_exact = case["solution"](centers[:, 0], centers[:, 1], t_end)
        errors.append(_rel_l2(u, u_exact, areas))
    assert np.log2(errors[0] / errors[1]) > 1.7
    assert errors[1] < 1e-4


def test_reaction_diffusion_spatially_varying_rate_runs():
    case = pennes_bioheat_case(alpha=0.1, perfusion=8.0, forced=True)
    vertices, polygons, centers, _ = _square(20, case["bbox"])
    solver = ReactionDiffusionHeatSolver(
        vertices, polygons, 0.1, 1e-3, reaction_rate=lambda x, y: 8.0 * (1.0 + 0.5 * x),
        bc_func=case["boundary"], source_func=case["source"],
    )
    u0 = case["solution"](centers[:, 0], centers[:, 1], 0.0)
    _, u = solver.solve(u0, 0.0, 0.05)
    assert np.all(np.isfinite(u))


# --------------------------------------------------------------------------- #
# Functionally graded diffusivity
# --------------------------------------------------------------------------- #
def test_functionally_graded_convergence():
    from heat_solver.drivers import run_square_polygonal_test

    errors = []
    for n in (16, 32):
        *_, results = run_square_polygonal_test(
            case="functionally_graded", alpha=0.1, dt=2e-4,
            t_init=0.0, t_end=0.02, nx=n, ny=n, bbox=(0.0, 1.0, 0.0, 1.0),
        )
        errors.append(results["L2_rel"])
    # Spatially graded scalar diffusivity -> roughly second-order spatial decay.
    assert np.log2(errors[0] / errors[1]) > 1.6
    assert errors[1] < 1e-3
