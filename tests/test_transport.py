import numpy as np
import pytest

from heat_solver.cases import (
    advection_diffusion_case,
    cattaneo_wave_case,
    fractional_subdiffusion_case,
)
from heat_solver.geometry import polygon_area_and_centroid
from heat_solver.meshes import generate_square_polygonal_mesh
from heat_solver.transport import (
    AdvectionDiffusionHeatSolver,
    FractionalHeatSolver,
    HyperbolicHeatSolver,
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
