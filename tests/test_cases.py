import numpy as np
import pytest
from heat_solver.cases import get_analytical_case

def test_get_analytical_case_heat_kernel():
    case = get_analytical_case("heat_kernel", alpha=0.1, t_end=0.05)
    assert case["name"] == "Heat Kernel"
    assert len(case["bbox"]) == 4
    
    solution = case["solution"]
    u_val = solution(0.0, 0.0, 0.05)
    # Peak heat kernel 1 / (4 * pi * alpha * t)
    expected = 1.0 / (4.0 * np.pi * 0.1 * 0.05)
    assert np.isclose(u_val, expected)

def test_get_analytical_case_anisotropic_heat_kernel():
    alpha = [[0.2, 0.0], [0.0, 0.1]]
    case = get_analytical_case("anisotropic_heat_kernel", alpha=alpha, t_end=0.1)
    assert case["name"] == "Anisotropic Heat Kernel"
    
    solution = case["solution"]
    # Check t=0 behavior (should return 1 at origin, 0 elsewhere)
    u_init = solution(np.array([0.0, 1.0]), np.array([0.0, 0.0]), 0)
    assert np.allclose(u_init, [1.0, 0.0])
    
    # Check t>0 behavior
    u_val = solution(0.0, 0.0, 0.1)
    det = 0.2 * 0.1 - 0.0
    expected = 1.0 / (4.0 * np.pi * 0.1 * np.sqrt(det))
    assert np.isclose(u_val, expected)

def test_get_analytical_case_steady_linear_neumann():
    case = get_analytical_case("steady_linear_neumann", alpha=0.1, t_end=1.0)
    assert case["bc_type"] == "neumann"
    boundary = case["boundary"]
    # 0.75 * nx - 0.5 * ny
    nx, ny = 1.0, 0.0
    bc_val = boundary(1.0, 0.5, 1.0, nx, ny)
    assert np.isclose(bc_val, 0.75)


def test_get_analytical_case_stefan_apparent_capacity():
    case = get_analytical_case("stefan_apparent_capacity", alpha=0.08, t_end=0.05)
    assert case["name"] == "Stefan Apparent Capacity"
    assert "phase_change_model" in case
    assert case["phase_change_model"] is not None
    x = np.array([-0.5, 0.0, 0.5])
    y = np.zeros_like(x)
    u = case["solution"](x, y, 0.02)
    q = case["source"](x, y, 0.02)
    assert u.shape == x.shape
    assert q.shape == x.shape


@pytest.mark.parametrize(
    "case_name, extra_keys",
    [
        ("hyperbolic_stefan_apparent_capacity", ("relaxation_time", "initial_rate")),
        ("fractional_stefan_apparent_capacity", ("beta",)),
    ],
)
def test_get_analytical_case_nonfourier_stefan(case_name, extra_keys):
    case = get_analytical_case(case_name, alpha=0.08, t_end=0.05)
    assert "Stefan Apparent Capacity" in case["name"]
    assert "phase_change_model" in case
    assert "phase_change_options" in case
    assert case["phase_change_model"] is not None
    for key in extra_keys:
        assert key in case

    if case_name.startswith("hyperbolic"):
        x = np.array([-0.42, -0.35, -0.28])
    else:
        x = np.array([0.40, 0.45, 0.50])
    y = np.zeros_like(x)
    u = case["solution"](x, y, 0.02)
    q = case["source"](x, y, 0.02)
    assert u.shape == x.shape
    assert q.shape == x.shape
    assert np.any(case["phase_change_model"].effective_heat_capacity(u) > 1.0)


def test_get_analytical_case_temperature_dependent_diffusivity():
    case = get_analytical_case("temperature_dependent_diffusivity", alpha=0.12, t_end=0.05)
    assert case["name"] == "Temperature-Dependent Diffusivity"
    assert case["temperature_dependent_diffusivity"] is True
    x = np.array([0.0, 0.5, 1.0])
    y = np.array([0.2, 0.2, 0.2])
    u = case["solution"](x, y, 0.02)
    q = case["source"](x, y, 0.02)
    k = case["alpha"](x, y, u)
    assert u.shape == x.shape
    assert q.shape == x.shape
    assert np.all(k > 0.0)


def test_get_analytical_case_radiative_manufactured():
    case = get_analytical_case("radiative_manufactured", alpha=0.1, t_end=0.03)
    assert case["bc_type"] == "radiative"
    x = np.array([0.0, 1.0])
    y = np.array([0.3, 0.7])
    nx = np.array([-1.0, 1.0])
    ny = np.array([0.0, 0.0])
    bc = case["boundary"](x, y, 0.01, nx, ny)
    assert "epsilon" in bc
    assert "sigma" in bc
    assert "t_inf" in bc


def test_get_analytical_case_functionally_graded():
    case = get_analytical_case("functionally_graded", alpha=0.1, t_end=0.02)
    assert case["name"] == "Functionally Graded Diffusivity"
    alpha = case["alpha"]
    # alpha(x) = alpha0 * exp(grade * x); increases with x.
    assert callable(alpha)
    assert np.isclose(alpha(0.0, 0.0), 0.1)
    assert alpha(1.0, 0.0) > alpha(0.0, 0.0)
    x = np.array([0.25, 0.5, 0.75])
    y = np.array([0.5, 0.5, 0.5])
    assert case["solution"](x, y, 0.0).shape == x.shape
    assert case["source"](x, y, 0.01).shape == x.shape


def test_pennes_bioheat_eigenmode_is_source_free():
    from heat_solver.cases import pennes_bioheat_case

    case = pennes_bioheat_case(alpha=0.1, perfusion=8.0)
    assert case["name"] == "Pennes Bioheat"
    x = np.array([0.25, 0.5, 0.75])
    y = np.array([0.25, 0.5, 0.75])
    # Source-free eigenmode: Q == 0 and the decay rate is 2 pi^2 alpha + k.
    assert np.allclose(case["source"](x, y, 0.1), 0.0)
    decay = 2.0 * np.pi**2 * 0.1 + 8.0
    u0 = case["solution"](x, y, 0.0)
    u1 = case["solution"](x, y, 0.5)
    assert np.allclose(u1, u0 * np.exp(-decay * 0.5))


def test_pennes_bioheat_forced_boundary_is_ambient():
    from heat_solver.cases import pennes_bioheat_case

    case = pennes_bioheat_case(alpha=0.1, perfusion=8.0, ambient=0.5, forced=True)
    # The sin-mode vanishes on the unit-square boundary, so u = ambient there.
    assert np.isclose(case["solution"](0.0, 0.5, 0.3), 0.5)
    assert np.isclose(case["solution"](0.5, 0.0, 0.3), 0.5)


def test_get_analytical_case_unknown():
    with pytest.raises(ValueError, match="Unknown analytical case"):
        get_analytical_case("unknown_case_name")
