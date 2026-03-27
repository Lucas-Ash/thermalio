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

def test_get_analytical_case_unknown():
    with pytest.raises(ValueError, match="Unknown analytical case"):
        get_analytical_case("unknown_case_name")
