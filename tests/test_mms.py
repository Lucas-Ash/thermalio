import numpy as np
import pytest

pytest.importorskip("sympy")

from heat_solver.cases import (
    advection_diffusion_case,
    cattaneo_wave_case,
    fractional_subdiffusion_case,
    functionally_graded_source,
    get_analytical_case,
    pennes_bioheat_case,
    source_driven_sine_source,
)
from heat_solver.mms import manufactured_case

# A grid of interior points (avoid the boundary where the sin-mode vanishes) and
# two sample times.
_X, _Y = np.meshgrid(np.linspace(0.1, 0.9, 7), np.linspace(0.1, 0.9, 7))
_TIMES = (0.1, 0.37)


def _max_source_diff(mms, hand_source, times=_TIMES):
    return max(
        float(np.max(np.abs(mms["source"](_X, _Y, t) - hand_source(_X, _Y, t))))
        for t in times
    )


def test_mms_reproduces_source_driven_sine():
    alpha = 0.1
    mms = manufactured_case("exp(-t)*sin(pi*x)*sin(pi*y)", alpha=alpha, model="diffusion")
    diff = _max_source_diff(mms, lambda x, y, t: source_driven_sine_source(x, y, t, alpha))
    assert diff < 1e-13


def test_mms_reproduces_functionally_graded():
    alpha0, grade = 0.1, 0.8
    mms = manufactured_case(
        "exp(-t)*sin(pi*x)*sin(pi*y)",
        alpha=f"{alpha0}*exp({grade}*x)",
        model="diffusion",
    )
    diff = _max_source_diff(mms, lambda x, y, t: functionally_graded_source(x, y, t, alpha0, grade))
    assert diff < 1e-13


def test_mms_reproduces_cattaneo():
    case = cattaneo_wave_case(alpha=0.1, tau=0.2)
    mms = manufactured_case(
        "exp(-t)*sin(pi*x)*sin(pi*y)", alpha=0.1, model="cattaneo", tau=0.2
    )
    assert _max_source_diff(mms, case["source"]) < 1e-13
    # Initial-rate du/dt(0) = -sin(pi x) sin(pi y) must also match.
    ir_mms = mms["initial_rate"](_X, _Y)
    ir_hand = case["initial_rate"](_X, _Y)
    assert np.max(np.abs(ir_mms - ir_hand)) < 1e-13


def test_mms_reproduces_advection_diffusion():
    case = advection_diffusion_case(alpha=0.05, velocity=(0.8, 0.4))
    mms = manufactured_case(
        "exp(-t)*sin(pi*x)*sin(pi*y)", alpha=0.05,
        model="advection_diffusion", velocity=(0.8, 0.4),
    )
    assert _max_source_diff(mms, case["source"]) < 1e-13


def test_mms_reproduces_pennes_forced():
    case = pennes_bioheat_case(alpha=0.1, perfusion=8.0, ambient=0.5, forced=True)
    mms = manufactured_case(
        "0.5 + exp(-t)*sin(pi*x)*sin(pi*y)", alpha=0.1,
        model="reaction_diffusion", reaction=8.0,
    )
    assert _max_source_diff(mms, case["source"]) < 1e-13


def test_mms_reproduces_fractional_subdiffusion():
    beta = 0.6
    case = fractional_subdiffusion_case(alpha=0.1, beta=beta)
    mms = manufactured_case(
        "t**2*sin(pi*x)*sin(pi*y)", alpha=0.1, model="fractional", beta=beta
    )
    assert _max_source_diff(mms, case["source"]) < 1e-13


def test_mms_caputo_cubic_polynomial_in_time():
    # Generalized Caputo handles arbitrary integer powers (here t**3): the source
    # must equal Gamma(4)/Gamma(4-beta) t^{3-beta} phi + 2 pi^2 alpha t^3 phi.
    from scipy.special import gamma as _gamma

    beta, alpha = 0.6, 0.1
    mms = manufactured_case(
        "t**3*sin(pi*x)*sin(pi*y)", alpha=alpha, model="fractional", beta=beta
    )

    def expected(x, y, t):
        phi = np.sin(np.pi * x) * np.sin(np.pi * y)
        caputo = _gamma(4.0) / _gamma(4.0 - beta) * t ** (3.0 - beta) * phi
        return caputo + 2.0 * np.pi**2 * alpha * (t**3) * phi

    diff = _max_source_diff(mms, expected)
    assert diff < 1e-13


def test_mms_caputo_fractional_power_in_time():
    # Non-integer power t**(1+beta): D_t^beta t^{1+beta} = Gamma(2+beta) t.
    from scipy.special import gamma as _gamma

    beta, alpha = 0.5, 0.1
    mms = manufactured_case(
        f"t**(1+{beta})*sin(pi*x)*sin(pi*y)", alpha=alpha, model="fractional", beta=beta
    )

    def expected(x, y, t):
        phi = np.sin(np.pi * x) * np.sin(np.pi * y)
        caputo = _gamma(2.0 + beta) * t * phi  # Gamma(2+beta)/Gamma(2) t^1, Gamma(2)=1
        return caputo + 2.0 * np.pi**2 * alpha * (t ** (1.0 + beta)) * phi

    diff = _max_source_diff(mms, expected, times=(0.2, 0.5))
    assert diff < 1e-12


def test_mms_caputo_rejects_transcendental_time():
    import pytest as _pytest

    with _pytest.raises(ValueError, match="powers of t"):
        manufactured_case(
            "exp(-t)*sin(pi*x)*sin(pi*y)", alpha=0.1, model="fractional", beta=0.5
        )


def test_mms_neumann_boundary_matches_linear_case():
    # Steady linear u = 1 + 0.75 x - 0.5 y has du/dn = 0.75 nx - 0.5 ny.
    mms = manufactured_case(
        "1 + 0.75*x - 0.5*y", alpha=0.1, model="diffusion", bc_type="neumann"
    )
    case = get_analytical_case("steady_linear_neumann", alpha=0.1)
    nx = np.array([1.0, 0.0, -1.0, 0.0])
    ny = np.array([0.0, 1.0, 0.0, -1.0])
    x = np.full_like(nx, 0.5)
    y = np.full_like(nx, 0.5)
    got = mms["boundary"](x, y, 0.0, nx, ny)
    want = case["boundary"](x, y, 0.0, nx, ny)
    assert np.max(np.abs(got - want)) < 1e-13


def test_mms_robin_boundary_matches_linear_case():
    alpha, beta = 0.1, 2.0
    mms = manufactured_case(
        "1 + 0.75*x - 0.5*y", alpha=alpha, model="diffusion",
        bc_type="robin", robin_beta=beta,
    )
    case = get_analytical_case("steady_linear_robin", alpha=alpha)
    nx = np.array([1.0, 0.0, -1.0, 0.0])
    ny = np.array([0.0, 1.0, 0.0, -1.0])
    x = np.full_like(nx, 0.5)
    y = np.full_like(nx, 0.5)
    beta_got, val_got = mms["boundary"](x, y, 0.0, nx, ny)
    beta_want, val_want = case["boundary"](x, y, 0.0, nx, ny)
    assert np.allclose(beta_got, beta_want)
    assert np.max(np.abs(val_got - val_want)) < 1e-13


def test_mms_source_drives_observed_second_order():
    # A fresh MMS case run through the real solver should converge at ~2.
    from heat_solver.drivers import run_square_polygonal_test
    from heat_solver.geometry import polygon_area_and_centroid
    from heat_solver.meshes import generate_square_polygonal_mesh
    from heat_solver.polygonal import PolygonalHeatSolver

    mms = manufactured_case(
        "exp(-0.5*t)*sin(pi*x)*sin(pi*y)", alpha=0.2, model="diffusion"
    )
    errs = []
    for n in (20, 40):
        v, p, c = generate_square_polygonal_mesh(nx=n, ny=n, bbox=mms["bbox"])
        areas = np.array([polygon_area_and_centroid(v[poly])[0] for poly in p])
        solver = PolygonalHeatSolver(
            v, p, 0.2, 1e-4, bc_type="dirichlet",
            bc_func=mms["boundary"], source_func=mms["source"],
        )
        u0 = mms["solution"](c[:, 0], c[:, 1], 0.0)
        _, u = solver.solve(u0, 0.0, 0.02)
        ue = mms["solution"](c[:, 0], c[:, 1], 0.02)
        errs.append(float(np.sqrt(np.sum(areas * (u - ue) ** 2) / np.sum(areas * ue**2))))
    assert np.log2(errs[0] / errs[1]) > 1.7
