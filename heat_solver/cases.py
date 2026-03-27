import numpy as np
from scipy.special import erf

from .phase_change import ApparentHeatCapacityModel


def heat_kernel(x, y, t, alpha):
    return (1.0 / (4.0 * np.pi * alpha * t)) * np.exp(-(x * x + y * y) / (4.0 * alpha * t))


def sine_mode_solution(x, y, t, alpha):
    decay = np.exp(-2.0 * np.pi**2 * alpha * t)
    return np.sin(np.pi * x) * np.sin(np.pi * y) * decay


def harmonic_polynomial_solution(x, y, t, alpha):
    del t, alpha
    return x**2 - y**2 + 0.25 * x * y + x - 0.5 * y


def source_driven_sine_solution(x, y, t, alpha):
    del alpha
    return np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y)


def source_driven_sine_source(x, y, t, alpha):
    solution = source_driven_sine_solution(x, y, t, alpha)
    return (2.0 * np.pi**2 * alpha - 1.0) * solution


def laplace_equation_solution(x, y, t, alpha):
    del t, alpha
    # Steady state Laplace's equation solution
    # Domain: [0, 1] x [0, 1]
    # BC: u(x, 0) = 0, u(x, 1) = sin(pi * x), u(0, y) = 0, u(1, y) = 0
    # True solution is u(x,y) = sin(pi * x) * sinh(pi * y) / sinh(pi)
    return np.sin(np.pi * x) * np.sinh(np.pi * y) / np.sinh(np.pi)


def steady_linear_boundary_solution(x, y, t, alpha):
    del t, alpha
    return 1.0 + 0.75 * x - 0.5 * y


def steady_linear_neumann_bc(x, y, t, nx, ny, alpha):
    del x, y, t, alpha
    return 0.75 * nx - 0.5 * ny


def steady_linear_robin_bc(x, y, t, nx, ny, alpha, beta=2.0):
    solution = steady_linear_boundary_solution(x, y, t, alpha)
    value = alpha * (0.75 * nx - 0.5 * ny) + beta * solution
    return beta * np.ones_like(solution, dtype=float), value


def linear_patch_solution():
    """1. Exact Steady-State Linear Patch Test"""
    def solution(x, y, t):
        return 2.0 * x - 3.0 * y + 1.0
    return solution

def hot_block_solution(alpha, x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5):
    """2. Discontinuous Hot Block (Maximum Principle Test)"""
    def solution(x, y, t):
        if t <= 0.0:
            u = np.zeros_like(x)
            mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
            u[mask] = 1.0
            return u
        
        denom = np.sqrt(4.0 * alpha * t)
        x_term = erf((x - x_min) / denom) - erf((x - x_max) / denom)
        y_term = erf((y - y_min) / denom) - erf((y - y_max) / denom)
        return 0.25 * x_term * y_term
    return solution

def off_axis_plane_wave_solution(alpha, k=np.pi, theta=np.pi/8.0):
    """3. Off-Axis Plane Wave (Rotational Invariance Test)"""
    def solution(x, y, t):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        spatial = np.sin(k * (x * cos_t + y * sin_t))
        decay = np.exp(-alpha * (k**2) * t)
        return spatial * decay
    return solution

def nyquist_oscillation_solution(alpha, m=12, n=12):
    """4. High-Frequency Nyquist Oscillations (Grid limit test)"""
    def solution(x, y, t):
        spatial = np.cos(m * np.pi * x) * np.cos(n * np.pi * y)
        decay = np.exp(-alpha * (np.pi**2) * (m**2 + n**2) * t)
        return spatial * decay
    return solution

def point_source_solution(alpha):
    """5. Single-Cell Point Source (Heat Kernel)"""
    def solution(x, y, t):
        if t <= 0.0:
            # Note: For automated drivers evaluating at t=0, we approximate
            # the delta by dropping 1.0 at the origin node. For true discrete
            # delta behavior, start the driver at t_init > 0.
            u = np.zeros_like(x)
            r2 = x**2 + y**2
            u[r2 == r2.min()] = 1.0
            return u
        return (1.0 / (4.0 * np.pi * alpha * t)) * np.exp(-(x**2 + y**2) / (4.0 * alpha * t))
    return solution

def anisotropic_heat_kernel_solution(alpha):
    """Anisotropic Heat Kernel Test"""
    alpha = np.asarray(alpha, dtype=float)
    if alpha.ndim == 0 or (alpha.ndim == 1 and alpha.size == 1):
        alpha = np.array([[float(alpha), 0.0], [0.0, float(alpha)]])
    elif alpha.shape != (2, 2):
        raise ValueError("anisotropic_heat_kernel_solution requires a scalar or a 2x2 matrix for alpha")
    det = np.linalg.det(alpha)
    inv_alpha = np.linalg.inv(alpha)
    def solution(x, y, t):
        if t <= 0.0:
            u = np.zeros_like(x, dtype=float)
            r2 = x**2 + y**2
            u[r2 == r2.min()] = 1.0
            return u
        rT_inv_r = inv_alpha[0, 0] * x**2 + (inv_alpha[0, 1] + inv_alpha[1, 0]) * x * y + inv_alpha[1, 1] * y**2
        return (1.0 / (4.0 * np.pi * t * np.sqrt(det))) * np.exp(-rT_inv_r / (4.0 * t))
    return solution

def green_function_source_solution(alpha, epsilon=0.01):
    """Exact convolution of a constant regularized point source over time."""
    from scipy.special import exp1
    def solution(x, y, t):
        if t <= 0.0:
            return np.zeros_like(x)
        r2 = x**2 + y**2
        u = np.zeros_like(r2)
        mask = r2 > 1e-12
        # For r > 0
        u[mask] = (1.0 / (4.0 * np.pi * alpha)) * (exp1(r2[mask] / (4.0 * alpha * t + epsilon)) - exp1(r2[mask] / epsilon))
        # For r ~ 0 (limit log expression)
        u[~mask] = (1.0 / (4.0 * np.pi * alpha)) * np.log((4.0 * alpha * t + epsilon) / epsilon)
        return u
    return solution


def green_function_source_source(alpha, epsilon=0.01):
    """Regularized spatial Dirac delta function."""
    del alpha
    def source(x, y, t):
        del t
        r2 = x**2 + y**2
        return (1.0 / (np.pi * epsilon)) * np.exp(-r2 / epsilon)
    return source


def stefan_apparent_capacity_solution(x, y, t, amplitude=0.8, interface_width=0.18, speed=0.55, x0=-0.35):
    del y
    z = (x - x0 - speed * t) / interface_width
    return amplitude * np.tanh(z)


def stefan_apparent_capacity_source(alpha, phase_change_model, amplitude=0.8, interface_width=0.18, speed=0.55, x0=-0.35):
    def source(x, y, t):
        del y
        z = (x - x0 - speed * t) / interface_width
        tanh_z = np.tanh(z)
        sech2_z = 1.0 - tanh_z**2
        temperature = amplitude * tanh_z
        dT_dt = -(amplitude * speed / interface_width) * sech2_z
        d2T_dx2 = -(2.0 * amplitude / (interface_width**2)) * tanh_z * sech2_z
        capacity = phase_change_model.effective_heat_capacity(temperature)
        return capacity * dT_dt - alpha * d2T_dx2

    return source


def temp_dependent_diffusivity_solution(x, y, t):
    del y
    return np.exp(-t) * (1.0 + 0.25 * x)


def temp_dependent_diffusivity_alpha(x, y, temperature):
    del y
    base = 0.12
    beta = 0.6
    gamma = 0.35
    return base * (1.0 + gamma * x) * (1.0 + beta * temperature)


def temp_dependent_diffusivity_source(x, y, t):
    del y
    temperature = temp_dependent_diffusivity_solution(x, 0.0, t)
    tx = 0.25 * np.exp(-t) * np.ones_like(x, dtype=float)
    alpha = temp_dependent_diffusivity_alpha(x, 0.0, temperature)
    base = 0.12
    beta = 0.6
    gamma = 0.35
    dalpha_dx = base * (gamma * (1.0 + beta * temperature) + (1.0 + gamma * x) * beta * tx)
    div_flux = dalpha_dx * tx
    return -temperature - div_flux


def radiative_manufactured_solution(x, y, t):
    amplitude = 0.25 * np.exp(-t)
    return 1.5 + amplitude * np.sin(np.pi * x) * np.sin(np.pi * y)


def radiative_manufactured_source(x, y, t, alpha):
    amp = 0.25 * np.exp(-t)
    spatial = np.sin(np.pi * x) * np.sin(np.pi * y)
    dT_dt = -amp * spatial
    laplacian = -2.0 * (np.pi**2) * amp * spatial
    return dT_dt - alpha * laplacian


def radiative_manufactured_bc(x, y, t, nx, ny, alpha):
    epsilon = np.ones_like(x, dtype=float)
    sigma = 5.0 * np.ones_like(x, dtype=float)
    temperature = radiative_manufactured_solution(x, y, t)
    grad_x = 0.25 * np.exp(-t) * np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
    grad_y = 0.25 * np.exp(-t) * np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
    dT_dn = grad_x * nx + grad_y * ny
    t_inf_pow4 = np.maximum(temperature**4 + (alpha / sigma) * dT_dn, 1e-12)
    t_inf = t_inf_pow4 ** 0.25
    return {"epsilon": epsilon, "sigma": sigma, "t_inf": t_inf}


def get_analytical_case(case="heat_kernel", alpha=0.1, t_end=0.15):
    if case == "heat_kernel":
        L = 4.0 * np.sqrt(4.0 * alpha * t_end)
        return {
            "name": "Heat Kernel",
            "bbox": (-L, L, -L, L),
            "solution": lambda x, y, t: heat_kernel(x, y, t, alpha),
        }
    if case == "sine_mode":
        return {
            "name": "Sine Mode",
            "bbox": (-1.0, 1.0, -1.0, 1.0),
            "solution": lambda x, y, t: sine_mode_solution(x, y, t, alpha),
        }
    if case == "harmonic_polynomial":
        return {
            "name": "Harmonic Polynomial",
            "bbox": (-1.0, 1.0, -1.0, 1.0),
            "solution": lambda x, y, t: harmonic_polynomial_solution(x, y, t, alpha),
        }
    if case == "source_driven_sine":
        return {
            "name": "Source-Driven Sine",
            "bbox": (-1.0, 1.0, -1.0, 1.0),
            "solution": lambda x, y, t: source_driven_sine_solution(x, y, t, alpha),
            "source": lambda x, y, t: source_driven_sine_source(x, y, t, alpha),
        }
    if case == "steady_linear_neumann":
        return {
            "name": "Steady Linear Neumann",
            "bbox": (0.0, 1.0, 0.0, 1.0),
            "solution": lambda x, y, t: steady_linear_boundary_solution(x, y, t, alpha),
            "bc_type": "neumann",
            "boundary": lambda x, y, t, nx, ny: steady_linear_neumann_bc(x, y, t, nx, ny, alpha),
        }
    if case == "steady_linear_robin":
        return {
            "name": "Steady Linear Robin",
            "bbox": (0.0, 1.0, 0.0, 1.0),
            "solution": lambda x, y, t: steady_linear_boundary_solution(x, y, t, alpha),
            "bc_type": "robin",
            "boundary": lambda x, y, t, nx, ny: steady_linear_robin_bc(x, y, t, nx, ny, alpha),
        }
    if case == "linear_patch":
        return {
            "name": "Linear Patch",
            "bbox": (-1.0, 1.0, -1.0, 1.0),
            "solution": linear_patch_solution()
        }
    elif case == "hot_block":
        return {
            "name": "Hot Block",
            "bbox": (-1.0, 1.0, -1.0, 1.0),
            "solution": hot_block_solution(alpha)
        }
    elif case == "off_axis_wave":
        return {
            "name": "Off-Axis Plane Wave",
            "bbox": (-1.0, 1.0, -1.0, 1.0),
            "solution": off_axis_plane_wave_solution(alpha)
        }
    elif case == "nyquist_oscillations":
        return {
            "name": "Nyquist Oscillations",
            "bbox": (-1.0, 1.0, -1.0, 1.0),
            "solution": nyquist_oscillation_solution(alpha)
        }
    elif case == "point_source":
        return {
            "name": "Point Source",
            "bbox": (-1.0, 1.0, -1.0, 1.0),
            "solution": point_source_solution(alpha)
        }
    elif case == "green_function_source":
        return {
            "name": "Green Function Source",
            "bbox": (-1.0, 1.0, -1.0, 1.0),
            "solution": green_function_source_solution(alpha),
            "source": green_function_source_source(alpha),
        }
    elif case == "laplace_equation":
        return {
            "name": "Laplace Equation",
            "bbox": (0.0, 1.0, 0.0, 1.0),
            "solution": lambda x, y, t: laplace_equation_solution(x, y, t, alpha),
        }
    elif case == "anisotropic_heat_kernel":
        if np.isscalar(alpha) or np.asarray(alpha).ndim == 0:
            max_alpha = float(alpha)
        else:
            max_alpha = np.max(np.linalg.eigvals(alpha))
        L = 4.0 * np.sqrt(4.0 * max_alpha * t_end)
        return {
            "name": "Anisotropic Heat Kernel",
            "bbox": (-L, L, -L, L),
            "solution": anisotropic_heat_kernel_solution(alpha),
        }
    elif case == "stefan_apparent_capacity":
        phase_change_model = ApparentHeatCapacityModel(
            solidus_temperature=-0.05,
            liquidus_temperature=0.05,
            latent_heat=6.0,
            specific_heat=1.0,
        )
        return {
            "name": "Stefan Apparent Capacity",
            "bbox": (-1.0, 1.0, -1.0, 1.0),
            "solution": lambda x, y, t: stefan_apparent_capacity_solution(x, y, t),
            "source": stefan_apparent_capacity_source(alpha, phase_change_model),
            "phase_change_model": phase_change_model,
            "phase_change_options": {"max_iters": 40, "tol": 1e-10, "relaxation": 1.0},
            "polygonal_only": True,
        }
    elif case == "temperature_dependent_diffusivity":
        return {
            "name": "Temperature-Dependent Diffusivity",
            "bbox": (0.0, 1.0, 0.0, 1.0),
            "solution": temp_dependent_diffusivity_solution,
            "source": temp_dependent_diffusivity_source,
            "alpha": temp_dependent_diffusivity_alpha,
            "temperature_dependent_diffusivity": True,
            "nonlinear_options": {"max_iters": 35, "tol": 1e-10, "relaxation": 0.9},
            "polygonal_only": True,
        }
    elif case == "radiative_manufactured":
        return {
            "name": "Radiative Manufactured",
            "bbox": (0.0, 1.0, 0.0, 1.0),
            "solution": radiative_manufactured_solution,
            "source": lambda x, y, t: radiative_manufactured_source(x, y, t, alpha),
            "bc_type": "radiative",
            "boundary": lambda x, y, t, nx, ny: radiative_manufactured_bc(x, y, t, nx, ny, alpha),
            "nonlinear_options": {"max_iters": 35, "tol": 1e-10, "relaxation": 0.9},
            "polygonal_only": True,
        }
    raise ValueError(
        "Unknown analytical case "
        f"'{case}'. Available cases: heat_kernel, sine_mode, harmonic_polynomial, "
        "source_driven_sine, steady_linear_neumann, steady_linear_robin, "
        "linear_patch, hot_block, off_axis_wave, nyquist_oscillations, point_source, "
        "green_function_source, laplace_equation, anisotropic_heat_kernel, stefan_apparent_capacity, "
        "temperature_dependent_diffusivity, radiative_manufactured"
    )


def build_error_report(weights, diff, u_exact, t_final, bbox, case, **extra):
    l2 = np.sqrt(np.sum(weights * diff**2))
    l2_ref = np.sqrt(np.sum(weights * u_exact**2)) + 1e-16
    linf = np.max(np.abs(diff))
    return {
        "case": case,
        "L2": l2,
        "L2_rel": l2 / l2_ref,
        "Linf": linf,
        "Linf_rel": linf / (np.max(np.abs(u_exact)) + 1e-16),
        "t_final": t_final,
        "bbox": bbox,
        **extra,
    }
