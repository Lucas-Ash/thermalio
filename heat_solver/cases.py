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


def cattaneo_wave_case(alpha=0.1, tau=0.2):
    """Manufactured solution for the hyperbolic (Cattaneo--Vernotte) model.

    ``tau u_tt + u_t - alpha laplacian(u) = Q`` on the unit square with
    ``u(x, y, t) = exp(-t) sin(pi x) sin(pi y)`` (homogeneous Dirichlet data).

    For this ``u``: ``u_t = -u``, ``u_tt = u``, ``laplacian(u) = -2 pi^2 u``, so
    ``Q = (tau - 1 + 2 pi^2 alpha) u`` and the initial rate is ``du/dt|_0 = -u_0``.
    """
    alpha = float(alpha)

    def phi(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def solution(x, y, t):
        return np.exp(-t) * phi(x, y)

    def source(x, y, t):
        return (tau - 1.0 + 2.0 * np.pi**2 * alpha) * solution(x, y, t)

    def initial_rate(x, y):
        return -phi(x, y)

    return {
        "name": "Cattaneo Thermal Wave",
        "bbox": (0.0, 1.0, 0.0, 1.0),
        "solution": solution,
        "source": source,
        "boundary": solution,
        "initial_rate": initial_rate,
        "relaxation_time": float(tau),
        "alpha": alpha,
    }


def advection_diffusion_case(alpha=0.05, velocity=(0.8, 0.4)):
    """Manufactured solution for ``u_t + v . grad(u) - alpha laplacian(u) = Q``.

    Uses ``u(x, y, t) = exp(-t) sin(pi x) sin(pi y)`` (homogeneous Dirichlet
    data) with a constant velocity ``v = (vx, vy)``.
    """
    alpha = float(alpha)
    vx, vy = float(velocity[0]), float(velocity[1])

    def solution(x, y, t):
        return np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y)

    def source(x, y, t):
        decay = np.exp(-t)
        s = np.sin(np.pi * x) * np.sin(np.pi * y)
        u = decay * s
        u_t = -u
        grad_x = decay * np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
        grad_y = decay * np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
        laplacian = -2.0 * np.pi**2 * u
        return u_t + vx * grad_x + vy * grad_y - alpha * laplacian

    return {
        "name": "Advection-Diffusion",
        "bbox": (0.0, 1.0, 0.0, 1.0),
        "solution": solution,
        "source": source,
        "boundary": solution,
        "velocity": (vx, vy),
        "alpha": alpha,
    }


def fractional_subdiffusion_case(alpha=0.1, beta=0.6):
    """Manufactured solution for Caputo subdiffusion ``D_t^beta u - alpha lap(u) = Q``.

    Uses ``u(x, y, t) = t^2 sin(pi x) sin(pi y)`` (homogeneous Dirichlet data,
    ``u(.,.,0) = 0``).  The Caputo derivative of ``t^2`` is
    ``Gamma(3)/Gamma(3-beta) t^{2-beta} = 2/Gamma(3-beta) t^{2-beta}``.
    """
    from scipy.special import gamma as _gamma

    alpha = float(alpha)
    beta = float(beta)

    def phi(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def solution(x, y, t):
        return (t**2) * phi(x, y)

    def source(x, y, t):
        caputo = (2.0 / _gamma(3.0 - beta)) * (t ** (2.0 - beta)) * phi(x, y)
        laplacian = -2.0 * np.pi**2 * (t**2) * phi(x, y)
        return caputo - alpha * laplacian

    return {
        "name": "Fractional Subdiffusion",
        "bbox": (0.0, 1.0, 0.0, 1.0),
        "solution": solution,
        "source": source,
        "boundary": solution,
        "beta": beta,
        "alpha": alpha,
    }


def functionally_graded_alpha(x, y, alpha0=0.1, grade=0.8):
    """Exponentially graded thermal diffusivity ``alpha(x) = alpha0 * exp(grade * x)``.

    Models a functionally graded material (e.g. a thermal-barrier coating) whose
    conductivity varies smoothly with position.
    """
    del y
    return alpha0 * np.exp(grade * np.asarray(x, dtype=float))


def functionally_graded_solution(x, y, t):
    return np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y)


def functionally_graded_source(x, y, t, alpha0=0.1, grade=0.8):
    """Source closing ``u_t - div(alpha(x) grad u) = Q`` for the graded case.

    With ``div(alpha grad u) = alpha * laplacian(u) + (d alpha/dx) * du/dx`` and
    ``d alpha/dx = grade * alpha``.
    """
    u = functionally_graded_solution(x, y, t)
    alpha = functionally_graded_alpha(x, y, alpha0, grade)
    du_dx = np.exp(-t) * np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
    laplacian = -2.0 * np.pi**2 * u
    div_flux = alpha * laplacian + grade * alpha * du_dx
    return -u - div_flux


def pennes_bioheat_case(alpha=0.1, perfusion=8.0, ambient=0.0, forced=False):
    """Manufactured solution for the Pennes bioheat / reaction-diffusion equation.

    ``u_t - alpha laplacian(u) + k (u - u_a) = q_met`` on the unit square, written
    in the solver's form ``u_t - div(alpha grad u) + k u = Q`` with
    ``Q = k u_a + q_met``.  Here ``k`` is the perfusion (reaction) rate and
    ``u_a`` the arterial/ambient temperature.

    ``forced=False`` (default) returns the *source-free* decaying eigenmode
    ``u = exp(-(2 pi^2 alpha + k) t) sin(pi x) sin(pi y)`` with ``u_a = 0`` and
    ``Q = 0``: perfusion makes the mode decay faster than pure diffusion, which
    is the physical signature of the reaction term.  ``forced=True`` returns a
    manufactured solution ``u = u_a + exp(-t) sin(pi x) sin(pi y)`` whose
    boundary trace is the (nonzero) ambient temperature ``u_a``, with closing
    source ``Q = (-1 + 2 pi^2 alpha + k) exp(-t) sin(pi x) sin(pi y) + k u_a``,
    exercising the ambient/metabolic source path and nonzero Dirichlet data.
    """
    alpha = float(alpha)
    perfusion = float(perfusion)
    ambient = float(ambient)

    def phi(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    if not forced:
        decay = 2.0 * np.pi**2 * alpha + perfusion

        def solution(x, y, t):
            return np.exp(-decay * t) * phi(x, y)

        source = lambda x, y, t: np.zeros_like(np.asarray(x, dtype=float))
    else:
        def solution(x, y, t):
            return ambient + np.exp(-t) * phi(x, y)

        def source(x, y, t):
            w = np.exp(-t) * phi(x, y)
            return (-1.0 + 2.0 * np.pi**2 * alpha + perfusion) * w + perfusion * ambient

    return {
        "name": "Pennes Bioheat",
        "bbox": (0.0, 1.0, 0.0, 1.0),
        "solution": solution,
        "source": source,
        "boundary": solution,
        "alpha": alpha,
        "perfusion": perfusion,
        "ambient": ambient,
    }


def transport_linear_boundary_case(
    model="cattaneo",
    bc_type="neumann",
    alpha=0.1,
    tau=0.2,
    velocity=(0.4, 0.3),
    beta=0.6,
    robin_beta=2.0,
):
    """Manufactured linear-profile cases for non-Dirichlet transport boundaries.

    Spatial profile ``phi = 1 + 0.75 x - 0.5 y`` (``laplacian(phi) = 0``, so the
    TPFA fluxes are spatially exact for constant scalar ``alpha`` and the error
    is dominated by the time discretization), combined with a model-specific
    time factor:

    - ``model='cattaneo'``:   ``u = exp(-t) phi``, ``Q = (tau - 1) u``
    - ``model='advection'``:  ``u = exp(-t) phi``,
      ``Q = -u + exp(-t) (0.75 vx - 0.5 vy)``
    - ``model='fractional'``: ``u = t^2 phi``,
      ``Q = 2 t^{2-beta} / Gamma(3-beta) * phi``

    ``bc_type`` selects the boundary data fed to the transport solvers:

    - ``'neumann'``: ``du/dn = g(t) (0.75 nx - 0.5 ny)``
    - ``'flux'``:    inward heat flux ``alpha * du/dn``
    - ``'robin'``:   ``alpha du/dn + robin_beta u = value``
    """
    from scipy.special import gamma as _gamma

    alpha = float(alpha)

    def profile(x, y):
        return 1.0 + 0.75 * x - 0.5 * y

    def dprofile_dn(nx, ny):
        return 0.75 * nx - 0.5 * ny

    extras = {}
    if model == "cattaneo":
        tau = float(tau)

        def time_factor(t):
            return np.exp(-t)

        def source(x, y, t):
            return (tau - 1.0) * np.exp(-t) * profile(x, y)

        extras["relaxation_time"] = tau
        extras["initial_rate"] = lambda x, y: -profile(x, y)
    elif model == "advection":
        vx, vy = float(velocity[0]), float(velocity[1])

        def time_factor(t):
            return np.exp(-t)

        def source(x, y, t):
            return np.exp(-t) * (-profile(x, y) + (0.75 * vx - 0.5 * vy))

        extras["velocity"] = (vx, vy)
    elif model == "fractional":
        beta = float(beta)

        def time_factor(t):
            return t**2

        def source(x, y, t):
            return (2.0 / _gamma(3.0 - beta)) * (t ** (2.0 - beta)) * profile(x, y)

        extras["beta"] = beta
    else:
        raise ValueError("model must be 'cattaneo', 'advection', or 'fractional'.")

    def solution(x, y, t):
        return time_factor(t) * profile(x, y)

    bc_type = str(bc_type).lower().strip()
    if bc_type == "neumann":
        def boundary(x, y, t, nx, ny):
            return time_factor(t) * dprofile_dn(nx, ny)
    elif bc_type == "flux":
        def boundary(x, y, t, nx, ny):
            return alpha * time_factor(t) * dprofile_dn(nx, ny)
    elif bc_type == "robin":
        def boundary(x, y, t, nx, ny):
            value = alpha * time_factor(t) * dprofile_dn(nx, ny) + robin_beta * solution(x, y, t)
            return robin_beta * np.ones_like(np.asarray(x, dtype=float)), value
    else:
        raise ValueError("bc_type must be 'neumann', 'flux', or 'robin'.")

    return {
        "name": f"Transport Linear Boundary ({model}, {bc_type})",
        "bbox": (0.0, 1.0, 0.0, 1.0),
        "solution": solution,
        "source": source,
        "boundary": boundary,
        "bc_type": bc_type,
        "alpha": alpha,
        **extras,
    }


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
            "phase_change_options": {"max_iters": 100, "tol": 1e-9, "relaxation": 0.95, "anderson_depth": 5, "linearize_cp": True},
        }
    elif case == "temperature_dependent_diffusivity":
        return {
            "name": "Temperature-Dependent Diffusivity",
            "bbox": (0.0, 1.0, 0.0, 1.0),
            "solution": temp_dependent_diffusivity_solution,
            "source": temp_dependent_diffusivity_source,
            "alpha": temp_dependent_diffusivity_alpha,
            "temperature_dependent_diffusivity": True,
            "nonlinear_options": {"max_iters": 35, "tol": 1e-10, "relaxation": 0.9, "anderson_depth": 5},
            "polygonal_only": True,
        }
    elif case == "functionally_graded":
        grade = 0.8
        if np.isscalar(alpha) or np.asarray(alpha).ndim == 0:
            alpha0 = float(alpha)
        else:
            alpha0 = 0.1
        return {
            "name": "Functionally Graded Diffusivity",
            "bbox": (0.0, 1.0, 0.0, 1.0),
            "solution": functionally_graded_solution,
            "source": lambda x, y, t: functionally_graded_source(x, y, t, alpha0, grade),
            "alpha": lambda x, y: functionally_graded_alpha(x, y, alpha0, grade),
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
        }
    raise ValueError(
        "Unknown analytical case "
        f"'{case}'. Available cases: heat_kernel, sine_mode, harmonic_polynomial, "
        "source_driven_sine, steady_linear_neumann, steady_linear_robin, "
        "linear_patch, hot_block, off_axis_wave, nyquist_oscillations, point_source, "
        "green_function_source, laplace_equation, anisotropic_heat_kernel, stefan_apparent_capacity, "
        "temperature_dependent_diffusivity, radiative_manufactured, functionally_graded"
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
