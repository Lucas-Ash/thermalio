"""Standard test problems for every boundary-condition type.

Classical exact solutions of the heat equation ``u_t = alpha Lap u (+ Q)``,
one per ``bc_type`` (:func:`heat_problem`):

``dirichlet``  decaying sine mode on ``[0, 1]^2``,
               ``u = exp(-2 pi^2 alpha t) sin(pi x) sin(pi y)``, ``u = 0`` on the boundary.
``neumann``    insulated plate on ``[0, 1]^2``,
               ``u = exp(-2 pi^2 alpha t) cos(pi x) cos(pi y)``, ``du/dn = 0``.
``robin``      convective cooling of a square ``[-1, 1]^2`` into ambient 0
               (Carslaw & Jaeger), ``alpha du/dn + h u = 0``:
               ``u = cos(lam x) cos(lam y) exp(-2 alpha lam^2 t)``, ``lam tan(lam) = h / alpha``.
``flux``       uniform heating through all walls of ``[-1, 1]^2`` with inward
               flux ``q``: ``u = q (x^2 + y^2) / (2 alpha) + 2 q t``.
``radiative``  radiating plate (manufactured): ``u = 1 + 0.4 exp(-t) cos(pi x / 2) cos(pi y / 2)``
               on ``[-1, 1]^2`` with ``eps = 0.5, sigma = 1`` and ambient temperature
               chosen so the Stefan--Boltzmann law holds exactly (source closes the PDE).

:func:`transport_problem` builds manufactured problems for the extended
models (Cattaneo, Caputo fractional, advection--diffusion, Pennes) with any
``bc_type``, using ``u = c0 + (1 + t + t^2) S(x, y)`` so the Caputo
derivative is exact.
"""

from __future__ import annotations

from math import gamma

import numpy as np
from scipy.optimize import brentq

from .boundary_conditions import manufactured_bc_func, normalize_bc_type


def heat_problem(bc_type, alpha=0.1, h=0.6, q=1.0, manufactured_boundary=False):
    """Classical heat-equation problem for ``bc_type`` (see module docstring).

    The classical boundary data assume the axis-aligned box ``bbox``.  With
    ``manufactured_boundary=True`` the boundary data are generated from the
    exact gradient instead, so the same exact solution is valid on any domain
    (e.g. the sheared parallelogram of the non-orthogonal mesh generator).
    """
    prob = _heat_problem(bc_type, alpha, h, q)
    if manufactured_boundary and prob["bc_type"] != "dirichlet":
        kw = {}
        if prob["bc_type"] == "robin":
            kw["robin_beta"] = h
        if prob["bc_type"] == "radiative":
            kw.update(emissivity=0.5, sigma=1.0)
        prob = dict(prob, bc_func=manufactured_bc_func(prob["bc_type"], prob["exact"], prob["grad"], alpha, **kw))
    return prob


def _heat_problem(bc_type, alpha, h, q):
    bc_type = normalize_bc_type(bc_type)
    pi = np.pi
    zero = lambda x, y, t: np.zeros_like(np.asarray(x, dtype=float))
    if bc_type == "dirichlet":
        k = 2 * pi**2 * alpha
        exact = lambda x, y, t: np.exp(-k * t) * np.sin(pi * x) * np.sin(pi * y)
        grad = lambda x, y, t: (pi * np.exp(-k * t) * np.cos(pi * x) * np.sin(pi * y),
                                pi * np.exp(-k * t) * np.sin(pi * x) * np.cos(pi * y))
        return dict(name="decaying sine mode", bbox=(0.0, 1.0, 0.0, 1.0), exact=exact, grad=grad, source=zero,
                    bc_type="dirichlet", bc_func=exact, alpha=alpha)
    if bc_type == "neumann":
        k = 2 * pi**2 * alpha
        exact = lambda x, y, t: np.exp(-k * t) * np.cos(pi * x) * np.cos(pi * y)
        grad = lambda x, y, t: (-pi * np.exp(-k * t) * np.sin(pi * x) * np.cos(pi * y),
                                -pi * np.exp(-k * t) * np.cos(pi * x) * np.sin(pi * y))
        return dict(name="insulated plate", bbox=(0.0, 1.0, 0.0, 1.0), exact=exact, grad=grad, source=zero,
                    bc_type="neumann", bc_func=lambda x, y, t, nx, ny: np.zeros_like(np.asarray(x, float)),
                    alpha=alpha)
    if bc_type == "robin":
        bi = h / alpha
        lam = brentq(lambda z: z * np.tan(z) - bi, 1e-9, pi / 2 - 1e-12)
        exact = lambda x, y, t: np.cos(lam * x) * np.cos(lam * y) * np.exp(-2 * alpha * lam**2 * t)
        grad = lambda x, y, t: (-lam * np.sin(lam * x) * np.cos(lam * y) * np.exp(-2 * alpha * lam**2 * t),
                                -lam * np.cos(lam * x) * np.sin(lam * y) * np.exp(-2 * alpha * lam**2 * t))
        return dict(name=f"convective cooling (Bi={bi:.1f})", bbox=(-1.0, 1.0, -1.0, 1.0), exact=exact, grad=grad,
                    source=zero, bc_type="robin", alpha=alpha, eigenvalue=lam,
                    bc_func=lambda x, y, t, nx, ny: (h * np.ones_like(np.asarray(x, float)),
                                                     np.zeros_like(np.asarray(x, float))))
    if bc_type == "flux":
        exact = lambda x, y, t: q * (np.asarray(x) ** 2 + np.asarray(y) ** 2) / (2 * alpha) + 2 * q * t
        grad = lambda x, y, t: (q * np.asarray(x, float) / alpha, q * np.asarray(y, float) / alpha)
        return dict(name="uniform wall heating", bbox=(-1.0, 1.0, -1.0, 1.0), exact=exact, grad=grad, source=zero,
                    bc_type="flux", alpha=alpha,
                    bc_func=lambda x, y, t, nx, ny: q * np.ones_like(np.asarray(x, float)))
    # radiative (manufactured)
    a = pi / 2

    def exact(x, y, t):
        return 1.0 + 0.4 * np.exp(-t) * np.cos(a * x) * np.cos(a * y)

    def grad(x, y, t):
        e = 0.4 * np.exp(-t)
        return -e * a * np.sin(a * x) * np.cos(a * y), -e * a * np.cos(a * x) * np.sin(a * y)

    def source(x, y, t):
        s = 0.4 * np.exp(-t) * np.cos(a * x) * np.cos(a * y)
        return -s + alpha * 2 * a**2 * s

    return dict(name="radiating plate (manufactured)", bbox=(-1.0, 1.0, -1.0, 1.0), exact=exact, grad=grad,
                source=source, bc_type="radiative", alpha=alpha,
                bc_func=manufactured_bc_func("radiative", exact, grad, alpha, emissivity=0.5, sigma=1.0))


def transport_problem(model, bc_type, alpha=0.1, tau=0.05, beta=0.6, perfusion=1.5,
                      velocity=(0.3, -0.2), robin_beta=2.0):
    """Manufactured problem for an extended transport model with any ``bc_type``.

    ``u = c0 + T(t) S(x, y)`` with ``T = 1 + t + t^2`` and
    ``S = cos(pi x / 2) cos(pi y / 2)`` on ``[-1, 1]^2`` (``c0 = 1`` keeps the
    temperature positive for radiation).  ``model`` in ``{"cattaneo",
    "fractional", "advection", "pennes"}``.
    """
    bc_type = normalize_bc_type(bc_type)
    a = np.pi / 2
    c0 = 1.0
    T = lambda t: 1.0 + t + t * t
    dT = lambda t: 1.0 + 2.0 * t
    S = lambda x, y: np.cos(a * x) * np.cos(a * y)
    Sx = lambda x, y: -a * np.sin(a * x) * np.cos(a * y)
    Sy = lambda x, y: -a * np.cos(a * x) * np.sin(a * y)
    lap = lambda x, y: -2 * a * a * S(x, y)

    def exact(x, y, t):
        return c0 + 0.3 * T(t) * S(x, y)

    def grad(x, y, t):
        return 0.3 * T(t) * Sx(x, y), 0.3 * T(t) * Sy(x, y)

    def time_term(x, y, t):
        if model == "cattaneo":
            return 0.3 * (tau * 2.0 + dT(t)) * S(x, y)
        if model == "fractional":
            # Caputo of 1 + t + t^2: t^{1-b}/Gamma(2-b) + 2 t^{2-b}/Gamma(3-b)
            c = t ** (1 - beta) / gamma(2 - beta) + 2 * t ** (2 - beta) / gamma(3 - beta)
            return 0.3 * c * S(x, y)
        return 0.3 * dT(t) * S(x, y)

    def source(x, y, t):
        q = time_term(x, y, t) - alpha * 0.3 * T(t) * lap(x, y)
        if model == "advection":
            gx, gy = grad(x, y, t)
            q = q + velocity[0] * gx + velocity[1] * gy  # div(v u) = v . grad u (constant v)
        if model == "pennes":
            q = q + perfusion * exact(x, y, t)
        return q

    def initial_rate(x, y):
        return 0.3 * dT(0.0) * S(x, y)

    return dict(model=model, bbox=(-1.0, 1.0, -1.0, 1.0), exact=exact, grad=grad, source=source,
                initial_rate=initial_rate, bc_type=bc_type, alpha=alpha, tau=tau, beta=beta,
                perfusion=perfusion, velocity=velocity,
                bc_func=manufactured_bc_func(bc_type, exact, grad, alpha, robin_beta=robin_beta,
                                             emissivity=0.5, sigma=1.0))
