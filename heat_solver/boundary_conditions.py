"""Boundary-condition types shared by every Thermalio solver.

All solvers accept the same ``bc_type`` values and callback conventions:

=============  ==============================================  ===================================
``bc_type``    ``bc_func`` returns                             condition on the boundary
=============  ==============================================  ===================================
``dirichlet``  ``u``                                           ``u = g``
``neumann``    ``du/dn`` (outward normal derivative)           ``du/dn = g``
``robin``      ``(beta, value)`` or ``{"beta", "value"}``      ``alpha du/dn + beta u = value``
``flux``       ``q_in`` (inward heat flux per unit length)     ``alpha du/dn = q_in``
``radiative``  ``(eps, T_inf[, sigma])`` or a dict             ``alpha du/dn = -eps sigma (u^4 - T_inf^4)``
=============  ==============================================  ===================================

Non-Dirichlet callbacks may take the outward unit normal as extra arguments:
``bc_func(x, y, t, nx, ny)``.

Every non-Dirichlet condition is a prescription of the *inward* boundary heat
flux ``q_in = alpha du/dn``, which this module expresses in the linearized
form ``q_in ~ c - a u`` (exact for the linear types, a Newton linearization
about the current iterate for radiation).  Each discretization (cell-centred,
vertex-centred, structured grid) integrates that form with its own geometry.
"""

from __future__ import annotations

import inspect

import numpy as np

BC_TYPES = ("dirichlet", "neumann", "robin", "flux", "radiative")
STEFAN_BOLTZMANN = 5.670374419e-8


def normalize_bc_type(bc_type, supported=BC_TYPES):
    name = str(bc_type).lower().strip()
    if name not in supported:
        raise ValueError(f"bc_type must be one of {tuple(supported)}; got {bc_type!r}.")
    return name


def callback_accepts_normals(func):
    """Whether ``func`` can be called as ``func(x, y, t, nx, ny)``."""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return True
    params = list(signature.parameters.values())
    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
        return True
    positional = [p for p in params if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                                   inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    return len(positional) >= 5


def evaluate(func, x, y, t, normals=None, accepts_normals=None):
    if accepts_normals is None:
        accepts_normals = callback_accepts_normals(func)
    if accepts_normals and normals is not None:
        normals = np.asarray(normals, dtype=float)
        return func(x, y, t, normals[..., 0], normals[..., 1])
    return func(x, y, t)


def parse_robin(data, count):
    if isinstance(data, dict):
        beta = data.get("beta")
        value = data.get("value", data.get("gamma", data.get("rhs")))
    else:
        try:
            beta, value = data
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Robin boundary conditions require bc_func to return (beta, value) "
                "for alpha * du/dn + beta * u = value."
            ) from exc
    if beta is None or value is None:
        raise ValueError("Robin boundary conditions require bc_func to provide both beta and value.")
    return (np.broadcast_to(np.asarray(beta, dtype=float), (count,)),
            np.broadcast_to(np.asarray(value, dtype=float), (count,)))


def parse_radiative(data, count):
    if isinstance(data, dict):
        emissivity = data.get("epsilon", data.get("emissivity"))
        t_inf = data.get("t_inf", data.get("ambient_temperature", data.get("T_inf")))
        sigma = data.get("sigma", STEFAN_BOLTZMANN)
    else:
        try:
            if len(data) == 2:
                emissivity, t_inf = data
                sigma = STEFAN_BOLTZMANN
            else:
                emissivity, t_inf, sigma = data
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Radiative boundary conditions require bc_func to return "
                "(epsilon, T_inf) or (epsilon, T_inf, sigma), or a dict with "
                "'epsilon'/'emissivity' and 't_inf'/'ambient_temperature'."
            ) from exc
    if emissivity is None or t_inf is None:
        raise ValueError("Radiative boundary conditions require both emissivity and ambient temperature.")
    emissivity = np.broadcast_to(np.asarray(emissivity, dtype=float), (count,))
    t_inf = np.broadcast_to(np.asarray(t_inf, dtype=float), (count,))
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (count,))
    if np.any(emissivity < 0.0):
        raise ValueError("Radiative emissivity must be non-negative.")
    if np.any(sigma <= 0.0):
        raise ValueError("Stefan-Boltzmann coefficient sigma must be positive.")
    return emissivity, t_inf, sigma


def inflow_linearization(bc_type, data, count, alpha_n=None, temperature=None):
    """Return ``(a, c)`` with inward flux ``q_in = alpha du/dn ~ c - a u``.

    ``alpha_n`` (normal conductivity) is needed for ``neumann`` data, which is
    a derivative rather than a flux; ``temperature`` (the boundary values of
    the current iterate) is needed for ``radiative``.
    """
    if bc_type == "neumann":
        g = np.broadcast_to(np.asarray(data, dtype=float), (count,))
        return np.zeros(count), np.asarray(alpha_n, dtype=float) * g
    if bc_type == "flux":
        return np.zeros(count), np.broadcast_to(np.asarray(data, dtype=float), (count,)).astype(float)
    if bc_type == "robin":
        beta, value = parse_robin(data, count)
        return beta.astype(float), value.astype(float)
    if bc_type == "radiative":
        if temperature is None:
            raise ValueError("Radiative boundary assembly requires the current temperature iterate.")
        eps, t_inf, sigma = parse_radiative(data, count)
        tk = np.maximum(np.asarray(temperature, dtype=float), 0.0)
        a = 4.0 * eps * sigma * tk**3
        c = 3.0 * eps * sigma * tk**4 + eps * sigma * np.maximum(t_inf, 0.0) ** 4
        return a, c
    raise ValueError(f"no inflow linearization for bc_type={bc_type!r}.")


def radiative_inflow(data, count, temperature):
    """Exact (nonlinear) radiative inflow ``-eps sigma (u^4 - T_inf^4)`` and its ``d/du``."""
    eps, t_inf, sigma = parse_radiative(data, count)
    u = np.maximum(np.asarray(temperature, dtype=float), 0.0)
    q = -eps * sigma * (u**4 - np.maximum(t_inf, 0.0) ** 4)
    dq = -4.0 * eps * sigma * u**3
    return q, dq


# ---------------------------------------------------------------------------
# Manufactured boundary data (any bc_type from an exact solution)
# ---------------------------------------------------------------------------


def manufactured_bc_func(bc_type, exact, grad, alpha, robin_beta=2.0, emissivity=0.5,
                         sigma=1.0):
    """Boundary callback reproducing ``exact`` for any ``bc_type``.

    ``exact(x, y, t)`` and ``grad(x, y, t) -> (u_x, u_y)``; ``alpha`` is a
    scalar conductivity (the normal flux is ``alpha grad u . n``).  For
    ``radiative`` the ambient temperature is chosen so that
    ``alpha du/dn = -eps sigma (u^4 - T_inf^4)`` holds exactly; this requires
    ``u^4 + alpha du/dn / (eps sigma) >= 0`` on the boundary.
    """
    bc_type = normalize_bc_type(bc_type)
    alpha = float(alpha)
    if bc_type == "dirichlet":
        return lambda x, y, t: exact(x, y, t)

    def dudn(x, y, t, nx, ny):
        gx, gy = grad(x, y, t)
        return gx * nx + gy * ny

    if bc_type == "neumann":
        return lambda x, y, t, nx, ny: dudn(x, y, t, nx, ny)
    if bc_type == "flux":
        return lambda x, y, t, nx, ny: alpha * dudn(x, y, t, nx, ny)
    if bc_type == "robin":
        return lambda x, y, t, nx, ny: (
            robin_beta * np.ones_like(np.asarray(x, dtype=float)),
            alpha * dudn(x, y, t, nx, ny) + robin_beta * exact(x, y, t),
        )

    def radiative(x, y, t, nx, ny):
        u = exact(x, y, t)
        t_inf4 = u**4 + alpha * dudn(x, y, t, nx, ny) / (emissivity * sigma)
        if np.any(t_inf4 < 0.0):
            raise ValueError("manufactured radiative data needs u^4 + alpha du/dn/(eps sigma) >= 0.")
        return emissivity, t_inf4 ** 0.25, sigma

    return radiative
