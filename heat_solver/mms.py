"""Symbolic method-of-manufactured-solutions (MMS) source derivation.

Given a closed-form temperature field ``u(x, y, t)`` and a PDE operator, this
module symbolically derives the forcing source ``Q = L[u]`` and the Dirichlet /
Neumann / Robin boundary data, then lambdifies them into NumPy callables with the
**exact signatures** the solvers and drivers expect
(``solution(x, y, t)``, ``source(x, y, t)``, ``boundary(x, y, t[, nx, ny])``).
The returned dict matches the shape of ``heat_solver.cases.get_analytical_case``.

This removes hand-derivation error from manufactured cases: an auto-derived
source can be checked against an existing hand-coded one to machine precision.

SymPy is an **optional, test-time dependency**.  It is imported lazily inside the
functions here, and nothing in ``heat_solver/__init__.py`` imports this module, so
the core solvers never require SymPy.  The *outputs* are plain NumPy callables, so
once a case is built nothing downstream touches SymPy.

Supported operators (``model=``):
  - ``"diffusion"``           : u_t - div(alpha grad u)
  - ``"advection_diffusion"`` : u_t + v . grad u - div(alpha grad u)
  - ``"reaction_diffusion"``  : u_t - div(alpha grad u) + k u   (Pennes bioheat)
  - ``"cattaneo"``            : tau u_tt + u_t - div(alpha grad u)
  - ``"fractional"``          : D_t^beta u - div(alpha grad u)   (Caputo; u must
                                be a sum of real powers of t, e.g. t**2 or
                                t**(1+beta))

``alpha`` may be a number, a string expression in ``x, y`` (and ``u`` for
temperature-dependent diffusivity), or a 2x2 nested list (anisotropic tensor).
``velocity`` and ``reaction`` may be numbers or string expressions.
"""

from __future__ import annotations

import numpy as np


def _require_sympy():
    try:
        import sympy as sp
    except ImportError as exc:  # pragma: no cover - exercised only without sympy
        raise ImportError(
            "heat_solver.mms requires SymPy (an optional V&V dependency). "
            "Install it with `pip install sympy` (see requirements-dev.txt)."
        ) from exc
    return sp


def _sympify(expr, symbols, sp):
    """Turn a number / string into a SymPy expression with x, y, t, u in scope."""
    if isinstance(expr, str):
        return sp.sympify(expr, locals=symbols)
    return sp.sympify(expr)


def _grad(u, x, y, sp):
    return sp.Matrix([sp.diff(u, x), sp.diff(u, y)])


def _conductivity_matrix(alpha, x, y, u_sym, u_expr, symbols, sp):
    """Return a 2x2 SymPy conductivity matrix K(x, y) (with u substituted in)."""
    if isinstance(alpha, (list, tuple)):
        rows = [[_sympify(a, symbols, sp) for a in row] for row in alpha]
        K = sp.Matrix(rows)
    else:
        a = _sympify(alpha, symbols, sp)
        K = sp.Matrix([[a, 0], [0, a]])
    # Substitute the known solution for temperature-dependent diffusivity.
    return K.applyfunc(lambda e: e.subs(u_sym, u_expr))


def _diffusion_divergence(u, x, y, K, sp):
    """Return div(K grad u) as a SymPy expression."""
    gu = _grad(u, x, y, sp)
    flux = K * gu  # (K grad u)
    return sp.diff(flux[0], x) + sp.diff(flux[1], y)


def _caputo(u, t, beta, sp):
    """Caputo derivative D_t^beta u for u a sum of (real) powers of t.

    Each additive term must factor as ``c(x, y) * t**p`` with a constant exponent
    ``p`` (integer or real); then
    ``D_t^beta t^p = Gamma(p+1)/Gamma(p+1-beta) t^{p-beta}`` for ``p > 0`` and 0
    for ``p == 0``.  This covers polynomials in ``t`` (any degree) and fractional
    powers such as ``t**(1+beta)`` that are standard in fractional MMS, but not
    genuinely transcendental time dependence (e.g. ``exp(t)``), which would need a
    Mittag-Leffler / series treatment.
    """
    u = sp.expand(u)
    result = sp.Integer(0)
    for term in u.as_ordered_terms():
        coeff, tpart = term.as_independent(t)
        if tpart == sp.Integer(1):
            continue  # constant in t -> Caputo derivative is 0
        base, exp = tpart.as_base_exp()
        if base != t or exp.has(t):
            raise ValueError(
                "Caputo derivation supports sums of powers of t "
                f"(c(x,y)*t**p); got non-power time dependence in term {term}."
            )
        p = exp
        result += coeff * sp.gamma(p + 1) / sp.gamma(p + 1 - beta) * t ** (p - beta)
    return result


def _operator(model, u, x, y, t, K, *, velocity, reaction, tau, beta, symbols, sp):
    diffusion = _diffusion_divergence(u, x, y, K, sp)
    if model == "diffusion":
        return sp.diff(u, t) - diffusion
    if model == "advection_diffusion":
        vx = _sympify(velocity[0], symbols, sp)
        vy = _sympify(velocity[1], symbols, sp)
        gu = _grad(u, x, y, sp)
        return sp.diff(u, t) + (vx * gu[0] + vy * gu[1]) - diffusion
    if model == "reaction_diffusion":
        k = _sympify(reaction, symbols, sp)
        return sp.diff(u, t) - diffusion + k * u
    if model == "cattaneo":
        return tau * sp.diff(u, t, 2) + sp.diff(u, t) - diffusion
    if model == "fractional":
        return _caputo(u, t, beta, sp) - diffusion
    raise ValueError(
        f"Unknown model {model!r}. Use diffusion, advection_diffusion, "
        "reaction_diffusion, cattaneo, or fractional."
    )


def _broadcast(f, nargs):
    """Wrap a lambdified callable so it always returns an array shaped like x."""

    def wrapped(*args):
        val = f(*args)
        x0 = np.asarray(args[0], dtype=float)
        return np.asarray(val, dtype=float) + np.zeros_like(x0)

    return wrapped


def manufactured_case(
    u,
    *,
    alpha,
    model="diffusion",
    bc_type="dirichlet",
    velocity=None,
    reaction=None,
    tau=None,
    beta=None,
    robin_beta=2.0,
    bbox=(0.0, 1.0, 0.0, 1.0),
    name="MMS",
):
    """Build a manufactured-solution case dict from a symbolic ``u``.

    Parameters
    ----------
    u : str
        Solution expression in ``x, y, t`` (e.g. ``"exp(-t)*sin(pi*x)*sin(pi*y)"``).
    alpha : number | str | 2x2 list
        Diffusivity; string may depend on ``x, y`` and on ``u`` (temperature
        dependent).
    model : str
        Operator selector (see module docstring).
    bc_type : str
        ``"dirichlet"``, ``"neumann"``, or ``"robin"``.
    velocity, reaction, tau, beta : optional
        Parameters for the corresponding models.

    Returns
    -------
    dict with keys ``name, bbox, solution, source, boundary, bc_type`` and the
    relevant extras (``relaxation_time``, ``velocity``, ``beta``, ...), all as
    NumPy callables matching the solver/driver signatures.
    """
    sp = _require_sympy()
    x, y, t = sp.symbols("x y t", real=True)
    u_sym = sp.Symbol("u", real=True)
    nx, ny = sp.symbols("nx ny", real=True)
    symbols = {"x": x, "y": y, "t": t, "u": u_sym}

    u_expr = _sympify(u, symbols, sp)
    K = _conductivity_matrix(alpha, x, y, u_sym, u_expr, symbols, sp)
    q_expr = sp.expand(
        _operator(
            model, u_expr, x, y, t, K,
            velocity=velocity, reaction=reaction, tau=tau, beta=beta,
            symbols=symbols, sp=sp,
        )
    )

    solution = _broadcast(sp.lambdify((x, y, t), u_expr, modules=["numpy"]), 3)
    source = _broadcast(sp.lambdify((x, y, t), q_expr, modules=["numpy"]), 3)

    bc_type = str(bc_type).lower()
    case = {
        "name": name,
        "bbox": tuple(float(b) for b in bbox),
        "solution": solution,
        "source": source,
        "bc_type": bc_type,
    }

    gu = _grad(u_expr, x, y, sp)
    if bc_type == "dirichlet":
        case["boundary"] = solution
    elif bc_type == "neumann":
        dudn = gu[0] * nx + gu[1] * ny
        case["boundary"] = _broadcast(sp.lambdify((x, y, t, nx, ny), dudn, modules=["numpy"]), 5)
    elif bc_type == "robin":
        a_scalar = K[0, 0]  # robin closure uses scalar alpha
        beta_sym = _sympify(robin_beta, symbols, sp)
        value = a_scalar * (gu[0] * nx + gu[1] * ny) + beta_sym * u_expr
        value_fn = _broadcast(sp.lambdify((x, y, t, nx, ny), value, modules=["numpy"]), 5)
        beta_val = float(robin_beta)

        def boundary(x_, y_, t_, nx_, ny_):
            return (
                beta_val * np.ones_like(np.asarray(x_, dtype=float)),
                value_fn(x_, y_, t_, nx_, ny_),
            )

        case["boundary"] = boundary
    else:
        raise ValueError("bc_type must be 'dirichlet', 'neumann', or 'robin'.")

    if model == "cattaneo":
        case["relaxation_time"] = float(tau)
        case["initial_rate"] = _broadcast(
            sp.lambdify((x, y), sp.diff(u_expr, t).subs(t, 0), modules=["numpy"]), 2
        )
    if model == "advection_diffusion":
        case["velocity"] = (float(velocity[0]), float(velocity[1])) if all(
            not isinstance(v, str) for v in velocity
        ) else velocity
    if model == "fractional":
        case["beta"] = float(beta)
    if model == "reaction_diffusion":
        case["reaction"] = reaction

    return case
