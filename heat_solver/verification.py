"""Grid-convergence verification utilities: observed order of accuracy,
Richardson extrapolation, and the Grid Convergence Index (GCI).

These are pure-numpy post-processing helpers (no solver dependency) used by the
convergence sweep in ``tests.py`` and by unit tests.  They support two related
but distinct workflows:

* **Method-of-manufactured-solutions verification** (the exact solution is
  known): the headline metric is the *observed order of accuracy* computed from
  a sequence of error norms ``e_i = ||u_h_i - u_exact||`` that should behave like
  ``e_i ~ C h_i^p``.  ``triplet_report`` is tailored to this case.

* **Solution-functional uncertainty estimation** (exact solution unknown): the
  classic three-grid Richardson / GCI machinery (Roache; ASME V&V 20) applied to
  a scalar functional ``f_i`` of the numerical solution.  ``richardson_order``,
  ``richardson_extrapolate``, ``gci`` and ``asymptotic_ratio`` cover this and are
  reusable by later validation work (directions B/C/D).

Grids are ordered **coarsest first** throughout; refinement ratio ``r = h_coarse
/ h_fine > 1``.
"""

from __future__ import annotations

import math


def observed_order(e_coarse, e_fine, r):
    """Observed order of accuracy from two error norms one refinement apart.

    For ``e ~ C h^p`` with ``r = h_coarse / h_fine > 1``,
    ``p = ln(e_coarse / e_fine) / ln(r)``.  Returns ``None`` if the inputs are
    non-positive or ``r <= 1``.
    """
    e_coarse = float(e_coarse)
    e_fine = float(e_fine)
    r = float(r)
    if e_coarse <= 0.0 or e_fine <= 0.0 or r <= 1.0:
        return None
    return math.log(e_coarse / e_fine) / math.log(r)


def richardson_order(f_coarse, f_medium, f_fine, r_cm, r_mf, max_iter=100, tol=1e-12):
    """Observed order from three solution-functional values (Roache general form).

    ``f_coarse, f_medium, f_fine`` are a scalar functional on three grids ordered
    coarsest -> finest.  ``r_cm = h_coarse / h_medium`` and ``r_mf = h_medium /
    h_fine`` are the (possibly unequal) refinement ratios, both > 1.

    Solves the fixed-point equation
    ``p = |ln|eps_cm / eps_mf| + q(p)| / ln(r_mf)`` with
    ``q(p) = ln((r_mf^p - s) / (r_cm^p - s))`` and ``s = sign(eps_cm / eps_mf)``.
    For equal ratios this reduces to ``p = ln|eps_cm/eps_mf| / ln(r)``.

    Returns ``(p, oscillatory)``.  ``oscillatory`` is True when the functional is
    non-monotone across the three grids (``s < 0``), in which case ``p`` is still
    returned but should be treated with caution.
    """
    eps_mf = float(f_fine) - float(f_medium)
    eps_cm = float(f_medium) - float(f_coarse)
    r_cm = float(r_cm)
    r_mf = float(r_mf)
    if eps_mf == 0.0 or eps_cm == 0.0 or r_cm <= 1.0 or r_mf <= 1.0:
        return None, False
    ratio = eps_cm / eps_mf
    s = 1.0 if ratio > 0.0 else -1.0
    oscillatory = s < 0.0

    if abs(r_cm - r_mf) < 1e-12:
        p = math.log(abs(ratio)) / math.log(r_mf)
        return p, oscillatory

    p = math.log(abs(ratio)) / math.log(r_mf)  # initial guess (q = 0)
    for _ in range(max_iter):
        denom = r_cm**p - s
        numer = r_mf**p - s
        if numer == 0.0 or denom == 0.0:
            break
        q = math.log(abs(numer / denom))
        p_new = abs(math.log(abs(ratio)) + q) / math.log(r_mf)
        if abs(p_new - p) < tol:
            p = p_new
            break
        p = p_new
    return p, oscillatory


def richardson_extrapolate(f_fine, f_coarse, p, r):
    """Richardson-extrapolated (h -> 0) value: ``f_fine + (f_fine - f_coarse)/(r^p - 1)``."""
    p = float(p)
    r = float(r)
    denom = r**p - 1.0
    if denom == 0.0:
        return None
    return float(f_fine) + (float(f_fine) - float(f_coarse)) / denom


def gci(f_coarse, f_fine, p, r, fs=1.25):
    """Grid Convergence Index for the fine grid (fractional, multiply by 100 for %).

    ``GCI_fine = fs * |(f_fine - f_coarse) / f_fine| / (r^p - 1)``.  ``fs`` is the
    safety factor (1.25 for three or more grids, 3.0 for two).
    """
    f_fine = float(f_fine)
    denom = (float(r) ** float(p)) - 1.0
    if f_fine == 0.0 or denom == 0.0:
        return None
    rel = abs((f_fine - float(f_coarse)) / f_fine)
    return float(fs) * rel / denom


def asymptotic_ratio(gci_coarse, gci_fine, r, p):
    """Asymptotic-range indicator ``GCI_coarse / (r^p * GCI_fine)``; ~1 means the
    triplet is in the asymptotic convergence range."""
    if gci_coarse is None or gci_fine is None:
        return None
    denom = (float(r) ** float(p)) * float(gci_fine)
    if denom == 0.0:
        return None
    return float(gci_coarse) / denom


def triplet_report(hs, errors, fs=1.25):
    """MMS verification report for three grids (error-vs-exact sequence).

    Parameters
    ----------
    hs : sequence of 3 mesh sizes (any order).
    errors : matching error norms ``e_i = ||u_h_i - u_exact||``.

    Returns a dict with:
      ``p_obs``        observed order from the two finest grids,
      ``p_obs_3grid``  three-grid order (Roache form on the error sequence),
      ``extrap_error`` Richardson-extrapolated error (≈0 confirms power-law decay),
      ``asymptotic``   True if errors decrease monotonically and the two pairwise
                       orders agree to within 20%,
      ``oscillatory``  True if the error sequence is non-monotone.

    Note: the GCI is intentionally *not* reported here.  GCI estimates
    discretization uncertainty when the exact solution is unknown; applied to an
    error sequence that converges to zero it degenerates to the safety factor and
    carries no information.  Use the standalone :func:`gci` on a solution
    functional for no-exact-solution (validation) settings.
    """
    order = sorted(range(3), key=lambda i: hs[i], reverse=True)  # coarse -> fine
    h = [float(hs[i]) for i in order]
    e = [float(errors[i]) for i in order]
    r_cm = h[0] / h[1]
    r_mf = h[1] / h[2]

    p_cm = observed_order(e[0], e[1], r_cm)
    p_mf = observed_order(e[1], e[2], r_mf)
    p3, oscillatory = richardson_order(e[0], e[1], e[2], r_cm, r_mf)

    extrap = None
    if p_mf is not None:
        extrap = richardson_extrapolate(e[2], e[1], p_mf, r_mf)

    monotone = e[0] > e[1] > e[2]
    agree = (
        p_cm is not None
        and p_mf is not None
        and p_cm > 0.0
        and abs(p_cm - p_mf) <= 0.2 * max(abs(p_cm), abs(p_mf), 1e-12)
    )
    return {
        "p_obs": p_mf,
        "p_obs_3grid": p3,
        "extrap_error": extrap,
        "asymptotic": bool(monotone and agree),
        "oscillatory": bool(oscillatory or not monotone),
    }
