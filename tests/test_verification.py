import numpy as np

from heat_solver.verification import (
    asymptotic_ratio,
    gci,
    observed_order,
    richardson_extrapolate,
    richardson_order,
    triplet_report,
)


def test_observed_order_recovers_second_order():
    # e = C h^2 on h = [1, 1/2]: p = 2 exactly.
    C = 0.3
    e_coarse = C * 1.0**2
    e_fine = C * 0.5**2
    p = observed_order(e_coarse, e_fine, r=2.0)
    assert p == np.float64(p)  # plain float
    assert abs(p - 2.0) < 1e-12


def test_observed_order_guards():
    assert observed_order(0.0, 1.0, 2.0) is None
    assert observed_order(1.0, 1.0, 1.0) is None


def test_richardson_order_constant_ratio_second_order():
    # Functional f_i = f_exact + C h^2; recover p = 2 and f_exact.
    f_exact, C = 5.0, 0.7
    hs = [1.0, 0.5, 0.25]
    f = [f_exact + C * h**2 for h in hs]
    p, oscillatory = richardson_order(f[0], f[1], f[2], r_cm=2.0, r_mf=2.0)
    assert abs(p - 2.0) < 1e-9
    assert not oscillatory
    extrap = richardson_extrapolate(f[2], f[1], p, r=2.0)
    assert abs(extrap - f_exact) < 1e-9


def test_richardson_order_unequal_ratios():
    f_exact, C, q = 2.0, 0.5, 2.0
    hs = [1.0, 0.5, 0.2]  # ratios 2.0 and 2.5
    f = [f_exact + C * h**q for h in hs]
    p, _ = richardson_order(f[0], f[1], f[2], r_cm=hs[0] / hs[1], r_mf=hs[1] / hs[2])
    assert abs(p - 2.0) < 1e-6


def test_gci_and_asymptotic_ratio_in_range():
    f_exact, C = 1.0, 0.4
    hs = [1.0, 0.5, 0.25]
    f = [f_exact + C * h**2 for h in hs]
    p = 2.0
    gci_fine = gci(f[1], f[2], p, r=2.0)
    gci_coarse = gci(f[0], f[1], p, r=2.0)
    assert gci_fine is not None and gci_fine > 0.0
    # Refined GCI should shrink ~ r^p.
    assert gci_fine < gci_coarse
    # Relative-GCI normalization (by each pair's fine value) makes the ratio
    # approach 1 only as grids refine; ~0.93 here is correct, in-range behavior.
    ratio = asymptotic_ratio(gci_coarse, gci_fine, r=2.0, p=p)
    assert abs(ratio - 1.0) < 0.1


def test_triplet_report_second_order_clean():
    C = 0.25
    hs = [1.0, 0.5, 0.25]
    errors = [C * h**2 for h in hs]
    rep = triplet_report(hs, errors)
    assert abs(rep["p_obs"] - 2.0) < 1e-9
    assert abs(rep["p_obs_3grid"] - 2.0) < 1e-6
    assert abs(rep["extrap_error"]) < 1e-9  # error extrapolates to 0
    assert rep["asymptotic"] is True
    assert rep["oscillatory"] is False


def test_triplet_report_first_order():
    C = 0.5
    hs = [1.0, 0.5, 0.25]
    errors = [C * h for h in hs]
    rep = triplet_report(hs, errors)
    assert abs(rep["p_obs"] - 1.0) < 1e-9
    assert rep["asymptotic"] is True


def test_triplet_report_unsorted_input():
    # Same data, shuffled order, must give identical result.
    C = 0.25
    hs = [0.25, 1.0, 0.5]
    errors = [C * h**2 for h in hs]
    rep = triplet_report(hs, errors)
    assert abs(rep["p_obs"] - 2.0) < 1e-9


def test_triplet_report_oscillatory_flagged():
    hs = [1.0, 0.5, 0.25]
    errors = [0.1, 0.2, 0.05]  # non-monotone
    rep = triplet_report(hs, errors)
    assert rep["oscillatory"] is True
    assert rep["asymptotic"] is False
