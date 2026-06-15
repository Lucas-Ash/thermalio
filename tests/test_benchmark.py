import pytest

pytest.importorskip("sympy")

from benchmark_suite import run_benchmark


def test_benchmark_suite_quick_passes():
    # End-to-end V&V benchmark (verification + mms + nversion + reference_fd):
    # every panel entry must report ~2nd-order convergence and pass its checks.
    report = run_benchmark(quick=True)
    assert report["all_pass"], report
    for entry in report["entries"]:
        assert entry["observed_order"] is not None
        assert entry["observed_order"] > 1.6
