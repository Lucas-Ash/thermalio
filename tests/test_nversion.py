import numpy as np

from heat_solver.nversion import SCHEME_VARIANTS, run_nversion


def test_nversion_agreement_smooth_dirichlet():
    # Smooth Dirichlet diffusion: independent flux schemes on the same mesh must
    # agree tightly, and every run must be accurate.
    report = run_nversion(
        "source_driven_sine", alpha=0.1, dt=1e-4, t_init=0.0, t_end=0.02,
        bbox=(-1.0, 1.0, -1.0, 1.0), n=32, tol=2e-2,
    )
    assert report["all_agree"], report["within_mesh_max_spread"]
    assert report["accuracy_ok"], report["errors_L2_rel"]
    # Every mesh ran multiple schemes, so each has a recorded spread.
    assert set(report["within_mesh_max_spread"]) == {
        "square_polygonal",
        "nonorthogonal_tiled_polygonal",
    }


def test_nversion_cross_mesh_agreement():
    # Independent mesh types (square vs skewed tiled), interpolated onto a common
    # grid, must agree to within the looser cross-mesh tolerance.
    report = run_nversion(
        "source_driven_sine", alpha=0.1, dt=1e-4, t_init=0.0, t_end=0.02,
        bbox=(-1.0, 1.0, -1.0, 1.0), n=32, tol=2e-2, cross_tol=5e-2,
    )
    assert report["cross_mesh_max_spread"] is not None
    assert report["cross_mesh_ok"], report["cross_mesh_max_spread"]
    # Cross-mesh spread is looser than within-mesh (different discretizations).
    assert report["cross_mesh_max_spread"] >= max(report["within_mesh_max_spread"].values())


def test_nversion_skips_unsupported_variant():
    # Explicitly request the invalid mpfa+reconstructed combo; it must be skipped.
    report = run_nversion(
        "source_driven_sine", n=20, meshes=("square_polygonal",),
        variants=(("tpfa", "tpfa"), ("mpfa", "reconstructed")),
    )
    reasons = {s["reason"] for s in report["skipped"]}
    assert any("mpfa+reconstructed" in r for r in reasons)


def test_nversion_records_solver_failure_as_skip():
    # MPFA is singular on the mixed tiled mesh; the harness records it instead of
    # crashing -- a useful robustness finding, not a fatal error.
    report = run_nversion(
        "source_driven_sine", n=24, meshes=("nonorthogonal_tiled_polygonal",), tol=3e-2,
    )
    failed = [s for s in report["skipped"] if "solver failed" in s["reason"]]
    assert any(s["variant"] == "mpfa/tpfa" for s in failed)
    # The surviving TPFA-family schemes still agree.
    assert report["all_agree"]


def test_nversion_square_only_three_schemes_present():
    report = run_nversion(
        "source_driven_sine", n=24, meshes=("square_polygonal",), tol=2e-2,
    )
    labels = {key.split("[")[1].rstrip("] ") for key in report["errors_L2_rel"]}
    assert "tpfa/tpfa" in labels
    assert "tpfa/reconstructed" in labels
    assert "mpfa/tpfa" in labels
    assert report["all_agree"]
