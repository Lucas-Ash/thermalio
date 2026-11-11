import numpy as np
import pytest

from heat_solver.afc import AFCMonotoneSolver
from heat_solver.dmp import anisotropic_tensor, bound_excursion, run_dmp_case
from heat_solver.geometry import polygon_area_and_centroid
from heat_solver.meshes import (
    generate_nonorthogonal_polygonal_mesh,
    generate_nonorthogonal_tiled_polygonal_mesh,
)
from heat_solver.polygonal import PolygonalHeatSolver


def _block_ic(centers, blk=(-0.35, 0.35, -0.35, 0.35)):
    x, y = centers[:, 0], centers[:, 1]
    return ((x >= blk[0]) & (x <= blk[1]) & (y >= blk[2]) & (y <= blk[3])).astype(float)


def _rel_l2(u, ue, areas):
    return float(np.sqrt(np.sum(areas * (u - ue) ** 2) / np.sum(areas * ue**2)))


def test_afc_is_bound_preserving_where_highorder_overshoots():
    # Steep front + strong anisotropy on a skewed mesh: the reconstructed flux
    # overshoots; AFC must stay within [0, 1] exactly.
    alpha = anisotropic_tensor(ratio=20.0, angle=np.pi / 6)
    v, p, c = generate_nonorthogonal_tiled_polygonal_mesh(nx_tiles=8, ny_tiles=8, bbox=(-1, 1, -1, 1), skew=0.4)

    recon = run_dmp_case(v, p, c, alpha, {
        "flux_scheme": "tpfa", "flux_discretization": "reconstructed", "nonorthogonal_correction": True,
    }, dt=1.5e-3, t_end=9e-3)
    assert recon["bounds"]["overshoot"] > 1e-3  # confirm the regime really violates the DMP

    solver = AFCMonotoneSolver(
        v, p, alpha, 1.5e-3, bc_func=lambda x, y, t: np.zeros_like(x),
        source_func=lambda x, y, t: np.zeros_like(x), flux_discretization="reconstructed",
    )
    _, u = solver.solve(_block_ic(c), 0.0, 9e-3)
    be = bound_excursion(u, 0.0, 1.0)
    assert be["overshoot"] == 0.0
    assert be["undershoot"] == 0.0


def test_afc_more_accurate_than_linear_monotone():
    # On a smooth anisotropic solution on a skewed mesh, AFC recovers much of the
    # high-order accuracy lost by the linear M-matrix projection, while both are
    # bound-preserving.
    pytest.importorskip("sympy")
    from heat_solver.mms import manufactured_case

    alpha = anisotropic_tensor(ratio=5.0, angle=np.pi / 6)
    case = manufactured_case("exp(-t)*sin(pi*x)*sin(pi*y)", alpha=alpha.tolist(), model="diffusion")
    v, p, c = generate_nonorthogonal_polygonal_mesh(nx=24, ny=24, bbox=(-1, 1, -1, 1), skew=0.3)
    areas = np.array([polygon_area_and_centroid(v[poly])[0] for poly in p])
    u0 = case["solution"](c[:, 0], c[:, 1], 0.0)
    ue = case["solution"](c[:, 0], c[:, 1], 0.004)

    def run_poly(monotone):
        s = PolygonalHeatSolver(
            v, p, alpha, 2e-5, bc_type="dirichlet", bc_func=case["boundary"],
            source_func=case["source"], flux_discretization="reconstructed",
            nonorthogonal_correction=True, monotone=monotone,
        )
        _, u = s.solve(u0, 0.0, 0.004)
        return _rel_l2(u, ue, areas)

    err_recon = run_poly(False)
    err_monotone = run_poly(True)
    s = AFCMonotoneSolver(v, p, alpha, 2e-5, bc_func=case["boundary"],
                          source_func=case["source"], flux_discretization="reconstructed")
    _, u_afc = s.solve(u0, 0.0, 0.004)
    err_afc = _rel_l2(u_afc, ue, areas)

    assert err_afc < 0.7 * err_monotone          # clearly better than linear monotone
    assert err_afc < 3.0 * err_recon             # within a small factor of high-order


def test_afc_recovers_high_order_when_no_limiting_needed():
    # Smooth solution on a square isotropic mesh has no spurious extrema, so AFC
    # should track the high-order reconstructed result closely.
    pytest.importorskip("sympy")
    from heat_solver.mms import manufactured_case
    from heat_solver.meshes import generate_square_polygonal_mesh

    case = manufactured_case("exp(-t)*sin(pi*x)*sin(pi*y)", alpha=0.1, model="diffusion")
    v, p, c = generate_square_polygonal_mesh(nx=24, ny=24, bbox=(-1, 1, -1, 1))
    areas = np.array([polygon_area_and_centroid(v[poly])[0] for poly in p])
    u0 = case["solution"](c[:, 0], c[:, 1], 0.0)
    ue = case["solution"](c[:, 0], c[:, 1], 0.01)

    s_recon = PolygonalHeatSolver(v, p, 0.1, 2e-5, bc_type="dirichlet", bc_func=case["boundary"],
                                  source_func=case["source"], flux_discretization="reconstructed")
    _, u_recon = s_recon.solve(u0, 0.0, 0.01)
    s_afc = AFCMonotoneSolver(v, p, 0.1, 2e-5, bc_func=case["boundary"],
                              source_func=case["source"], flux_discretization="reconstructed")
    _, u_afc = s_afc.solve(u0, 0.0, 0.01)

    assert _rel_l2(u_afc, ue, areas) < 1.5e-3
    # Both should be close on this benign case.
    assert _rel_l2(u_afc, u_recon, areas) < 5e-2


def test_smoothness_relaxation_recovers_highorder_accuracy():
    # The strict Zalesak limiter clips the smooth peak; the smoothness relaxation
    # (smoothness_factor > 0) recovers the high-order scheme's accuracy.
    pytest.importorskip("sympy")
    from heat_solver.mms import manufactured_case

    alpha = anisotropic_tensor(ratio=5.0, angle=np.pi / 6)
    case = manufactured_case("exp(-t)*sin(pi*x)*sin(pi*y)", alpha=alpha.tolist(), model="diffusion")
    v, p, c = generate_nonorthogonal_polygonal_mesh(nx=24, ny=24, bbox=(-1, 1, -1, 1), skew=0.3)
    areas = np.array([polygon_area_and_centroid(v[poly])[0] for poly in p])
    u0 = case["solution"](c[:, 0], c[:, 1], 0.0)
    ue = case["solution"](c[:, 0], c[:, 1], 0.004)

    def afc_err(sf):
        s = AFCMonotoneSolver(v, p, alpha, 2e-5, bc_func=case["boundary"],
                              source_func=case["source"], flux_discretization="reconstructed",
                              smoothness_factor=sf)
        _, u = s.solve(u0, 0.0, 0.004)
        return _rel_l2(u, ue, areas)

    err_strict = afc_err(0.0)
    err_relaxed = afc_err(1.0)
    s_recon = PolygonalHeatSolver(v, p, alpha, 2e-5, bc_type="dirichlet", bc_func=case["boundary"],
                                  source_func=case["source"], flux_discretization="reconstructed",
                                  nonorthogonal_correction=True)
    _, u_recon = s_recon.solve(u0, 0.0, 0.004)
    err_recon = _rel_l2(u_recon, ue, areas)

    assert err_relaxed < 0.6 * err_strict          # clipping penalty removed
    assert err_relaxed < 1.5 * err_recon           # matches the high-order scheme


def test_smoothness_relaxation_overshoot_is_mesh_vanishing():
    # With smoothness_factor > 0 the steep-front overshoot is essentially
    # non-oscillatory: it shrinks under refinement and is small on a fine mesh.
    alpha = anisotropic_tensor(ratio=20.0, angle=np.pi / 6)

    def front_overshoot(tiles):
        v, p, c = generate_nonorthogonal_tiled_polygonal_mesh(
            nx_tiles=tiles, ny_tiles=tiles, bbox=(-1, 1, -1, 1), skew=0.4)
        s = AFCMonotoneSolver(v, p, alpha, 1.5e-3, bc_func=lambda x, y, t: np.zeros_like(x),
                              source_func=lambda x, y, t: np.zeros_like(x),
                              flux_discretization="reconstructed", smoothness_factor=1.0)
        _, u = s.solve(_block_ic(c), 0.0, 9e-3)
        be = bound_excursion(u, 0.0, 1.0)
        return max(be["overshoot"], be["undershoot"])

    coarse = front_overshoot(10)
    fine = front_overshoot(24)
    assert fine < coarse          # overshoot shrinks under refinement
    assert fine < 5e-3            # and is small on the fine mesh


def test_global_mpp_strictly_bound_preserving_at_coarse_resolution():
    # The global maximum-principle limiter enforces [m, M] exactly even on a
    # coarse mesh where the relaxed local limiter overshoots; bounds auto-detect
    # to [0, 1] here (block in [0, 1], zero Dirichlet data).
    alpha = anisotropic_tensor(ratio=20.0, angle=np.pi / 6)
    v, p, c = generate_nonorthogonal_tiled_polygonal_mesh(nx_tiles=10, ny_tiles=10, bbox=(-1, 1, -1, 1), skew=0.4)
    s = AFCMonotoneSolver(v, p, alpha, 1.5e-3, bc_func=lambda x, y, t: np.zeros_like(x),
                          source_func=lambda x, y, t: np.zeros_like(x),
                          flux_discretization="reconstructed", limiter="global")
    _, u = s.solve(_block_ic(c), 0.0, 9e-3)
    be = bound_excursion(u, 0.0, 1.0)
    assert be["overshoot"] == 0.0
    assert be["undershoot"] == 0.0


def test_global_mpp_recovers_highorder_accuracy():
    # The global limiter does not clip smooth interior extrema (they lie inside
    # [m, M]), so it recovers the high-order scheme's accuracy that the strict
    # local limiter loses -- while remaining strictly bound-preserving.
    pytest.importorskip("sympy")
    from heat_solver.mms import manufactured_case

    alpha = anisotropic_tensor(ratio=5.0, angle=np.pi / 6)
    case = manufactured_case("exp(-t)*sin(pi*x)*sin(pi*y)", alpha=alpha.tolist(), model="diffusion")
    v, p, c = generate_nonorthogonal_polygonal_mesh(nx=24, ny=24, bbox=(-1, 1, -1, 1), skew=0.3)
    areas = np.array([polygon_area_and_centroid(v[poly])[0] for poly in p])
    u0 = case["solution"](c[:, 0], c[:, 1], 0.0)
    ue = case["solution"](c[:, 0], c[:, 1], 0.004)

    def afc_err(limiter):
        s = AFCMonotoneSolver(v, p, alpha, 2e-5, bc_func=case["boundary"], source_func=case["source"],
                              flux_discretization="reconstructed", limiter=limiter)
        _, u = s.solve(u0, 0.0, 0.004)
        return _rel_l2(u, ue, areas), u

    err_local, _ = afc_err("local")
    err_global, u_global = afc_err("global")
    s_recon = PolygonalHeatSolver(v, p, alpha, 2e-5, bc_type="dirichlet", bc_func=case["boundary"],
                                  source_func=case["source"], flux_discretization="reconstructed",
                                  nonorthogonal_correction=True)
    _, u_recon = s_recon.solve(u0, 0.0, 0.004)
    err_recon = _rel_l2(u_recon, ue, areas)

    assert err_global < 0.6 * err_local      # no smooth-extremum clipping
    assert err_global < 1.5 * err_recon      # matches the high-order scheme
    # Still strictly within the auto-detected global bounds [-1, 1].
    assert u_global.min() >= -1.0 - 1e-9 and u_global.max() <= 1.0 + 1e-9


def test_afc_rejects_bad_limiter():
    v, p, _ = generate_nonorthogonal_polygonal_mesh(nx=4, ny=4, bbox=(-1, 1, -1, 1))
    with pytest.raises(ValueError):
        AFCMonotoneSolver(v, p, 0.1, 0.01, limiter="bogus")


# --- PR5: sign-changing sources ---------------------------------------------

def _heater_sink(q0):
    def src(x, y, t):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        heat = ((x - 0.55) ** 2 + (y + 0.55) ** 2 < 0.09).astype(float)
        sink = ((x + 0.55) ** 2 + (y - 0.55) ** 2 < 0.09).astype(float)
        return q0 * (heat - sink)
    return src


def _zero(x, y, t):
    return np.zeros_like(np.asarray(x, float))


def test_source_envelope_reduces_to_global_without_source():
    alpha = anisotropic_tensor(ratio=20.0, angle=np.pi / 6)
    v, p, c = generate_nonorthogonal_tiled_polygonal_mesh(nx_tiles=8, ny_tiles=8, bbox=(-1, 1, -1, 1), skew=0.4)
    out = {}
    for lim in ("global", "source_envelope"):
        s = AFCMonotoneSolver(v, p, alpha, 1.5e-3, bc_func=_zero, source_func=_zero, limiter=lim)
        out[lim] = s.solve(_block_ic(c), 0.0, 6e-3)[1]
    assert np.max(np.abs(out["global"] - out["source_envelope"])) < 1e-13


def test_source_limiters_respect_rigorous_envelope_where_highorder_violates_it():
    alpha = anisotropic_tensor(ratio=20.0, angle=np.pi / 6)
    v, p, c = generate_nonorthogonal_tiled_polygonal_mesh(nx_tiles=10, ny_tiles=10, bbox=(-1, 1, -1, 1), skew=0.4)
    src, dt, n = _heater_sink(1.0), 1.5e-3, 12
    high = PolygonalHeatSolver(v, p, alpha, dt, bc_type="dirichlet", bc_func=_zero, source_func=src,
                               flux_discretization="reconstructed", nonorthogonal_correction=True)
    q_int = src(c[:, 0], c[:, 1], 0.0)[~high.is_boundary]
    lo, hi, u, worst = 0.0, 1.0, _block_ic(c), 0.0
    los, his = [], []
    for k in range(n):
        u = high.solve(u, k * dt, (k + 1) * dt)[1]
        hi += dt * max(q_int.max(), 0.0)
        lo -= dt * max((-q_int).max(), 0.0)
        los.append(lo)
        his.append(hi)
        worst = max(worst, u.max() - hi, lo - u.min())
    assert worst > 1e-2  # the high-order scheme leaves the physical envelope
    for lim in ("source_envelope", "source_local"):
        s = AFCMonotoneSolver(v, p, alpha, dt, bc_func=_zero, source_func=src, limiter=lim)
        s.solve(_block_ic(c), 0.0, n * dt)
        for (t, umin, umax), l_, h_ in zip(s.extrema_history, los, his):
            assert umax <= h_ + 1e-10 and umin >= l_ - 1e-10
        assert min(m for _, m, _ in s.extrema_history) < -1e-3  # the sink's minimum is NOT clipped


def test_source_envelope_keeps_high_order_accuracy_when_solution_leaves_initial_range():
    pytest.importorskip("sympy")
    from heat_solver.mms import manufactured_case

    alpha = anisotropic_tensor(ratio=5.0, angle=np.pi / 6)
    case = manufactured_case("(0.2+4*t)*sin(pi*x)*sin(pi*y)", alpha=alpha.tolist(), model="diffusion")
    v, p, c = generate_nonorthogonal_polygonal_mesh(nx=24, ny=24, bbox=(-1, 1, -1, 1), skew=0.3)
    areas = np.array([polygon_area_and_centroid(v[poly])[0] for poly in p])
    u0 = case["solution"](c[:, 0], c[:, 1], 0.0)
    ue = case["solution"](c[:, 0], c[:, 1], 0.05)
    errs = {}
    for lim in ("global", "source_envelope"):
        s = AFCMonotoneSolver(v, p, alpha, 1e-3, bc_func=case["boundary"], source_func=case["source"], limiter=lim)
        errs[lim] = _rel_l2(s.solve(u0, 0.0, 0.05)[1], ue, areas)
    high = PolygonalHeatSolver(v, p, alpha, 1e-3, bc_type="dirichlet", bc_func=case["boundary"],
                               source_func=case["source"], flux_discretization="reconstructed",
                               nonorthogonal_correction=True)
    err_high = _rel_l2(high.solve(u0, 0.0, 0.05)[1], ue, areas)
    assert errs["source_envelope"] < 1.01 * err_high
    assert errs["global"] > 1.8 * errs["source_envelope"]  # PR4 clips once u leaves [-0.2, 0.2]
