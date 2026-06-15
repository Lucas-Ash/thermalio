import numpy as np

from heat_solver import interface_diagnostics as idiag
from heat_solver.geometry import polygon_area_and_centroid
from heat_solver.meshes import generate_square_polygonal_mesh
from heat_solver.phase_change import ApparentHeatCapacityModel


def _mesh(nx, ny, bbox=(0.0, 1.0, 0.0, 0.4)):
    v, p, c = generate_square_polygonal_mesh(nx=nx, ny=ny, bbox=bbox)
    a = np.array([polygon_area_and_centroid(v[poly])[0] for poly in p])
    return v, p, c, a


# --------------------------------------------------------------------------- #
# Step 4: sharp-interface diagnostics (verified against an analytic tanh front)
# --------------------------------------------------------------------------- #
def test_isotherm_crossings_linear():
    s = np.array([0.0, 1.0, 2.0, 3.0])
    vals = np.array([-2.0, -1.0, 1.0, 2.0])  # crosses zero between s=1 and s=2
    crossings = idiag.isotherm_crossings(s, vals, 0.0)
    assert len(crossings) == 1
    assert np.isclose(crossings[0], 1.5)


def test_interface_position_matches_tanh_front():
    pcm = ApparentHeatCapacityModel(-0.25, 0.25, 4.0, 1.0)  # melt isotherm at 0
    A, w, xc = 0.8, 0.18, 0.6
    _, _, c, _ = _mesh(120, 4)
    u = A * np.tanh((c[:, 0] - xc) / w)
    pos = idiag.interface_position(c, u, pcm, axis="x", coord=0.2)
    assert abs(pos - xc) < 0.02  # within ~one cell width


def test_mushy_zone_thickness_matches_analytic():
    pcm = ApparentHeatCapacityModel(-0.25, 0.25, 4.0, 1.0)
    A, w, xc, hw = 0.8, 0.18, 0.6, 0.25
    _, _, c, _ = _mesh(160, 4)
    u = A * np.tanh((c[:, 0] - xc) / w)
    expected = 2.0 * w * np.arctanh(hw / A)
    thickness = idiag.mushy_zone_thickness(c, u, pcm, axis="x", coord=0.2)
    assert abs(thickness - expected) < 0.03


def test_enthalpy_budget_splits_total():
    pcm = ApparentHeatCapacityModel(-0.25, 0.25, 4.0, 1.0)
    _, _, c, a = _mesh(40, 8)
    u = 0.6 * np.tanh((c[:, 0] - 0.5) / 0.2)
    b = idiag.enthalpy_budget(pcm, u, a)
    assert np.isclose(b["sensible"] + b["latent"], b["total"], rtol=1e-12, atol=1e-12)
    lf = idiag.liquid_volume_fraction(pcm, u, a)
    assert 0.0 <= lf <= 1.0


def test_phase_fractions_sum_to_one():
    pcm = ApparentHeatCapacityModel(-0.25, 0.25, 4.0, 1.0)
    _, _, c, a = _mesh(30, 6)
    u = 0.6 * np.tanh((c[:, 0] - 0.5) / 0.2)
    pf = idiag.phase_fractions(pcm, u, a)
    assert np.isclose(pf["solid"] + pf["mushy"] + pf["liquid"], 1.0, atol=1e-12)


def test_front_speed_constant_velocity():
    times = np.array([0.0, 1.0, 2.0, 3.0])
    positions = 0.5 * times  # speed 0.5
    speeds = idiag.front_speed(times, positions)
    assert np.allclose(speeds, 0.5)


# --------------------------------------------------------------------------- #
# Step 5: application-study runners (physical sanity, small configs)
# --------------------------------------------------------------------------- #
def test_pulsed_laser_melting_runner():
    import direction_c_applications as dca

    summary, rows = dca.pulsed_laser_melting(
        nx=24, ny=6, nt=80, n_snapshots=3, t_end=0.4, t_pulse=0.2)
    assert summary["scenario"] == "pulsed_laser_melting"
    assert summary["final_injected_energy"] > 0.0
    # Energy closure: enthalpy rise should match injected boundary energy well.
    assert abs(summary["final_energy_closure_residual"]) < 0.15 * summary["final_injected_energy"]
    # Some melting occurred and the liquid fraction is non-decreasing in time.
    lf = [r["liquid_fraction"] for r in rows]
    assert lf[-1] >= lf[0] >= 0.0
    assert summary["final_liquid_fraction"] > 0.0


def test_cryosurgery_freezing_runner():
    import direction_c_applications as dca

    summary, rows = dca.cryosurgery_freezing(
        nx=20, ny=20, nt=48, n_snapshots=3, t_end=0.4)
    assert summary["scenario"] == "cryosurgery_freezing"
    assert summary["final_frozen_fraction"] > 0.0      # the probe freezes some tissue
    assert summary["enthalpy_removed"] > 0.0           # energy is extracted
    frozen = [r["frozen_fraction"] for r in rows]
    assert frozen[-1] >= frozen[0]                     # freezing advances in time


def test_moving_scan_melt_pool_runner():
    import direction_c_applications as dca

    summary, rows = dca.moving_scan_melt_pool(
        nx=20, ny=10, nt=36, n_snapshots=3, t_end=0.28)
    assert summary["scenario"] == "moving_scan_melt_pool"
    assert summary["final_source_energy"] > 0.0
    assert summary["final_liquid_fraction"] > 0.0
    assert summary["final_melt_pool_length"] > 0.0
    assert rows[-1]["laser_x"] > rows[0]["laser_x"]


def test_dual_pulse_remelting_runner():
    import direction_c_applications as dca

    summary, rows = dca.dual_pulse_remelting(
        nx=22, ny=6, nt=42, n_snapshots=4, t_end=0.42)
    assert summary["scenario"] == "dual_pulse_remelting"
    assert summary["final_injected_energy"] > 0.0
    assert summary["peak_liquid_fraction"] > 0.0
    assert max(r["peak_temperature"] for r in rows) > rows[0]["peak_temperature"]


def test_rapid_solidification_quench_runner():
    import direction_c_applications as dca

    summary, rows = dca.rapid_solidification_quench(
        nx=18, ny=10, nt=34, n_snapshots=3, t_end=0.24)
    assert summary["scenario"] == "rapid_solidification_quench"
    assert summary["final_liquid_fraction"] < summary["initial_liquid_fraction"]
    assert summary["enthalpy_removed"] > 0.0
    assert rows[-1]["solid_fraction"] >= rows[0]["solid_fraction"]


def test_buried_hot_inclusion_relaxation_runner():
    import direction_c_applications as dca

    summary, rows = dca.buried_hot_inclusion_relaxation(
        nx=20, ny=20, nt=36, n_snapshots=3, t_end=0.24)
    assert summary["scenario"] == "buried_hot_inclusion_relaxation"
    assert summary["final_liquid_fraction"] < summary["initial_liquid_fraction"]
    assert summary["enthalpy_removed"] > 0.0
    assert rows[-1]["peak_temperature"] <= rows[0]["peak_temperature"]
