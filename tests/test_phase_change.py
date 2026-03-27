import numpy as np

from heat_solver.phase_change import ApparentHeatCapacityModel


def test_apparent_heat_capacity_piecewise_behavior():
    model = ApparentHeatCapacityModel(
        solidus_temperature=-0.1,
        liquidus_temperature=0.1,
        latent_heat=4.0,
        specific_heat=2.0,
    )
    temperature = np.array([-0.2, -0.05, 0.0, 0.05, 0.2])
    cp_eff = model.effective_heat_capacity(temperature)
    expected_mushy_cp = 2.0 + 4.0 / 0.2
    assert np.isclose(cp_eff[0], 2.0)
    assert np.isclose(cp_eff[-1], 2.0)
    assert np.allclose(cp_eff[1:4], expected_mushy_cp)


def test_enthalpy_temperature_inverse_consistency():
    model = ApparentHeatCapacityModel(
        solidus_temperature=-0.05,
        liquidus_temperature=0.05,
        latent_heat=6.0,
        specific_heat=1.5,
    )
    temperature = np.linspace(-0.3, 0.3, 41)
    enthalpy = model.enthalpy(temperature)
    reconstructed_temperature = model.temperature_from_enthalpy(enthalpy)
    assert np.allclose(reconstructed_temperature, temperature, atol=1e-12, rtol=1e-12)
