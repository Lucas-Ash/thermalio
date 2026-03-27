from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ApparentHeatCapacityModel:
    """
    Apparent heat-capacity phase-change model.

    The latent heat contribution is distributed uniformly over
    [solidus_temperature, liquidus_temperature].
    """

    solidus_temperature: float
    liquidus_temperature: float
    latent_heat: float
    specific_heat: float = 1.0

    def __post_init__(self):
        if self.liquidus_temperature < self.solidus_temperature:
            raise ValueError("liquidus_temperature must be >= solidus_temperature.")
        if self.latent_heat < 0.0:
            raise ValueError("latent_heat must be non-negative.")
        if self.specific_heat <= 0.0:
            raise ValueError("specific_heat must be strictly positive.")

    @property
    def transition_width(self):
        return self.liquidus_temperature - self.solidus_temperature

    def liquid_fraction(self, temperature):
        temperature = np.asarray(temperature, dtype=float)
        width = self.transition_width
        if width <= 0.0:
            return (temperature >= self.liquidus_temperature).astype(float)
        return np.clip((temperature - self.solidus_temperature) / width, 0.0, 1.0)

    def d_liquid_fraction_d_temperature(self, temperature):
        temperature = np.asarray(temperature, dtype=float)
        width = self.transition_width
        if width <= 0.0:
            return np.zeros_like(temperature)
        inside = (temperature >= self.solidus_temperature) & (temperature <= self.liquidus_temperature)
        return np.where(inside, 1.0 / width, 0.0)

    def effective_heat_capacity(self, temperature):
        return self.specific_heat + self.latent_heat * self.d_liquid_fraction_d_temperature(temperature)

    def enthalpy(self, temperature):
        temperature = np.asarray(temperature, dtype=float)
        return self.specific_heat * temperature + self.latent_heat * self.liquid_fraction(temperature)

    def temperature_from_enthalpy(self, enthalpy):
        enthalpy = np.asarray(enthalpy, dtype=float)
        width = self.transition_width
        if width <= 0.0:
            temp = np.where(
                enthalpy <= self.specific_heat * self.solidus_temperature,
                enthalpy / self.specific_heat,
                (enthalpy - self.latent_heat) / self.specific_heat,
            )
            return temp

        h_solidus = self.specific_heat * self.solidus_temperature
        h_liquidus = self.specific_heat * self.liquidus_temperature + self.latent_heat
        coeff = self.specific_heat + self.latent_heat / width

        temp = np.empty_like(enthalpy, dtype=float)
        solid_mask = enthalpy <= h_solidus
        liquid_mask = enthalpy >= h_liquidus
        mushy_mask = ~(solid_mask | liquid_mask)

        temp[solid_mask] = enthalpy[solid_mask] / self.specific_heat
        temp[liquid_mask] = (enthalpy[liquid_mask] - self.latent_heat) / self.specific_heat
        temp[mushy_mask] = (
            enthalpy[mushy_mask] + self.latent_heat * self.solidus_temperature / width
        ) / coeff
        return temp
