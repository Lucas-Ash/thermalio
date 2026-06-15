"""Sharp-interface diagnostics on top of the apparent-capacity phase-change model.

Direction C, computing-capability step 4.  Even without explicit front tracking,
these post-processing helpers extract the physically meaningful interface
quantities from a cell-centered temperature field and an
:class:`~heat_solver.phase_change.ApparentHeatCapacityModel`:

* liquid-fraction field and area-weighted liquid volume fraction,
* enthalpy budget split into sensible and latent parts,
* melt-isotherm interface position(s) along a centerline,
* mushy-zone thickness (extent of the solidus..liquidus band),
* front speed from a time series of interface positions.

These make comparisons against Stefan theory and front-position data
straightforward, and feed the Direction C application runners (step 5).
"""

from __future__ import annotations

import numpy as np


def liquid_fraction_field(model, u):
    """Per-cell liquid fraction in [0, 1]."""
    return np.asarray(model.liquid_fraction(np.asarray(u, dtype=float)), dtype=float)


def liquid_volume_fraction(model, u, areas):
    """Area-weighted mean liquid fraction over the domain."""
    areas = np.asarray(areas, dtype=float)
    lf = liquid_fraction_field(model, u)
    return float(np.sum(areas * lf) / max(np.sum(areas), 1e-300))


def phase_fractions(model, u, areas):
    """Area fractions of the solid / mushy / liquid phases."""
    u = np.asarray(u, dtype=float)
    areas = np.asarray(areas, dtype=float)
    total = max(float(np.sum(areas)), 1e-300)
    solid = u < model.solidus_temperature
    liquid = u > model.liquidus_temperature
    mushy = ~(solid | liquid)
    return {
        "solid": float(np.sum(areas[solid]) / total),
        "mushy": float(np.sum(areas[mushy]) / total),
        "liquid": float(np.sum(areas[liquid]) / total),
    }


def enthalpy_budget(model, u, areas):
    """Total / sensible / latent enthalpy integrated over the domain.

    Sensible enthalpy is ``c_p * T`` and latent enthalpy is
    ``L * liquid_fraction(T)``; their sum is the model enthalpy ``H(T)``.
    """
    u = np.asarray(u, dtype=float)
    areas = np.asarray(areas, dtype=float)
    sensible = float(np.sum(areas * model.specific_heat * u))
    latent = float(np.sum(areas * model.latent_heat * liquid_fraction_field(model, u)))
    total = float(np.sum(areas * np.asarray(model.enthalpy(u), dtype=float)))
    return {"total": total, "sensible": sensible, "latent": latent}


def melt_temperature(model):
    """Representative melt isotherm (mid-point of the phase interval)."""
    return 0.5 * (model.solidus_temperature + model.liquidus_temperature)


def centerline(centers, values, axis="x", coord=0.0):
    """Extract ``(s, values)`` along the mesh row closest to ``coord`` on the
    other axis, sorted by the requested ``axis`` coordinate."""
    centers = np.asarray(centers, dtype=float)
    values = np.asarray(values, dtype=float)
    other = 1 if axis == "x" else 0
    main = 0 if axis == "x" else 1
    score = np.abs(centers[:, other] - coord)
    row = score <= np.min(score) + 1e-12
    s = centers[row, main]
    v = values[row]
    order = np.argsort(s)
    return s[order], v[order]


def isotherm_crossings(s, values, level):
    """Linearly-interpolated coordinates where ``values`` crosses ``level``."""
    s = np.asarray(s, dtype=float)
    values = np.asarray(values, dtype=float) - float(level)
    crossings = []
    for i in range(len(s) - 1):
        a, b = values[i], values[i + 1]
        if a == 0.0:
            crossings.append(float(s[i]))
        elif a * b < 0.0:
            crossings.append(float(s[i] - a * (s[i + 1] - s[i]) / (b - a)))
    if len(s) and values[-1] == 0.0:
        crossings.append(float(s[-1]))
    return np.array(crossings, dtype=float)


def interface_position(centers, u, model, axis="x", coord=0.0, level=None, pick="first"):
    """Melt-isotherm interface coordinate(s) along a centerline.

    ``pick`` selects ``"first"``, ``"last"``, ``"all"`` crossings (``None`` if
    the isotherm is not crossed on the line).
    """
    level = melt_temperature(model) if level is None else float(level)
    s, v = centerline(centers, u, axis=axis, coord=coord)
    crossings = isotherm_crossings(s, v, level)
    if pick == "all":
        return crossings
    if crossings.size == 0:
        return None
    return float(crossings[0] if pick == "first" else crossings[-1])


def mushy_zone_thickness(centers, u, model, axis="x", coord=0.0):
    """Spatial extent of the solidus..liquidus band along a centerline (0 if the
    band is not resolved on the line)."""
    s, v = centerline(centers, u, axis=axis, coord=coord)
    inside = (v >= model.solidus_temperature) & (v <= model.liquidus_temperature)
    if not np.any(inside):
        return 0.0
    return float(np.max(s[inside]) - np.min(s[inside]))


def front_speed(times, positions):
    """Finite-difference front speed ``d(position)/dt`` (NaN-safe)."""
    times = np.asarray(times, dtype=float)
    positions = np.asarray(positions, dtype=float)
    return np.gradient(positions, times)
