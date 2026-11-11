"""Manufactured nonlinear (temperature-dependent conductivity) cases with curvature.

The legacy ``temperature_dependent_diffusivity`` case in :mod:`heat_solver.cases`
has a solution that is linear in space, which every consistent scheme
reproduces exactly; it cannot measure spatial order.  This case has a curved
solution and an anisotropic, temperature-dependent tensor.
"""

from __future__ import annotations

import numpy as np

from .dmp import anisotropic_tensor


def anisotropic_nonlinear_case(beta=0.5, ratio=5.0, angle=np.pi / 6, scale=0.1):
    """``u = 1 + e^{-t} sin(pi x) sin(pi y)`` on ``[-1, 1]^2`` with ``K(u) = (1 + beta u) K0``.

    ``Q = u_t - (1 + beta u) K0 : grad^2 u - beta grad u . K0 grad u``.
    """
    K0 = anisotropic_tensor(ratio=ratio, angle=angle) * scale

    def exact(x, y, t):
        return 1.0 + np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y)

    def alpha(x, y, temperature):
        T = np.asarray(temperature, dtype=float)
        return (1.0 + beta * T)[..., None, None] * K0

    def source(x, y, t):
        e = np.exp(-t)
        sx, sy, cx, cy = np.sin(np.pi * x), np.sin(np.pi * y), np.cos(np.pi * x), np.cos(np.pi * y)
        u = 1.0 + e * sx * sy
        ut = -e * sx * sy
        gx, gy = e * np.pi * cx * sy, e * np.pi * sx * cy
        uxx = uyy = -np.pi**2 * e * sx * sy
        uxy = np.pi**2 * e * cx * cy
        lap_k = K0[0, 0] * uxx + (K0[0, 1] + K0[1, 0]) * uxy + K0[1, 1] * uyy
        grad_k_grad = K0[0, 0] * gx * gx + (K0[0, 1] + K0[1, 0]) * gx * gy + K0[1, 1] * gy * gy
        return ut - (1.0 + beta * u) * lap_k - beta * grad_k_grad

    return {"exact": exact, "alpha": alpha, "source": source, "K0": K0, "beta": beta,
            "bbox": (-1.0, 1.0, -1.0, 1.0)}
