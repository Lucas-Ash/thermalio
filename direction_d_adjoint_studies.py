"""Direction D follow-up: discrete adjoints beyond scalar Pennes perfusion.

Generates figures/JSON under ``test_plots/direction_D_inverse_problems/adjoints/``:

1. ``adjoint_taylor_tests``          Taylor-remainder verification of every adjoint
   (linear Dirichlet/Robin/flux field adjoint, fractional L1, enthalpy Stefan
   with tau = 0 and tau > 0).
2. ``adjoint_cost_scaling``          gradient cost vs number of parameters,
   adjoint vs finite differences.
3. ``conductivity_field_inversion``  49-coefficient conductivity field recovered
   from sparse noisy sensors (4 heater experiments) with L-BFGS-B + adjoint.
4. ``nonlinear_parameter_recovery``  latent heat + diffusivity from melting
   thermocouple data (enthalpy-Stefan adjoint), objective profiles with
   adjoint tangents for the Cattaneo relaxation time and fractional order.

Run: ``MPLCONFIGDIR=/tmp/matplotlib python direction_d_adjoint_studies.py``
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from heat_solver.adjoint import (
    LinearHeatAdjoint,
    enthalpy_stefan_adjoint,
    fractional_heat_adjoint,
    taylor_test,
)
from heat_solver.enthalpy import EnthalpyStefanSolver, SmoothPhaseChangeModel
from heat_solver.meshes import generate_square_polygonal_mesh

OUT = Path("test_plots/direction_D_inverse_problems/adjoints")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.grid": True, "grid.alpha": 0.3})


def gauss(x0, y0, s):
    return lambda x, y: np.exp(-((np.asarray(x) - x0) ** 2 + (np.asarray(y) - y0) ** 2) / (2 * s * s))


def ones(x, y):
    return np.ones_like(np.asarray(x, dtype=float))


def mode(c):
    return np.sin(np.pi * c[:, 0]) * np.sin(np.pi * c[:, 1])


TIGHT = {"tol": 1e-14, "rtol_residual": 1e-14}


# ---------------------------------------------------------------------------
# 1. Taylor tests
# ---------------------------------------------------------------------------


def study_taylor():
    v, p, _ = generate_square_polygonal_mesh(nx=12, ny=12, bbox=(0.0, 1.0, 0.0, 1.0))
    hs = 0.05 * 0.5 ** np.arange(8)
    results = {}
    for bc in ("dirichlet", "robin", "flux"):
        adj = LinearHeatAdjoint(
            v, p, 0.02, [ones, gauss(0.3, 0.3, 0.25), gauss(0.7, 0.6, 0.25)],
            perfusion_basis=[gauss(0.5, 0.5, 0.25)], source_basis=[gauss(0.4, 0.6, 0.15)],
            source_time_profile=lambda t: np.exp(-t), bc_type=bc,
            bc_func=None if bc == "robin" else (lambda x, y, t: 0.1 * np.sin(np.pi * x)), robin_ambient=0.2,
        )
        truth = np.array([0.05, 0.03, 0.02, 0.5, 2.0] + ([3.0] if bc == "robin" else []))
        u0 = mode(adj.cell_centers)
        times, sens = [0.1, 0.2, 0.3], np.arange(0, adj.M, 7)
        st, dt, _, _ = adj.forward(truth, u0, 0.3)
        obs = np.vstack([st[round(t / dt)][sens] for t in times])
        th = 1.3 * truth
        g = adj.gradient(th, u0, 0.3, times, obs, sens).gradient
        results[f"linear ({bc})"] = taylor_test(lambda q: adj.objective(q, u0, 0.3, times, obs, sens), g, th, hs=hs)
    u0 = mode(generate_square_polygonal_mesh(nx=12, ny=12, bbox=(0, 1, 0, 1))[2])
    sens = np.arange(0, 144, 5)
    obs = fractional_heat_adjoint(v, p, 0.1, 0.01, 0.6, u0, 0.2, [0.1, 0.2], np.zeros((2, sens.size)), sens).predicted
    f = lambda q: fractional_heat_adjoint(v, p, q[1], 0.01, q[0], u0, 0.2, [0.1, 0.2], obs, sens).objective
    th = np.array([0.5, 0.13])
    g = fractional_heat_adjoint(v, p, th[1], 0.01, th[0], u0, 0.2, [0.1, 0.2], obs, sens).gradient
    results["fractional L1 (beta, alpha)"] = taylor_test(f, g, th, hs=hs)
    for tau in (0.0, 0.03):
        params = ("latent_heat", "alpha") + (("tau",) if tau > 0 else ())

        def make(q, tau=tau):
            model = SmoothPhaseChangeModel(0.0, 0.15, latent_heat=q[0])
            pulse = lambda x, y, t: np.where(x < 1e-9, 2.0 * np.exp(-t / 0.1), 0.0)
            return EnthalpyStefanSolver(v, p, q[1], 0.01, model, relaxation_time=q[2] if tau > 0 else 0.0,
                                        bc_type="flux", bc_func=pulse, newton_options=TIGHT)
        T0 = -0.3 * np.ones(144)
        sens = np.arange(0, 144, 3)
        truth = np.array([2.0, 0.05] + ([tau] if tau > 0 else []))
        obs = enthalpy_stefan_adjoint(make(truth), T0, 0.3, [0.1, 0.2, 0.3], np.zeros((3, sens.size)), sens,
                                      parameters=params).predicted
        th = truth * np.array([1.2, 0.8] + ([1.3] if tau > 0 else []))
        g = enthalpy_stefan_adjoint(make(th), T0, 0.3, [0.1, 0.2, 0.3], obs, sens, parameters=params).gradient
        f = lambda q, make=make, params=params: enthalpy_stefan_adjoint(make(q), T0, 0.3, [0.1, 0.2, 0.3], obs, sens,
                                                                         parameters=params).objective
        label = "enthalpy Stefan (L, alpha)" if tau == 0 else "Cattaneo-Stefan (L, alpha, tau)"
        results[label] = taylor_test(f, g, th, hs=hs)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6))
    for ax, (name, r) in zip(axes.ravel(), results.items()):
        h = r["h"]
        ax.loglog(h, r["remainder0"], "o--", ms=4, label=r"$|J(\theta+h d)-J(\theta)|$")
        ax.loglog(h, np.maximum(r["remainder1"], 1e-30), "s-", ms=4, label=r"$|J(\theta+h d)-J(\theta)-h\,\nabla J\cdot d|$")
        ax.loglog(h, r["remainder0"][0] * h / h[0], "k:", lw=0.8)
        ax.loglog(h, r["remainder1"][0] * (h / h[0]) ** 2, "k--", lw=0.8)
        ax.set_title(f"{name}\nrates: {r['rate0'][-1]:.2f} / {r['rate1'][-1]:.2f}")
        ax.set_xlabel("$h$")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Taylor tests: first-order remainder O(h), second-order remainder O(h$^2$) $\\Rightarrow$ exact discrete gradient", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "adjoint_taylor_tests.png", dpi=170)
    plt.close(fig)
    return {k: {"rate0": r["rate0"].tolist(), "rate1": r["rate1"].tolist(),
                "remainder1_min": float(np.min(r["remainder1"]))} for k, r in results.items()}


# ---------------------------------------------------------------------------
# 2. Cost scaling
# ---------------------------------------------------------------------------


def rbf_grid(n_side, sigma=None, bbox=(0.0, 1.0)):
    centers = np.linspace(bbox[0], bbox[1], n_side + 2)[1:-1] if n_side > 1 else np.array([0.5])
    sigma = sigma or (0.6 / max(n_side, 1))
    if n_side == 1:
        return [ones], np.array([[0.5, 0.5]])
    funcs, cs = [], []
    for cy in centers:
        for cx in centers:
            funcs.append(gauss(cx, cy, sigma))
            cs.append((cx, cy))
    return funcs, np.array(cs)


def study_cost():
    v, p, _ = generate_square_polygonal_mesh(nx=32, ny=32, bbox=(0.0, 1.0, 0.0, 1.0))
    heater = gauss(0.5, 0.5, 0.15)
    rows = []
    for n_side in (1, 2, 3, 4, 6, 8, 10, 12):
        basis, _ = rbf_grid(n_side)
        adj = LinearHeatAdjoint(v, p, 0.01, basis, background_source=lambda x, y, t: 5.0 * heater(x, y))
        theta = np.full(len(basis), 0.05 if n_side == 1 else 0.02)
        sens = np.arange(0, adj.M, 13)
        times = [0.1, 0.2, 0.3, 0.4, 0.5]
        st, dt, _, _ = adj.forward(theta * 1.1, np.zeros(adj.M), 0.5)
        obs = np.vstack([st[round(t / dt)][sens] for t in times])
        t0 = time.perf_counter(); res = adj.gradient(theta, np.zeros(adj.M), 0.5, times, obs, sens); t_adj = time.perf_counter() - t0
        t0 = time.perf_counter(); f0 = adj.objective(theta, np.zeros(adj.M), 0.5, times, obs, sens); t_fwd = time.perf_counter() - t0
        row = {"n_params": len(basis), "t_adjoint": t_adj, "t_forward": t_fwd, "t_fd_estimate": 2 * len(basis) * t_fwd}
        if len(basis) <= 16:
            t0 = time.perf_counter()
            fd = np.zeros(len(basis))
            for j in range(len(basis)):
                e = np.zeros(len(basis)); e[j] = 1e-6 * theta[j]
                fd[j] = (adj.objective(theta + e, np.zeros(adj.M), 0.5, times, obs, sens)
                         - adj.objective(theta - e, np.zeros(adj.M), 0.5, times, obs, sens)) / (2 * e[j])
            row["t_fd"] = time.perf_counter() - t0
            row["max_rel_diff"] = float(np.max(np.abs(fd - res.gradient)) / np.max(np.abs(fd)))
        rows.append(row)
        del f0
    n = np.array([r["n_params"] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].loglog(n, [r["t_adjoint"] for r in rows], "s-", label="adjoint gradient (1 forward + 1 adjoint)")
    meas = [r for r in rows if "t_fd" in r]
    ax[0].loglog([r["n_params"] for r in meas], [r["t_fd"] for r in meas], "o-", label="central differences (measured)")
    ax[0].loglog(n, [r["t_fd_estimate"] for r in rows], "o:", alpha=0.6, label=r"central differences ($2n$ forward solves)")
    ax[0].set_xlabel("number of conductivity parameters $n$"); ax[0].set_ylabel("wall time per gradient [s]")
    ax[0].set_title("(a) gradient cost, 1024-cell mesh, 50 steps"); ax[0].legend(fontsize=7)
    ax[1].semilogx([r["n_params"] for r in meas], [r["max_rel_diff"] for r in meas], "o-")
    ax[1].set_yscale("log"); ax[1].set_xlabel("$n$")
    ax[1].set_ylabel(r"$\|g_{\mathrm{FD}}-g_{\mathrm{adj}}\|_\infty / \|g_{\mathrm{FD}}\|_\infty$")
    ax[1].set_title("(b) adjoint vs central differences")
    fig.tight_layout()
    fig.savefig(OUT / "adjoint_cost_scaling.png", dpi=170)
    plt.close(fig)
    return rows


# ---------------------------------------------------------------------------
# 3. Conductivity field inversion
# ---------------------------------------------------------------------------


def study_field_inversion(noise=0.01, seed=3, beta=2e-4, sigma=0.08, ridge=0.0):
    v, p, _ = generate_square_polygonal_mesh(nx=28, ny=28, bbox=(0.0, 1.0, 0.0, 1.0))
    bumps, centers = rbf_grid(7, sigma=sigma)
    basis = [ones] + bumps  # kappa = c_0 + sum_j c_j phi_j
    nb = len(basis)
    kappa_true = lambda x, y: 0.02 + 0.03 * gauss(0.32, 0.66, 0.12)(x, y) + 0.02 * gauss(0.7, 0.3, 0.1)(x, y)
    heaters = [gauss(0.3, 0.3, 0.12), gauss(0.7, 0.7, 0.12), gauss(0.3, 0.7, 0.12), gauss(0.7, 0.3, 0.12)]
    sens_xy = np.array([(x, y) for y in np.linspace(0.12, 0.88, 7) for x in np.linspace(0.12, 0.88, 7)])
    times = [0.25, 0.5, 0.75, 1.0]
    rng = np.random.default_rng(seed)
    experiments, data, truths = [], [], []
    for hfun in heaters:
        truth_solver = LinearHeatAdjoint(v, p, 0.025, [kappa_true],
                                         background_source=lambda x, y, t, h=hfun: 4.0 * h(x, y))
        sens = np.array([int(np.argmin(np.sum((truth_solver.cell_centers - s) ** 2, axis=1))) for s in sens_xy])
        st, dt, _, _ = truth_solver.forward([1.0], np.zeros(truth_solver.M), 1.0)
        clean = np.vstack([st[round(t / dt)][sens] for t in times])
        data.append(clean + noise * np.max(np.abs(clean)) * rng.standard_normal(clean.shape))
        truths.append(clean)
        experiments.append(LinearHeatAdjoint(v, p, 0.025, basis,
                                             background_source=lambda x, y, t, h=hfun: 4.0 * h(x, y)))
    ex0 = experiments[0]
    # Smoothness regularization on the 7x7 coefficient grid (discrete gradient).
    side = 7
    D = []
    for j in range(side):
        for i in range(side):
            k = 1 + j * side + i
            if i + 1 < side:
                r = np.zeros(nb); r[k] = -1; r[k + 1] = 1; D.append(r)
            if j + 1 < side:
                r = np.zeros(nb); r[k] = -1; r[k + side] = 1; D.append(r)
    D = np.array(D)
    if ridge > 0.0:  # mild amplitude penalty on the bump coefficients
        D = np.vstack([D, np.sqrt(ridge) * np.eye(nb)[1:]])
    scale = sum(0.5 * np.sum(d**2) for d in data)
    c_ref = 0.02  # background prior level for the regularizer
    history = []

    def fun(c):
        J, g = 0.0, np.zeros(nb)
        for adj, d in zip(experiments, data):
            r = adj.gradient(c, np.zeros(adj.M), 1.0, times, d, sens)
            J += r.objective; g += r.gradient
        Dc = D @ c
        J = J / scale + 0.5 * beta * float(Dc @ Dc) / c_ref**2
        g = g / scale + beta * (D.T @ Dc) / c_ref**2
        history.append(J)
        return J, g

    c0 = np.zeros(nb)
    c0[0] = 0.03  # homogeneous initial guess (truth background 0.02, peak 0.05)
    t0 = time.perf_counter()
    res = minimize(fun, c0, jac=True, method="L-BFGS-B", bounds=[(1e-4, None)] + [(-0.02, None)] * (nb - 1),
                   options={"maxiter": 150, "ftol": 1e-12, "gtol": 1e-10})
    wall = time.perf_counter() - t0
    xc, yc = ex0.cell_centers[:, 0], ex0.cell_centers[:, 1]
    Phi_c = np.column_stack([f(xc, yc) for f in basis])
    k_rec = Phi_c @ res.x
    k_true = kappa_true(xc, yc)
    k_init = Phi_c @ c0
    rel_err = float(np.linalg.norm(k_rec - k_true) / np.linalg.norm(k_true))
    rel_err0 = float(np.linalg.norm(k_init - k_true) / np.linalg.norm(k_true))
    interior = (xc > 0.1) & (xc < 0.9) & (yc > 0.1) & (yc < 0.9)
    rel_err_int = float(np.linalg.norm((k_rec - k_true)[interior]) / np.linalg.norm(k_true[interior]))

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    vmin, vmax = k_true.min(), k_true.max()
    for a, field, title in ((ax[0], k_true, r"(a) true $\kappa(x)$"), (ax[1], k_rec, r"(b) recovered $\kappa$ (1 + 49 coeffs)")):
        im = a.tripcolor(xc, yc, field, shading="gouraud", cmap="magma", vmin=vmin, vmax=vmax)
        a.set_title(title); a.set_aspect("equal"); a.grid(False)
        fig.colorbar(im, ax=a, fraction=0.046)
    ax[0].plot(sens_xy[:, 0], sens_xy[:, 1], "c+", ms=5, label="sensors")
    ax[0].plot(*np.array([[0.3, 0.3], [0.7, 0.7], [0.3, 0.7], [0.7, 0.3]]).T, "w*", ms=9, label="heaters")
    ax[0].legend(fontsize=6, loc="lower left")
    ax[1].plot(centers[:, 0], centers[:, 1], "w.", ms=2)
    im = ax[2].tripcolor(xc, yc, k_rec - k_true, shading="gouraud", cmap="RdBu_r")
    ax[2].set_title(f"(c) error (rel. L2 {rel_err:.1%}, interior {rel_err_int:.1%})"); ax[2].set_aspect("equal"); ax[2].grid(False)
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    ax[3].semilogy(np.array(history) - min(history) + 1e-12 * max(history), color="tab:blue", label=r"$J_k - J_{\min}$")
    ax[3].set_xlabel("objective/gradient evaluations"); ax[3].set_title(f"(d) L-BFGS-B, {res.nit} iterations")
    ax[3].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "conductivity_field_inversion.png", dpi=170)
    plt.close(fig)
    return {"n_params": nb, "noise": noise, "iterations": int(res.nit), "evaluations": int(res.nfev),
            "rel_l2_error": rel_err, "rel_l2_error_initial": rel_err0, "rel_l2_error_interior": rel_err_int,
            "wall_time": wall, "regularization": beta, "n_sensors": int(len(sens_xy)), "n_experiments": len(heaters),
            "success": bool(res.success), "message": str(res.message)}


# ---------------------------------------------------------------------------
# 4. Nonlinear parameter recovery
# ---------------------------------------------------------------------------


def stefan_setup():
    v, p, _ = generate_square_polygonal_mesh(nx=24, ny=12, bbox=(0.0, 1.0, 0.0, 0.5))
    pulse = lambda x, y, t: np.where((x < 1e-9) & (np.abs(y - 0.25) < 0.15), 3.0 * (t < 0.3), 0.0)
    probe = EnthalpyStefanSolver(v, p, 0.05, 0.02, SmoothPhaseChangeModel(0.0, 0.05, latent_heat=1.0),
                                 bc_type="flux", bc_func=pulse)
    sens_xy = [(0.1, 0.25), (0.2, 0.25), (0.3, 0.25), (0.15, 0.4), (0.25, 0.1)]
    sens = np.array([int(np.argmin(np.sum((probe.cell_centers - s) ** 2, axis=1))) for s in sens_xy])
    times = list(np.round(np.arange(1, 31) * 0.02, 10))
    T0 = -0.2 * np.ones(probe.M)

    def make(q, tau=0.0):
        model = SmoothPhaseChangeModel(0.0, 0.05, latent_heat=q[0])
        return EnthalpyStefanSolver(v, p, q[1], 0.02, model, relaxation_time=tau, bc_type="flux", bc_func=pulse,
                                    newton_options=TIGHT)
    return make, sens, times, T0, probe, np.array(sens_xy)


def study_nonlinear():
    make, sens, times, T0, probe, sens_xy = stefan_setup()
    truth = np.array([2.0, 0.05])
    clean = enthalpy_stefan_adjoint(make(truth), T0, 0.6, times, np.zeros((len(times), sens.size)), sens).predicted
    rng = np.random.default_rng(7)
    data = clean + 0.01 * np.max(np.abs(clean)) * rng.standard_normal(clean.shape)
    q_scale = truth.copy()
    path = []

    def fun(z, data=data):
        q = z * q_scale
        r = enthalpy_stefan_adjoint(make(q), T0, 0.6, times, data, sens)
        return r.objective, r.gradient * q_scale

    q0 = np.array([0.8, 0.09])
    path.append(q0.copy())
    res = minimize(fun, q0 / q_scale, jac=True, method="L-BFGS-B", bounds=[(0.05, 5.0), (0.2, 5.0)],
                   callback=lambda zk: path.append(zk * q_scale),
                   options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 60})
    q_hat = res.x * q_scale
    path = np.array(path)
    # Cost landscape.
    Ls = np.linspace(0.6, 3.2, 21)
    As = np.linspace(0.03, 0.1, 21)
    Jgrid = np.array([[enthalpy_stefan_adjoint(make([L, a]), T0, 0.6, times, data, sens).objective for L in Ls] for a in As])
    fit0 = enthalpy_stefan_adjoint(make(q0), T0, 0.6, times, data, sens)
    fit = enthalpy_stefan_adjoint(make(q_hat), T0, 0.6, times, data, sens)
    # Noise study.
    noise_rows = []
    for lvl in (0.0, 0.005, 0.01, 0.02, 0.05):
        for s in range(3 if lvl > 0 else 1):
            d = clean + lvl * np.max(np.abs(clean)) * np.random.default_rng(100 + s).standard_normal(clean.shape)
            rr = minimize(lambda z, d=d: (lambda r: (r.objective, r.gradient * q_scale))(
                enthalpy_stefan_adjoint(make(z * q_scale), T0, 0.6, times, d, sens)),
                q0 / q_scale, jac=True, method="L-BFGS-B", bounds=[(0.05, 5.0), (0.2, 5.0)],
                options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 60})
            qq = rr.x * q_scale
            noise_rows.append({"noise": lvl, "seed": s, "L": float(qq[0]), "alpha": float(qq[1]),
                               "L_err": float(abs(qq[0] / truth[0] - 1)), "alpha_err": float(abs(qq[1] / truth[1] - 1)),
                               "iterations": int(rr.nit)})

    # Cattaneo tau profile with adjoint tangents.
    tau_true = 0.02
    data_tau = enthalpy_stefan_adjoint(make(truth, tau=tau_true), T0, 0.6, times,
                                       np.zeros((len(times), sens.size)), sens, parameters=("tau",)).predicted
    taus = np.linspace(0.005, 0.05, 31)
    J_tau = [enthalpy_stefan_adjoint(make(truth, tau=t), T0, 0.6, times, data_tau, sens, parameters=("tau",)).objective for t in taus]
    tangents_tau = []
    for t in (0.008, 0.035, 0.045):
        r = enthalpy_stefan_adjoint(make(truth, tau=t), T0, 0.6, times, data_tau, sens, parameters=("tau",))
        tangents_tau.append((t, r.objective, float(r.gradient[0])))
    # Fractional beta profile.
    vf, pf, cf = generate_square_polygonal_mesh(nx=16, ny=16, bbox=(0.0, 1.0, 0.0, 1.0))
    u0f = mode(cf)
    sensf = np.arange(0, 256, 9)
    tf = [0.1, 0.2, 0.3]
    data_b = fractional_heat_adjoint(vf, pf, 0.1, 0.01, 0.7, u0f, 0.3, tf, np.zeros((3, sensf.size)), sensf).predicted
    betas = np.linspace(0.4, 0.95, 31)
    J_b = [fractional_heat_adjoint(vf, pf, 0.1, 0.01, b, u0f, 0.3, tf, data_b, sensf).objective for b in betas]
    tangents_b = []
    for b in (0.5, 0.85):
        r = fractional_heat_adjoint(vf, pf, 0.1, 0.01, b, u0f, 0.3, tf, data_b, sensf)
        tangents_b.append((b, r.objective, float(r.gradient[0])))

    # Adjoint field snapshot for the recovered model.
    lam = fit.adjoint_states
    k_snap = len(lam) // 3

    fig, ax = plt.subplots(2, 3, figsize=(14, 7.6))
    a = ax[0, 0]
    cs = a.contourf(Ls, As, np.log10(Jgrid), levels=25, cmap="viridis")
    fig.colorbar(cs, ax=a, label=r"$\log_{10} J$")
    a.plot(path[:, 0], path[:, 1], "w.-", ms=5, label="L-BFGS-B iterates")
    a.plot(*truth, "r*", ms=12, label="truth")
    a.plot(*q_hat, "wo", mfc="none", ms=9, label="estimate")
    a.set_xlabel("latent heat $L$"); a.set_ylabel(r"diffusivity $\alpha$")
    a.set_xlim(Ls[0], Ls[-1]); a.set_ylim(As[0], As[-1])
    a.set_title(f"(a) Stefan misfit, {res.nit} iterations"); a.legend(fontsize=7); a.grid(False)
    a = ax[0, 1]
    tt = np.array(times)
    for j, col in zip(range(sens.size), plt.cm.tab10.colors):
        a.plot(tt, data[:, j], "o", ms=2.5, color=col)
        a.plot(tt, fit.predicted[:, j], "-", color=col)
        a.plot(tt, fit0.predicted[:, j], ":", color=col, alpha=0.6)
    a.set_xlabel("$t$"); a.set_ylabel("thermocouple $T$")
    a.set_title("(b) data (dots), initial (dotted), fitted (solid)")
    a = ax[0, 2]
    c = probe.cell_centers
    lim = float(np.max(np.abs(lam[k_snap])))
    im = a.tripcolor(c[:, 0], c[:, 1], lam[k_snap], shading="gouraud", cmap="RdBu_r", vmin=-lim, vmax=lim)
    a.plot(sens_xy[:, 0], sens_xy[:, 1], "k+", ms=7)
    a.set_aspect("equal"); a.grid(False); fig.colorbar(im, ax=a, fraction=0.03)
    a.set_title(rf"(c) adjoint $\lambda$ at $t={fit.extras['times'][k_snap + 1]:.2f}$")
    a = ax[1, 0]
    nr = noise_rows
    for key, mk, lab in (("L_err", "o", "$L$"), ("alpha_err", "s", r"$\alpha$")):
        a.semilogy([r["noise"] * 100 for r in nr], [max(r[key], 1e-12) for r in nr], mk, label=lab, alpha=0.8)
    a.set_xlabel("noise level [% of max signal]"); a.set_ylabel("relative parameter error")
    a.set_title("(d) recovery vs noise (3 seeds)"); a.legend(fontsize=7)
    for a, xs, Js, tans, lab, truthv, ttl in (
        (ax[1, 1], taus, J_tau, tangents_tau, r"relaxation time $\tau$", tau_true, "(e) Cattaneo-Stefan: $J(\\tau)$"),
        (ax[1, 2], betas, J_b, tangents_b, r"fractional order $\beta$", 0.7, "(f) Caputo L1: $J(\\beta)$"),
    ):
        a.plot(xs, Js, "k-", label="objective")
        span = 0.15 * (xs[-1] - xs[0])
        for x0, j0, g0 in tans:
            xx = np.array([x0 - span, x0 + span])
            a.plot(xx, j0 + g0 * (xx - x0), "r-", lw=1.5)
            a.plot(x0, j0, "ro", ms=4)
        a.axvline(truthv, color="tab:green", ls="--", lw=0.8, label="truth")
        a.set_ylim(-0.05 * max(Js), 1.1 * max(Js))
        a.set_xlabel(lab); a.set_title(ttl + " with adjoint tangents"); a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "nonlinear_parameter_recovery.png", dpi=170)
    plt.close(fig)
    return {"truth": truth.tolist(), "initial": q0.tolist(), "estimate": q_hat.tolist(), "iterations": int(res.nit),
            "noise": 0.01, "noise_study": noise_rows, "tau_true": tau_true, "beta_true": 0.7,
            "n_sensors": int(sens.size), "n_times": len(times)}


def main():
    t0 = time.perf_counter()
    summary = {"taylor": study_taylor()}
    print("taylor", time.perf_counter() - t0)
    summary["cost"] = study_cost()
    print("cost", time.perf_counter() - t0)
    summary["field_inversion"] = study_field_inversion()
    print("field", time.perf_counter() - t0, summary["field_inversion"])
    summary["nonlinear"] = study_nonlinear()
    print("nonlinear", time.perf_counter() - t0, summary["nonlinear"]["estimate"])
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
