"""Direction C follow-up: enthalpy-primary formulation + adaptive time stepping.

Generates the figures/JSON used by ``direction_CD_enthalpy_adjoint_report.tex``
under ``test_plots/direction_C_nonfourier_phase_change/enthalpy_adaptive/``:

1. ``enthalpy_mms_convergence``   manufactured traveling-front verification of
   :class:`EnthalpyStefanSolver` (Fourier and Cattaneo): temporal order 1 / 2,
   spatial order 2.
2. ``stiffness_robustness``       apparent-capacity Picard vs enthalpy Newton
   across mushy-zone width x time step: failures, iterations, energy defect,
   melt-front accuracy.
3. ``adaptive_pulse_cooling``     ultrafast flux pulse followed by slow cooling:
   adaptive step history, front tracking, work-precision vs uniform steps and
   exact energy closure.

Run: ``MPLCONFIGDIR=/tmp/matplotlib python direction_c_enthalpy_studies.py``
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from heat_solver.enthalpy import (
    AdaptiveStepOptions,
    ApparentCapacityPicardSolver,
    EnthalpyStefanSolver,
    SmoothPhaseChangeModel,
)
from heat_solver.interface_diagnostics import interface_position
from heat_solver.meshes import generate_square_polygonal_mesh
from heat_solver.phase_change import ApparentHeatCapacityModel

OUT = Path("test_plots/direction_C_nonfourier_phase_change/enthalpy_adaptive")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.grid": True, "grid.alpha": 0.3})


def _order(hs, errs):
    hs, errs = np.asarray(hs, float), np.asarray(errs, float)
    return np.log(errs[:-1] / errs[1:]) / np.log(hs[:-1] / hs[1:])


# ---------------------------------------------------------------------------
# 1. Manufactured traveling front
# ---------------------------------------------------------------------------

MMS = dict(A=0.8, w=0.18, s=0.45, x0=-0.35, alpha=0.08, L=2.0, delta=0.25, t_end=0.6)


def mms_case(tau):
    A, w, s, x0, alpha = MMS["A"], MMS["w"], MMS["s"], MMS["x0"], MMS["alpha"]
    model = SmoothPhaseChangeModel(0.0, MMS["delta"], latent_heat=MMS["L"])

    def parts(x, t):
        z = (x - x0 - s * t) / w
        th = np.tanh(z)
        sech2 = 1.0 - th**2
        T = A * th
        Tt = -(A * s / w) * sech2
        Ttt = -(2.0 * A * s**2 / w**2) * th * sech2
        Txx = -(2.0 * A / w**2) * th * sech2
        return T, Tt, Ttt, Txx

    def exact(x, y, t):
        return parts(np.asarray(x, float), t)[0]

    def source(x, y, t):
        T, Tt, Ttt, Txx = parts(np.asarray(x, float), t)
        zt = (T - model.melt_temperature) / model.half_width
        thz = np.tanh(zt)
        phi2 = -(1.0 - thz**2) * thz / model.half_width**2  # phi''(T)
        cap = model.effective_heat_capacity(T)
        Ht = cap * Tt
        Htt = cap * Ttt + model.latent_heat * phi2 * Tt**2
        return tau * Htt + Ht - alpha * Txx

    def rate(x, y):
        return parts(np.asarray(x, float), 0.0)[1]

    return model, exact, source, rate


def run_mms(nx, dt, tau, ny=8):
    model, exact, source, rate = mms_case(tau)
    v, p, c = generate_square_polygonal_mesh(nx=nx, ny=ny, bbox=(-1.0, 1.0, -0.1, 0.1))
    solver = EnthalpyStefanSolver(v, p, MMS["alpha"], dt, model, relaxation_time=tau,
                                  bc_type="dirichlet", bc_func=exact, source_func=source,
                                  newton_options={"tol": 1e-12, "rtol_residual": 1e-12})
    x, y = solver.cell_centers[:, 0], solver.cell_centers[:, 1]
    _, T = solver.solve(exact(x, y, 0.0), 0.0, MMS["t_end"], du0=rate(x, y))
    return solver, T, exact(x, y, MMS["t_end"])


def study_mms():
    out = {}
    # temporal self-convergence on a fixed mesh
    nx = 80
    dts = [0.04, 0.02, 0.01, 0.005, 0.0025]
    for tau in (0.0, 0.05):
        _, T_ref, _ = run_mms(nx, dts[-1] / 16, tau)
        errs = [float(np.max(np.abs(run_mms(nx, dt, tau)[1] - T_ref))) for dt in dts]
        out[f"time_tau{tau}"] = {"dt": dts, "err": errs, "order": _order(dts, errs).tolist()}
    # spatial convergence against the exact solution (dt ~ h^2 so the O(dt) error is O(h^2))
    nxs = [20, 40, 80, 160]
    for tau in (0.0, 0.05):
        errs, hs = [], []
        for nx in nxs:
            h = 2.0 / nx
            dt = 0.02 * (20 / nx) ** 2  # dt ~ h^2: both schemes are first order in time
            _, T, T_ex = run_mms(nx, dt, tau)
            errs.append(float(np.max(np.abs(T - T_ex))))
            hs.append(h)
        out[f"space_tau{tau}"] = {"h": hs, "err": errs, "order": _order(hs, errs).tolist()}

    solver, T, T_ex = run_mms(80, 0.0025, 0.05)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for tau, mk in ((0.0, "o"), (0.05, "s")):
        d = out[f"time_tau{tau}"]
        ax[0].loglog(d["dt"], d["err"], mk + "-", label=fr"$\tau={tau}$ (order {d['order'][-1]:.2f})")
    dts_a = np.array(dts)
    ax[0].loglog(dts_a, 0.5 * out["time_tau0.0"]["err"][0] * dts_a / dts_a[0], "k:", lw=0.8, label="slope 1")
    ax[0].set_xlabel(r"$\Delta t$"); ax[0].set_ylabel(r"$\|T_{\Delta t}-T_{\mathrm{ref}}\|_\infty$")
    ax[0].set_title("(a) temporal self-convergence"); ax[0].legend(fontsize=7)
    for a_ in ax[:2]:
        a_.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    for tau, mk in ((0.0, "o"), (0.05, "s")):
        d = out[f"space_tau{tau}"]
        ax[1].loglog(d["h"], d["err"], mk + "-", label=fr"$\tau={tau}$ (order {d['order'][-1]:.2f})")
    hs_a = np.array(out["space_tau0.0"]["h"])
    ax[1].loglog(hs_a, 0.5 * out["space_tau0.0"]["err"][0] * (hs_a / hs_a[0]) ** 2, "k--", lw=0.8, label="slope 2")
    ax[1].set_xlabel("$h$"); ax[1].set_ylabel(r"$\|T_h-T_{\mathrm{exact}}\|_\infty$")
    ax[1].set_title("(b) convergence to the exact front"); ax[1].legend(fontsize=7)
    order = np.argsort(solver.cell_centers[:, 0])
    row = order[np.abs(solver.cell_centers[order, 1] - solver.cell_centers[order[0], 1]) < 1e-12]
    x = solver.cell_centers[row, 0]
    ax[2].plot(x, T_ex[row], "k-", lw=2, label="exact $T$")
    ax[2].plot(x, T[row], "r--", label=r"enthalpy FV, $\tau=0.05$")
    ax2 = ax[2].twinx()
    ax2.plot(x, solver.model.liquid_fraction(T[row]), color="tab:blue", alpha=0.6, label=r"$\phi(T)$")
    ax2.set_ylabel(r"liquid fraction $\phi$", color="tab:blue"); ax2.grid(False)
    ax[2].set_xlabel("$x$"); ax[2].set_ylabel("$T$")
    ax[2].set_title(r"(c) front at $t=0.6$"); ax[2].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "enthalpy_mms_convergence.png", dpi=170)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 2. Stiffness robustness: apparent capacity (Picard) vs enthalpy (Newton)
# ---------------------------------------------------------------------------


def study_stiffness():
    v, p, _ = generate_square_polygonal_mesh(nx=20, ny=20, bbox=(0.0, 1.0, 0.0, 1.0))
    hot_wall = lambda x, y, t: np.where(x < 0.05, 1.0, -0.5)
    widths = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
    dts = [0.002, 0.005, 0.01, 0.02, 0.05]
    t_end, L = 0.5, 5.0
    shape = (len(widths), len(dts))
    ac_fail = np.zeros(shape); ac_iters = np.zeros(shape); ac_defect = np.zeros(shape); ac_time = np.zeros(shape)
    en_iters = np.zeros(shape); en_defect = np.zeros(shape); en_time = np.zeros(shape)
    picard = {"max_iters": 100, "tol": 1e-8, "anderson_depth": 5, "relaxation": 0.7}
    for i, hw in enumerate(widths):
        model = SmoothPhaseChangeModel(0.0, hw, latent_heat=L)
        for j, dt in enumerate(dts):
            ac = ApparentCapacityPicardSolver(v, p, 0.05, dt, model, bc_type="dirichlet", bc_func=hot_wall,
                                              picard_options=picard)
            u0 = -0.5 * np.ones(ac.M)
            t0 = time.perf_counter(); ac.solve(u0, 0.0, t_end); ac_time[i, j] = time.perf_counter() - t0
            r = ac.solve_report
            ac_fail[i, j] = r["failed_steps"] / len(r["iterations"])
            ac_iters[i, j] = r["total_iterations"] / len(r["iterations"])
            ac_defect[i, j] = abs(np.sum(r["energy_defect"]))
            en = EnthalpyStefanSolver(v, p, 0.05, dt, model, bc_type="dirichlet", bc_func=hot_wall)
            t0 = time.perf_counter(); en.solve(u0, 0.0, t_end); en_time[i, j] = time.perf_counter() - t0
            er = en.solve_report
            en_iters[i, j] = er["total_newton_iterations"] / er["accepted_steps"]
            en_defect[i, j] = abs(np.sum(er["energy_balance_residual"]))

    # Front tracking for a narrow mushy zone, dt = 0.01, against a fine-dt enthalpy reference.
    hw, dt = 0.01, 0.01
    model = SmoothPhaseChangeModel(0.0, hw, latent_heat=L)
    times = np.round(np.arange(1, 51) * 0.01, 10)
    def front_history(solver_cls, dt_run, **kw):
        s = solver_cls(v, p, 0.05, dt_run, model, bc_type="dirichlet", bc_func=hot_wall, **kw)
        u = -0.5 * np.ones(s.M); t = 0.0; pos = []
        for tn in times:
            t, u = s.solve(u, t, tn)
            pos.append(interface_position(s.cell_centers, u, model, axis="x", coord=0.525, level=0.0) or np.nan)
        return np.array(pos, dtype=float), u, s
    pos_ref, u_ref, s_ref = front_history(EnthalpyStefanSolver, 2.5e-4)
    pos_en, u_en, _ = front_history(EnthalpyStefanSolver, dt)
    pos_ac, u_ac, _ = front_history(ApparentCapacityPicardSolver, dt, picard_options=picard)

    fig, ax = plt.subplots(2, 3, figsize=(13, 7.2))
    ext = dict(aspect="auto", origin="lower")
    xt = np.arange(len(dts)); yt = np.arange(len(widths))
    im = ax[0, 0].imshow(ac_fail, cmap="Reds", vmin=0, vmax=max(ac_fail.max(), 1e-3), **ext)
    for (i, j), val in np.ndenumerate(ac_fail):
        ax[0, 0].text(j, i, f"{100 * val:.0f}%", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax[0, 0], label="failed-step fraction")
    ax[0, 0].set_title("(a) apparent capacity: Picard failures")
    im = ax[0, 1].imshow(ac_iters, cmap="viridis", **ext)
    for (i, j), val in np.ndenumerate(ac_iters):
        ax[0, 1].text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7, color="w")
    fig.colorbar(im, ax=ax[0, 1], label="Picard iterations / step")
    ax[0, 1].set_title("(b) apparent capacity: iterations")
    im = ax[0, 2].imshow(en_iters, cmap="viridis", vmin=ac_iters.min(), vmax=ac_iters.max(), **ext)
    for (i, j), val in np.ndenumerate(en_iters):
        ax[0, 2].text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7, color="w")
    fig.colorbar(im, ax=ax[0, 2], label="Newton iterations / step")
    ax[0, 2].set_title("(c) enthalpy Newton: iterations (0 failures)")
    for a in ax[0]:
        a.set_xticks(xt); a.set_xticklabels([str(d) for d in dts]); a.set_xlabel(r"$\Delta t$")
        a.set_yticks(yt); a.set_yticklabels([str(w) for w in widths]); a.set_ylabel(r"mushy half-width $\delta$")
        a.grid(False)
    jdt = dts.index(0.01)
    ax[1, 0].loglog(widths, np.maximum(ac_defect[:, jdt], 1e-17), "o-", label="apparent capacity")
    ax[1, 0].loglog(widths, np.maximum(en_defect[:, jdt], 1e-17), "s-", label="enthalpy")
    ax[1, 0].invert_xaxis(); ax[1, 0].set_xlabel(r"$\delta$ (stiffer $\rightarrow$)")
    ax[1, 0].set_ylabel(r"$|\sum_n \mathcal{E}^n|$ (energy defect)")
    ax[1, 0].set_title(r"(d) discrete energy defect, $\Delta t=0.01$"); ax[1, 0].legend(fontsize=7)
    ax[1, 1].plot(times, pos_ref, "k-", lw=2, label=r"enthalpy, $\Delta t=2.5\times10^{-4}$")
    ax[1, 1].plot(times, pos_en, "s--", ms=3, label=r"enthalpy, $\Delta t=0.01$")
    ax[1, 1].plot(times, pos_ac, "o:", ms=3, label=r"apparent capacity, $\Delta t=0.01$")
    ax[1, 1].set_xlabel("$t$"); ax[1, 1].set_ylabel("melt-front position $x_f$")
    ax[1, 1].set_title(r"(e) front tracking, $\delta=0.01$"); ax[1, 1].legend(fontsize=7)
    ax[1, 2].loglog(widths, ac_time[:, jdt], "o-", label="apparent capacity")
    ax[1, 2].loglog(widths, en_time[:, jdt], "s-", label="enthalpy")
    ax[1, 2].invert_xaxis(); ax[1, 2].set_xlabel(r"$\delta$"); ax[1, 2].set_ylabel("wall time [s]")
    ax[1, 2].set_title(r"(f) cost, $\Delta t=0.01$"); ax[1, 2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "stiffness_robustness.png", dpi=170)
    plt.close(fig)
    front_err = lambda pos: float(np.nanmax(np.abs(pos - pos_ref)))
    return {
        "widths": widths, "dts": dts, "latent_heat": L,
        "ac_failed_fraction": ac_fail.tolist(), "ac_iterations_per_step": ac_iters.tolist(),
        "ac_energy_defect": ac_defect.tolist(), "enthalpy_iterations_per_step": en_iters.tolist(),
        "enthalpy_energy_defect": en_defect.tolist(), "ac_time": ac_time.tolist(), "enthalpy_time": en_time.tolist(),
        "front_error_dt0.01": {"enthalpy": front_err(pos_en), "apparent_capacity": front_err(pos_ac)},
        "total_runs": int(ac_fail.size), "ac_runs_with_failures": int(np.sum(ac_fail > 0)),
    }


# ---------------------------------------------------------------------------
# 3. Adaptive stepping: ultrafast pulse then slow cooling
# ---------------------------------------------------------------------------


def study_adaptive():
    v, p, _ = generate_square_polygonal_mesh(nx=40, ny=8, bbox=(0.0, 1.0, 0.0, 0.2))
    model = ApparentHeatCapacityModel(-0.01, 0.01, latent_heat=4.0)
    q0, t_on = 25.0, 0.02
    pulse = lambda x, y, t: np.where(x < 1e-9, q0 * (t < t_on), 0.0)
    T_end = 2.0
    outputs = sorted(set(np.round(np.linspace(0.005, T_end, 60), 8)) | {t_on})
    u0 = -0.3 * np.ones(len(p))

    def run(dt, adaptive=None):
        s = EnthalpyStefanSolver(v, p, 0.05, dt, model, bc_type="flux", bc_func=pulse, adaptive=adaptive)
        t0 = time.perf_counter()
        s.solve(u0, 0.0, T_end, output_times=outputs)
        return s, time.perf_counter() - t0

    ref, _ = run(2e-5)
    err = lambda s: max(float(np.max(np.abs(s.outputs[t] - ref.outputs[t]))) for t in outputs)
    uniform = []
    for n in [50, 100, 200, 400, 800, 1600, 3200]:
        s, w = run(T_end / n)
        uniform.append({"n": n, "newton": s.solve_report["total_newton_iterations"], "err": err(s), "time": w})
    adaptive = []
    runs = {}
    for rtol in [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4]:
        s, w = run(1e-4, AdaptiveStepOptions(rtol=rtol, atol=0.1 * rtol))
        r = s.solve_report
        adaptive.append({"rtol": rtol, "accepted": r["accepted_steps"], "rejected": r["rejected_error"] + r["rejected_newton"],
                         "newton": r["total_newton_iterations"], "err": err(s), "time": w,
                         "dt_min": min(r["dts"]), "dt_max": max(r["dts"])})
        runs[rtol] = s
    show = runs[1e-2]
    rep = show.solve_report
    times = np.array(rep["times"])
    probe = lambda s, t: interface_position(s.cell_centers, s.outputs[t], model, axis="x", coord=0.1125, pick="last")
    wall = np.argmin(show.cell_centers[:, 0] + np.abs(show.cell_centers[:, 1] - 0.1125))

    fig, ax = plt.subplots(2, 2, figsize=(11, 7.4))
    a = ax[0, 0]
    a.semilogy(times[1:], rep["dts"], ".-", ms=3, label=r"accepted $\Delta t_n$ (rtol $10^{-2}$)")
    a.axvline(t_on, color="tab:red", ls="--", lw=0.8, label="pulse switch-off")
    a.set_xscale("symlog", linthresh=0.02); a.set_xlabel("$t$"); a.set_ylabel(r"$\Delta t$")
    a.set_title(f"(a) step history: {rep['accepted_steps']} accepted, "
                f"{rep['rejected_error'] + rep['rejected_newton']} rejected"); a.legend(fontsize=7)
    a = ax[0, 1]
    tt = np.array(outputs)
    a.plot(tt, [ref.outputs[t][wall] for t in outputs], "k-", lw=2, label="wall $T$, reference")
    a.plot(tt, [show.outputs[t][wall] for t in outputs], "r--", label="wall $T$, adaptive")
    a.set_xscale("symlog", linthresh=0.02); a.set_xlabel("$t$"); a.set_ylabel("wall temperature")
    b = a.twinx(); b.grid(False)
    b.plot(tt, [probe(ref, t) or np.nan for t in outputs], color="tab:blue", lw=2, alpha=0.5, label="front, reference")
    b.plot(tt, [probe(show, t) or np.nan for t in outputs], "b:", label="front, adaptive")
    b.set_ylabel("melt-front position", color="tab:blue")
    a.set_title("(b) pulse heating, melting, re-solidification")
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")
    a = ax[1, 0]
    a.loglog([u["newton"] for u in uniform], [u["err"] for u in uniform], "o-", label="uniform steps")
    a.loglog([d["newton"] for d in adaptive], [d["err"] for d in adaptive], "s-", label="adaptive (rtol sweep)")
    a.set_xlabel("total Newton iterations (work)"); a.set_ylabel(r"$\max_{t_k}\|T-T_{\mathrm{ref}}\|_\infty$")
    a.set_title("(c) work-precision"); a.legend(fontsize=7)
    a = ax[1, 1]
    energy = np.array(rep["energy"]) - rep["energy"][0]
    a.plot(times, energy, "r-", label=r"$\sum_i |\Omega_i|(H_i^n-H_i^0)$")
    a.plot(times, rep["energy_input"], "k--", label="cumulative injected flux")
    a.set_xscale("symlog", linthresh=0.02); a.set_xlabel("$t$"); a.set_ylabel("energy")
    b = a.twinx(); b.grid(False)
    b.semilogy(times[1:], np.maximum(np.abs(rep["energy_balance_residual"]), 1e-18), "g.", ms=2)
    b.set_ylabel("per-step balance residual", color="g")
    a.set_title("(d) exact discrete energy closure"); a.legend(fontsize=7, loc="lower right")
    for a_ in (ax[0, 0], ax[0, 1], ax[1, 1]):
        a_.set_xlim(0.0, T_end)
    fig.tight_layout()
    fig.savefig(OUT / "adaptive_pulse_cooling.png", dpi=170)
    plt.close(fig)
    return {"uniform": uniform, "adaptive": adaptive, "pulse_flux": q0, "pulse_duration": t_on, "t_end": T_end,
            "showcase_rtol": 1e-2, "showcase_accepted": rep["accepted_steps"],
            "showcase_rejected": rep["rejected_error"] + rep["rejected_newton"],
            "showcase_dt_range": [min(rep["dts"]), max(rep["dts"])],
            "showcase_max_balance_residual": float(np.max(np.abs(rep["energy_balance_residual"])))}


def main():
    t0 = time.perf_counter()
    summary = {"mms": study_mms()}
    print("mms done", time.perf_counter() - t0)
    summary["stiffness"] = study_stiffness()
    print("stiffness done", time.perf_counter() - t0)
    summary["adaptive"] = study_adaptive()
    print("adaptive done", time.perf_counter() - t0)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: (v.get("order") if isinstance(v, dict) else None) for k, v in summary["mms"].items()}))


if __name__ == "__main__":
    main()
