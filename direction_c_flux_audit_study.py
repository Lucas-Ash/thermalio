"""Direction C capability 7: consistent Dirichlet boundary-flux reconstruction.

Re-audits the cryosurgery-freezing and rapid-quench scenarios with energies
computed from the *reconstructed boundary flux* instead of being inferred from
the enthalpy change (which made the old closure residual zero by
construction).  Writes
``test_plots/direction_C_nonfourier_phase_change/flux_audit/{flux_audit.png,summary.json}``.

Run: ``MPLCONFIGDIR=/tmp/matplotlib python direction_c_flux_audit_study.py``
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from heat_solver.boundary_flux import enthalpy_solver_flux_audit, hyperbolic_stefan_flux_audit
from heat_solver.enthalpy import EnthalpyStefanSolver
from heat_solver.meshes import generate_square_polygonal_mesh
from heat_solver.phase_change import ApparentHeatCapacityModel
from heat_solver.transport import HyperbolicStefanSolver

OUT = Path("test_plots/direction_C_nonfourier_phase_change/flux_audit")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.grid": True, "grid.alpha": 0.3})


def scenario(name, n=24):
    if name == "cryosurgery":
        bbox, alpha, tau, t_end = (0.0, 1.0, 0.0, 1.0), 0.08, 0.005, 0.5
        pcm = ApparentHeatCapacityModel(-0.15, 0.15, latent_heat=4.0)
        dx = 1.0 / n
        bc = lambda x, y, t: np.where(x < 0.75 * dx, -0.8, 0.6)
        T0 = 0.6
        v, p, _ = generate_square_polygonal_mesh(nx=n, ny=n, bbox=bbox)
    else:  # rapid solidification quench
        bbox, alpha, tau, t_end = (0.0, 1.2, 0.0, 0.6), 0.075, 0.006, 0.36
        pcm = ApparentHeatCapacityModel(-0.18, 0.18, latent_heat=3.6)
        bc = lambda x, y, t: -0.65 * np.ones_like(np.asarray(x, float))
        T0 = 0.68
        v, p, _ = generate_square_polygonal_mesh(nx=n, ny=n // 2, bbox=bbox)
    return {"v": v, "p": p, "alpha": alpha, "tau": tau, "t_end": t_end, "pcm": pcm, "bc": bc, "T0": T0}


OPTS = {"max_iters": 200, "tol": 1e-8, "relaxation": 0.5, "anderson_depth": 6, "raise_on_nonconvergence": False}


def run_apparent(sc, nt):
    s = HyperbolicStefanSolver(sc["v"], sc["p"], sc["alpha"], sc["t_end"] / nt, sc["tau"], sc["pcm"],
                               bc_type="dirichlet", bc_func=sc["bc"], phase_change_options=OPTS)
    s.record_history = True
    u0 = np.full(s.M, sc["T0"])
    s.solve(u0, 0.0, sc["t_end"], du0=np.zeros_like(u0))
    audit = hyperbolic_stefan_flux_audit(s, sc["pcm"])
    # Enthalpy-loss history for comparison with the cumulative flux.
    area = s.cell_areas
    H = np.array([np.sum(area * sc["pcm"].enthalpy(u)) for u in s.history["u"]])
    audit["enthalpy_history"] = H[1:] - H[0]
    return s, audit


def run_enthalpy(sc, nt):
    s = EnthalpyStefanSolver(sc["v"], sc["p"], sc["alpha"], sc["t_end"] / nt, sc["pcm"],
                             bc_type="dirichlet", bc_func=sc["bc"])
    s.solve(np.full(s.M, sc["T0"]), 0.0, sc["t_end"], store_history=True)
    audit = enthalpy_solver_flux_audit(s)
    area = s.cell_areas
    H = np.array([np.sum(area * h) for h in s.history["H"]])
    audit["enthalpy_history"] = H[1:] - H[0]
    return s, audit


def main():
    t0 = time.perf_counter()
    summary = {}
    show = {}
    for name in ("cryosurgery", "quench"):
        sc = scenario(name)
        s_ac, a_ac = run_apparent(sc, 100)
        s_en, a_en = run_enthalpy(sc, 100)
        show[name] = (s_ac, a_ac, s_en, a_en)
        sweep = []
        for nt in (25, 50, 100, 200, 400):
            _, aa = run_apparent(sc, nt)
            _, ae = run_enthalpy(sc, nt)
            sweep.append({"nt": nt, "dt": sc["t_end"] / nt,
                          "ac_closure": aa["energy_closure_residual"], "ac_capacity_defect": aa["capacity_defect"],
                          "ac_nonlinear_residual": aa["nonlinear_residual_energy"], "ac_failed_steps": aa["failed_steps"],
                          "ac_identity": aa["scheme_identity_residual"],
                          "ac_interior_closure": aa["interior_energy_closure_residual"],
                          "ac_interior_capacity_defect": aa["interior_capacity_defect"],
                          "ac_band_capacity_defect": aa["band_capacity_defect"],
                          "enthalpy_closure": ae["energy_closure_residual"],
                          "ac_enthalpy_change": aa["enthalpy_change"], "enthalpy_change": ae["enthalpy_change"]})
            print(name, nt, {k: f"{v:.3e}" for k, v in sweep[-1].items() if isinstance(v, float)})
        keys = ("enthalpy_change", "boundary_flux_energy", "relaxation_storage", "capacity_work",
                "capacity_defect", "nonlinear_residual_energy", "scheme_identity_residual",
                "energy_closure_residual", "relative_energy_closure_residual", "band_interface_energy",
                "failed_steps", "interior_enthalpy_change", "interior_capacity_defect",
                "interior_energy_closure_residual", "band_capacity_defect")
        summary[name] = {
            "apparent_capacity_nt100": {k: a_ac[k] for k in keys},
            "enthalpy_nt100": {k: a_en[k] for k in ("enthalpy_change", "boundary_flux_energy",
                                                    "energy_closure_residual")},
            "dt_sweep": sweep,
        }

    fig, ax = plt.subplots(2, 2, figsize=(12, 8.6))
    a = ax[0, 0]
    s_ac, a_ac, s_en, a_en = show["cryosurgery"]
    a.plot(a_ac["times"], -a_ac["enthalpy_history"], "b-", lw=2, label="apparent capacity: enthalpy loss")
    a.plot(a_ac["times"], -a_ac["flux_history"], "b--", label="apparent capacity: reconstructed boundary flux")
    a.plot(a_en["times"], -a_en["enthalpy_history"], "r-", lw=2, alpha=0.6, label="enthalpy solver: enthalpy loss")
    a.plot(a_en["times"], -a_en["flux_history"], "k:", label="enthalpy solver: reconstructed boundary flux")
    a.set_xlabel("$t$"); a.set_ylabel("energy extracted by the cryoprobe")
    a.set_title("(a) cryosurgery: two independent measures of extracted energy"); a.legend(fontsize=7)
    a = ax[0, 1]
    labels = ["$\\Delta E$", "flux in", "band latent\njump", "interior\ncapacity defect",
              "unconverged\nresidual", "closure\nresidual"]
    width = 0.38
    for off, name, col in ((-width / 2, "cryosurgery", "tab:blue"), (width / 2, "quench", "tab:orange")):
        r = summary[name]["apparent_capacity_nt100"]
        vals = [r["enthalpy_change"], r["boundary_flux_energy"], r["band_capacity_defect"],
                r["interior_capacity_defect"], r["nonlinear_residual_energy"], r["energy_closure_residual"]]
        a.bar(np.arange(len(vals)) + off, vals, width, color=col, label=f"{name} (apparent capacity)")
    a.axhline(0, color="k", lw=0.8)
    a.set_xticks(np.arange(len(labels))); a.set_xticklabels(labels, fontsize=7.5)
    a.set_ylabel("energy")
    a.set_title("(b) audit decomposition (closure = band jump + interior defect + residual)", fontsize=9)
    a.legend(fontsize=7)
    a = ax[1, 0]
    for name, mk in (("cryosurgery", "o"), ("quench", "s")):
        sw = summary[name]["dt_sweep"]
        dts = [r["dt"] for r in sw]
        a.loglog(dts, [abs(r["ac_closure"]) for r in sw], mk + "-", label=f"apparent capacity, full domain, {name}")
        a.loglog(dts, [abs(r["ac_interior_closure"]) for r in sw], mk + ":", label=f"apparent capacity, interior, {name}")
        a.loglog(dts, [max(abs(r["enthalpy_closure"]), 1e-17) for r in sw], mk + "--", color="k", alpha=0.6,
                 label=f"enthalpy solver, {name}")
    a.set_xlabel(r"$\Delta t$"); a.set_ylabel("|energy closure residual|")
    a.set_title("(c) closure residual vs time step"); a.legend(fontsize=6.5)
    a = ax[1, 1]
    for (s, aud, lab, sty) in ((s_ac, a_ac, "apparent capacity", "b-"), (s_en, a_en, "enthalpy solver", "r--")):
        b = s._boundary_idx
        c = s.cell_centers[b]
        left = c[:, 0] < 0.5 / 24 + 1e-9
        order = np.argsort(c[left, 1])
        a.plot(c[left, 1][order], -aud["per_boundary_cell_energy"][left][order], sty, label=lab)
    a.set_xlabel("$y$ along the cryoprobe wall"); a.set_ylabel("energy extracted per boundary cell")
    a.set_title("(d) where heat leaves: per-cell reaction flux (cryosurgery)"); a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "flux_audit.png", dpi=170)
    plt.close(fig)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("done", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
