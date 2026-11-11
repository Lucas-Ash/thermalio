"""Order-of-accuracy study for the quadratic (2-exact) diffusion flux.

Fixes the skewed-mesh anisotropic order loss found by the cross-code study.
Compares ``reconstructed`` (averaged LSQ gradients), ``tpfa`` +
non-orthogonal correction and the new ``quadratic`` flux on four mesh
families with a rotated anisotropic tensor, using consistent cell-average
data (initial data, source, Dirichlet band, error metric) and Crank--Nicolson
so the spatial error is isolated.

Writes ``test_plots/flux_reconstruction/quadratic_flux.png`` and ``.json``.
Run: ``MPLCONFIGDIR=/tmp/matplotlib python quadratic_flux_study.py``
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import spsolve

from heat_solver.dmp import anisotropic_tensor
from heat_solver.meshes import (
    generate_hexagonal_polygonal_mesh,
    generate_nonorthogonal_polygonal_mesh,
    generate_nonorthogonal_tiled_polygonal_mesh,
    generate_square_polygonal_mesh,
)
from heat_solver.polygonal import PolygonalHeatSolver
from heat_solver.quadratic_flux import polygon_second_moments

OUT = Path("test_plots/flux_reconstruction")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.grid": True, "grid.alpha": 0.3})

K = anisotropic_tensor(ratio=5.0, angle=np.pi / 6) * 0.1
T_END, DT = 0.02, 1e-4


def exact(x, y, t):
    return np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y)


def source(x, y, t):
    u = exact(x, y, t)
    uxy = np.pi**2 * np.exp(-t) * np.cos(np.pi * x) * np.cos(np.pi * y)
    return -u + np.pi**2 * (K[0, 0] + K[1, 1]) * u - (K[0, 1] + K[1, 0]) * uxy


def perturbed(n, seed=0):
    v, p, _ = generate_square_polygonal_mesh(nx=n, ny=n, bbox=(-1, 1, -1, 1))
    v = v.copy()
    inner = (np.abs(v[:, 0]) < 1 - 1e-9) & (np.abs(v[:, 1]) < 1 - 1e-9)
    v[inner] += 0.2 * (2.0 / n) * np.random.default_rng(seed).uniform(-1, 1, (inner.sum(), 2))
    return v, p


MESHES = {
    "skewed (shear 0.3)": (lambda n: generate_nonorthogonal_polygonal_mesh(nx=n, ny=n, bbox=(-1, 1, -1, 1), skew=0.3)[:2],
                           (16, 32, 64, 128)),
    "hexagonal": (lambda n: generate_hexagonal_polygonal_mesh(nx=n, ny=n, spacing=2.0 / n, bbox=(-1, 1, -1, 1))[:2],
                  (16, 32, 64, 128)),
    "randomly perturbed quads": (perturbed, (16, 32, 64, 128)),
    "tiled mixed polygons": (lambda n: generate_nonorthogonal_tiled_polygonal_mesh(
        nx_tiles=n // 3, ny_tiles=n // 3, bbox=(-1, 1, -1, 1), skew=0.3)[:2], (12, 24, 48, 96)),
}
SCHEMES = {
    "reconstructed": (dict(flux_discretization="reconstructed", cell_average_data=True), "o-", "tab:gray"),
    "tpfa + nonorth. corr.": (dict(flux_discretization="tpfa", nonorthogonal_correction=True, cell_average_data=True),
                              "s-", "tab:blue"),
    "quadratic (new)": (dict(flux_discretization="quadratic"), "^-", "tab:red"),
}


def solve_error(v, p, kw):
    s = PolygonalHeatSolver(v, p, K, DT, bc_type="dirichlet", bc_func=exact, source_func=source,
                            time_scheme="crank_nicolson", **kw)
    u = s.solve(s.cell_average(exact, 0.0), 0.0, T_END)[1]
    ue = s.cell_average(exact, T_END)
    A = s.cell_areas
    return float(np.sqrt(np.sum(A * (u - ue) ** 2) / np.sum(A * ue**2))), float(np.sqrt(np.mean(A))), s.A.nnz / s.M


def quadratic_truncation(v, p, kw):
    """Max interior truncation error of the steady operator on q = x^2 + 3xy - 2y^2."""
    s = PolygonalHeatSolver(v, p, K, 1.0, **kw)
    s._assemble_system()
    mom = [polygon_second_moments(v[q]) for q in p]
    avg = np.array([c[0] ** 2 + 3 * c[0] * c[1] - 2 * c[1] ** 2 + M[0, 0] + 3 * M[0, 1] - 2 * M[1, 1]
                    for _, c, M in mom])
    q_src = -(2 * K[0, 0] + 3 * (K[0, 1] + K[1, 0]) - 4 * K[1, 1])
    r = (s.A @ avg) / s.cell_areas - q_src
    A = s.A.tolil()
    b = s.cell_areas * q_src
    for i in np.where(s.is_boundary)[0]:
        A.rows[i] = [i]
        A.data[i] = [1.0]
        b[i] = avg[i]
    u = spsolve(A.tocsr(), b)
    return float(np.max(np.abs(r[~s.is_boundary]))), float(np.max(np.abs(u - avg)))


def main():
    t0 = time.perf_counter()
    results = {}
    for mname, (mk, ns) in MESHES.items():
        results[mname] = {}
        for sname, (kw, _, _) in SCHEMES.items():
            rows = [solve_error(*mk(n), kw) for n in ns]
            e = np.array([r[0] for r in rows]); h = np.array([r[1] for r in rows])
            trunc, steady = quadratic_truncation(*mk(ns[1]), kw)
            results[mname][sname] = {
                "n": list(ns), "h": h.tolist(), "err": e.tolist(),
                "pair_orders": (np.log(e[:-1] / e[1:]) / np.log(h[:-1] / h[1:])).tolist(),
                "nnz_per_row": rows[-1][2], "quadratic_truncation": trunc, "quadratic_steady_error": steady,
            }
            print(mname, sname, ["%.2e" % x for x in e], np.round(results[mname][sname]["pair_orders"], 2),
                  "trunc %.1e" % trunc, f"{time.perf_counter() - t0:.0f}s")

    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    for a, (mname, res) in zip([ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]], results.items()):
        for sname, (_, mk_, col) in SCHEMES.items():
            r = res[sname]
            a.loglog(r["h"], r["err"], mk_, color=col, label=f"{sname} (last order {r['pair_orders'][-1]:.2f})")
        h = np.array(res["quadratic (new)"]["h"])
        a.loglog(h, res["quadratic (new)"]["err"][-1] * (h / h[-1]) ** 2, "k:", lw=0.8, label="slope 2")
        a.set_title(mname); a.set_xlabel("$h$"); a.set_ylabel(r"relative $L^2$ error"); a.legend(fontsize=7)
    a = ax[0, 2]
    names = list(results)
    x = np.arange(len(names))
    for k, (sname, (_, _, col)) in enumerate(SCHEMES.items()):
        a.bar(x + (k - 1) * 0.27, [max(results[m][sname]["quadratic_truncation"], 1e-16) for m in names], 0.27,
              color=col, label=sname)
    a.set_yscale("log"); a.set_xticks(x); a.set_xticklabels([m.replace(" (", "\n(").replace("randomly perturbed", "randomly\nperturbed").replace("tiled mixed", "tiled\nmixed") for m in names], fontsize=7)
    a.set_ylabel("max interior truncation error")
    a.set_title("steady quadratic field $x^2+3xy-2y^2$:\nconsistency of the discrete operator"); a.legend(fontsize=7, loc="center left")
    a = ax[1, 2]
    for k, (sname, (_, _, col)) in enumerate(SCHEMES.items()):
        a.bar(x + (k - 1) * 0.27, [results[m][sname]["err"][-1] for m in names], 0.27, color=col, label=sname)
    a.set_yscale("log"); a.set_xticks(x); a.set_xticklabels([m.replace(" (", "\n(").replace("randomly perturbed", "randomly\nperturbed").replace("tiled mixed", "tiled\nmixed") for m in names], fontsize=7)
    a.set_ylabel(r"relative $L^2$ error, finest mesh"); a.set_title("finest-mesh error"); a.legend(fontsize=7)
    fig.suptitle("Anisotropic diffusion (ratio 5, 30$^\\circ$): flux discretizations across mesh families", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "quadratic_flux.png", dpi=170)
    plt.close(fig)
    (OUT / "quadratic_flux.json").write_text(json.dumps(results, indent=2))
    print("done", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
