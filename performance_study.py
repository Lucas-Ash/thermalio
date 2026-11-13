"""How much do acceleration techniques and parallelism matter for Thermalio?

A measured, guided study (companion to ``docs/performance_guide.md``).  All
numbers come from this machine; nothing is assumed except where a panel says
"extrapolated".  Four questions, one figure each, written to
``test_plots/benchmark/performance/``:

1. ``where_time_goes.png``      Where does a Thermalio run spend its time?
   Wall-clock phases (setup, operator assembly, factorization, time loop,
   framework overhead) of a 100-step transient solve; the legacy per-face
   assembly loop (still available as ``_assemble_tpfa_loop``) against the
   vectorized assembly; and the end-to-end run before vectorization (previous
   measurements, ``summary_before_vectorization.json``) against now.
1b. ``boundary_closures.png``  Robin / radiative boundaries: reusing the
   factorization (frozen Jacobian for radiation) against refactorizing every
   step, as the solvers did before.
2. ``linear_solvers.png``       Algorithmic acceleration of the linear solve:
   refactorize every step vs factorize once vs CG (Jacobi) vs CG + algebraic
   multigrid (pyamg), in the accuracy-balanced transient regime
   (``dt ~ h^2``) and the stiff large-step regime (``dt = 1``), up to 1M cells.
3. ``parallelism.png``          Parallel speedup actually obtained on this
   machine: ensembles of independent solves over processes, threaded
   memory-bound (vector triad, sparse mat-vec) and compute-bound (dense
   matrix multiply) kernels, and Amdahl's law from the measured serial fraction.
4. ``accuracy_as_acceleration.png``  Higher order as acceleration: time steps
   needed by backward Euler vs Crank--Nicolson, and degrees of freedom needed
   by first- vs second-order-in-error spatial schemes (from the external-code
   comparison) for a target error.

Run: ``MPLCONFIGDIR=/tmp/matplotlib python performance_study.py``
(``pip install -r requirements-external.txt`` for pyamg / threadpoolctl; the
multigrid and BLAS-thread panels are skipped without them).
"""

from __future__ import annotations

import json
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg, factorized, spsolve

from heat_solver.mesh_library import generate
from heat_solver.polygonal import PolygonalHeatSolver

try:
    import pyamg
except ImportError:  # optional
    pyamg = None
try:
    from threadpoolctl import threadpool_limits
except ImportError:  # optional
    threadpool_limits = None

OUT = Path("test_plots/benchmark/performance")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.grid": True, "grid.alpha": 0.3})

ALPHA = 0.1
BBOX = (-1.0, 1.0, -1.0, 1.0)
CORES = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()


def exact(x, y, t):
    return np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y)


def source(x, y, t):
    return (2 * ALPHA * np.pi**2 - 1.0) * exact(x, y, t)


def _best(fn, repeat=3):
    best = np.inf
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


# ---------------------------------------------------------------------------
# 1. Where the time goes
# ---------------------------------------------------------------------------


BEFORE = OUT / "summary_before_vectorization.json"  # measured with the per-face loop (commit 45dd494)


def _legacy_identity_rows(K, mask):
    """The LIL row replacement the solver used before vectorization (for timing only)."""
    lil = K.tolil()
    for i in np.flatnonzero(mask):
        lil.rows[i] = [i]
        lil.data[i] = [1.0]
    return lil.tocsr()


def where_time_goes(n, nsteps=100, dt=2.5e-4):
    mesh = generate("square", nx=n, ny=n, bbox=BBOX)
    make = lambda: PolygonalHeatSolver(*mesh, ALPHA, dt, bc_type="dirichlet", bc_func=exact, source_func=source)
    t0 = time.perf_counter()
    s = make()
    setup = time.perf_counter() - t0
    # Second construction on the same mesh: derived geometry comes from the
    # content-hash cache (what a mesh loaded from the mesh library gets).
    t0 = time.perf_counter()
    s2 = make()
    setup_cached = time.perf_counter() - t0
    t0 = time.perf_counter()
    A_loop = s._assemble_tpfa_loop()  # the legacy per-face loop, kept as a reference
    assembly_loop = time.perf_counter() - t0
    t0 = time.perf_counter()
    s._assemble_system()  # vectorized
    assembly = time.perf_counter() - t0
    diff = abs(s.A - A_loop).max() / abs(A_loop).max()
    K = s.M_diag + dt * s.A
    t0 = time.perf_counter()
    _legacy_identity_rows(K, s.is_boundary)
    lhs_legacy = time.perf_counter() - t0
    t0 = time.perf_counter()
    lhs = s._build_linear_lhs_csr(dt)
    lhs_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    lu = factorized(lhs.tocsc())
    factor = time.perf_counter() - t0
    # The lean time loop: exactly the per-step work of the cached linear path.
    c = s.cell_centers
    area = s.cell_areas
    bidx = s._boundary_idx
    u = exact(c[:, 0], c[:, 1], 0.0)
    t_rhs = t_solve = 0.0
    t = 0.0
    for _ in range(nsteps):
        t += dt
        a = time.perf_counter()
        rhs = area * u + dt * area * source(c[:, 0], c[:, 1], t)
        rhs[bidx] = exact(c[bidx, 0], c[bidx, 1], t)
        b = time.perf_counter()
        u = lu(rhs)
        t_rhs += b - a
        t_solve += time.perf_counter() - b
    # The public solve(): first call assembles, builds the LHS, factorizes and loops;
    # a repeated call reuses the cached operator and factorization.
    u0 = exact(c[:, 0], c[:, 1], 0.0)
    t0 = time.perf_counter()
    _, u_full = s2.solve(u0, 0.0, nsteps * dt)
    solve_first = time.perf_counter() - t0
    t0 = time.perf_counter()
    s2.solve(u0, 0.0, nsteps * dt)
    solve_repeat = time.perf_counter() - t0
    framework = max(solve_first - (assembly + lhs_build + factor + t_rhs + t_solve), 0.0)
    out = {"n": n, "cells": int(s.M), "nsteps": nsteps, "setup": setup, "setup_cached": setup_cached,
           "assembly_loop": assembly_loop, "assembly": assembly, "lhs_legacy": lhs_legacy, "lhs_build": lhs_build,
           "factorization": factor, "rhs_per_run": t_rhs, "backsolve_per_run": t_solve,
           "solve_first": solve_first, "solve_repeat": solve_repeat, "framework_overhead": framework,
           "vectorized_vs_loop_rel_diff": float(diff),
           "lean_vs_solve_rel_diff": float(np.max(np.abs(u - u_full)) / np.max(np.abs(u_full)))}
    print(f"[1] n={n:4d} " + " ".join(f"{k}={v:.3g}" for k, v in out.items() if isinstance(v, float)))
    return out


def _before_rows():
    if not BEFORE.exists():
        return None
    return {r["cells"]: r for r in json.loads(BEFORE.read_text())["where_time_goes"]}


def plot_where_time_goes(rows):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    labels = [f"{r['cells']:,}" for r in rows]
    parts = [("setup_cached", "mesh geometry (cached, as from the mesh library)", "0.75"),
             ("assembly", "operator assembly (vectorized)", "tab:red"),
             ("lhs_build", "LHS + Dirichlet rows", "tab:pink"),
             ("factorization", "sparse LU factorization", "tab:purple"),
             ("rhs_per_run", "right-hand sides (100 steps)", "tab:olive"),
             ("backsolve_per_run", "LU back-solves (100 steps)", "tab:blue"),
             ("framework_overhead", "other solve() overhead", "tab:orange")]
    a = ax[0]
    bottom = np.zeros(len(rows))
    tot = np.array([sum(r[k] for k, _, _ in parts) for r in rows])  # phases are timed separately
    for key, lab, col in parts:
        frac = np.array([r[key] for r in rows]) / tot
        a.bar(labels, frac, bottom=bottom, color=col, label=lab)
        bottom += frac
    a.set_ylabel("fraction of measured phases"); a.set_xlabel("cells")
    a.set_ylim(0, 1)
    a.set_title("(a) a 100-step run now: linear algebra dominates")
    a.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    a = ax[1]
    cells = np.array([r["cells"] for r in rows])
    a.loglog(cells, [r["assembly_loop"] for r in rows], "o-", color="tab:red", label="per-face Python loop (before)")
    a.loglog(cells, [r["assembly"] for r in rows], "s-", color="tab:green", label="vectorized assembly (now)")
    a.loglog(cells, [r["lhs_legacy"] for r in rows], "o:", color="tab:pink", label="Dirichlet rows, LIL loop (before)")
    a.loglog(cells, [r["lhs_build"] for r in rows], "s:", color="tab:olive", label="Dirichlet rows, vectorized (now)")
    a.loglog(cells, [r["factorization"] for r in rows], "^--", color="tab:purple", label="LU factorization (C)")
    r = rows[-1]
    a.annotate(f"{r['assembly_loop'] / r['assembly']:.0f}x faster,\nrel. diff {r['vectorized_vs_loop_rel_diff']:.0e}",
               xy=(r["cells"], r["assembly"]), xytext=(0.45, 0.1), textcoords="axes fraction",
               fontsize=7, arrowprops={"arrowstyle": "->", "lw": 0.7})
    a.set_xlabel("cells"); a.set_ylabel("wall time [s]")
    a.set_title("(b) assembly and Dirichlet rows: loop vs vectorized"); a.legend(fontsize=6.5)
    a = ax[2]
    before = _before_rows()
    if before is not None:
        bc = [c for c in cells if c in before]
        a.loglog(bc, [before[c]["setup"] + before[c]["solve_wall"] for c in bc], "o-", color="tab:red",
                 label="before: solve() + first-build setup")
    now = np.array([r["setup"] + r["solve_first"] for r in rows])
    cached = np.array([r["setup_cached"] + r["solve_first"] for r in rows])
    repeat = np.array([r["solve_repeat"] for r in rows])
    kernel = np.array([r["factorization"] + r["backsolve_per_run"] for r in rows])
    a.loglog(cells, now, "s-", color="tab:orange", label="now: solve() + first-build setup")
    a.loglog(cells, cached, "D-", color="tab:green", label="now: solve() with cached geometry")
    a.loglog(cells, repeat, "v-", color="tab:cyan", label="now: repeated solve() (operator + LU reused)")
    a.loglog(cells, kernel, "^--", color="tab:blue", label="linear-algebra kernels only (LU + back-solves)")
    if before is not None and cells[-1] in before:
        b = before[cells[-1]]
        a.set_title(f"(c) 100-step run at {cells[-1]:,} cells: "
                    f"{(b['setup'] + b['solve_wall']) / cached[-1]:.1f}x faster with cached geometry", fontsize=9)
    a.set_xlabel("cells"); a.set_ylabel("wall time [s], 100 steps")
    a.legend(fontsize=6.5)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.27)
    fig.savefig(OUT / "where_time_goes.png", dpi=160)
    plt.close(fig)


def boundary_closure_costs(sizes=(64, 128, 256), nsteps=100, dt=2.5e-4):
    """100-step Robin / radiative solves: factorization reuse vs refactorizing every step."""
    from heat_solver.bc_test_problems import heat_problem

    out = []
    for n in sizes:
        for bc in ("robin", "radiative"):
            p = heat_problem(bc)
            m = generate("square", nx=n, ny=n, bbox=p["bbox"])
            row = {"n": n, "cells": n * n, "bc": bc}
            for mode in ("refactorize", "reuse"):
                kw = {}
                if bc == "radiative" and mode == "refactorize":
                    kw["nonlinear_options"] = {"radiative_jacobian": "newton"}
                s = PolygonalHeatSolver(*m, p["alpha"], dt, bc_type=bc, bc_func=p["bc_func"],
                                        source_func=p["source"], **kw)
                if bc == "robin" and mode == "refactorize":
                    s._factor_cache.lookup = lambda *a, **k: None  # the old behaviour: no reuse
                c = s.cell_centers
                u0 = p["exact"](c[:, 0], c[:, 1], 0.0)
                s.solve(u0, 0.0, dt)  # assembly and first factorization outside the timing
                t0 = time.perf_counter()
                s.solve(u0, 0.0, nsteps * dt)
                row[mode] = time.perf_counter() - t0
                row[mode + "_stats"] = dict(s.last_solve_stats)
            out.append(row)
            print(f"[1b] n={n} {bc}: refactorize {row['refactorize']:.2f}s reuse {row['reuse']:.2f}s")
    return out


def plot_boundary_closures(rows):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for bc, col in (("robin", "tab:blue"), ("radiative", "tab:red")):
        rr = [r for r in rows if r["bc"] == bc]
        cells = [r["cells"] for r in rr]
        ax.loglog(cells, [r["refactorize"] for r in rr], "o--", color=col,
                  label=f"{bc}: refactorize every step" + (" (full Newton)" if bc == "radiative" else ""))
        ax.loglog(cells, [r["reuse"] for r in rr], "s-", color=col,
                  label=f"{bc}: reuse factorization" + (" (frozen Jacobian)" if bc == "radiative" else ""))
        r = rr[-1]
        ax.annotate(f"{r['refactorize'] / r['reuse']:.0f}x", (r["cells"], r["reuse"]), textcoords="offset points",
                    xytext=(6, -4), fontsize=8, color=col)
    ax.set_xlabel("cells"); ax.set_ylabel("wall time, 100 steps [s]")
    ax.set_title("Robin / radiative boundaries: factorization reuse")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "boundary_closures.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Linear solvers (algorithmic acceleration)
# ---------------------------------------------------------------------------


def interior_system(n, dt):
    """SPD interior system of Thermalio's backward-Euler step on an n x n square mesh.

    Thermalio holds the outer cell band at the Dirichlet data; eliminating it
    leaves ``(h^2 I + dt alpha L5) u_I = rhs`` with the 5-point Laplacian ``L5``
    on the ``(n-2)^2`` interior cells.  :func:`check_interior_system` verifies
    this against Thermalio's assembled matrix.
    """
    h = (BBOX[1] - BBOX[0]) / n
    m = n - 2
    T = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(m, m))
    I = sp.identity(m)
    L = sp.kron(I, T) + sp.kron(T, I)
    return (h * h * sp.identity(m * m) + dt * ALPHA * L).tocsr()


def check_interior_system(n=32, dt=1e-3):
    mesh = generate("square", nx=n, ny=n, bbox=BBOX)
    s = PolygonalHeatSolver(*mesh, ALPHA, dt, bc_type="dirichlet", bc_func=exact, source_func=source)
    s._assemble_system()
    K = (s.M_diag + dt * s.A).tocsr()
    c = s.cell_centers
    interior = np.where(~s.is_boundary)[0]
    order = interior[np.lexsort((c[interior, 0], c[interior, 1]))]
    Ki = K[order][:, order]
    return float(abs(Ki - interior_system(n, dt)).max())


def solver_costs(n, regime, nsolve=3):
    h = (BBOX[1] - BBOX[0]) / n
    dt = 0.5 * h * h / ALPHA if regime == "transient" else 1.0
    A = interior_system(n, dt)
    N = A.shape[0]
    rng = np.random.default_rng(0)
    rhs = [A @ rng.standard_normal(N) for _ in range(nsolve)]
    out = {"n": n, "N": N, "regime": regime, "dt": dt}
    if N <= 300_000:
        t0 = time.perf_counter()
        lu = factorized(A.tocsc())
        out["lu_factor"] = time.perf_counter() - t0
        out["lu_solve"] = _best(lambda: [lu(r) for r in rhs], 1) / nsolve
        out["spsolve"] = out["lu_factor"] + out["lu_solve"]  # refactorize every step
        del lu
    dinv = 1.0 / A.diagonal()
    jac = LinearOperator(A.shape, matvec=lambda x: dinv * x)
    its = []

    def run_cg(M):
        its.clear()
        for r in rhs:
            k = [0]
            cg(A, r, rtol=1e-8, atol=0.0, M=M, maxiter=20000, callback=lambda _: k.__setitem__(0, k[0] + 1))
            its.append(k[0])

    out["cg_jacobi"] = _best(lambda: run_cg(jac), 1) / nsolve
    out["cg_jacobi_iters"] = float(np.mean(its))
    if pyamg is not None:
        t0 = time.perf_counter()
        ml = pyamg.smoothed_aggregation_solver(A.tocsr())
        out["amg_setup"] = time.perf_counter() - t0
        Mamg = ml.aspreconditioner(cycle="V")
        out["cg_amg"] = _best(lambda: run_cg(Mamg), 1) / nsolve
        out["cg_amg_iters"] = float(np.mean(its))
    print(f"[2] {regime:9s} n={n:5d} N={N:8d} " + " ".join(
        f"{k}={v:.3g}" for k, v in out.items() if k not in ("n", "N", "regime")))
    return out


def plot_linear_solvers(rows, nsteps=200):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    styles = {"spsolve": ("refactorize every step (spsolve)", "x-", "tab:red"),
              "lu_solve": ("factorize once, back-solve per step", "o-", "tab:purple"),
              "cg_jacobi": ("CG + Jacobi", "s-", "tab:orange"),
              "cg_amg": ("CG + algebraic multigrid (pyamg)", "D-", "tab:green")}
    for a, regime, title in ((ax[0], "transient", r"(a) per-step solve, $\Delta t = h^2/(2\alpha)$ (accuracy-balanced)"),
                             (ax[1], "stiff", r"(b) per-step solve, $\Delta t = 1$ (large steps / near-steady)")):
        rr = [r for r in rows if r["regime"] == regime]
        for key, (lab, mk, col) in styles.items():
            pts = [(r["N"], r[key]) for r in rr if key in r]
            if pts:
                a.loglog(*zip(*pts), mk, color=col, label=lab)
        Ns = np.array([r["N"] for r in rr], float)
        a.loglog(Ns, 1.2 * rr[-1]["cg_amg" if "cg_amg" in rr[-1] else "cg_jacobi"] * Ns / Ns[-1], "k:", lw=0.8,
                 label=r"$\propto N$")
        for r in rr:
            for key, dy in (("cg_jacobi_iters", 1.25), ("cg_amg_iters", 0.7)):
                base = key.replace("_iters", "")
                if key in r:
                    a.annotate(f"{r[key]:.0f}", (r["N"], r[base] * dy), fontsize=6, ha="center",
                               color="tab:orange" if "jacobi" in key else "tab:green")
        a.set_xlabel("unknowns $N$"); a.set_ylabel("wall time per time step [s]"); a.set_title(title, fontsize=9)
        a.legend(fontsize=6.5)
    a = ax[2]
    for regime, ls in (("transient", "-"), ("stiff", "--")):
        rr = [r for r in rows if r["regime"] == regime]
        runs = {
            "factorize once": [(r["N"], r["lu_factor"] + nsteps * r["lu_solve"]) for r in rr if "lu_factor" in r],
            "CG + Jacobi": [(r["N"], nsteps * r["cg_jacobi"]) for r in rr],
            "CG + AMG": [(r["N"], r["amg_setup"] + nsteps * r["cg_amg"]) for r in rr if "amg_setup" in r],
        }
        cols = {"factorize once": "tab:purple", "CG + Jacobi": "tab:orange", "CG + AMG": "tab:green"}
        for name, pts in runs.items():
            if len(pts) < 2:
                continue
            N_, T_ = map(np.array, zip(*pts))
            p = np.polyfit(np.log(N_[-3:]), np.log(T_[-3:]), 1)
            a.loglog(N_, T_, ls, marker="o", ms=3, color=cols[name],
                     label=f"{name}, {regime}: $\\propto N^{{{p[0]:.2f}}}$")
            Ne = np.array([N_[-1], 1e7])
            a.loglog(Ne, np.exp(np.polyval(p, np.log(Ne))), ":", color=cols[name], lw=0.9)
    a.axvspan(1.1e6, 1e7, color="0.92", zorder=0)
    a.text(1.3e6, a.get_ylim()[0] * 3 if a.get_ylim()[0] > 0 else 1e-2, "extrapolated\n(dotted)", fontsize=7)
    a.set_xlabel("unknowns $N$"); a.set_ylabel(f"wall time for {nsteps} steps incl. setup [s]")
    a.set_title(f"(c) total cost of a {nsteps}-step run; fitted exponents"); a.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "linear_solvers.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Parallelism
# ---------------------------------------------------------------------------


def _ensemble_member(alpha):
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    mesh = generate("square", nx=48, ny=48, bbox=BBOX)
    s = PolygonalHeatSolver(*mesh, alpha, 2.5e-4, bc_type="dirichlet", bc_func=exact, source_func=source)
    c = s.cell_centers
    _, u = s.solve(exact(c[:, 0], c[:, 1], 0.0), 0.0, 0.02)
    return float(np.sum(s.cell_areas * u))


def ensemble_scaling(members=16, workers=(1, 2, 4, 8)):
    alphas = np.linspace(0.05, 0.2, members)
    out = []
    for w in workers:
        t0 = time.perf_counter()
        if w == 1:
            res = [_ensemble_member(a) for a in alphas]
        else:
            with ProcessPoolExecutor(max_workers=w) as pool:
                res = list(pool.map(_ensemble_member, alphas))
        wall = time.perf_counter() - t0
        out.append({"workers": w, "wall": wall, "checksum": float(np.sum(res))})
        print(f"[3] ensemble workers={w} wall={wall:.2f}s")
    return out


def threaded_kernels(threads=(1, 2, 4)):
    res = {"triad": [], "spmv": [], "gemm": []}
    # Memory-bound: STREAM-like triad a = b + s c on 2 x 10^7 doubles, split over threads.
    nvec = 20_000_000
    b_, c_ = np.random.rand(nvec), np.random.rand(nvec)
    a_ = np.empty(nvec)
    # Memory-bound: 5-point sparse mat-vec with 4 x 10^6 unknowns, row blocks over threads.
    A = interior_system(2002, 1.0)
    x = np.random.rand(A.shape[0])
    y = np.empty(A.shape[0])
    for k in threads:
        edges = np.linspace(0, nvec, k + 1).astype(int)

        def triad(i):
            lo, hi = edges[i], edges[i + 1]
            np.multiply(c_[lo:hi], 3.0, out=a_[lo:hi])
            np.add(a_[lo:hi], b_[lo:hi], out=a_[lo:hi])

        redges = np.linspace(0, A.shape[0], k + 1).astype(int)
        blocks = [A[redges[i]:redges[i + 1]] for i in range(k)]

        def spmv(i):
            y[redges[i]:redges[i + 1]] = blocks[i] @ x

        with ThreadPoolExecutor(k) as pool:
            res["triad"].append(_best(lambda: list(pool.map(triad, range(k))), 5))
            res["spmv"].append(_best(lambda: list(pool.map(spmv, range(k))), 5))
    # Compute-bound: dense matrix multiply with OpenBLAS threads.
    if threadpool_limits is not None:
        M1, M2 = np.random.rand(2000, 2000), np.random.rand(2000, 2000)
        for k in threads:
            with threadpool_limits(limits=k, user_api="blas"):
                res["gemm"].append(_best(lambda: M1 @ M2, 3))
    res["threads"] = list(threads)
    res["triad_GBs"] = [3 * 8 * nvec / t / 1e9 for t in res["triad"]]
    print("[3] kernels", {k: [round(v, 4) for v in vals] for k, vals in res.items() if k != "threads"})
    return res


def plot_parallelism(ens, kern, breakdown):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    a = ax[0]
    w = np.array([e["workers"] for e in ens])
    sp_ = ens[0]["wall"] / np.array([e["wall"] for e in ens])
    a.plot(w, sp_, "o-", color="tab:blue", label="16 independent Thermalio solves (processes)")
    a.plot([1, CORES], [1, CORES], "k:", lw=0.8, label="ideal")
    a.axvline(CORES, color="0.6", ls="--", lw=0.8)
    a.text(CORES * 1.03, 1.1, f"{CORES} cores", fontsize=7)
    for wi, si in zip(w, sp_):
        a.annotate(f"{si:.2f}x", (wi, si), textcoords="offset points", xytext=(4, -10), fontsize=7)
    a.set_xlabel("worker processes"); a.set_ylabel("speedup"); a.set_title("(a) task parallelism: ensembles / sweeps / UQ")
    a.legend(fontsize=6.5)
    a = ax[1]
    th = np.array(kern["threads"])
    for key, lab, col in (("gemm", "dense matrix multiply (compute-bound)", "tab:green"),
                          ("spmv", "sparse mat-vec, 4M unknowns (memory-bound)", "tab:orange"),
                          ("triad", "vector triad (pure memory bandwidth)", "tab:red")):
        if kern[key]:
            s_ = kern[key][0] / np.array(kern[key])
            a.plot(th, s_, "o-", color=col, label=lab)
            a.annotate(f"{s_[-1]:.1f}x", (th[-1], s_[-1]), textcoords="offset points", xytext=(4, 0), fontsize=7)
    a.plot([1, th[-1]], [1, th[-1]], "k:", lw=0.8, label="ideal")
    a.set_xlabel("threads"); a.set_ylabel("speedup"); a.set_title("(b) data parallelism inside kernels")
    a.legend(fontsize=6.5)
    a = ax[2]
    r = breakdown
    P = np.unique(np.logspace(0, 8, 60, base=2).astype(int))
    curves = []
    before = _before_rows()
    if before is not None and r["cells"] in before:
        b = before[r["cells"]]
        curves.append(((b["factorization"] + b["backsolve_per_run"]) / (b["setup"] + b["solve_wall"]),
                       "before (only LU + back-solves parallel)", "tab:red"))
    # Fractions of the separately measured phases (as in where_time_goes.png panel a): LU
    # factorization and back-solves count as parallelizable in principle (parallel sparse
    # direct solvers exist, but triangular solves scale poorly).
    serial_now = r["setup_cached"] + r["lhs_build"] + r["framework_overhead"]
    par_now = r["assembly"] + r["rhs_per_run"] + r["factorization"] + r["backsolve_per_run"]
    curves.append((par_now / (par_now + serial_now), "now (assembly, RHS, LU parallel)", "tab:green"))
    curves.append((0.99, "well-engineered parallel code", "tab:blue"))
    for f, lab, col in curves:
        S = 1.0 / ((1 - f) + f / P)
        a.loglog(P, S, color=col, label=f"{lab}: f = {f:.2f}, limit {1 / (1 - f):.1f}x")
    a.loglog(P, P, "k:", lw=0.8, label="ideal")
    a.set_xlabel("cores $P$"); a.set_ylabel("speedup $S(P)$")
    a.set_title(f"(c) Amdahl's law with the measured serial fraction ({r['cells']:,} cells)")
    a.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "parallelism.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Accuracy as acceleration
# ---------------------------------------------------------------------------


def time_order_study(n=48, t_end=0.5, steps=(5, 10, 20, 40, 80, 160, 320)):
    mesh = generate("square", nx=n, ny=n, bbox=BBOX)

    def run(scheme, k):
        s = PolygonalHeatSolver(*mesh, ALPHA, t_end / k, bc_type="dirichlet", bc_func=exact, source_func=source,
                                time_scheme=scheme)
        c = s.cell_centers
        t0 = time.perf_counter()
        _, u = s.solve(exact(c[:, 0], c[:, 1], 0.0), 0.0, t_end)
        return u, time.perf_counter() - t0, s.cell_areas

    ref, _, area = run("crank_nicolson", 2560)  # same spatial discretization: isolates the time error
    out = {}
    for scheme in ("backward_euler", "bdf2", "crank_nicolson"):
        errs, walls = [], []
        for k in steps:
            u, wall, _ = run(scheme, k)
            errs.append(float(np.sqrt(np.sum(area * (u - ref) ** 2) / np.sum(area * ref**2))))
            walls.append(wall)
        out[scheme] = {"steps": list(steps), "errors": errs, "walls": walls}
        print(f"[4] {scheme} errors {np.array(errs)}")
    return out


def plot_accuracy(timeord, ext):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    a = ax[0]
    target = 1e-5
    for scheme, lab, col in (("backward_euler", "backward Euler (order 1)", "tab:red"),
                             ("bdf2", "BDF2 (order 2, damping; driver default)", "tab:purple"),
                             ("crank_nicolson", "Crank-Nicolson (order 2)", "tab:green")):
        d = timeord[scheme]
        a.loglog(d["steps"], d["errors"], "o-", color=col, label=lab)
        k = np.interp(np.log(target), np.log(d["errors"][::-1]), np.log(d["steps"][::-1]))
        if d["errors"][-1] <= target <= d["errors"][0]:
            a.plot(np.exp(k), target, "*", color=col, ms=10)
            a.annotate(f"{np.exp(k):.0f} steps", (np.exp(k), target), textcoords="offset points",
                       xytext=(5, 6), fontsize=7, color=col)
        elif d["errors"][-1] > target:  # extrapolate with the observed order of the last pair
            order = np.log(d["errors"][-2] / d["errors"][-1]) / np.log(d["steps"][-1] / d["steps"][-2])
            k_ext = d["steps"][-1] * (d["errors"][-1] / target) ** (1 / order)
            a.loglog([d["steps"][-1], k_ext], [d["errors"][-1], target], ":", color=col, lw=0.9)
            a.plot(k_ext, target, "*", color=col, ms=10, mfc="none")
            a.annotate(f"~{k_ext:,.0f} steps\n(extrapolated)", (k_ext, target), textcoords="offset points",
                       xytext=(-60, 10), fontsize=7, color=col)
    a.axhline(target, color="0.6", ls="--", lw=0.8)
    a.set_xlabel("time steps"); a.set_ylabel("temporal error (vs fine-$\\Delta t$ reference)")
    a.set_title("(a) higher order in time: fewer steps for the same accuracy"); a.legend(fontsize=7)
    a = ax[1]
    if ext is not None:
        rows = ext["isotropic"]
        for code, lab, col, mk in (("skfem_p1", "scikit-fem P1 (error ~ h^2, nodal)", "tab:green", "^-"),
                                   ("thermalio", "Thermalio FV (error ~ h^2)", "tab:blue", "o-"),
                                   ("fipy", "FiPy FV (error ~ h^2)", "tab:orange", "s-"),
                                   ("skfem_p2", "scikit-fem P2 (error ~ h^3)", "tab:red", "v-")):
            dofs = [r.get(f"{code}_dofs", r["thermalio_dofs"]) for r in rows]
            a.loglog(dofs, [r[code] for r in rows], mk, color=col, label=lab)
        a.axhline(1e-4, color="0.6", ls="--", lw=0.8)
        a.set_xlabel("degrees of freedom"); a.set_ylabel("relative $L^2$ error vs exact")
        a.set_title("(b) higher order in space: fewer unknowns per digit"); a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_as_acceleration.png", dpi=160)
    plt.close(fig)


def main():
    t_start = time.perf_counter()
    info = {"machine": platform.machine(), "cores": CORES, "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "pyamg": getattr(pyamg, "__version__", None)}
    print(info)
    info["interior_system_check"] = check_interior_system()
    print("[2] interior system vs Thermalio matrix, max abs diff", info["interior_system_check"])
    breakdown = [where_time_goes(n) for n in (32, 64, 128, 256, 512)]
    plot_where_time_goes(breakdown)
    closures = boundary_closure_costs()
    plot_boundary_closures(closures)
    solvers = [solver_costs(n, regime) for regime in ("transient", "stiff") for n in (64, 128, 256, 512, 1024)]
    plot_linear_solvers(solvers)
    ens = ensemble_scaling()
    kern = threaded_kernels()
    plot_parallelism(ens, kern, breakdown[-2])
    timeord = time_order_study()
    ext_path = Path("test_plots/benchmark/external_codes/summary.json")
    ext = json.loads(ext_path.read_text()) if ext_path.exists() else None
    plot_accuracy(timeord, ext)
    summary = {"environment": info, "where_time_goes": breakdown, "boundary_closures": closures,
               "linear_solvers": solvers,
               "ensemble": ens, "threaded_kernels": kern, "time_order": timeord}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("done", time.perf_counter() - t_start)


def replot():
    """Redraw the figures from ``summary.json`` without re-measuring."""
    summary = json.loads((OUT / "summary.json").read_text())
    rows = summary["where_time_goes"]
    plot_where_time_goes(rows)
    if "boundary_closures" in summary:
        plot_boundary_closures(summary["boundary_closures"])
    plot_linear_solvers(summary["linear_solvers"])
    plot_parallelism(summary["ensemble"], summary["threaded_kernels"],
                     next(r for r in rows if r["cells"] == 65536))
    ext_path = Path("test_plots/benchmark/external_codes/summary.json")
    plot_accuracy(summary["time_order"], json.loads(ext_path.read_text()) if ext_path.exists() else None)


if __name__ == "__main__":
    import sys

    replot() if "--replot" in sys.argv else main()
