"""Import a mesh from a file, solve a heat problem on it, export the result.

1. Builds an L-shaped quadrilateral mesh (a domain no built-in generator makes)
   and writes it as a Gmsh 2.2 file ``examples/meshes/l_shape.msh`` with a
   tagged ``hot_wall`` boundary -- standing in for a mesh made in Gmsh.
2. Imports it, stores it in the mesh library under a name, and reloads it.
3. Solves ``u_t = div(K grad u)`` with the quadratic flux (hot re-entrant
   corner walls held at 1, everything else at 0).
4. Exports the temperature field to ``examples/meshes/l_shape_solution.vtu``
   (open in ParaView) and a PNG preview.

Run: ``MPLCONFIGDIR=/tmp/matplotlib python mesh_io_demo.py``
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection

from heat_solver import Mesh2D, PolygonalHeatSolver, export_vtu, load_mesh, load_named, save_named
from heat_solver.dmp import anisotropic_tensor

OUT = Path("examples/meshes")
OUT.mkdir(parents=True, exist_ok=True)


def build_l_shape(n=24):
    """Quads on [0,1]^2 minus the upper-right quarter; tag the re-entrant walls."""
    h = 1.0 / n
    vid = {}
    verts = []
    for j in range(n + 1):
        for i in range(n + 1):
            if i * h > 0.5 + 1e-12 and j * h > 0.5 + 1e-12:
                continue
            vid[(i, j)] = len(verts)
            verts.append((i * h, j * h))
    quads = []
    for j in range(n):
        for i in range(n):
            if i >= n // 2 and j >= n // 2:
                continue
            quads.append([vid[(i, j)], vid[(i + 1, j)], vid[(i + 1, j + 1)], vid[(i, j + 1)]])
    m = n // 2
    hot = [[vid[(m, j)], vid[(m, j + 1)]] for j in range(m, n)] + [[vid[(i, m)], vid[(i + 1, m)]] for i in range(m, n)]
    mesh = Mesh2D(np.array(verts), quads, metadata={"description": "L-shaped domain"})
    mesh.boundary_tags = {"hot_wall": np.array(hot)}
    return mesh


def main():
    msh = build_l_shape().save(OUT / "l_shape.msh")
    mesh = load_mesh(msh)  # as if produced by Gmsh
    save_named("l_shape", mesh)  # store once in the library
    mesh = load_named("l_shape")  # every later problem imports it
    print(mesh.summary())

    segments = mesh.vertices[mesh.boundary_tags["hot_wall"]]  # (k, 2, 2) tagged wall edges
    h = 1.0 / 24

    def wall_temperature(x, y, t):
        """1 on the cells along the tagged hot wall, 0 elsewhere on the boundary."""
        p = np.stack([np.asarray(x, float), np.asarray(y, float)], axis=-1)[..., None, :]
        a, b = segments[:, 0], segments[:, 1]
        s = np.clip(np.sum((p - a) * (b - a), -1) / np.sum((b - a) ** 2, -1), 0.0, 1.0)
        d = np.min(np.linalg.norm(p - (a + s[..., None] * (b - a)), axis=-1), axis=-1)
        return np.where(d <= h + 1e-12, 1.0, 0.0)

    K = anisotropic_tensor(ratio=4.0, angle=np.pi / 4) * 0.05
    solver = PolygonalHeatSolver(*mesh, K, 2e-3, bc_type="dirichlet", bc_func=wall_temperature,
                                 flux_discretization="quadratic")
    _, u = solver.solve(np.zeros(mesh.n_cells), 0.0, 1.5)
    export_vtu(mesh, OUT / "l_shape_solution.vtu", cell_data={"temperature": u})

    fig, ax = plt.subplots(figsize=(5, 4.6))
    pc = PolyCollection([mesh.vertices[p] for p in mesh.polygons], array=u, cmap="inferno", edgecolors="none")
    ax.add_collection(pc)
    ax.add_collection(LineCollection(segments, colors="cyan", linewidths=2.5, label="tagged hot_wall"))
    ax.legend(loc="lower left", fontsize=8)
    ax.set_aspect("equal"); ax.autoscale()
    fig.colorbar(pc, ax=ax, label="temperature")
    ax.set_title("L-shaped mesh imported from Gmsh\nanisotropic diffusion, quadratic flux", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "l_shape_solution.png", dpi=150)
    print("wrote", OUT / "l_shape.msh", OUT / "l_shape_solution.vtu", OUT / "l_shape_solution.png")


if __name__ == "__main__":
    main()
