import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.tri import Triangulation


def visualize_polygonal_mesh(vertices, polygons, values=None, ax=None, cmap="viridis", edgecolor="k"):
    if ax is None:
        _, ax = plt.subplots()
    verts = [vertices[poly] for poly in polygons]
    coll = PolyCollection(
        verts,
        array=values,
        cmap=cmap,
        edgecolor=edgecolor,
        linewidths=0.7,
        antialiaseds=False,
        closed=True,
    )
    ax.add_collection(coll)
    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")
    if values is not None:
        plt.colorbar(coll, ax=ax)
    return ax


def create_polygonal_figure(vertices, polygons, u_num, u_exact, diff, case_name, mesh_title):
    vmin = min(u_num.min(), u_exact.min())
    vmax = max(u_num.max(), u_exact.max())
    errmax = np.max(np.abs(diff))
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    ax0, ax1, ax2, ax3 = axs.flat
    visualize_polygonal_mesh(vertices, polygons, None, ax=ax0)
    ax0.set_title(mesh_title)
    visualize_polygonal_mesh(vertices, polygons, u_num, ax=ax1)
    ax1.set_title("Numerical Solution")
    ax1.collections[-1].set_clim(vmin, vmax)
    visualize_polygonal_mesh(vertices, polygons, u_exact, ax=ax2)
    ax2.set_title(f"Analytical Solution ({case_name})")
    ax2.collections[-1].set_clim(vmin, vmax)
    visualize_polygonal_mesh(vertices, polygons, diff, ax=ax3, cmap="coolwarm")
    ax3.set_title(f"Error (max |e| = {errmax:.2e})")
    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    return fig


def create_delaunay_figure(points, tris, u_num, u_exact, diff, case_name):
    tri = Triangulation(points[:, 0], points[:, 1], tris)
    vmin = min(u_num.min(), u_exact.min())
    vmax = max(u_num.max(), u_exact.max())
    errmax = np.max(np.abs(diff))
    fig = plt.figure(figsize=(12, 10))
    ax0 = fig.add_subplot(2, 2, 1)
    ax0.triplot(tri, color="gray", lw=0.5, alpha=0.8)
    ax0.set_title("Unstructured Delaunay Mesh")
    ax0.set_aspect("equal", adjustable="box")
    ax1 = fig.add_subplot(2, 2, 2)
    tpc1 = ax1.tricontourf(tri, u_num, levels=30, cmap="viridis", vmin=vmin, vmax=vmax)
    fig.colorbar(tpc1, ax=ax1)
    ax1.set_title("Numerical Solution")
    ax1.set_aspect("equal", adjustable="box")
    ax2 = fig.add_subplot(2, 2, 3)
    tpc2 = ax2.tricontourf(tri, u_exact, levels=30, cmap="viridis", vmin=vmin, vmax=vmax)
    fig.colorbar(tpc2, ax=ax2)
    ax2.set_title(f"Analytical Solution ({case_name})")
    ax2.set_aspect("equal", adjustable="box")
    ax3 = fig.add_subplot(2, 2, 4)
    tpc3 = ax3.tricontourf(tri, diff, levels=30, cmap="coolwarm")
    fig.colorbar(tpc3, ax=ax3)
    ax3.set_title(f"Error (max |e| = {errmax:.2e})")
    ax3.set_aspect("equal", adjustable="box")
    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    return fig

def create_curvilinear_figure(X, Y, u_num, u_exact, diff, case_name):
    vmin = min(u_num.min(), u_exact.min())
    vmax = max(u_num.max(), u_exact.max())
    errmax = np.max(np.abs(diff))
    fig = plt.figure(figsize=(12, 10))
    
    ax0 = fig.add_subplot(2, 2, 1)
    for i in range(X.shape[0]):
        ax0.plot(X[i, :], Y[i, :], color="gray", lw=0.5, alpha=0.8)
    for j in range(X.shape[1]):
        ax0.plot(X[:, j], Y[:, j], color="gray", lw=0.5, alpha=0.8)
    ax0.set_title("Curvilinear Quadrilateral Mesh")
    ax0.set_aspect("equal", adjustable="box")
    
    ax1 = fig.add_subplot(2, 2, 2)
    pcm1 = ax1.pcolormesh(X, Y, u_num, shading="gouraud", cmap="viridis", vmin=vmin, vmax=vmax)
    fig.colorbar(pcm1, ax=ax1)
    ax1.set_title("Numerical Solution")
    ax1.set_aspect("equal", adjustable="box")
    
    ax2 = fig.add_subplot(2, 2, 3)
    pcm2 = ax2.pcolormesh(X, Y, u_exact, shading="gouraud", cmap="viridis", vmin=vmin, vmax=vmax)
    fig.colorbar(pcm2, ax=ax2)
    ax2.set_title(f"Analytical Solution ({case_name})")
    ax2.set_aspect("equal", adjustable="box")
    
    ax3 = fig.add_subplot(2, 2, 4)
    pcm3 = ax3.pcolormesh(X, Y, diff, shading="gouraud", cmap="coolwarm")
    fig.colorbar(pcm3, ax=ax3)
    ax3.set_title(f"Error (max |e| = {errmax:.2e})")
    ax3.set_aspect("equal", adjustable="box")
    
    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    return fig

