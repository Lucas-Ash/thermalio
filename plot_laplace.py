import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from heat_solver import (
    run_curvilinear_test,
    run_test as run_delaunay_test,
    run_square_polygonal_test,
    create_delaunay_figure,
    create_polygonal_figure,
    create_curvilinear_figure,
)

def main():
    case = "laplace_equation"
    alpha = 0.1
    dt = 1e-3
    t_end = 0.02
    
    # 1. Square Polygonal
    print("Testing Square Polygonal...")
    vertices, polygons, centers, u_num, u_exact, diff, results = run_square_polygonal_test(
        case=case, alpha=alpha, dt=dt, t_init=0.0, t_end=t_end, nx=16, ny=16,
        bbox=(0.0, 1.0, 0.0, 1.0)
    )
    fig1 = create_polygonal_figure(vertices, polygons, u_num, u_exact, diff, results["case"], "Square Polygonal Mesh")
    fig1.savefig("laplace_square.png", dpi=200, bbox_inches="tight")
    plt.close(fig1)

    # 2. Delaunay
    print("Testing Delaunay...")
    points, tris, u_num, u_exact, diff, results = run_delaunay_test(
        case=case, alpha=alpha, dt=dt, t_init=0.0, t_end=t_end, nx=16, ny=16, mesh_type="delaunay",
        bbox=(0.0, 1.0, 0.0, 1.0)
    )
    fig2 = create_delaunay_figure(points, tris, u_num, u_exact, diff, results["case"])
    fig2.savefig("laplace_delaunay.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)

    # 3. Curvilinear
    print("Testing Curvilinear...")
    X, Y, u_num, u_exact, diff, results = run_curvilinear_test(
        case=case, alpha=alpha, dt=dt, t_init=0.0, t_end=t_end, nx=16, ny=16, warp=0.1,
        bbox=(0.0, 1.0, 0.0, 1.0)
    )
    fig3 = create_curvilinear_figure(X, Y, u_num, u_exact, diff, results["case"])
    fig3.savefig("laplace_curvilinear.png", dpi=200, bbox_inches="tight")
    plt.close(fig3)

    print("Plots saved: laplace_square.png, laplace_delaunay.png, laplace_curvilinear.png")

if __name__ == "__main__":
    main()
