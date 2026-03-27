import numpy as np
from heat_solver import NonUniformHeatSolver, get_analytical_case
from tests import _delaunay_config, RESOLUTION_LEVELS, CASE_SETTINGS
from heat_solver import generate_nonuniform_delaunay

# Test the unit heat kernel on coarse triangle mesh
case = "anisotropic_heat_kernel"
settings = CASE_SETTINGS[case]
level = RESOLUTION_LEVELS[4]
config = _delaunay_config(case, settings, level)

# Extract config args
bbox = config["bbox"]
nx = config["nx"]
ny = config["ny"]
jitter = config["jitter"]
seed = config["seed"]

points, tris = generate_nonuniform_delaunay(nx=nx, ny=ny, bbox=bbox, jitter=jitter, seed=seed)
alpha = config["alpha"]
dt = config["dt"]
t_init = config["t_init"]
t_end = config["t_end"]
case_info = get_analytical_case(case, alpha, t_end)
solution = case_info["solution"]
source = case_info.get("source")

solver = NonUniformHeatSolver(
    points, tris, alpha, dt, bc_func=solution, source_func=source
)

u0 = solution(points[:, 0], points[:, 1], t_init)
t_final, u_num = solver.solve(u0, t_init, t_end)
u_exact = solution(points[:, 0], points[:, 1], t_final)

diff = u_num - u_exact
l2 = np.sqrt(np.mean(diff**2))
l2_ref = np.sqrt(np.mean(u_exact**2))
print(f"L2 error: {l2/l2_ref}")
