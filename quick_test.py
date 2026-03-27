import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from heat_solver.drivers import run_square_polygonal_test
from heat_solver.plotting import create_polygonal_figure

vertices, polygons, centers, u_num, u_exact, diff, results = run_square_polygonal_test(
    nx=100,
    ny=100,
    case='sine_mode',
    alpha=0.1,
    dt=5e-3,
    t_init=0.0,
    t_end=0.02,
    bbox=(-1.0, 1.0, -1.0, 1.0),
    nonorthogonal_correction=True,
    linear_solver='bicgstab',
    linear_solver_options={'rtol': 1e-11, 'maxiter': 50_000},
)
title = 'Square polygonal 60×60 — implicit BiCGSTAB'
fig = create_polygonal_figure(vertices, polygons, u_num, u_exact, diff, results['case'], title)
fig.suptitle('Polygonal heat solver (iterative linear solve)', fontsize=14, y=1.02)
out = 'test_plots/iterative_solver_60x60.png'
fig.tight_layout()
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print('Wrote', out)
print('L2_rel', results['L2_rel'], 'Linf', results['Linf'])