from dolfinx.mesh import create_unit_square
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.plot import vtk_mesh
import ufl
from configs import *
from mpi4py import MPI
import numpy as np
import pyvista


# Managing a mesh
def create_mesh(N=N):
    return mesh.create_unit_square(MPI.COMM_WORLD, N, N)


def plot_mesh(mesh):
    """
    Given a DOLFINx mesh, create a `pyvista.UnstructuredGrid`,
    and plot it and the mesh nodes
    """
    plotter = pyvista.Plotter()
    ugrid = pyvista.UnstructuredGrid(*vtk_mesh(mesh))
    if mesh.geometry.cmaps[0].degree > 1:
        plotter.add_mesh(ugrid, style="points", color="b", point_size=10)
        ugrid = ugrid.tessellate()
        show_edges = False
    else:
        show_edges = True
    plotter.add_mesh(ugrid, show_edges=show_edges)

    plotter.show_axes()
    plotter.view_xy()
    plotter.show()


def plot_subdomains(domain, cells_H, cells_T):
    # Filter out ghosted cells
    num_cells_local = domain.topology.index_map(domain.topology.dim).size_local
    marker = np.zeros(num_cells_local, dtype=np.int32)
    cells_H = cells_H[cells_H < num_cells_local]
    cells_T = cells_T[cells_T < num_cells_local]
    marker[cells_H] = 1
    marker[cells_T] = 0
    topology, cell_types, x = vtk_mesh(
        domain, domain.topology.dim, np.arange(num_cells_local, dtype=np.int32)
    )

    p = pyvista.Plotter(window_size=[500, 500])
    grid = pyvista.UnstructuredGrid(topology, cell_types, x)
    grid.cell_data["Marker"] = marker
    grid.set_active_scalars("Marker")
    p.add_mesh(grid, show_edges=True)
    p.show()


def plot_function(function, VectorSpace):
    # Create a topology that has a 1-1 correspondence with the
    # degrees-of-freedom in the function space V
    cells, types, x = plot.vtk_mesh(VectorSpace)

    # Create a pyvista mesh and attach the values of u
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["function"] = function.x.array
    grid.set_active_scalars("function")

    # We visualize the data
    plotter = pyvista.Plotter()
    plotter.add_text(
        f"Function {function}", position="upper_edge", font_size=14, color="black"
    )
    plotter.add_mesh(grid, show_edges=True)
    plotter.add_mesh(grid, style="points", point_size=8, render_points_as_spheres=True)
    plotter.view_xy()
    plotter.show()


def heart_marker(x):
    return np.array((x[0] - Hx) ** 2 + (x[1] - Hy) ** 2 <= R**2)


def torso_marker(x):
    return np.array((x[0] - Hx) ** 2 + (x[1] - Hy) ** 2 >= R**2)

####################################################################################
# MAYBE ERASE?


# solver and its parameters
def custom_solver(
    F, v, atol=1e-2, rtol=1e-2, max_iter=10, relax_par=1.0, conv_crit="incremental"
):
    # Jacobian
    J = ufl.derivative(F, v)

    # Problem solver
    problem = NonlinearProblem(F, v, J=J)
    solver = NewtonSolver(MPI.COMM_WORLD, problem)

    # Problem solver parameters
    solver.convergence_criterion = conv_crit
    solver.rtol = rtol
    solver.atol = atol
    solver.max_iter = max_iter
    solver.report = True

    return solver
