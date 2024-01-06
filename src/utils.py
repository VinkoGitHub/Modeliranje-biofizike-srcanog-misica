from dolfinx import mesh, fem, geometry, io
from dolfinx.plot import vtk_mesh
from configs import *
import numpy as np
import pyvista
from mpi4py import MPI


# Managing meshes
def create_mesh(*args):
    return mesh.create_unit_square(*args)


def import_mesh(filename: str):
    """Import mesh from an .xdmf file."""
    with io.XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
        return xdmf.read_mesh(name="Grid")


# fem.Function utilities
def plot_function(
    function: fem.Function,
    VectorSpace: fem.FunctionSpaceBase,
    function_name: str = "function",
    camera_direction: list[float] = [1, 1, 1],
    shadow: bool = False,
):
    # Create a pyvista mesh and attach the values of u
    grid = pyvista.UnstructuredGrid(*vtk_mesh(VectorSpace))
    grid.point_data["function"] = function.x.array
    grid.set_active_scalars("function")

    # We visualize the data
    plotter = pyvista.Plotter()
    plotter.add_text(
        f"{function_name}", position="upper_edge", font_size=14, color="black"
    )
    plotter.add_mesh(grid, show_edges=False, lighting=shadow)
    plotter.view_vector(camera_direction)
    plotter.show()


def evaluate_function_at_points(
    function: fem.Function, mesh: mesh.Mesh, points: np.ndarray
) -> np.ndarray:
    """Takes a fem Function, mesh domain and a set of points of shape (num_points, 3)
    and evaluates the function at each point."""
    bb_tree = geometry.bb_tree(mesh, mesh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(mesh, cell_candidates, points)
    cells = []
    for i, _ in enumerate(points):
        if len(colliding_cells.links(i)) > 0:
            cells.append(colliding_cells.links(i)[0])
    return function.eval(points, cells)


def evaluate_function_at_point(
    function: fem.Function, mesh: mesh.Mesh, point: list
) -> float:
    point = np.array(point)
    """Takes a fem Function, mesh domain and a point of shape (num_points, 3)
    and evaluates the function at the point."""
    bb_tree = geometry.bb_tree(mesh, mesh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, point)
    colliding_cells = geometry.compute_colliding_cells(mesh, cell_candidates, point)
    return function.eval(point, colliding_cells[0])[0]
