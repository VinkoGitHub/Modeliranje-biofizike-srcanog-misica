from dolfinx import mesh, fem, geometry
from dolfinx.plot import vtk_mesh
from configs import *
import numpy as np
import pyvista


# Managing a mesh
def create_mesh(*args):
    return mesh.create_unit_square(*args)


def plot_function(
    function: fem.Function,
    VectorSpace: fem.FunctionSpaceBase,
    function_name: str = "function",
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
    plotter.add_mesh(grid, show_edges=True)
    plotter.view_xy()
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


# defining cell models


class FitzHughNagumo:
    def __init__(self):
        pass

    @staticmethod
    def f(V: np.ndarray, w: np.ndarray, a=0.7, b=0.8, tau=12.5) -> np.ndarray:
        return 1 / tau * (V + a - b * w)

    @staticmethod
    def I_ion(V: np.ndarray, w: np.ndarray, R=0.1, I_ext=0.5) -> np.ndarray:
        return V - V**3 / 3 - w + R * I_ext


class ModifiedFitzHughNagumo:
    def __init__(self):
        pass

    @staticmethod
    def f(V: np.ndarray, w: np.ndarray) -> np.ndarray:
        return l * (V - V_REST - b * w)

    @staticmethod
    def I_ion(V: np.ndarray, w: np.ndarray) -> np.ndarray:
        return -(k * (V - V_REST) * (V - V_TH) * (V - V_PEAK) - k * (V - V_REST) * w)


class CustomCellModel:
    def __init__(self):
        pass

    @staticmethod
    def f(V: np.ndarray, w: np.ndarray) -> np.ndarray:
        return l * (V - V_REST - b * w)

    @staticmethod
    def I_ion(V: np.ndarray, w: np.ndarray) -> np.ndarray:
        return -k * (V - V_REST) * (V - V_TH) * (V - V_PEAK) - w
