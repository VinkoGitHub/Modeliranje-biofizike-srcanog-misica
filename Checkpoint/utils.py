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
    camera_direction: list[float] = [1, 1, 1],
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
    plotter.add_mesh(grid, show_edges=False)
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


# defining cell models

class ReparametrizedFitzHughNagumo:
    def __init__(self):
        pass

    @staticmethod
    def f(V: np.ndarray, w: np.ndarray, b: int = 0.013, c3: int = 1.0) -> np.ndarray:
        return b * (V - V_REST - c3 * w)

    @staticmethod
    def I_ion(
        V: np.ndarray, w: np.ndarray, a: int = 0.13, c1: int = 0.26, c2: int = 0.1
    ) -> np.ndarray:
        V_AMP = V_PEAK - V_REST
        V_TH = V_REST + a * V_AMP
        return (
            c1 / V_AMP**2 * (V - V_REST) * (V - V_TH) * (V_PEAK - V)
            - c2 / V_AMP * (V - V_REST) * w
        )
