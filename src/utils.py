from dolfinx import mesh, fem, geometry, io
from dolfinx.plot import vtk_mesh
import numpy as np
import pyvista
from typing import Callable
from mpi4py import MPI
import ufl


# mesh.Mesh utilities
def create_square(Nx: int, Ny: int):
    """Create a unit square mesh which contains Nx points
    in x-direction and Ny points in y-direction."""
    return mesh.create_unit_square(MPI.COMM_WORLD, Nx, Ny)


def create_cube(Nx: int, Ny: int, Nz: int):
    """Create a unit box mesh which contains Nx points
    in x-direction, Ny points in y-direction and Nz points in
    z-direction."""
    return mesh.create_unit_cube(MPI.COMM_WORLD, Nx, Ny, Nz)


def import_mesh(filename: str):
    """Import mesh from an .xdmf file."""
    with io.XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
        return xdmf.read_mesh(name="Grid")


def plot_mesh(
    domain: mesh.Mesh,
    mesh_name: str = "Mesh",
    camera_direction: list[float] = [1, 1, 1],
    zoom: float = 1.0,
    shadow: bool = False,
    save_to: str | None = None,
):
    """Plot a dolfinx Mesh."""
    # Create a pyvista mesh and attach the values of u
    grid = pyvista.UnstructuredGrid(*vtk_mesh(domain))

    # We visualize the data
    plotter = pyvista.Plotter()
    plotter.add_text(f"{mesh_name}", position="upper_edge", font_size=14, color="black")
    plotter.add_mesh(grid, show_edges=True, lighting=shadow)
    plotter.view_vector(camera_direction)
    plotter.camera.zoom(zoom)
    if save_to is not None:
        plotter.save_graphic(save_to)
    plotter.show()


# fem.Function utilities
def evaluate_function_at_points(
    function: fem.Function, points: np.ndarray
) -> np.ndarray:
    """Takes a fem Function, mesh domain and a set of points of shape (num_points, 3)
    and evaluates the function at each point."""
    bb_tree = geometry.bb_tree(
        function.function_space.mesh, function.function_space.mesh.topology.dim
    )
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(
        function.function_space.mesh, cell_candidates, points
    )
    cells = []
    for i, _ in enumerate(points):
        if len(colliding_cells.links(i)) > 0:
            cells.append(colliding_cells.links(i)[0])
    return function.eval(points, cells)


def evaluate_function_at_point(function: fem.Function, point: list) -> float:
    point = np.array(point)
    """Takes a fem Function, mesh domain and a point of shape (num_points, 3)
    and evaluates the function at the point."""
    bb_tree = geometry.bb_tree(
        function.function_space.mesh, function.function_space.mesh.topology.dim
    )
    cell_candidates = geometry.compute_collisions_points(bb_tree, point)
    colliding_cells = geometry.compute_colliding_cells(
        function.function_space.mesh, cell_candidates, point
    )
    return function.eval(point, colliding_cells[0])[0]


def plot_function(
    function: fem.Function,
    function_name: str = "function",
    camera_direction: list[float] = [1, 1, 1],
    zoom: float = 1.0,
    shadow: bool = False,
    show_mesh: bool = True,
    save_to: str | None = None,
):
    """Plot a dolfinx fem Function."""
    # Create a pyvista mesh and attach the values of u
    grid = pyvista.UnstructuredGrid(*vtk_mesh(function.function_space))
    grid.point_data["function"] = function.x.array
    grid.set_active_scalars("function")

    # We visualize the data
    plotter = pyvista.Plotter()
    plotter.add_text(
        f"{function_name}", position="upper_edge", font_size=14, color="black"
    )
    plotter.add_mesh(grid, show_edges=show_mesh, lighting=shadow)
    plotter.view_vector(camera_direction)
    plotter.camera.zoom(zoom)
    if save_to is not None:
        plotter.save_graphic(save_to)
    plotter.show()


def plot_vector_field(
    domain: mesh.Mesh,
    vector_field: Callable[[ufl.SpatialCoordinate], list],
    tolerance: float = 0.05,
    factor: float = 0.3,
    camera_direction: list[float] = [1, 1, 1],
    zoom: float = 1.0,
    save_to: str | None = None,
):
    """A function that plots a vector field defined on a given domain.
    A vector field must be 3-dimensional.

    Parameters
    ----------
    domain: mesh.Mesh
        A mesh domain on which we plot the vector field.
    vector_field: Callable
        Input a function that intakes x and returns a vector field as a list.\n
    Example:
        >>> lambda x: [x[1], -x[0], 0]
    tolerance: float
        A paremeter which controls amount of plotted glyphs.
    factor: float
        A parameter which scales each of the glyphs by the given amount.
    """
    vector_field_new = lambda x: [
        val * (x[0] ** 2 + 1) / (x[0] ** 2 + 1) for val in vector_field(x)
    ]

    VS = fem.VectorFunctionSpace(domain, ("CG", 2), 3)
    v = fem.Function(VS)

    v.interpolate(vector_field_new)

    mesh = pyvista.UnstructuredGrid(*vtk_mesh(VS))
    mesh["vectors"] = v.x.array.reshape((-1, 3))
    mesh.set_active_vectors("vectors")
    arrows = mesh.glyph(
        scale="vectors",
        orient="vectors",
        tolerance=tolerance,
        factor=factor,
    )
    plotter = pyvista.Plotter()
    plotter.add_mesh(arrows, color="black")
    plotter.add_mesh(
        mesh,
        color="firebrick",
        show_scalar_bar=False,
    )
    plotter.view_vector(camera_direction)
    plotter.camera.zoom(zoom)
    if save_to is not None:
        plotter.save_graphic(save_to)  
    plotter.show()


# Other utilities
def RK2_step(f: Callable, dt: float, v: np.ndarray, *args) -> np.ndarray:
    """Napisati dokumentaciju!"""
    k1 = f(v, *args)
    k2 = f(v + dt * k1, *args)
    return v + dt / 2 * (k1 + k2)


def RK3_step(f: Callable, dt: float, v: np.ndarray, *args) -> np.ndarray:
    """Napisati dokumentaciju!"""
    k1 = f(v, *args)
    k2 = f(v + dt / 2 * k1, *args)
    k3 = f(v - dt * k1 + 2 * dt * k2, *args)
    return v + dt / 6 * (k1 + 4 * k2 + k3)


def RK4_step(f: Callable, dt: float, v: np.ndarray, *args) -> np.ndarray:
    """Napisati dokumentaciju!"""
    k1 = f(v, *args)
    k2 = f(v + dt / 2 * k1, *args)
    k3 = f(v + dt / 2 * k2, *args)
    k4 = f(v + dt * k3, *args)
    return v + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
