from dolfinx import mesh, fem, geometry, io
from dolfinx.plot import vtk_mesh
import numpy as np
import pyvista
from mpi4py import MPI


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
    shadow: bool = False,
):
    """Plot a dolfinx Mesh."""
    # Create a pyvista mesh and attach the values of u
    grid = pyvista.UnstructuredGrid(*vtk_mesh(domain))

    # We visualize the data
    plotter = pyvista.Plotter()
    plotter.add_text(f"{mesh_name}", position="upper_edge", font_size=14, color="black")
    plotter.add_mesh(grid, show_edges=True, lighting=shadow)
    plotter.view_vector(camera_direction)
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
    shadow: bool = False,
    show_mesh: bool = True,
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
    plotter.show()

def plot_vector_field(
    domain: mesh.Mesh, vector_field: list = [0, 0, 0], tolerance: float = 0.05, factor: float = 0.3
):
    """A function that plots a vector field defined on a given domain.
    In order to use this function and x, y and z coordinates,
    you must define them before you call the function.\n
    
    They are defined as:
    >>> x = domain.geometry.x[:, 0]
    >>> y = domain.geometry.x[:, 1]
    >>> z = domain.geometry.x[:, 2]

    Parameters
    ----------
    vector_field: list
        Input a vector field, eg [0, 0, 1] or [y, -x, 0].
    tolerance: float
        A paremeter which controls amount of plotted glyphs.
    factor: float
        A parameter which scales each of the glyphs by the given amount.
    """

    mesh = pyvista.UnstructuredGrid(*vtk_mesh(domain))
    shape = domain.geometry.x.shape[0]

    vectors = np.vstack(
        (
            np.zeros(shape) + vector_field[0],
            np.zeros(shape) + vector_field[1],
            np.zeros(shape) + vector_field[2],
        )
    ).T

    # add and scale
    mesh["vectors"] = vectors
    mesh.set_active_vectors("vectors")

    arrows = mesh.glyph(
        scale="vectors",
        orient="vectors",
        tolerance=tolerance,
        factor=factor,
    )
    pl = pyvista.Plotter()
    pl.add_mesh(arrows, color="black")
    pl.add_mesh(
        mesh,
        color="firebrick",
        show_scalar_bar=False,
    )
    pl.show()
