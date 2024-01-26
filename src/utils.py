from dolfinx import mesh, fem, geometry, io
from dolfinx.plot import vtk_mesh
from dolfinx.io import gmshio
from typing import Callable
from mpi4py import MPI
import numpy as np
import gmsh
import pyvista
import ufl


# mesh.Mesh utilities
def heart_ventricle(coarseness: float = 0.2) -> mesh.Mesh:
    # Initialize gmsh:
    gmsh.initialize()
    model = gmsh.model()
    # paremeters for outer ellipse
    R1 = 3.0
    R2 = -8.0
    Point1 = model.occ.add_point(0.0, 0.0, 0.0, coarseness)
    Point2 = model.occ.add_point(R1 * np.cos(0), R2 * np.sin(0), 0.0, coarseness)
    Point3 = model.occ.add_point(
        R1 * np.cos(np.pi / 2), R2 * np.sin(np.pi / 2), 0.0, coarseness
    )
    Point4 = model.occ.add_point(
        R1 * np.cos(np.pi / 2), R2 * np.sin(np.pi / 2), 0.0, coarseness
    )
    Ellipse_arc = model.occ.add_ellipse_arc(Point2, Point1, Point3, Point4)
    # parameters for inner ellipse
    r1 = 2.0
    r2 = -7.0
    point1 = model.occ.add_point(0.0, 0.0, 0.0, coarseness)
    point2 = model.occ.add_point(r1 * np.cos(0), r2 * np.sin(0), 0.0, coarseness)
    point3 = model.occ.add_point(
        r1 * np.cos(np.pi / 2), r2 * np.sin(np.pi / 2), 0.0, coarseness
    )
    point4 = model.occ.add_point(
        r1 * np.cos(np.pi / 2), r2 * np.sin(np.pi / 2), 0.0, coarseness
    )
    ellipse_arc = model.occ.add_ellipse_arc(point2, point1, point3, point4)
    # define lines to close a surface
    line1 = model.occ.add_line(point2, Point2)
    line2 = model.occ.add_line(Point4, point4)
    # define a loop
    loop1 = model.occ.add_curve_loop([line1, Ellipse_arc, line2, -ellipse_arc])
    # define surface:
    surface = model.occ.add_plane_surface([loop1])
    # revolve the surface to get the volume
    # revolve the surface to get the volume
    model.occ.revolve(
        dimTags=[(2, surface)], x=0, y=0, z=0, ax=0, ay=1, az=0, angle=4 * np.pi / 3
    )
    # rotate the volume to be symmetric about z-axis
    model.occ.rotate(dimTags=[(3, 1)], x=0, y=0, z=0, ax=1, ay=0, az=0, angle=np.pi / 2)
    model.occ.rotate(dimTags=[(3, 1)], x=0, y=0, z=0, ax=0, ay=0, az=1, angle=np.pi / 2)
    # Create the relevant data structures from Gmsh model
    model.occ.synchronize()
    # add the volume to a physical group
    model.addPhysicalGroup(3, [1], name="Left ventricle")
    # Generate mesh:
    model.mesh.generate(3)
    # Creates graphical user interface
    # if "close" not in sys.argv:  # this should be included to
    #    gmsh.fltk.run()  # visualize mesh in gmsh GUI
    # Convert to Dolfinx mesh format
    domain, cell_markers, facet_markers = gmshio.model_to_mesh(
        model, MPI.COMM_WORLD, 0, 3
    )
    # It finalize the Gmsh API
    gmsh.finalize()
    return domain


def heart_slice(coarseness: float = 0.1) -> mesh.Mesh:
    # Initialize gmsh:
    gmsh.initialize()
    model = gmsh.model()
    # Define points:
    lc = coarseness
    point1 = model.geo.add_point(1, 4.4, 0, lc)
    point2 = model.geo.add_point(2.1, 4.6, 0, lc)
    point3 = model.geo.add_point(3.3, 4.3, 0, lc)
    point4 = model.geo.add_point(4.2, 3.6, 0, lc)
    point5 = model.geo.add_point(4.9, 2.4, 0, lc)
    point6 = model.geo.add_point(4.9, 0.7, 0, lc)
    point7 = model.geo.add_point(4.3, -0.5, 0, lc)
    point8 = model.geo.add_point(3.3, -1, 0, lc)
    point9 = model.geo.add_point(1.8, -1.6, 0, lc)
    point10 = model.geo.add_point(-0.2, -1.8, 0, lc)
    point11 = model.geo.add_point(-1.8, -1.4, 0, lc)
    point12 = model.geo.add_point(-2.9, -0.5, 0, lc)
    point13 = model.geo.add_point(-3.2, 1, 0, lc)
    point14 = model.geo.add_point(-2.7, 2.5, 0, lc)
    point15 = model.geo.add_point(-1.8, 3.3, 0, lc)
    point16 = model.geo.add_point(-0.7, 4.1, 0, lc)
    point17 = model.geo.add_point(-0.3, 2.5, 0, lc)
    point18 = model.geo.add_point(-1, 1.9, 0, lc)
    point19 = model.geo.add_point(-1.4, 0.9, 0, lc)
    point20 = model.geo.add_point(-1, -0.2, 0, lc)
    point21 = model.geo.add_point(0.3, -0.3, 0, lc)
    point22 = model.geo.add_point(1.4, 0.3, 0, lc)
    point23 = model.geo.add_point(1.6, 1.1, 0, lc)
    point24 = model.geo.add_point(1.5, 2, 0, lc)
    point25 = model.geo.add_point(0.5, 2.6, 0, lc)
    point26 = model.geo.add_point(2.0, 3.5, 0, lc)
    point27 = model.geo.add_point(2.3, 2.3, 0, lc)
    point28 = model.geo.add_point(2.4, 0.9, 0, lc)
    point29 = model.geo.add_point(2.3, -0.5, 0, lc)
    point30 = model.geo.add_point(3.4, 0.0, 0, lc)
    point31 = model.geo.add_point(4.1, 1, 0, lc)
    point32 = model.geo.add_point(4.2, 1.9, 0, lc)
    point33 = model.geo.add_point(3.8, 3.0, 0, lc)
    # Define curves:
    spline1 = model.geo.add_spline(
        [
            point1,
            point2,
            point3,
            point4,
            point5,
            point6,
            point7,
            point8,
            point9,
            point10,
            point11,
            point12,
            point13,
            point14,
            point15,
            point16,
            point1,
        ]
    )
    spline2 = model.geo.add_spline(
        [
            point17,
            point18,
            point19,
            point20,
            point21,
            point22,
            point23,
            point24,
            point25,
            point17,
        ]
    )
    spline3 = model.geo.add_spline(
        [
            point26,
            point27,
            point28,
            point29,
            point30,
            point31,
            point32,
            point33,
            point26,
        ]
    )
    # Define loops:
    loop1 = model.geo.add_curve_loop([spline1])
    loop2 = model.geo.add_curve_loop([spline2])
    loop3 = model.geo.add_curve_loop([spline3])
    # Define surface:
    surface = model.geo.add_plane_surface([loop1, loop2, loop3])
    gmsh.model.addPhysicalGroup(2, [surface], 1)
    # Create the relevant data structures from Gmsh model
    model.geo.synchronize()
    # Generate mesh:
    model.mesh.generate()
    # Creates graphical user interface
    # if "close" not in sys.argv: # this should be included to
    #    gmsh.fltk.run()         # visualize mesh in gmsh GUI
    # Convert to Dolfinx mesh format
    domain, cell_markers, facet_markers = gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, 0, 2
    )
    # It finalize the Gmsh API
    gmsh.finalize()
    return domain


def rectangle(x: float = 1, y: float = 1, Nx: int = 32, Ny: int = 32) -> mesh.Mesh:
    """Create a unit square mesh which contains Nx points
    in x-direction and Ny points in y-direction."""
    return mesh.create_rectangle(MPI.COMM_WORLD, [(0, 0), (x, y)], [Nx, Ny])


def box(
    x: float = 1, y: float = 1, z: float = 1, Nx: int = 32, Ny: int = 32, Nz: int = 32
) -> mesh.Mesh:
    """Create a unit box mesh which contains Nx points
    in x-direction, Ny points in y-direction and Nz points in
    z-direction."""
    return mesh.create_box(MPI.COMM_WORLD, [(0, 0, 0), (x, y, z)], [Nx, Ny, Nz])


def import_mesh(filename: str):
    """Import mesh from an .xdmf file."""
    with io.XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
        return xdmf.read_mesh(name="Grid")


def plot_mesh(
    domain: mesh.Mesh,
    mesh_name: str = "Mesh",
    camera_direction: list[float] | str = [1, 1, 1],
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
    if type(camera_direction) == list:
        plotter.view_vector(camera_direction)
    elif type(camera_direction) == str:
        plotter.camera_position = camera_direction
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
    cmap: str = "coolwarm",
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
    sargs = dict(
        title="",
        height=0.5,
        vertical=True,
        position_x=0.85,
        position_y=0.25,
        font_family="times",
    )
    plotter.add_mesh(
        grid,
        show_edges=show_mesh,
        lighting=shadow,
        cmap=cmap,
        scalar_bar_args=sargs,
    )
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
