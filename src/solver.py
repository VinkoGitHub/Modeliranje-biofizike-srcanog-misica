from dolfinx.fem.petsc import LinearProblem
from dolfinx import mesh, fem, plot
import matplotlib.pyplot as plt
from models.base_models import *
import numpy as np
import pyvista
import utils
import ufl


def solve(
    T: float,
    NUM_STEPS: int,
    domain: mesh.Mesh,
    cell_model: BaseCellModel,
    dynamics_model: BaseDynamicsModel,
    V_m_0: list[list[float], float, float],
    ischemia: list[list[float], float, float] | None = None,
    longitudinal_fibres: list = [0, 0, 0],
    transversal_fibres: list = [0, 0, 0],
    signal_point: list[float] | None = None,
):
    """
    Main function for solving the heart dynamics models.
    Calling the function solves given equation and saves output .gif file in plots folder.

    Parameters
    ----------
    domain : mesh.Mesh
        Domain on which the equations are solved.
    cell_model : BaseCellModel
        One of cell models in cell_models module.
    dynamics_model : BaseDynamicsModel
        One of dynamics models in dynamics_models module.
    V_m_0 : list[list[float], float, float]
        Initial value for transmembrane potential.
        First value in the list is the center [x,y,z] in which we define V_m_0.
        Second value in the list is the radius in which V_m_0 is defined.
        Third value in the list is the value of V_m_0 in the given domain.
    ischemia : list[list[float], float, float] or None
        If given, makes  value for transmembrane potential.
        First value in the list is the center [x,y,z] of ischemia.
        Second value in the list is the radius in which ischemia is defined.
        Third value in the list is the conductivity reduction factor.
    longitudinal_sheets: list
        If given, defines the vector field that tells the direction of cardiac sheets.
        Input can be constants or coordinate-dependent values.
        e.g. [-1, 0, 0] or [-x[1], x[0], 0] where x[0] means x, x[1] means y and x[2] means z.
    transversal_sheets: list
        If given, defines the vector field that tells the direction of the normal of the cardiac sheets.
        Input can be constants or coordinate-dependent values.
        e.g. [-1, 0, 0] or [-x[1], x[0], 0] where x[0] means x, x[1] means y and x[2] means z.
    signal_point: list[float]
        A point at which we track V_m.


    Returns
    -------
    V_m_n : fem.Function
        A dolfinx Function containing V_m at time T.
    signal: list[float]
        A list containing the values of V_m at all time points at a given signal_point


    """

    element = ufl.FiniteElement("P", domain.ufl_cell(), degree=2)
    W = fem.FunctionSpace(domain, ufl.MixedElement(element, element))  # P1 FEM space
    V1, sub1 = W.sub(0).collapse()
    V2, sub2 = W.sub(1).collapse()

    # Define test functions
    psi, phi = ufl.TestFunctions(W)

    # Define trialfunctions
    V_m, U_e = ufl.TrialFunctions(W)

    # Define functions
    v_ = fem.Function(W)
    w, w_n = fem.Function(V1), fem.Function(V1)
    V_m_n = fem.Function(V1)

    x = ufl.SpatialCoordinate(domain)
    d = domain.topology.dim

    def impulse_location(x):
        return (x[0] - V_m_0[0][0]) ** 2 + (x[1] - V_m_0[0][1]) ** 2 + (
            x[2] - V_m_0[0][2]
        ) ** 2 < V_m_0[1] ** 2

    cells = fem.locate_dofs_geometrical(V1, impulse_location)
    V_m_n.x.array[:] = dynamics_model.V_REST
    V_m_n.x.array[cells] = np.full_like(cells, V_m_0[2])

    # Muscle sheets
    sheet_l = ufl.as_vector(longitudinal_fibres)
    sheet_n = ufl.as_vector(transversal_fibres)

    # Healthy conductivities
    M_i = (
        SIGMA_IT * ufl.Identity(d)
        + (SIGMA_IL - SIGMA_IT) * ufl.outer(sheet_l, sheet_l)
        + (SIGMA_IN - SIGMA_IT) * ufl.outer(sheet_n, sheet_n)
    )
    M_e = (
        SIGMA_ET * ufl.Identity(d)
        + (SIGMA_EL - SIGMA_ET) * ufl.outer(sheet_l, sheet_l)
        + (SIGMA_EN - SIGMA_ET) * ufl.outer(sheet_n, sheet_n)
    )

    # Ishemic conductivities
    if ischemia != None:
        tissue_location = ischemia[0]
        tissue_radius = ischemia[1]
        reduction_factor = ischemia[2]

        M_i = ufl.conditional(
            (x[0] - tissue_location[0]) ** 2 + (x[1] - tissue_location[1]) ** 2
            < tissue_radius**2,
            M_i / reduction_factor,
            M_i,
        )

        M_e = ufl.conditional(
            (x[0] - tissue_location[0]) ** 2 + (x[1] - tissue_location[1]) ** 2
            < tissue_radius**2,
            M_e / reduction_factor,
            M_e,
        )

    problem = LinearProblem(
        a=ufl.lhs(dynamics_model.F()), L=ufl.rhs(dynamics_model.F()), u=v_
    )

    grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V1))
    plotter = pyvista.Plotter(notebook=True, off_screen=False)
    plotter.open_gif("V_m_time.gif", fps=int(NUM_STEPS / 10))
    grid.point_data["V_m"] = V_m_n.x.array
    plotter.add_mesh(
        grid,
        show_edges=False,
        lighting=True,
        smooth_shading=True,
        clim=[-100, 50],
    )

    # Defining a cell model
    CellModel = utils.ReparametrizedFitzHughNagumo()

    # List of signal values for each time step
    DT = T/NUM_STEPS
    signal = []
    t = 0.0

    # Iterate through time
    while t < T:
        # Appending the transmembrane potential value at some point to a list
        # signal.append(utils.evaluate_function_at_point(V_m_n, domain, [1.5, 1.5, 0.0]))

        # 1st step of Strang splitting
        k1_V = CellModel.I_ion(V_m_n.x.array[:], w.x.array[:])
        k2_V = CellModel.I_ion(V_m_n.x.array[:] + DT / 2 * k1_V, w.x.array[:])
        V_m_n.x.array[:] = V_m_n.x.array[:] + DT / 4 * (k1_V + k2_V)

        k1_w = CellModel.f(V_m_n.x.array[:], w.x.array[:])
        k2_w = CellModel.f(V_m_n.x.array[:], w.x.array[:] + DT / 2 * k1_w)
        w.x.array[:] = w.x.array[:] + DT / 4 * (k1_w + k2_w)

        # 2nd step of Strang splitting
        problem.solve()
        v_.x.array[sub2] = v_.x.array[sub2] - np.mean(
            v_.x.array[sub2]
        )  # Normalize U_e to zero mean

        # Update solution for V_m
        V_m_n.x.array[:] = v_.x.array[sub1]

        # 3rd step of Strang splitting
        k1_V = CellModel.I_ion(V_m_n.x.array[:], w.x.array[:])
        k2_V = CellModel.I_ion(V_m_n.x.array[:] + DT * k1_V, w.x.array[:])
        V_m_n.x.array[:] = V_m_n.x.array[:] + DT / 2 * (k1_V + k2_V)

        k1_w = CellModel.f(V_m_n.x.array[:], w.x.array[:])
        k2_w = CellModel.f(V_m_n.x.array[:], w.x.array[:] + DT * k1_w)
        w.x.array[:] = w.x.array[:] + DT / 2 * (k1_w + k2_w)

        # Print time
        print("t = %.3f" % t)
        # Update plot
        plotter.clear()
        grid.point_data["V_m"] = V_m_n.x.array[:]
        plotter.add_mesh(
            grid,
            show_edges=False,
            lighting=True,
            smooth_shading=True,
            clim=[-100, 50],
        )
        plotter.add_title("t = %.3f" % t, font_size=24)
        plotter.view_vector([1, -1, -1])
        plotter.write_frame()
        t += DT

    plotter.close()

    return V_m_n, signal
