from src.models.base_models import BaseCellModel, BaseDynamicsModel
from dolfinx.fem.petsc import LinearProblem
from dolfinx import fem, mesh, plot
import matplotlib.pyplot as plt
import src.utils as utils
from tqdm import tqdm
import numpy as np
import pyvista
import ufl


class BidomainModel(BaseDynamicsModel):
    """A model that solves bidomain equations to simulate
    electric impulse conduction in heart."""

    # model parameters
    CHI = 2000  # cm^-1
    C_M = 1  # ms*mS/cm^2
    V_REST = -85.0  # mV
    V_PEAK = 40.0  # mV

    # conductivities
    sigma_il = 3.0  # mS/cm
    sigma_it = 1.0  # mS/cm
    sigma_in = 0.31525  # mS/cm
    sigma_el = 2.0  # mS/cm
    sigma_et = 1.65  # mS/cm
    sigma_en = 1.3514  # mS/cm

    SIGMA_IL = sigma_il / C_M / CHI  # cm^2/ms
    SIGMA_IT = sigma_it / C_M / CHI  # cm^2/ms
    SIGMA_IN = sigma_in / C_M / CHI  # cm^2/ms
    SIGMA_EL = sigma_el / C_M / CHI  # cm^2/ms
    SIGMA_ET = sigma_et / C_M / CHI  # cm^2/ms
    SIGMA_EN = sigma_en / C_M / CHI  # cm^2/ms

    def __init__(self):
        pass

    def conductivity(
        self,
        longitudinal_fibres: list[float] | None = None,
        transversal_fibres: list[float] | None = None,
    ):
        self.conductivity.__doc__

        if longitudinal_fibres is not None and transversal_fibres is not None:
            # Muscle sheets
            self.sheet_l = ufl.as_vector(longitudinal_fibres)
            self.sheet_n = ufl.as_vector(transversal_fibres)

            # Healthy conductivities
            self.M_i = (
                self.SIGMA_IT * ufl.Identity(len(longitudinal_fibres))
                + (self.SIGMA_IL - self.SIGMA_IT)
                * ufl.outer(self.sheet_l, self.sheet_l)
                + (self.SIGMA_IN - self.SIGMA_IT)
                * ufl.outer(self.sheet_n, self.sheet_n)
            )
            self.M_e = (
                self.SIGMA_ET * ufl.Identity(len(transversal_fibres))
                + (self.SIGMA_EL - self.SIGMA_ET)
                * ufl.outer(self.sheet_l, self.sheet_l)
                + (self.SIGMA_EN - self.SIGMA_ET)
                * ufl.outer(self.sheet_n, self.sheet_n)
            )

        else:
            self.M_i = self.SIGMA_IT * ufl.Identity(self.d)
            self.M_e = self.SIGMA_ET * ufl.Identity(self.d)

    def setup(
        self,
        domain: mesh.Mesh,
        longitudinal_fibres: list | None = None,
        transversal_fibres: list | None = None,
    ):
        self.setup.__doc__

        # Setting up meshes, function spaces and functions
        self.element = ufl.FiniteElement("P", domain.ufl_cell(), degree=2)
        self.W = fem.FunctionSpace(domain, ufl.MixedElement(self.element, self.element))
        self.V1, self.sub1 = self.W.sub(0).collapse()
        self.V2, self.sub2 = self.W.sub(1).collapse()

        # Define test and trial functions
        self.psi, self.phi = ufl.TestFunctions(self.W)
        self.V_m, self.U_e = ufl.TrialFunctions(self.W)

        # Define fem functions, spatial coordinates and the dimension of a mesh
        self.v_, self.w, self.V_m_n = (
            fem.Function(self.W),
            fem.Function(self.V1),
            fem.Function(self.V1),
        )

        self.x = ufl.SpatialCoordinate(domain)
        self.d = domain.topology.dim

        # Defining initial conditions for transmembrane potential
        cells = fem.locate_dofs_geometrical(self.V1, self.initial_V_m()[0])
        self.V_m_n.x.array[:] = self.V_REST
        self.V_m_n.x.array[cells] = np.full_like(cells, self.initial_V_m()[1])

        # Define conductivity tensors
        self.conductivity()

        # Ishemic conductivities
        if self.ischemia() is not None:
            self.M_i = ufl.conditional(
                self.ischemia()[0](self.x),
                self.ischemia()[1],
                self.M_i,
            )

            self.M_e = ufl.conditional(
                self.ischemia()[0](self.x),
                self.ischemia()[1],
                self.M_e,
            )

    def solve(
        self,
        T: float,
        steps: int,
        cell_model: BaseCellModel,
        signal_point: list[float] | None = None,
        camera: list[float] | None = None,
        gif_name: str = "V_m.gif",
    ):
        self.solve.__doc__

        dt = T / steps

        # Defining a ufl weak form and corresponding linear problem
        F = (
            (self.V_m - self.V_m_n) / dt * self.phi * ufl.dx
            + ufl.inner(
                ufl.dot(self.M_i, ufl.grad(self.V_m / 2 + self.V_m_n / 2 + self.U_e)),
                ufl.grad(self.phi),
            )
            * ufl.dx
            + (
                ufl.inner(
                    ufl.dot(self.M_i + self.M_e, ufl.grad(self.U_e)), ufl.grad(self.psi)
                )
                * ufl.dx
                + ufl.inner(
                    ufl.dot(self.M_i, ufl.grad(self.V_m / 2 + self.V_m_n / 2)),
                    ufl.grad(self.psi),
                )
                * ufl.dx
            )
        )

        problem = LinearProblem(a=ufl.lhs(F), L=ufl.rhs(F), u=self.v_)

        # Making a plotting environment
        grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(self.V1))
        plotter = pyvista.Plotter(notebook=True, off_screen=False)
        plotter.open_gif("gifs/" + gif_name, fps=int(steps / 10))
        grid.point_data["V_m"] = self.V_m_n.x.array
        plotter.add_mesh(
            grid,
            show_edges=False,
            lighting=True,
            smooth_shading=True,
            clim=[-100, 50],
        )

        # List of signal values for each time step
        self.signal_point = signal_point
        self.signal = []
        self.time = np.linspace(0, T, steps)

        # Iterate through time
        for t in tqdm(self.time, desc="Solving problem"):
            # Appending the transmembrane potential value at some point to a list
            if signal_point != None:
                self.signal.append(
                    utils.evaluate_function_at_point(self.V_m_n, signal_point)
                )

            # 1st step of Strang splitting
            self.V_m_n.x.array[:], self.w.x.array[:] = cell_model.step(
                dt / 2, self.V_m_n.x.array, self.w.x.array
            )

            # 2nd step of Strang splitting
            problem.solve()
            # Normalize U_e to zero mean
            self.v_.x.array[self.sub2] = self.v_.x.array[self.sub2] - np.mean(
                self.v_.x.array[self.sub2]
            )
            # Update solution for V_m
            self.V_m_n.x.array[:] = self.v_.x.array[self.sub1]

            # 3rd step of Strang splitting
            self.V_m_n.x.array[:], self.w.x.array[:] = cell_model.step(
                dt / 2, self.V_m_n.x.array, self.w.x.array
            )

            # Update plot
            plotter.clear()
            grid.point_data["V_m"] = self.V_m_n.x.array[:]
            plotter.add_mesh(
                grid,
                show_edges=False,
                lighting=True,
                smooth_shading=True,
                clim=[-100, 50],
            )
            plotter.add_title("t = %.3f" % t, font_size=24)
            if camera != None:
                plotter.view_vector([1, -1, -1])
            plotter.write_frame()

        plotter.close()

    def plot_ischemia(self, *args):
        """A function that plots ischemia parts of the domain.\n
        Plotting parameters can be passed."""

        if self.ischemia() is None:
            raise NotImplementedError("Ischemia function not implemented!")

        fun = fem.Function(self.V1)
        cells = fem.locate_dofs_geometrical(self.V1, self.ischemia()[0])

        fun.x.array[:] = 0
        fun.x.array[cells] = np.full_like(cells, 1)

        utils.plot_function(fun, "ischemia", *args)

    def plot_initial_V_m(self, *args):
        """A function that plots initial transmembrane potential.\n
        Plotting parameters can be passed."""
        utils.plot_function(self.V_m_n, "initial V_m", *args)

    def plot_signal(self, *args):
        """A function that plots transmembrane potential at a point
        previously defined as signal point.\n
        Plotting parameters can be passed."""
        if self.signal_point == None:
            raise ValueError("Signal point must be specified when solving the model.")

        plt.plot(self.time, self.signal)
        plt.xlabel("time [ms]")
        plt.ylabel("signal [mV]")
        plt.title(
            "Time dependence of $V_m$ at " + str(self.signal_point),
        )


class MonodomainModel(BaseDynamicsModel):
    def __init__(self):
        super().__init__()
