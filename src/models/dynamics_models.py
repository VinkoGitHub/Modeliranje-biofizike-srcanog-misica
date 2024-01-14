from dolfinx.fem.petsc import LinearProblem
from src.models.base_models import *
import matplotlib.pyplot as plt
from dolfinx import fem, plot, mesh
import src.utils as utils
from tqdm import tqdm
import numpy as np
import pyvista
import ufl


class BidomainModel(Common, BaseDynamicsModel):
    """A model that solves bidomain equations to simulate
    electric impulse conduction in heart."""

    # model parameters
    chi = 2000  # cm^-1
    V_REST = -85.0  # mV
    V_PEAK = 40.0  # mV

    # conductivities
    sigma_il = 3.0  # mS/cm
    sigma_it = 1.0  # mS/cm
    sigma_in = 0.31525  # mS/cm
    sigma_el = 2.0  # mS/cm
    sigma_et = 1.65  # mS/cm
    sigma_en = 1.3514  # mS/cm

    def __init__(self, domain: mesh.Mesh, cell_model: BaseCellModel):
        super().__init__(domain)
        self.cell_model = cell_model

        # Define test and trial functions
        self.psi, self.phi = ufl.TestFunctions(self.W)
        self.V_m, self.U_e = ufl.TrialFunctions(self.W)

        # Define fem functions, spatial coordinates and the dimension of a mesh
        self.v_, self.V_m_n = (
            fem.Function(self.W),
            fem.Function(self.V1),
        )

        # Defining initial conditions for transmembrane potential
        self.initial_V_m()

        # Define conductivity values with respect to the cell model
        self.SIGMA_IL = self.sigma_il / cell_model.C_m / self.chi  # cm^2/ms
        self.SIGMA_IT = self.sigma_it / cell_model.C_m / self.chi  # cm^2/ms
        self.SIGMA_IN = self.sigma_in / cell_model.C_m / self.chi  # cm^2/ms
        self.SIGMA_EL = self.sigma_el / cell_model.C_m / self.chi  # cm^2/ms
        self.SIGMA_ET = self.sigma_et / cell_model.C_m / self.chi  # cm^2/ms
        self.SIGMA_EN = self.sigma_en / cell_model.C_m / self.chi  # cm^2/ms

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
        signal_point: list[float] | None = None,
        camera: list[float] | None = None,
        save_to: str = "V_m.gif",
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

        if steps < 600:
            fps = int(steps / 10)
            sparser = 1
        else:
            fps = 60
            sparser = int(steps / 900)

        if save_to[-4:] == ".gif":
            plotter.open_gif("animations/" + save_to, loop=steps, fps=fps)
        elif save_to[-4:] == ".mp4":
            plotter.open_movie("animations/" + save_to, framerate=fps)

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

        iteration_number = 0
        # Iterate through time
        for t in tqdm(self.time, desc="Solving problem"):
            # Appending the transmembrane potential value at some point to a list
            if signal_point != None:
                self.signal.append(
                    utils.evaluate_function_at_point(self.V_m_n, signal_point)
                )

            # 1st step of Strang splitting
            self.V_m_n.x.array[:] = self.cell_model.step_V_m(dt / 2, self.V_m_n.x.array)

            # 2nd step of Strang splitting
            problem.solve()
            # Normalize U_e to zero mean
            self.v_.x.array[self.sub2] = self.v_.x.array[self.sub2] - np.mean(
                self.v_.x.array[self.sub2]
            )
            # Update solution for V_m
            self.V_m_n.x.array[:] = self.v_.x.array[self.sub1]

            # 3rd step of Strang splitting
            self.V_m_n.x.array[:] = self.cell_model.step_V_m(dt / 2, self.V_m_n.x.array)

            if iteration_number % sparser == 0:
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

            iteration_number += 1

        plotter.close()

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

    def plot_ischemia(
        self,
        camera_direction: list[float] = [1, 1, 1],
        zoom: float = 1.0,
        shadow: bool = False,
        show_mesh: bool = True,
        save_to: str | None = None,
    ):
        """A function that plots ischemia parts of the domain.\n
        Plotting parameters can be passed."""

        if self.ischemia() is None:
            raise NotImplementedError("Ischemia function not implemented!")

        fun = fem.Function(self.V1)
        cells = fem.locate_dofs_geometrical(self.V1, self.ischemia()[0])

        fun.x.array[:] = 0
        fun.x.array[cells] = np.full_like(cells, 1)

        utils.plot_function(
            fun, "ischemia", camera_direction, zoom, shadow, show_mesh, save_to
        )

    def plot_initial_V_m(
        self,
        camera_direction: list[float] = [1, 1, 1],
        zoom: float = 1.0,
        shadow: bool = False,
        show_mesh: bool = True,
        save_to: str | None = None,
    ):
        """A function that plots initial transmembrane potential.\n
        Plotting parameters can be passed."""
        utils.plot_function(
            self.V_m_n,
            "initial V_m",
            camera_direction,
            zoom,
            shadow,
            show_mesh,
            save_to,
        )

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
