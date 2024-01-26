from dolfinx.fem.petsc import LinearProblem
from src.models.base_models import *
import matplotlib.pyplot as plt
from dolfinx import fem, plot, mesh
import src.utils as utils
from tqdm import tqdm
import numpy as np
import pyvista
import ufl

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


class BidomainModel(Common, BaseDynamicsModel):
    """A model that solves bidomain equations to simulate
    electric impulse conduction in heart."""

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
        self.SIGMA_IL = sigma_il / cell_model.C_m / chi  # cm^2/ms
        self.SIGMA_IT = sigma_it / cell_model.C_m / chi  # cm^2/ms
        self.SIGMA_IN = sigma_in / cell_model.C_m / chi  # cm^2/ms
        self.SIGMA_EL = sigma_el / cell_model.C_m / chi  # cm^2/ms
        self.SIGMA_ET = sigma_et / cell_model.C_m / chi  # cm^2/ms
        self.SIGMA_EN = sigma_en / cell_model.C_m / chi  # cm^2/ms

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
        camera_direction: str | None = None,
        cmap: str = "plasma",
        save_to: str = "V_m.gif",
        checkpoints: list[int] = [],
        checkpoint_file: str = "checkpoint",
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

        if steps <= 600:
            fps = int(steps / 10)
            sparser = 1
        else:
            fps = 60
            sparser = int(steps / 900)

        if save_to[-4:] == ".gif":
            plotter.open_gif("animations/" + save_to, loop=steps, fps=fps)
        elif save_to[-4:] == ".mp4":
            plotter.open_movie("animations/" + save_to, framerate=fps)

        # List of signal values for each time step
        self.signal_point = signal_point
        self.signal = []
        self.time = np.linspace(0, T, steps)

        iteration_number = 0
        # Iterate through time
        for t in tqdm(self.time, desc="Solving problem"):
            # Update plot
            if iteration_number % sparser == 0:
                grid.point_data["V_m"] = self.V_m_n.x.array[:]
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
                    show_edges=False,
                    lighting=True,
                    smooth_shading=True,
                    clim=[-100, 50],
                    cmap=cmap,
                    scalar_bar_args=sargs,
                )
                plotter.add_title("t = %.3f" % t, font_size=24)
                plotter.enable_parallel_projection()
                if camera_direction != None:
                    plotter.camera_position = camera_direction
                plotter.write_frame()
                # plotter.clear()
                for cp in checkpoints:
                    if t < cp + 0.5 and t > cp - 0.5:
                        plotter.save_graphic(checkpoint_file + "_" + str(cp) + ".pdf")
                        break

            # Appending the transmembrane potential value at some point to a list
            if signal_point != None:
                self.signal.append(
                    utils.evaluate_function_at_point(self.V_m_n, signal_point)
                )

            # 1st step of Strang splitting
            self.V_m_n.x.array[:] = self.cell_model.step_V_m(
                dt / 2, t, self.V_m_n.x.array
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
            self.V_m_n.x.array[:] = self.cell_model.step_V_m(
                dt / 2, t, self.V_m_n.x.array
            )

            iteration_number += 1

        plotter.close()

    def conductivity(self):
        self.conductivity.__doc__
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
        function_name: str = "initial_V_m",
        zoom: float = 1.0,
        shadow: bool = False,
        show_mesh: bool = True,
        cmap: str = "plasma",
        save_to: str | None = None,
    ):
        """A function that plots initial transmembrane potential.\n
        Plotting parameters can be passed."""
        utils.plot_function(
            self.V_m_n,
            function_name,
            camera_direction,
            zoom,
            shadow,
            show_mesh,
            cmap,
            save_to,
        )

    def plot_signal(self, save_to: str | None = None):
        """A function that plots transmembrane potential at a point
        previously defined as signal point.\n
        Plotting parameters can be passed."""
        if self.signal_point == None:
            raise ValueError("Signal point must be specified when solving the model.")

        plt.plot(self.time, self.signal)
        plt.xlabel("$t$ [ms]")
        plt.ylabel("$V_m$ [mV]")
        if save_to is not None:
            plt.savefig(save_to)


class MonodomainModel(Common, BaseDynamicsModel):
    """A model that solves monodomain equations to simulate
    electric impulse conduction in heart."""

    def __init__(self, domain: mesh.Mesh, cell_model: BaseCellModel):
        super().__init__(domain)
        self.cell_model = cell_model

        # Define test and trial functions
        self.phi = ufl.TestFunction(self.V1)
        self.V_m = ufl.TrialFunction(self.V1)

        # Define fem functions, spatial coordinates and the dimension of a mesh
        self.V_m_, self.V_m_n = (
            fem.Function(self.V1),
            fem.Function(self.V1),
        )

        # Defining initial conditions for transmembrane potential
        self.initial_V_m()

        # Define conductivity values with respect to the cell model
        self.SIGMA_IL = sigma_il / cell_model.C_m / chi  # cm^2/ms
        self.SIGMA_IT = sigma_it / cell_model.C_m / chi  # cm^2/ms
        self.SIGMA_IN = sigma_in / cell_model.C_m / chi  # cm^2/ms

        # Define conductivity tensors
        self.conductivity()

        # Ishemic conductivities
        if self.ischemia() is not None:
            self.M_i = ufl.conditional(
                self.ischemia()[0](self.x),
                self.ischemia()[1],
                self.M_i,
            )

    def solve(
        self,
        T: float,
        steps: int,
        lambda_: float,
        signal_point: list[float] | None = None,
        camera: list[float] | None = None,
        save_to: str = "V_m.gif",
    ):
        self.solve.__doc__

        dt = T / steps
        self.M = self.M_i * lambda_ / (1 + lambda_)

        # Defining a ufl weak form and corresponding linear problem
        F = (self.V_m - self.V_m_n) / dt * self.phi * ufl.dx + ufl.inner(
            ufl.dot(self.M, ufl.grad(self.V_m / 2 + self.V_m_n / 2)), ufl.grad(self.phi)
        ) * ufl.dx

        problem = LinearProblem(a=ufl.lhs(F), L=ufl.rhs(F), u=self.V_m_)

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

        # List of signal values for each time step
        self.signal_point = signal_point
        self.signal = []
        self.time = np.linspace(0, T, steps)

        iteration_number = 0
        # Iterate through time
        for t in tqdm(self.time, desc="Solving problem"):
            # Update plot
            if iteration_number % sparser == 0:
                grid.point_data["V_m"] = self.V_m_n.x.array[:]
                plotter.add_mesh(
                    grid,
                    show_edges=False,
                    lighting=True,
                    smooth_shading=True,
                    clim=[-100, 50],
                    cmap="plasma",
                )
                plotter.add_title("t = %.3f" % t, font_size=24)
                if camera != None:
                    plotter.view_vector([1, -1, -1])
                plotter.write_frame()
                # plotter.clear()

            # Appending the transmembrane potential value at some point to a list
            if signal_point != None:
                self.signal.append(
                    utils.evaluate_function_at_point(self.V_m_n, signal_point)
                )

            # 1st step of Strang splitting
            self.V_m_n.x.array[:] = self.cell_model.step_V_m(
                dt / 2, t, self.V_m_n.x.array
            )

            # 2nd step of Strang splitting
            problem.solve()
            # Update solution for V_m
            self.V_m_n.x.array[:] = self.V_m_.x.array[:]

            # 3rd step of Strang splitting
            self.V_m_n.x.array[:] = self.cell_model.step_V_m(
                dt / 2, t, self.V_m_n.x.array
            )

            iteration_number += 1

        plotter.close()

    def conductivity(self):
        self.conductivity.__doc__
        self.M_i = self.SIGMA_IL * ufl.Identity(self.d)

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
