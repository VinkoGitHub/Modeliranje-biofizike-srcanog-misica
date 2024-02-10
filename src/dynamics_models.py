from dolfinx.fem.petsc import LinearProblem
from dolfinx import fem, plot, mesh
from src.base_models import *
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


class BidomainModel(BaseDynamicsModel):
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

        # Apply ischemic conductivities
        self.ischemia()

    def solve(
        self,
        T: float,
        steps: int,
        signal_point: list[float] | None = None,
        camera_direction: str | None = None,
        zoom: float = 1.0,
        cmap: str = "jet",
        save_to: str = "V_m.mp4",
        checkpoints: list[float] = [],
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
            sparser = round(steps / 900)

        if save_to[-4:] == ".gif":
            plotter.open_gif(f"animations/{save_to}", loop=steps, fps=fps)
        elif save_to[-4:] == ".mp4":
            plotter.open_movie(f"animations/{save_to}", framerate=fps)

        # Writing a first frame
        grid.point_data["V_m"] = self.V_m_n.x.array
        sargs = dict(
            title="",
            height=0.5,
            vertical=True,
            position_x=0.85,
            position_y=0.25,
            font_family="times",
            label_font_size=40,
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
        plotter.add_title("t = 0.000", font_size=24, font="times")

        if type(camera_direction) == list:
            plotter.view_vector(camera_direction)
        elif type(camera_direction) == str:
            plotter.camera_position = camera_direction

        plotter.camera.zoom(zoom)
        plotter.write_frame()

        # List of signal values for each time step
        self.signal_point = signal_point
        self.signal = []
        self.time = np.linspace(0, T, steps)

        iteration_number = 0
        # Iterate through time
        for t in tqdm(self.time, desc="Solving problem"):
            # Update plot
            if iteration_number % sparser == 0:
                grid.point_data["V_m"] = self.V_m_n.x.array
                plotter.add_mesh(
                    grid,
                    show_edges=False,
                    lighting=True,
                    smooth_shading=True,
                    clim=[-100, 50],
                    cmap=cmap,
                    scalar_bar_args=sargs,
                )
                plotter.add_title("t = %.3f" % t, font_size=24, font="times")
                plotter.write_frame()

            for cp in checkpoints:
                if t <= cp + dt and t >= cp - dt:
                    plotter.add_title("")
                    plotter.show_bounds(
                        font_family="times",
                        xtitle="",
                        ytitle="",
                        ztitle="",
                        grid=False,
                        ticks="both",
                        minor_ticks=True,
                        location="outer",
                        font_size=25,
                        use_2d=self.d == 2,
                    )
                    plotter.save_graphic(f"figures/{checkpoint_file}_{cp}.pdf")
                    plotter.remove_bounds_axes()
                    break

            # Appending the transmembrane potential value at some point to a list
            if signal_point is not None:
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


class MonodomainModel(BaseDynamicsModel):
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

        # Apply ischemic conductivities
        self.ischemia()

    def solve(
        self,
        T: float,
        steps: int,
        lambda_: float,
        camera_direction: str | None = None,
        signal_point: list[float] | None = None,
        zoom: float = 1.0,
        cmap: str = "jet",
        save_to: str = "V_m.mp4",
        checkpoints: list[float] = [],
        checkpoint_file: str = "checkpoint",
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

        if steps <= 600:
            fps = int(steps / 10)
            sparser = 1
        else:
            fps = 60
            sparser = round(steps / 900)

        if save_to[-4:] == ".gif":
            plotter.open_gif(f"animations/{save_to}", loop=steps, fps=fps)
        elif save_to[-4:] == ".mp4":
            plotter.open_movie(f"animations/{save_to}", framerate=fps)

        # Writing a first frame
        grid.point_data["V_m"] = self.V_m_n.x.array
        sargs = dict(
            title="",
            height=0.5,
            vertical=True,
            position_x=0.85,
            position_y=0.25,
            font_family="times",
            label_font_size=40,
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
        plotter.add_title("t = 0.000", font_size=24, font="times")

        if type(camera_direction) == list:
            plotter.view_vector(camera_direction)
        elif type(camera_direction) == str:
            plotter.camera_position = camera_direction

        plotter.camera.zoom(zoom)
        plotter.write_frame()

        # List of signal values for each time step
        self.signal_point = signal_point
        self.signal = []
        self.time = np.linspace(0, T, steps)

        iteration_number = 0
        # Iterate through time
        for t in tqdm(self.time, desc="Solving problem"):
            # Update plot
            if iteration_number % sparser == 0:
                grid.point_data["V_m"] = self.V_m_n.x.array
                plotter.add_mesh(
                    grid,
                    show_edges=False,
                    lighting=True,
                    smooth_shading=True,
                    clim=[-100, 50],
                    cmap=cmap,
                    scalar_bar_args=sargs,
                )
                plotter.add_title("t = %.3f" % t, font_size=24, font="times")
                plotter.write_frame()

            for cp in checkpoints:
                if t <= cp + dt and t >= cp - dt:
                    plotter.add_title("")
                    plotter.show_bounds(
                        font_family="times",
                        xtitle="",
                        ytitle="",
                        ztitle="",
                        grid=False,
                        ticks="both",
                        minor_ticks=True,
                        location="outer",
                        font_size=25,
                        use_2d=self.d == 2,
                    )
                    plotter.save_graphic(f"figures/{checkpoint_file}_{cp}.pdf")
                    plotter.remove_bounds_axes()
                    break

            # Appending the transmembrane potential value at some point to a list
            if signal_point is not None:
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
