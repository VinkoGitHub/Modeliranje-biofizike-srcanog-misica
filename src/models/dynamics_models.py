from src.models.base_models import BaseCellModel, BaseDynamicsModel
from dolfinx.fem.petsc import LinearProblem
from dolfinx import fem, mesh, plot
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

    def initial_V_m(self):
        """A function used to define initial transmembrane potential.
        This function should return a tuple containing another function that
        takes ``x`` as an input and outputs a mathematical condition
        for an area in which initial ``V_m`` is different from the rest value
        and the initial value for V_m in that area.

        Example:

        >>> def initial_V_m():
        >>>     return (lambda x: x[0] < 0.5, 0.0)
        """
        raise NotImplementedError(
            "Method initial_V_m must be implemented for the model to work."
        )

    def ischemia(slf):
        """A function used to define ischemia on domain.
        This function should return a tuple containing a ``ufl`` form that
        outputs a mathematical condition for an area in which conductivity is different
        from the rest value (ischemia domain) and the value of M_i and M_e in that domain.

        Example:

        >>> def ischemia():
        >>>     return (x[0] < 0.5, 0.1, 0.05)
        """
        return None

    def solve(
        self,
        T: float,
        steps: int,
        domain: mesh.Mesh,
        cell_model: BaseCellModel,
        longitudinal_fibres: list | None = None,
        transversal_fibres: list | None = None,
        signal_point: list[float] | None = None,
        camera: list[float] | None = None,
        gif_name: str = "V_m.gif",
    ):
        self.solve.__doc__

        # Setting up meshes, function spaces and functions
        element = ufl.FiniteElement("P", domain.ufl_cell(), degree=2)
        W = fem.FunctionSpace(domain, ufl.MixedElement(element, element))
        V1, sub1 = W.sub(0).collapse()
        V2, sub2 = W.sub(1).collapse()

        # Define test and trial functions
        psi, phi = ufl.TestFunctions(W)
        V_m, U_e = ufl.TrialFunctions(W)

        # Define fem functions, spatial coordinates and the dimension of a mesh
        v_, w, V_m_n = fem.Function(W), fem.Function(V1), fem.Function(V1)
        x, self.d = ufl.SpatialCoordinate(domain), domain.topology.dim

        # Defining initial conditions for transmembrane potential
        cells = fem.locate_dofs_geometrical(V1, self.initial_V_m()[0])
        V_m_n.x.array[:] = cell_model.V_REST
        V_m_n.x.array[cells] = np.full_like(cells, self.initial_V_m()[1])

        if longitudinal_fibres is not None and transversal_fibres is not None:
            # Muscle sheets
            sheet_l = ufl.as_vector(longitudinal_fibres)
            sheet_n = ufl.as_vector(transversal_fibres)

            # Healthy conductivities
            self.M_i = (
                self.SIGMA_IT * ufl.Identity(self.d)
                + (self.SIGMA_IL - self.SIGMA_IT) * ufl.outer(sheet_l, sheet_l)
                + (self.SIGMA_IN - self.SIGMA_IT) * ufl.outer(sheet_n, sheet_n)
            )
            self.M_e = (
                self.SIGMA_ET * ufl.Identity(self.d)
                + (self.SIGMA_EL - self.SIGMA_ET) * ufl.outer(sheet_l, sheet_l)
                + (self.SIGMA_EN - self.SIGMA_ET) * ufl.outer(sheet_n, sheet_n)
            )

        else:
            self.M_i = self.SIGMA_IT * ufl.Identity(self.d)
            self.M_e = self.SIGMA_ET * ufl.Identity(self.d)

        # Ishemic conductivities
        if self.ischemia() is not None:
            self.M_i = ufl.conditional(
                self.ischemia()[0],
                self.ischemia()[1],
                self.M_i,
            )

            self.M_e = ufl.conditional(
                self.ischemia()[0],
                self.ischemia()[1],
                self.M_e,
            )

        DT = T / steps

        # Defining a ufl weak form and corresponding linear problem
        F = (
            (V_m - V_m_n) / DT * phi * ufl.dx
            + ufl.inner(
                ufl.dot(self.M_i, ufl.grad(V_m / 2 + V_m_n / 2 + U_e)), ufl.grad(phi)
            )
            * ufl.dx
            + (
                ufl.inner(ufl.dot(self.M_i + self.M_e, ufl.grad(U_e)), ufl.grad(psi))
                * ufl.dx
                + ufl.inner(
                    ufl.dot(self.M_i, ufl.grad(V_m / 2 + V_m_n / 2)), ufl.grad(psi)
                )
                * ufl.dx
            )
        )

        problem = LinearProblem(a=ufl.lhs(F), L=ufl.rhs(F), u=v_)

        # Making a plotting environment
        grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V1))
        plotter = pyvista.Plotter(notebook=True, off_screen=False)
        plotter.open_gif("gifs/" + gif_name, fps=int(steps / 10))
        grid.point_data["V_m"] = V_m_n.x.array
        plotter.add_mesh(
            grid,
            show_edges=False,
            lighting=True,
            smooth_shading=True,
            clim=[-100, 50],
        )

        # List of signal values for each time step
        signal = []
        time = np.linspace(0, T, steps)

        # Iterate through time
        for t in tqdm(time, desc="Solving problem"):
            # Appending the transmembrane potential value at some point to a list
            if signal_point != None:
                signal.append(
                    utils.evaluate_function_at_point(V_m_n, domain, signal_point)
                )

            # 1st step of Strang splitting
            k1_V = cell_model.I_ion(V_m_n.x.array[:], w.x.array[:])
            k2_V = cell_model.I_ion(V_m_n.x.array[:] + DT / 2 * k1_V, w.x.array[:])
            V_m_n.x.array[:] = V_m_n.x.array[:] + DT / 4 * (k1_V + k2_V)

            k1_w = cell_model.f(V_m_n.x.array[:], w.x.array[:])
            k2_w = cell_model.f(V_m_n.x.array[:], w.x.array[:] + DT / 2 * k1_w)
            w.x.array[:] = w.x.array[:] + DT / 4 * (k1_w + k2_w)

            # 2nd step of Strang splitting
            problem.solve()
            v_.x.array[sub2] = v_.x.array[sub2] - np.mean(
                v_.x.array[sub2]
            )  # Normalize U_e to zero mean

            # Update solution for V_m
            V_m_n.x.array[:] = v_.x.array[sub1]

            # 3rd step of Strang splitting
            k1_V = cell_model.I_ion(V_m_n.x.array[:], w.x.array[:])
            k2_V = cell_model.I_ion(V_m_n.x.array[:] + DT * k1_V, w.x.array[:])
            V_m_n.x.array[:] = V_m_n.x.array[:] + DT / 2 * (k1_V + k2_V)

            k1_w = cell_model.f(V_m_n.x.array[:], w.x.array[:])
            k2_w = cell_model.f(V_m_n.x.array[:], w.x.array[:] + DT * k1_w)
            w.x.array[:] = w.x.array[:] + DT / 2 * (k1_w + k2_w)

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
            if camera != None:
                plotter.view_vector([1, -1, -1])
            plotter.write_frame()

        plotter.close()

        return V_m_n, signal


class MonodomainModel(BaseDynamicsModel):
    def __init__(self):
        super().__init__()
