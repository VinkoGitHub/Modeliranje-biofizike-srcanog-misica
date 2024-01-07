from src.models.base_models import BaseCellModel, BaseDynamicsModel
from dolfinx.fem.petsc import LinearProblem
from dolfinx import fem, mesh, plot
import src.utils as utils
import numpy as np
import pyvista
import ufl
from tqdm import tqdm


class BidomainModel(BaseDynamicsModel):
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

    def initial_V_m(self, loc: list[float], V_m_0_val: float):
        """A function used to define initial transmembrane potential."""
        return lambda x: (x[0] - loc[0]) ** 2 + (x[1] - loc[1]) ** 2 < V_m_0_val**2

    def solve(
        self,
        T: float,
        NUM_STEPS: int,
        domain: mesh.Mesh,
        cell_model: BaseCellModel,
        V_m_0: list[list[float], float, float],
        ischemia: list[list[float], float, float] | None = None,
        longitudinal_fibres: list = [0, 0, 0],
        transversal_fibres: list = [0, 0, 0],
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

        # Define test functions
        psi, phi = ufl.TestFunctions(W)

        # Define trial functions
        V_m, U_e = ufl.TrialFunctions(W)

        # Define fem functions
        v_ = fem.Function(W)
        w, w_n = fem.Function(V1), fem.Function(V1)
        V_m_n = fem.Function(V1)

        x = ufl.SpatialCoordinate(domain)
        d = domain.topology.dim

        cells = fem.locate_dofs_geometrical(V1, self.initial_V_m(V_m_0[0], V_m_0[1]))
        V_m_n.x.array[:] = cell_model.V_REST
        V_m_n.x.array[cells] = np.full_like(cells, V_m_0[2])

        # Muscle sheets
        sheet_l = ufl.as_vector(longitudinal_fibres)
        sheet_n = ufl.as_vector(transversal_fibres)

        # Healthy conductivities
        M_i = (
            self.SIGMA_IT * ufl.Identity(d)
            + (self.SIGMA_IL - self.SIGMA_IT) * ufl.outer(sheet_l, sheet_l)
            + (self.SIGMA_IN - self.SIGMA_IT) * ufl.outer(sheet_n, sheet_n)
        )
        M_e = (
            self.SIGMA_ET * ufl.Identity(d)
            + (self.SIGMA_EL - self.SIGMA_ET) * ufl.outer(sheet_l, sheet_l)
            + (self.SIGMA_EN - self.SIGMA_ET) * ufl.outer(sheet_n, sheet_n)
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

        DT = T / NUM_STEPS

        F = (V_m - V_m_n) / DT * phi * ufl.dx + ufl.inner(
            ufl.dot(M_i, ufl.grad(V_m / 2 + V_m_n / 2 + U_e)), ufl.grad(phi)
        ) * ufl.dx
        F += (
            ufl.inner(ufl.dot(M_i + M_e, ufl.grad(U_e)), ufl.grad(psi)) * ufl.dx
            + ufl.inner(ufl.dot(M_i, ufl.grad(V_m / 2 + V_m_n / 2)), ufl.grad(psi))
            * ufl.dx
        )
        problem = LinearProblem(a=ufl.lhs(F), L=ufl.rhs(F), u=v_)

        grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V1))
        plotter = pyvista.Plotter(notebook=True, off_screen=False)
        plotter.open_gif("gifs/" + gif_name, fps=int(NUM_STEPS / 10))
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
        time = np.linspace(0, T, NUM_STEPS)

        # Iterate through time
        for t in tqdm(time, desc="Solving problem"):
            # Appending the transmembrane potential value at some point to a list
            if signal_point != None:
                signal.append(
                    utils.evaluate_function_at_point(V_m_n, domain, [1.5, 1.5, 0.0])
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
