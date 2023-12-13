from dolfinx.mesh import create_unit_square
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from ufl import derivative
from configs import *
from mpi4py import MPI
import numpy as np
import pyvista
from dolfinx.plot import vtk_mesh


# Managing a mesh
class MeshHandler:
    def __init__(
        self,
    ):
        pass

    def create(self, N=N):
        return create_unit_square(MPI.COMM_WORLD, N, N)

    def plot_mesh(self, mesh):
        """
        Given a DOLFINx mesh, create a `pyvista.UnstructuredGrid`,
        and plot it and the mesh nodes
        """
        plotter = pyvista.Plotter()
        ugrid = pyvista.UnstructuredGrid(*vtk_mesh(mesh))
        if mesh.geometry.cmaps[0].degree > 1:
            plotter.add_mesh(ugrid, style="points", color="b", point_size=10)
            ugrid = ugrid.tessellate()
            show_edges = False
        else:
            show_edges = True
        plotter.add_mesh(ugrid, show_edges=show_edges)

        plotter.show_axes()
        plotter.view_xy()
        plotter.show()

    def plot_subdomains(self, domain, cells_H, cells_T):
        # Filter out ghosted cells
        num_cells_local = domain.topology.index_map(domain.topology.dim).size_local
        marker = np.zeros(num_cells_local, dtype=np.int32)
        cells_H = cells_H[cells_H < num_cells_local]
        cells_T = cells_T[cells_T < num_cells_local]
        marker[cells_H] = 1
        marker[cells_T] = 0
        topology, cell_types, x = vtk_mesh(
            domain, domain.topology.dim, np.arange(num_cells_local, dtype=np.int32)
        )

        p = pyvista.Plotter(window_size=[500, 500])
        grid = pyvista.UnstructuredGrid(topology, cell_types, x)
        grid.cell_data["Marker"] = marker
        grid.set_active_scalars("Marker")
        p.add_mesh(grid, show_edges=True)
        p.show()

    def plot_function(self, domain, function):
        # Filter out ghosted cells
        num_cells_local = domain.topology.index_map(domain.topology.dim).size_local
        topology, cell_types, x = vtk_mesh(
            domain, domain.topology.dim, np.arange(num_cells_local, dtype=np.int32)
        )

        p = pyvista.Plotter(window_size=[500, 500])
        grid = pyvista.UnstructuredGrid(topology, cell_types, x)
        grid.cell_data['function'] = function.x.array
        grid.set_active_scalars('function')
        p.add_mesh(grid, show_edges=True)
        p.show()


def Omega_H(x):
    return np.array((x[0] - Hx) ** 2 + (x[1] - Hy) ** 2 <= R**2)


def Omega_T(x):
    return np.array((x[0] - Hx) ** 2 + (x[1] - Hy) ** 2 >= R**2)


# Ionic current
class eval_I_ion:
    def __init__(self, v, w):
        self.V_m = v.sub(0).collapse().x.array
        self.w = w.x.array

    def __call__(self, x):
        value = np.zeros(self.w.shape[0])
        value[:] = -self.w[:] / TAU_IN * (self.V_m[:] - V_MIN) ** 2 * (
            V_MAX - self.V_m[:]
        ) / (V_MAX - V_MIN) + 1 / TAU_OUT * (self.V_m[:] - V_MIN) / (V_MAX - V_MIN)
        return value


# Gating variable
class eval_g:
    def __init__(self, v, w):
        self.V_m = v.sub(0).collapse().x.array
        self.w = w.x.array

    def __call__(self, x):
        value = np.zeros(self.w.shape[0])
        for i, _ in enumerate(value):
            if self.V_m[i] < V_GATE:
                value[i] = self.w[i] / TAU_OPEN - 1 / TAU_OPEN / (V_MAX - V_MIN) ** 2
            else:
                value[i] = self.w[i] / TAU_CLOSE
        return value


# Fiber vector field
class eval_fibres:
    def __init__(
        self,
    ):
        pass

    def __call__(self, x):
        values = np.zeros((2, x.shape[1]))
        values[0] = x[1] / np.sqrt(x[0] ** 2 + x[1] ** 2 + 10**-10)
        values[1] = -x[0] / np.sqrt(x[0] ** 2 + x[1] ** 2 + 10**-10)
        return values


# Applied stimulus
class eval_I_app:
    def __init__(
        self,
        Hx=Hx,
        Hy=Hy,
        t_act=T_ACT,
        t=0,
        size=R / 2,
    ):
        self.Hx = Hx
        self.Hy = Hy
        self.t_act = t_act
        self.t = t
        self.size = size

    def __call__(self, x):
        return np.exp(
            -(((x[0] - self.Hx) / self.size) ** 2)
            - ((x[1] - self.Hy) / self.size) ** 2
            - (self.t / self.t_act) ** 2
        )  # nije kao u radu


# solver and its parameters
def custom_solver(
    F, v, atol=1e-2, rtol=1e-2, max_iter=10, relax_par=1.0, conv_crit="incremental"
):
    # Jacobian
    J = derivative(F, v)

    # Problem solver
    problem = NonlinearProblem(F, v, J=J)
    solver = NewtonSolver(MPI.COMM_WORLD, problem)

    # Problem solver parameters
    solver.convergence_criterion = conv_crit
    solver.rtol = rtol
    solver.atol = atol
    solver.max_iter = max_iter
    solver.report = True

    return solver
