from dolfinx import *
from configs import *


class MeshHandler:
    def __init__(self):
        pass

    def create_mesh(self, data=None):
        """Returns a mesh, a Meshfunction regarding
        the domains and the MeshFunction regarding the boundary"""

        # Create classes for defining parts of the boundaries and the interior
        # of the domain
        class TorsoOuterBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return (
                    near(x[0], 0.0)
                    or near(x[0], Tx)
                    or near(x[1], 0.0)
                    or near(x[1], Ty)
                )

        class HeartDomain(SubDomain):
            def inside(self, x, on_boundary):
                return between(x[1], (Hy - Y / 2, Hy + Y / 2)) and between(
                    x[0], (Hx - X / 2, Hx + X / 2)
                )

        # Initialize sub-domain instances
        torso_outer_boundary = TorsoOuterBoundary()
        heart_domain = HeartDomain()

        # Define mesh
        mesh = UnitSquareMesh(N, N)

        # Initialize mesh function for interior domains
        cellfunction = MeshFunction("size_t", mesh, 2)
        cellfunction.set_all(0)
        heart_domain.mark(cellfunction, 1)

        # Initialize mesh function for boundary domains
        facetfunction = MeshFunction("size_t", mesh, 1)
        facetfunction.set_all(0)
        torso_outer_boundary.mark(facetfunction, 1)

        self.mesh = mesh
        self.cellfunction = cellfunction
        self.facetfunction = facetfunction

        print("Mesh, cell function and facet function created!")


class FunctionSpaceHandler(MeshHandler):
    def __init__(self):
        super().__init__()
        MeshHandler.create_mesh(self)

    def create_spaces(self, element="Lagrange", deg=2):
        self.V1 = VectorFunctionSpace(self.mesh, element, deg, 1)
        self.V2 = VectorFunctionSpace(self.mesh, element, deg, 2)
        self.V3 = VectorFunctionSpace(self.mesh, element, deg, 3)


# solver and its parameters
def custom_solver(F, v, tol_abs=1e-2, tol_rel=1e-2, max_iter=10, relax_par=1.0):
    # Jacobian
    J = derivative(F, v)

    # Problem solver
    problem = NonlinearVariationalProblem(F, v, J=J)
    solver = NonlinearVariationalSolver(problem)

    # Problem solver parameters
    solver.parameters["newton_solver"]["absolute_tolerance"] = tol_abs
    solver.parameters["newton_solver"]["relative_tolerance"] = tol_rel
    solver.parameters["newton_solver"]["maximum_iterations"] = max_iter
    solver.parameters["newton_solver"]["relaxation_parameter"] = relax_par

    return solver


# upitno treba li koristiti ovo dalje


def step_1():
    pass


class g(Expression):
    def __init__(
        self,
        V_m,
        w,
        tau_open=TAU_OPEN,
        tau_close=TAU_CLOSE,
        V_gate=V_GATE,
        V_max=V_MAX,
        V_min=V_MIN,
    ):
        self.V_m = V_m
        self.w = w
        self.tau_open = tau_open
        self.tau_close = tau_close
        self.V_gate = V_gate
        self.V_max = V_max
        self.V_min = V_min

    def eval(self, values):
        if self.V_m < self.V_gate:
            values[0] = (
                self.w / self.tau_open
                - 1 / self.tau_open / (self.V_max - self.V_min) ** 2
            )
        else:
            values[0] = self.w / self.tau_close

    def value_shape(self):
        return (1,)
