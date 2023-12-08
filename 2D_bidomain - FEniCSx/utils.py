from dolfinx import *
from configs import *


def mesh_maker():
    """Returns a mesh, a Meshfunction regarding
    the domains and the MeshFunction regarding the boundary"""

    # Create classes for defining parts of the boundaries and the interior
    # of the domain
    class TorsoOuterBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return (
                near(x[0], 0.0) or near(x[0], Tx) or near(x[1], 0.0) or near(x[1], Ty)
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

    return mesh, cellfunction, facetfunction


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
