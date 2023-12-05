from fenics import *
from configs import *

def mesh_maker():
    '''Returns a mesh, a Meshfunction regarding 
    the domains and the MeshFunction regarding the boundary'''
    
    # Create classes for defining parts of the boundaries and the interior
    # of the domain
    class TorsoOuterBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0.0) or near(x[0], Tx) or near(x[1], 0.0) or near(x[1], Ty)

    class HeartDomain(SubDomain):
        def inside(self, x, on_boundary):
            return between(x[1], (Hy-Y/2, Hy+Y/2)) and between(x[0], (Hx-X/2, Hx+X/2))  

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