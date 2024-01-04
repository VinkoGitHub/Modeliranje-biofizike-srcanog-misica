from dolfinx import fem
import utils
import ufl
from configs import *
import numpy as np

# Create a mesh, elements and spaces
domain = utils.create_mesh(N=10)
vector_element = ufl.VectorElement("Lagrange", domain.ufl_cell(), 2, 2)
element = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)

W = fem.FunctionSpace(domain, vector_element)
V = fem.FunctionSpace(domain, element)

# Define test functions
phi, psi = ufl.TestFunctions(W)
theta = ufl.TestFunction(V)

# Define functions
v, v_n, v_nn = fem.Function(W), fem.Function(W), fem.Function(W)
w, w_n, w_nn = fem.Function(V), fem.Function(V), fem.Function(V)

V_m, u = v.split()
V_m_n, u_n = v_n.split()
V_m_nn, u_nn = v_nn.split()

x = ufl.SpatialCoordinate(domain)
d = domain.topology.dim
t = 0

# Define meshtags
num_cells = domain.topology.index_map(domain.topology.dim).size_local
body_cells = np.arange(num_cells)
markers = np.zeros(num_cells)

heart_cells = mesh.locate_entities(domain, d, utils.heart_marker)
markers[heart_cells] = np.full_like(heart_cells, 1)
tags = mesh.meshtags(domain, d, body_cells, markers)

# Define new measure such that dx(0) integrates torso and dx(1) integrates heart
dx = ufl.Measure("cell", domain, subdomain_data=tags)

#####################################################################33
# PROMIJENITI!!!

dt = DT
A_m = A_M
C_m = C_M