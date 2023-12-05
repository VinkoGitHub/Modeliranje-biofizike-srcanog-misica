from fenics import *

mesh = Mesh("meshes/average.xml")
print("Plotting a mesh")
plot(mesh, title="Mesh")