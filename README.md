### Solver workflow:
 1. Define a domain mesh - ``utils.create_mesh`` or ``utils.import_mesh`` can be used.\
 Then ``ufl.SpatialCoordinate(domain)`` can be defined if it will be used.

 2. Pick a cell model from cell_models module.

 3. Define a dynamics model class which inherits from one of dynamic modules from a dynamics_models module.\
 Define ``initial_V_m`` method for that class as it is requested in the description of the method.

 4. Run a ``solve`` method of a model with its parameters.
