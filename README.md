## Modelling biophysics of a heart tissue
### Vinko Dragušica
Diploma thesis

### Solver workflow:

 1. Define a domain mesh - ``utils.create_mesh`` or ``utils.import_mesh`` can be used.\
 Then ``ufl.SpatialCoordinate(domain)`` can be defined if it will be used.

 2. Pick a cell model from cell_models module which inherits from one of cell models from a `dynamics_models` module.\
 You can define a cell model `applied_current` method here.

 3. Define a dynamics model class which inherits from one of dynamic models from a `cell_models` module.\
 Define ``initial_V_m`` method for that class as it is requested in the description of the method.

 4. Instantiate `cell_model` and `dynamics_model` objects. After that, you can visualize initial conditions, applied currents and ischemias before solving the equations.

 4. In the end, run a ``solve`` method of a model with its parameters. You can access the solutions as attributes of the model.

 Demo solver can be found in the `solve` directory.


### Presentation:

To run the presentation in a web browser run following commands in the terminal:

- `export PYTHONPATH=.`
- `streamlit run presentation/🎓_Modeliranje_biofizike_srčanog_tkiva.py`