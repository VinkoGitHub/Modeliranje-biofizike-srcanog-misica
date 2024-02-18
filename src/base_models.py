from abc import ABC, abstractmethod
from dolfinx import fem
import numpy as np
import ufl


class Common:
    """Common class used by multiple other classes."""

    def __init__(self, domain):
        self.domain = domain
        # Setting up meshes, function spaces and functions
        self.element = ufl.FiniteElement("P", self.domain.ufl_cell(), degree=2)
        self.W = fem.FunctionSpace(
            self.domain, ufl.MixedElement(self.element, self.element)
        )
        self.V1, self.sub1 = self.W.sub(0).collapse()
        self.V2, self.sub2 = self.W.sub(1).collapse()
        self.x = ufl.SpatialCoordinate(self.domain)
        self.d = self.domain.topology.dim


class BaseCellModel(ABC, Common):
    def applied_current(self, t: float) -> fem.Function | None:
        """
        Default value of applied current is 0. In this function you can define
        the position of applying a current by modifying `self.I_app` as a function.
        Function input is time point value and function before every time step.
        Model output can be plotted using the `utils.plot_function` function.

        Example:
        ----------
        >>> def applied_current(self, t: float):
        >>>     locator = lambda x: (x[0] - 1) ** 2 < 0.1**2
        >>>     cells = fem.locate_dofs_geometrical(self.V1, locator)
        >>>     if 0 < t < 3:
        >>>         self.I_app.x.array[cells] = np.full_like(cells, 10)
        >>>     else:
        >>>         self.I_app.x.array[:] = 0
        >>>     return self.I_app
        """

    @abstractmethod
    def step_V_m(dt: float, t: float, V: np.ndarray, *args) -> np.ndarray:
        """
        A function that computes a solution of the cell dynamics equations
        for one timestep -> [``dV/dt = I_ion(V, w)``] and gating variables equations.

        Parameters
        ----------
        ``dt``: float
            Timestep in miliseconds.
        ``t``: float
            Time moment in miliseconds
        ``V``: np.ndarray
            Transmembrane potential array at a given moment.
        ``*args``: optional
            Gating variables arrays at a given moment.

        Returns:
        ----------
        ``V_new``: ndarray
            Transmembrane potential after solving the cell dynamics equations for one timestep.
        """
        pass

    @abstractmethod
    def visualize(self) -> None:
        """
        Visualize the action potential given by the model.
        Input parameters are final time point T, initial transmembrane
        potential value V_0 and initial gating variable value w_0.
        """
        pass


class BaseDynamicsModel(Common, ABC):
    @abstractmethod
    def initial_V_m(self) -> fem.Function | None:
        """A function used to define initial transmembrane potential.
        This function should impose an initial condition on the function
        V_m. Model can also return a `fem.Function` object
        that can be plotted using the `utils.plot_function` function.

        Example:
        ---------

        >>> def initial_V_m(self):
        >>>     # Function that expresses domain where potential is not at rest
        >>>     locator = lambda x: (x[0]-1)** 2 + (x[1]-2.7) ** 2 < 0.4**2
        >>>     # Assigning different values of V_m to that area
        >>>     cells = fem.locate_dofs_geometrical(self.V1, locator)
        >>>     self.V_m_n.x.array[:] = -84
        >>>     self.V_m_n.x.array[cells] = np.full_like(cells, -60)
        >>>     return self.V_m_n
        """
        raise NotImplementedError(
            "Method initial_V_m must be implemented for the model to work."
        )

    @abstractmethod
    def solve(self) -> None:
        """
        Main function for solving the heart dynamics models.
        Calling the function solves given equation and saves output .gif file in plots folder.
        Initial ``V_m``, conductivities ``M_i`` and ``M_e`` and ischemia location and type can be
        defined through parameters described below or by accesing them as attributes and methods of
        the class. First method is good for testing purposes while second method is better suited
        for more sophisticated models.

        Parameters
        ----------
        `T` : float
            Time point at which simulation ends.
        `steps` : int
            Number of steps in solving iterator. Also, the number of interpolation points
            in time interval [0, T].
        `signal_point`: list[float]
            If given, defines a point at which we track V_m.
            A point must be a part of a domain.
        `camera_direction`: list[float] | str | None
            Determines the direction of the camera.
        `zoom`: float
            Sets the zoom factor.
        `cmap`: str
            A colormap that Pyvista can interpret.
        `save_to`: str
            A path to the file directory where the plot will be saved in the `animations` directory.
        `checkpoints`: list[float]
            A set of time points for which plots will be saved.
        `checkpoint_file`: str
            A file path which is a blueprint for saving a checkpoints in the `figures` directory.
            Checkpoint value will be appended to a checkpoint_file name.\n
        Additionally, for the `monodomain` solver, parameter `lambda_` should be passed in.
        It represents proportionality constant between the intracellular and extracellular conductivities.
        """
        pass

    def ischemia(self) -> fem.Function | None:
        """A function used to define ischemia on domain.
        This function should transform conductivities or other physical factors and
        is called after conductivities are defined. Also, model can return a `fem.Function` object
        that can be plotted using the `utils.plot_function` function.

        Example:
        ----------

        >>> def ischemia():
        >>>     x_c, y_c, z_c = -0.7, -1.8, -4.0
        >>>     a, b, c = 1.0, 3.0, 0.5
        >>>     def reduce(x, reduce_factor:float=15):
        >>>         return 1 + (reduce_factor-1) * ufl.exp(
        >>>        (
        >>>            -(((x[0] - x_c) / a) ** 2)
        >>>            - ((x[1] - y_c) / b) ** 2
        >>>            - ((x[2] - z_c) / c) ** 2
        >>>        )
        >>>        * ufl.ln(10)
        >>>    )
        >>>
        >>>     def value(x, reduce_factor:float = 15):
        >>>         return 1 + (reduce_factor-1) * np.exp(
        >>>        (
        >>>            -(((x[0] - x_c) / a) ** 2)
        >>>            - ((x[1] - y_c) / b) ** 2
        >>>            - ((x[2] - z_c) / c) ** 2
        >>>        )
        >>>        * np.log(10)
        >>>    )
        >>>
        >>>     self.M_i = self.M_i / reduce(self.x)
        >>>     self.M_e = self.M_e / reduce(self.x)
        >>>
        >>>     fun = fem.Function(self.V1)
        >>>     fun.interpolate(value)
        >>>     return fun
        """
        return None

    def conductivity(self) -> None:
        """This function is called during setup and it defines conductivities ``M_i`` and ``M_e``.\n
        Deafult conductivities are predefined constants and if ``longitudinal_fibres``
        and ``transversal_fibres`` are defined as parameters, conductivities become
        tensor quantities. Also, whole method can be overloaded to define conductivities
        differently.

        Example:
        -----------

        >>> def conductivity(self):
        >>>    longitudinal_fibres = [1, 0, 0]
        >>>    transversal_fibres = [0, 1, 0]
        >>>
        >>>    # Muscle sheets
        >>>    self.sheet_l = ufl.as_vector(longitudinal_fibres)
        >>>    self.sheet_n = ufl.as_vector(transversal_fibres)
        >>>
        >>>    # Healthy conductivities
        >>>    self.M_i = (
        >>>        self.SIGMA_IT * ufl.Identity(len(longitudinal_fibres))
        >>>        + (self.SIGMA_IL - self.SIGMA_IT)
        >>>        * ufl.outer(self.sheet_l, self.sheet_l)
        >>>        + (self.SIGMA_IN - self.SIGMA_IT)
        >>>        * ufl.outer(self.sheet_n, self.sheet_n)
        >>>    )
        >>>    self.M_e = (
        >>>        self.SIGMA_ET * ufl.Identity(len(transversal_fibres))
        >>>        + (self.SIGMA_EL - self.SIGMA_ET)
        >>>        * ufl.outer(self.sheet_l, self.sheet_l)
        >>>        + (self.SIGMA_EN - self.SIGMA_ET)
        >>>        * ufl.outer(self.sheet_n, self.sheet_n)
        >>>    )
        """
        pass
