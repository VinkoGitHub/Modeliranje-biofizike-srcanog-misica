from abc import ABCMeta, abstractmethod
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


class BaseCellModel(metaclass=ABCMeta):
    @abstractmethod
    def step_V_m(dt: float, V: np.ndarray, *args) -> list[np.ndarray]:
        """
        A function that computes a solution of the cell dynamics equations
        for one timestep --> [``dV/dt = I_ion(V, w)``] and gating variables equations.

        Parameters
        ----------
        ``dt``: float
            Timestep in miliseconds.
        ``V``: np.ndarray
            Transmembrane potential array at a given moment.
        ``*args``: optional
            Gating variables arrays at a given moment.

        Returns:
        ----------
        ``V_new``: ndarray
            Transmembrane potential after solving the cell dynamics equations for one timestep.
        ``g_new``: ndarray
            Gating variable after solving the cell dynamics equations for one timestep.

        ... same for another gating variables
        """
        pass

    @abstractmethod
    def visualize(self):
        """
        Visualize the action potential given by the model.
        Input parameters are final time point T, initial transmembrane
        potential value V_0 and initial gating variable value w_0.
        """
        pass


class BaseDynamicsModel(metaclass=ABCMeta):
    @abstractmethod
    def initial_V_m(self):
        """A function used to define initial transmembrane potential.
        This function should impose an initial condition on the function
        V_m. The value of V_m can be checked by plotting the function
        with ``plot_function`` function from ``utils`` module.

        Example:

        >>> def initial_V_m():
        >>>     # Function that expresses domain where potential is not at rest
        >>>     locator = lambda x: (x[0]-1)** 2 + (x[1]-2.7) ** 2 < 0.4**2
        >>>     # Assigning different values of V_m to that area
        >>>     cells = fem.locate_dofs_geometrical(self.V1, locator)
        >>>     self.V_m_n.x.array[:] = -84
        >>>     self.V_m_n.x.array[cells] = np.full_like(cells, -60)
        """
        raise NotImplementedError(
            "Method initial_V_m must be implemented for the model to work."
        )

    @abstractmethod
    def solve(self):
        """
        Main function for solving the heart dynamics models.
        Calling the function solves given equation and saves output .gif file in plots folder.
        Initial ``V_m``, conductivities ``M_i`` and ``M_e`` and ischemia location and type can be
        defined through parameters described below or by accesing them as attributes and methods of
        the class. First method is good for testing purposes while second method is better suited
        for more sophisticated models.

        Parameters
        ----------
        T : float
            Time point at which simulation ends.
        steps : int
            Number of steps in solving iterator. Also, the number of interpolation points
            in time interval [0, T].
        domain : mesh.Mesh
            Domain on which the equations are solved.
        cell_model : BaseCellModel
            One of cell models in cell_models module.
        longitudinal_sheets: list
            If given, defines the vector field that tells the direction of cardiac sheets.
            Input can be constants or coordinate-dependent values.
            e.g. [-1, 0, 0] or [-x[1], x[0], 0] where x[0] means x, x[1] means y and x[2] means z.
        transversal_sheets: list
            If given, defines the vector field that tells the direction of the normal of the cardiac sheets.
            Input can be constants or coordinate-dependent values.
            e.g. [-1, 0, 0] or [-x[1], x[0], 0] where x[0] means x, x[1] means y and x[2] means z.
        signal_point: list[float]
            If given, defines a point at which we track V_m.
            A point must be a part of a domain.
        camera: list[float] | None = None
            Camera direction vector. Defines the angle from which final solution will be recorded.
        gif_name: str
            Name of the .gif file that will be saved.

        Returns
        -------
        V_m_n : fem.Function
            A dolfinx Function containing V_m at time T.
        signal: list[float]
            A list containing the values of V_m at all time points at a given signal_point
        """
        pass

    def ischemia(self):
        """A function used to define ischemia on domain.
        This function should return a triple containing another function that
        takes ``x`` as an input and outputs a mathematical condition for an
        area in which conductivity is differentfrom the rest value
        (ischemia domain) and the value of ``M_i`` and ``M_e`` in that domain.

        Example:

        >>> def ischemia():
        >>>     return (lambda x: x[0] < 0.5, 0.1, 0.05)
        """
        return None

    def conductivity(self):
        """This function is called during setup and it defines conductivities ``M_i`` and ``M_e``.\n
        Deafult conductivities are predefined constants and if ``longitudinal_fibres``
        and ``transversal_fibres`` are defined as parameters, conductivities become
        tensor quantities. Also, whole method can be overloaded to define conductivities
        differently."""
        pass
