from scipy.integrate import solve_ivp
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np


class BaseCellModel(ABC):
    V_REST = -85

    def __init__(self):
        pass

    @abstractmethod
    def f(V: np.ndarray, w: np.ndarray, *args) -> np.ndarray:
        """dw/dt = f(V, w)"""
        pass

    @abstractmethod
    def I_ion(V: np.ndarray, w: np.ndarray, *args) -> np.ndarray:
        """dV/dt = I_ion(V, w)"""
        pass

    def visualize(self, T: float, V_0: float, w_0: float):
        """Visualize the action potential given by the model.
        Input parameters are final time point T, initial transmembrane
        potential value V_0 and initial gating variable value w_0."""

        def fun(t, z):
            V, w = z
            return [
                self.I_ion(V, w),
                self.f(V, w),
            ]

        time = np.linspace(0, T, 500)
        sol = solve_ivp(fun, [0, T], [V_0, w_0], method="DOP853", t_eval=time)

        plt.plot(sol.t, sol.y[0])
        plt.xlabel("t")
        plt.legend(["V", "w"], shadow=True)
        plt.title("Action potential")
        plt.show()


class BaseDynamicsModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def initial_V_m(self):
        """A function used to define initial transmembrane potential.
        This function should return a tuple containing another function that
        takes ``x`` as an input and outputs a mathematical condition
        for an area in which initial ``V_m`` is different from the rest value
        and the initial value for V_m in that area.

        Example:

        >>> def initial_V_m():
        >>>     return (lambda x: x[0] < 0.5, 0.0)
        """
        raise NotImplementedError(
            "Method initial_V_m must be implemented for the model to work."
        )

    @abstractmethod
    def setup(self):
        """
        Calling this function sets up the meshes, function spaces, functions etc. After calling this
        function, you can visualize and inspect initial V_m and ischemia details before
        solving actual equations.\n
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
            Name of the .gif file that will be saved."""
        pass

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
