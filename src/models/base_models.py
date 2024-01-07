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
    def solve():
        """
        Main function for solving the heart dynamics models.
        Calling the function solves given equation and saves output .gif file in plots folder.

        Parameters
        ----------
        domain : mesh.Mesh
            Domain on which the equations are solved.
        cell_model : BaseCellModel
            One of cell models in cell_models module.
        V_m_0 : list[list[float], float, float]
            Initial value for transmembrane potential.
            First value in the list is the center [x,y,z] in which we define V_m_0.
            Second value in the list is the radius in which V_m_0 is defined.
            Third value in the list is the value of V_m_0 in the given domain.
        ischemia : list[list[float], float, float] or None
            If given, makes  value for transmembrane potential.
            First value in the list is the center [x,y,z] of ischemia.
            Second value in the list is the radius in which ischemia is defined.
            Third value in the list is the conductivity reduction factor.
        longitudinal_sheets: list
            If given, defines the vector field that tells the direction of cardiac sheets.
            Input can be constants or coordinate-dependent values.
            e.g. [-1, 0, 0] or [-x[1], x[0], 0] where x[0] means x, x[1] means y and x[2] means z.
        transversal_sheets: list
            If given, defines the vector field that tells the direction of the normal of the cardiac sheets.
            Input can be constants or coordinate-dependent values.
            e.g. [-1, 0, 0] or [-x[1], x[0], 0] where x[0] means x, x[1] means y and x[2] means z.
        signal_point: list[float]
            A point at which we track V_m.
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
