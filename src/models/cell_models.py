from src.models.base_models import BaseCellModel, Common
from src.utils import RK2_step
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from dolfinx import fem
import numpy as np


class ReparametrizedFitzHughNagumo(Common, BaseCellModel):
    a = 0.13
    b = 0.013
    c1 = 0.26
    c2 = 0.1
    c3 = 1.0
    V_PEAK = 40
    V_REST = -85
    V_AMP = V_PEAK - V_REST
    V_TH = V_REST + a * V_AMP
    V_AMP = V_PEAK - V_REST
    V_TH = V_REST + a * V_AMP

    def __init__(self, domain):
        super().__init__(domain)
        self.w = fem.Function(self.V1)

    def step_V_m(self, dt: float, V: np.ndarray) -> np.ndarray:
        w = self.w.x.array

        dVdt = lambda V: (
            self.c1
            / self.V_AMP**2
            * (V - self.V_REST)
            * (V - self.V_TH)
            * (self.V_PEAK - V)
            - self.c2 / self.V_AMP * (V - self.V_REST) * w
        )
        dwdt = lambda w: self.b * (V - self.V_REST - self.c3 * w)
        self.w.x.array[:] = RK2_step(dwdt, dt, w)

        return RK2_step(dVdt, dt, V)

    def visualize(self, T: float, V_0: float, w_0: float):
        def fun(t, z):
            V, w = z
            dVdt = (
                self.c1
                / self.V_AMP**2
                * (V - self.V_REST)
                * (V - self.V_TH)
                * (self.V_PEAK - V)
                - self.c2 / self.V_AMP * (V - self.V_REST) * w
            )
            dwdt = self.b * (V - self.V_REST - self.c3 * w)
            return [dVdt, dwdt]

        time = np.linspace(0, T, 500)
        sol = solve_ivp(fun, [0, T], [V_0, w_0], method="DOP853", t_eval=time)

        plt.plot(sol.t, sol.y[0])
        plt.xlabel("t")
        plt.legend(["V", "w"], shadow=True)
        plt.title("Action potential")
        plt.show()


class Noble(Common):
    C_m = 12.0
    g_Na = 400.0
    g_K2 = 1.2
    g_i = 0.14
    v_Na = 40.0
    v_K = -100

    def __init__(self, domain):
        super().__init__(domain)
        self.w = fem.Function(self.V1)

    def step_V_m(self, dt: float, V: np.ndarray) -> np.ndarray:
        I_app = 0.0
        return -1 / self.C_m * (self.g_Na * (V - self.v_Na) + (self.gk) * () + I_app)

# -- Ovako je implementirano kod MFN --
#
#        w = self.w.x.array
#
#        dVdt = lambda V: (
#            self.c1
#            / self.V_AMP**2
#            * (V - self.V_REST)
#            * (V - self.V_TH)
#            * (self.V_PEAK - V)
#            - self.c2 / self.V_AMP * (V - self.V_REST) * w
#        )
#        dwdt = lambda w: self.b * (V - self.V_REST - self.c3 * w)
#        self.w.x.array[:] = RK2_step(dwdt, dt, w)
#
#        return RK2_step(dVdt, dt, V)

    def visualize(self, T: float, V_0: float, w_0: float):
        def fun(t, z):
            V, w = z
            dVdt = (
                self.c1
                / self.V_AMP**2
                * (V - self.V_REST)
                * (V - self.V_TH)
                * (self.V_PEAK - V)
                - self.c2 / self.V_AMP * (V - self.V_REST) * w
            )
            dwdt = self.b * (V - self.V_REST - self.c3 * w)
            return [dVdt, dwdt]

        time = np.linspace(0, T, 500)
        sol = solve_ivp(fun, [0, T], [V_0, w_0], method="DOP853", t_eval=time)

        plt.plot(sol.t, sol.y[0])
        plt.xlabel("t")
        plt.legend(["V", "w"], shadow=True)
        plt.title("Action potential")
        plt.show()
