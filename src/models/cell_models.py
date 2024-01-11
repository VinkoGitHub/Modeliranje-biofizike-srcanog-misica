from src.models.base_models import BaseCellModel
from src.utils import RK2_step
import numpy as np


class ReparametrizedFitzHughNagumo(BaseCellModel):
    a = 0.13
    b = 0.013
    c1 = 0.26
    c2 = 0.1
    c3 = 1.0
    V_PEAK = 40
    V_REST = -85
    V_AMP = V_PEAK - V_REST
    V_TH = V_REST + a * V_AMP

    def __init__(self):
        pass

    def step(self, dt: float, V: np.ndarray, w: np.ndarray) -> list[np.ndarray]:
        self.step.__doc__
        V_AMP = self.V_PEAK - self.V_REST
        V_TH = self.V_REST + self.a * V_AMP

        return RK2_step(
            lambda V: (
                self.c1
                / V_AMP**2
                * (V - self.V_REST)
                * (V - V_TH)
                * (self.V_PEAK - V)
                - self.c2 / V_AMP * (V - self.V_REST) * w
            ),
            dt,
            V,
        ), RK2_step(lambda w: self.b * (V - self.V_REST - self.c3 * w), dt, w)


class Noble(BaseCellModel):
    C_m = 12.0
    g_Na = 400.0
    g_K2 = 1.2
    g_i = 0.14
    v_Na = 40.0
    v_K = -100

    def __init__(self):
        pass

    def step(self, V: np.ndarray) -> np.ndarray:
        return -1 / self.C_m * (self.g_Na * (V - self.v_Na) + (self.gk) * () + I_app)
