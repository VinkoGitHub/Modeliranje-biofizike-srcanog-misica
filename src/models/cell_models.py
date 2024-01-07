from src.models.base_models import BaseCellModel
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

    def f(self, V: np.ndarray, w: np.ndarray) -> np.ndarray:
        return self.b * (V - self.V_REST - self.c3 * w)

    def I_ion(self, V: np.ndarray, w: np.ndarray) -> np.ndarray:
        V_AMP = self.V_PEAK - self.V_REST
        V_TH = self.V_REST + self.a * V_AMP
        return (
            self.c1 / V_AMP**2 * (V - self.V_REST) * (V - V_TH) * (self.V_PEAK - V)
            - self.c2 / V_AMP * (V - self.V_REST) * w
        )