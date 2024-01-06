import numpy as np
from base_models import BaseCellModel

class ReparametrizedFitzHughNagumo(BaseCellModel):
    def __init__(self):
        pass

    def f(V: np.ndarray, w: np.ndarray, b=0.013, c3=1.0, V_REST=-85) -> np.ndarray:
        return b * (V - V_REST - c3 * w)

    def I_ion(V: np.ndarray, w: np.ndarray, a=0.13, c1=0.26, c2=0.1, V_REST=-85, V_PEAK=40) -> np.ndarray:
        V_AMP = V_PEAK - V_REST
        V_TH = V_REST + a * V_AMP
        return (
            c1 / V_AMP**2 * (V - V_REST) * (V - V_TH) * (V_PEAK - V)
            - c2 / V_AMP * (V - V_REST) * w
        )