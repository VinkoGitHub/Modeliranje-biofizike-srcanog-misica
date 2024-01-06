import numpy as np
from abc import ABC, abstractmethod

class BaseCellModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def f(V: np.ndarray, w: np.ndarray, *args) -> np.ndarray:
        '''dw/dt = f(V, w)'''
        pass

    @abstractmethod
    def I_ion(
        V: np.ndarray, w: np.ndarray, *args) -> np.ndarray:
         '''dV/dt = I_ion(V, w)'''
         pass

class BaseDynamicsModel(ABC):
    def __init__(self, V_REST: float):
        self.V_REST = V_REST
    
    @abstractmethod
    def F():
        '''Function should return ufl form F which is weak form formulated as F(u,v,t) = 0.'''
        pass

# deafult model parameters
CHI = 2000  # cm^-1
C_M = 1  # ms*mS/cm^2
V_REST = -85.0 # mV
V_PEAK = 40.0 # mV

# conductivities
sigma_il = 3.0  # mS/cm
sigma_it = 1.0 # mS/cm
sigma_in = 0.31525 # mS/cm
sigma_el = 2.0 # mS/cm
sigma_et = 1.65 # mS/cm
sigma_en = 1.3514 # mS/cm

SIGMA_IL = sigma_il/C_M/CHI # cm^2/ms
SIGMA_IT = sigma_it/C_M/CHI # cm^2/ms
SIGMA_IN = sigma_in/C_M/CHI # cm^2/ms
SIGMA_EL = sigma_el/C_M/CHI # cm^2/ms
SIGMA_ET = sigma_et/C_M/CHI # cm^2/ms
SIGMA_EN = sigma_en/C_M/CHI # cm^2/ms