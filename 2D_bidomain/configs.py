from fenics import *

# time stepping parameters
T = 5.0 # final time
NUM_STEPS = 5 # number of time steps
DT = T / NUM_STEPS # time step size

# mesh parameters
N = 64 # mesh resolution
Hx = 0.65 # heart locations in x and y directions (cm units)
Hy = 0.65
X = 0.2 # heart sizes in x and y directions (cm units)
Y = 0.2
Tx = 1.0 # torso sizes in x and y directions (cm units)
Ty = 1.0

# model parameters

A_M = 1 # promijenit!!!
B_M = 2 # promijenit!!!
C_M = 3 # promijenit!!!

# conductivities
SIGMA_IL = 3.0*10**-3
SIGMA_EL = 3.0*10**-3
SIGMA_IT = 3.0*10**-4
SIGMA_ET = 1.2*10**-3
SIGMA_TLT = 5.0*10**-3 # mean value of lung and rest of the body conductivities

# ionic model (Nagumo, no repolarization)
Vrest = -85.0
Vdep = +30.0
Vthre = -55.0
Rmax = 1.4e-3

def fion(Vm):
    return Rmax * (Vm - Vdep) * (Vm - Vrest) * (Vm - Vthre)

# modeling an ionic current and a gating variable

V_min = -80 # mV
V_max = 20 # mV
tau_in = 0.01
tau_out = 0.1

tau_open = 1
tau_close = 1
V_gate = 3

class I_ion_expression(UserExpression):

    def __init__(self, V_m, w):
        super().__init__()
        self.V_m = V_m
        self.w = w

    def eval_cell(self):
        I_in = -self.w/tau_in*(self.V_m - V_min)**2*(V_max - self.V_m)/(V_max - V_min)
        I_out = 1/tau_out*(self.V_m - V_min)/(V_max - V_min)
        return I_in + I_out
    
    def value_shape(self):
        return (1,)
        
class g_expression(UserExpression):
    def __init__(self, V_m, w):
        super().__init__()
        self.V_m = V_m
        self.w = w
    
    def eval_cell(self):
        if(self.V_m < V_gate):
            return self.w/tau_open - 1/tau_open/(V_max - V_min)**2
        elif(self.V_m >= V_gate):
            return self.w/tau_close
    def value_shape(self):
        return (1,)