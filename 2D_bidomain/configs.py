# time stepping parameters
T = 5.0 # final time
num_steps = 500 # number of time steps
dt = T / num_steps # time step size

# mesh parameters
N = 64 # mesh resolution
Hx = 0.65 # heart locations in x and y directions (cm units)
Hy = 0.65
X = 0.2 # heart sizes in x and y directions (cm units)
Y = 0.2
Tx = 1.0 # torso sizes in x and y directions (cm units)
Ty = 1.0

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

def I_ion(V_m, w):
    I_in = -w/tau_in*(V_m - V_min)**2*(V_max - V_m)/(V_max - V_min)
    I_out = 1/tau_out*(V_m - V_min)/(V_max - V_min)
    return I_in + I_out

def g(V_m, w, V_gate = V_gate):
    if(V_m < V_gate):
        return w/tau_open - 1/tau_open/(V_max - V_min)**2
    elif(V_m >= V_gate):
        return w/tau_close