from dolfinx import *

# time variable parameters
T = 10  # final time (ms)
DT = 0.25  # time step size (ms)
NUM_STEPS = int(T / DT)  # number of time steps
T_ACT = 10 # external stimulus acting time (ms)

# mesh parameters
N = 10  # mesh resolution
Hx = 0.65  # heart locations in x and y directions (cm units)
Hy = 0.65
R = 0.1  # heart radius (cm units)
Tx = 1.0  # torso sizes in x and y directions (cm units)
Ty = 1.0

# model parameters
A_M = 200.0  # cm^-1
C_M = 1e-3  # mF

# conductivities
SIGMA_IL = 3.0 * 10**-3
SIGMA_EL = 3.0 * 10**-3
SIGMA_IT = 3.0 * 10**-4
SIGMA_ET = 1.2 * 10**-3
SIGMA_TLT = 5.0 * 10**-3  # mean value of lung and rest of the body conductivities

# ionic current parameters
V_MIN = -80.0  # mV
V_MAX = 20.0  # mV
TAU_IN = 4.5  # ms
TAU_OUT = 90.0  # ms

# gating variable parameters
TAU_OPEN = 100  # ms
TAU_CLOSE = (
    110  # ms - mean value of different tau_close values in different heart regions
)
V_GATE = -67  # mV

##############

t = 0

dt = DT
A_m = A_M
C_m = C_M