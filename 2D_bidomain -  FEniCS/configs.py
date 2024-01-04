from dolfinx import *

# time variable parameters
T = 20  # final time (ms)
DT = 0.25  # time step size (ms)
NUM_STEPS = int(T / DT)  # number of time steps
T_ACT = 10  # external stimulus acting time (ms)

# mesh parameters
N = 128  # mesh resolution
Hx = 0.65  # heart locations in x and y directions (cm units)
Hy = 0.65
X = 0.2  # heart sizes in x and y directions (cm units)
Y = 0.2
Tx = 1.0  # torso sizes in x and y directions (cm units)
Ty = 1.0

# model parameters
A_M = 200  # cm^-1
C_M = 1e-3  # mF

# conductivities
SIGMA_IL = 3.0 * 10**-3
SIGMA_EL = 3.0 * 10**-3
SIGMA_IT = 3.0 * 10**-4
SIGMA_ET = 1.2 * 10**-3
SIGMA_TLT = 5.0 * 10**-3  # mean value of lung and rest of the body conductivities

# ionic current parameters
V_MIN = -80  # mV
V_MAX = 20  # mV
TAU_IN = 4.5  # ms
TAU_OUT = 90  # ms

# gating variable parameters
TAU_OPEN = 100  # ms
TAU_CLOSE = (
    110  # ms - mean value of different tau_close values in different heart regions
)
V_GATE = -67  # mV

# MODELLING EXPRESSIONS

# Ionic current
def I_ion(V) -> Expression:
    """V - transmembrane potential (V_m) \n
    v - gating variable w"""
    value = Expression(
        "-w/TAU_IN*pow((V_m - V_MIN),2)*(V_MAX - V_m)/(V_MAX - V_MIN) + 1/TAU_OUT*(V_m - V_MIN)/(V_MAX - V_MIN)",
        TAU_IN=TAU_IN,
        TAU_OUT=TAU_OUT,
        V_MIN=V_MIN,
        V_MAX=V_MAX,
        V_m=Function(V),
        w=Function(V),
        degree=2,
    )
    return value


# Gating variable
def g(V: Function, v: Function) -> Expression:
    """V - transmembrane potential (V_m) \n
    v - gating variable (w)"""
    return Expression(
        "V_m < V_GATE ? w/TAU_OPEN - 1/TAU_OPEN/pow((V_MAX - V_MIN),2) : w/TAU_CLOSE",
        TAU_OPEN=TAU_OPEN,
        TAU_CLOSE=TAU_CLOSE,
        V_GATE=V_GATE,
        V_MAX=V_MAX,
        V_MIN=V_MIN,
        V_m=V,
        w=v,
        degree=2,
    )


# Fiber vector field
fibers = Expression(
    (
        "x[1]/sqrt(pow(x[0],2) + pow(x[1],2) + DOLFIN_EPS)",
        "-x[0]/sqrt(pow(x[0],2) + pow(x[1],2) + DOLFIN_EPS)",
    ),
    degree=2,
)

# Applied stimulus
I_app = Expression(
    "exp(-pow(x[0]-X, 2)/(size+DOLFIN_EPS) - pow(x[1]-Y, 2)/(size+DOLFIN_EPS) - pow(t/t_act, 2))",
    X=X,
    Y=Y,
    t_act=T_ACT,
    t=0,
    size=(X + Y) / 2,
    degree=2,
)  # nije kao u radu

# EXPRESSING PARAMETERS IN UFL LANGUAGE

# conductivities
sigma_il = Constant(SIGMA_IL)
sigma_it = Constant(SIGMA_IT)
sigma_el = Constant(SIGMA_EL)
sigma_et = Constant(SIGMA_ET)
sigma_tlt = Constant(SIGMA_TLT)

sigma_i = sigma_it * Identity(2) + (sigma_il - sigma_it) * outer(fibers, fibers)
sigma_e = sigma_et * Identity(2) + (sigma_el - sigma_et) * outer(fibers, fibers)
sigma_t = sigma_tlt * Identity(2)

# Constants used in variational forms
dt = Constant(DT)
A_m = Constant(A_M)
C_m = Constant(C_M)