from dolfinx.fem import Function
import ufl
from configs import *
from setup import *


# gating variable
def g(V_m: Function, w: Function):
    condition = ufl.lt(V_m, V_GATE)
    true_statement = w / TAU_OPEN - 1 / TAU_OPEN / (V_MAX - V_MIN) ** 2
    false_statement = w / TAU_CLOSE
    return ufl.conditional(condition, true_statement, false_statement)


# fibre orientations
fibres = ufl.as_vector(
    [
        x[1] / ufl.sqrt(x[0] ** 2 + x[1] ** 2 + 1),
        -x[0] / ufl.sqrt(x[0] ** 2 + x[1] ** 2 + 1),
    ],
)


# ionic current
def I_ion(V_m: Function, w: Function):
    return -w / TAU_IN * (V_m - V_MIN) ** 2 * (V_MAX - V_m) / (
        V_MAX - V_MIN
    ) + 1 / TAU_OUT * (V_m - V_MIN) / (V_MAX - V_MIN)


# applied stimulus - custom? - may need change
def I_app(size=1, t_act=0.1) -> ufl:
    return ufl.exp(
        -(((x[0] - Hx) / size) ** 2) - ((x[1] - Hy) / size) ** 2 - (t / t_act) ** 2
    )


# conductivities
sigma_i = SIGMA_IT * ufl.Identity(d) + (SIGMA_IL - SIGMA_IT) * ufl.outer(fibres, fibres)
sigma_e = SIGMA_ET * ufl.Identity(d) + (SIGMA_EL - SIGMA_ET) * ufl.outer(fibres, fibres)
sigma_t = SIGMA_TLT * ufl.Identity(d)