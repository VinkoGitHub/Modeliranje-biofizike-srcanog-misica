# time variable parameters
T = 500  # final time (ms)
DT = 1.5  # time step size (ms)
NUM_STEPS = int(T / DT)  # number of time steps

# mesh parameters
WIDTH, HEIGHT, DEPTH = 5.0, 5.0, 5.0  # cm
Nx, Ny, Nz = 32, 32, 32  # mesh points density
Hx, Hy = 0.5, 0.5  # heart locations in x and y directions (cm units)
R = 1  # heart radius (cm units)

# model parameters
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