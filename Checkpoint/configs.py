# time variable parameters
T = 30  # final time (ms)
DT = 0.125  # time step size (ms)
NUM_STEPS = int(T / DT)  # number of time steps

# mesh parameters
WIDTH = 1.0  # cm
HEIGHT = 1.0  # cm
DEPTH = 1.0  # cm
Nx = 32  # mesh points density
Ny = 32  # mesh points density
Nz = 1  # mesh points density
Hx = 0.5  # heart locations in x and y directions (cm units)
Hy = 0.5
R = 0.2  # heart radius (cm units)

# model parameters
SURFACE_AREA = WIDTH * HEIGHT * 2 + HEIGHT * DEPTH * 2 + DEPTH * WIDTH * 2  # cm^2
VOLUME = WIDTH * HEIGHT * DEPTH  # cm^3
CHI = SURFACE_AREA / VOLUME  # cm^-1
C_M = 1e-3  # ohm ms/cm^2
R_M = 5000  # ohm*cm^2
V_0 = -30.0  # mV
I_0 = -10  # mA/cm^2

V_REST = -85.0
V_TH = -68.75
V_PEAK = 40
k = 0.001 #k = 0.000416
l = 1 #l = 0.625
b = 0.2 #b = 0.013

# conductivities
SIGMA_IL = 1.5e-3  # cm^2/ms
SIGMA_IT = 5.0e-3 # cm^2/ms
SIGMA_IN = 1.5762e-3 # cm^2/ms
SIGMA_EL = 1.0e-3 # cm^2/ms
SIGMA_ET = 8.25e-3 # cm^2/ms
SIGMA_EN = 6.757e-3 # cm^2/ms