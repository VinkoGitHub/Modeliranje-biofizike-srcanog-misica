from src.models.base_models import BaseCellModel, Common
from src.utils import RK2_step, RK3_step, RK4_step
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from dolfinx import fem, mesh
import numpy as np


class ReparametrizedFitzHughNagumo(Common, BaseCellModel):
    a = 0.13
    b = 0.013
    c1 = 0.26
    c2 = 0.1
    c3 = 1.0
    V_PEAK = 40
    V_REST = -85
    V_AMP = V_PEAK - V_REST
    V_TH = V_REST + a * V_AMP
    V_AMP = V_PEAK - V_REST
    V_TH = V_REST + a * V_AMP

    def __init__(self, domain, w_0: float = 0.0):
        super().__init__(domain)
        self.w = fem.Function(self.V1)

    def step_V_m(self, dt: float, V: np.ndarray) -> np.ndarray:
        w = self.w.x.array

        dVdt = lambda V: (
            self.c1
            / self.V_AMP**2
            * (V - self.V_REST)
            * (V - self.V_TH)
            * (self.V_PEAK - V)
            - self.c2 / self.V_AMP * (V - self.V_REST) * w
        )
        dwdt = lambda w: self.b * (V - self.V_REST - self.c3 * w)
        self.w.x.array[:] = RK2_step(dwdt, dt, w)

        return RK2_step(dVdt, dt, V)

    def visualize(self, T: float, V_0: float, w_0: float):
        def fun(t, z):
            V, w = z
            dVdt = (
                self.c1
                / self.V_AMP**2
                * (V - self.V_REST)
                * (V - self.V_TH)
                * (self.V_PEAK - V)
                - self.c2 / self.V_AMP * (V - self.V_REST) * w
            )
            dwdt = self.b * (V - self.V_REST - self.c3 * w)
            return [dVdt, dwdt]

        time = np.linspace(0, T, 500)
        sol = solve_ivp(fun, [0, T], [V_0, w_0], method="DOP853", t_eval=time)

        plt.plot(sol.t, sol.y[0])
        plt.xlabel("$t$")
        plt.ylabel("$V_m$")
        plt.title("Action potential")
        plt.show()


class Noble(Common):
    C_m = 12.0
    gbar_Na = 400.0
    gbar_K2 = 1.2
    g_i = 0.14
    v_Na = 40.0
    v_K = -100.0
    I_app = 0.0

    def __init__(
        self, domain: mesh.Mesh, m_0: float = 0.0, h_0: float = 0.0, n_0: float = 0.0
    ):
        super().__init__(domain)
        self.m = fem.Function(self.V1)
        self.m.x.array[:] = m_0
        self.h = fem.Function(self.V1)
        self.h.x.array[:] = h_0
        self.n = fem.Function(self.V1)
        self.n.x.array[:] = n_0

    def step_V_m(self, dt: float, V: np.ndarray) -> np.ndarray:
        m = self.m.x.array
        h = self.h.x.array
        n = self.n.x.array

        self.g_K1 = 1.2 * np.exp(-(V + 90) / 50) + 0.015 * np.exp((V + 90) / 60)
        self.g_K2 = self.gbar_K2 * n**4
        self.g_Na = self.gbar_Na * m**3 * h + self.g_i

        dVdt = lambda V: (
            -1
            / self.C_m
            * (
                self.g_Na * (V - self.v_Na)
                + (self.g_K1 + self.g_K2) * (V - self.v_K)
                + self.I_app
            )
        )

        def blueprint(C1, C2, C3, C4, C5, v0):
            return (C1 * np.exp(C2 * (V - v0)) + C3 * (V - v0)) / (
                1 + C4 * np.exp(C5 * (V - v0))
            )

        dmdt = lambda m: (
            blueprint(0.0, 0.0, 0.1, -1.0, -1 / 15, -48.0) * (1 - m)
            - blueprint(0.0, 0.0, -0.12, -1.0, 0.2, -8.0) * m
        )
        dhdt = lambda h: (
            blueprint(0.17, -1 / 20, 0.0, 0.0, 0.0, -90) * (1 - h)
            - blueprint(1.0, 0.0, 0.0, 1.0, -0.1, -42.0) * h
        )
        dndt = lambda n: (
            blueprint(0.0, 0.0, 0.0001, -1.0, -0.1, -50.0) * (1 - n)
            - blueprint(0.002, -1 / 80, 0.0, 0.0, 0.0, -90.0) * n
        )

        self.m.x.array[:] = RK3_step(dmdt, dt, m)
        self.h.x.array[:] = RK3_step(dhdt, dt, h)
        self.n.x.array[:] = RK3_step(dndt, dt, n)

        return RK2_step(dVdt, dt, V)

    def visualize(
        self, T: float, V_0: float, m_0: float = 0.0, h_0: float = 0.0, n_0: float = 0.0
    ):
        def fun(t, z):
            V, m, h, n = z

            self.g_K1 = 1.2 * np.exp(-(V + 90) / 50) + 0.015 * np.exp((V + 90) / 60)
            self.g_K2 = self.gbar_K2 * n**4
            self.g_Na = self.gbar_Na * m**3 * h + self.g_i

            dVdt = (
                -1
                / self.C_m
                * (
                    self.g_Na * (V - self.v_Na)
                    + (self.g_K1 + self.g_K2) * (V - self.v_K)
                    + self.I_app
                )
            )

            def blueprint(C1, C2, C3, C4, C5, v0):
                return (C1 * np.exp(C2 * (V - v0)) + C3 * (V - v0)) / (
                    1 + C4 * np.exp(C5 * (V - v0))
                )

            dmdt = (
                blueprint(0.0, 0.0, 0.1, -1.0, -1 / 15, -48.0) * (1 - m)
                - blueprint(0.0, 0.0, -0.12, -1.0, 0.2, -8.0) * m
            )
            dhdt = (
                blueprint(0.17, -1 / 20, 0.0, 0.0, 0.0, -90) * (1 - h)
                - blueprint(1.0, 0.0, 0.0, 1.0, -0.1, -42.0) * h
            )
            dndt = (
                blueprint(0.0, 0.0, 0.0001, -1.0, -0.1, -50.0) * (1 - n)
                - blueprint(0.002, -1 / 80, 0.0, 0.0, 0.0, -90.0) * n
            )
            return [dVdt, dmdt, dhdt, dndt]

        time = np.linspace(0, T, 500)
        sol = solve_ivp(fun, [0, T], [V_0, m_0, h_0, n_0], method="DOP853", t_eval=time)

        plt.plot(sol.t, sol.y[0])
        plt.xlabel("$t$")
        plt.ylabel("$V_m$")
        plt.title("Action potential")
        plt.show()
        print(
            "U trenutku t=1600 V, m, h i n su:",
            sol.y[0][400],
            sol.y[1][400],
            sol.y[2][400],
            sol.y[3][400],
        )


class BeelerReuter(Common):
    C_m = 1.0
    I_app = 0.0

    def __init__(
        self,
        domain: mesh.Mesh,
        c_0: float = 1,
        m_0: float = 0.011,
        h_0: float = 0.99,
        j_0: float = 0.97,
        d_0: float = 0.003,
        f_0: float = 1,
        x_0: float = 0.0074,
    ):
        super().__init__(domain)
        self.c = fem.Function(self.V1)
        self.c.x.array[:] = c_0
        self.m = fem.Function(self.V1)
        self.m.x.array[:] = m_0
        self.h = fem.Function(self.V1)
        self.h.x.array[:] = h_0
        self.j = fem.Function(self.V1)
        self.j.x.array[:] = j_0
        self.d = fem.Function(self.V1)
        self.d.x.array[:] = d_0
        self.f = fem.Function(self.V1)
        self.f.x.array[:] = f_0
        self.x1 = fem.Function(self.V1)
        self.x1.x.array[:] = x_0

    def step_V_m(self, dt: float, V: np.ndarray) -> np.ndarray:
        c = self.c.x.array
        m = self.m.x.array
        h = self.h.x.array
        j = self.j.x.array
        d = self.d.x.array
        f = self.f.x.array
        x1 = self.x1.x.array

        I_K1 = 0.35 * (
            4
            * (np.exp(0.04 * (V + 85)) - 1)
            / (np.exp(0.08 * (V + 53)) + np.exp(0.04 * (V + 53)))
            + 0.2 * (V + 23) / (1 - np.exp(-0.04 * (V + 23)))
        )
        I_x1 = 0.8 * x1 * (np.exp(0.04 * (V + 77)) - 1) / np.exp(0.04 * (V + 35))
        I_Na = (4 * m**3 * h * j + 0.003) * (V - 50)
        I_Ca = 0.09 * d * f * (V - 127.698 + 13.0287 * np.log(c))

        dVdt = lambda V: -1 / self.C_m * (I_K1 + I_x1 + I_Na + I_Ca - self.I_app)

        dcdt = lambda c: 0.07 * (1 - c) - I_Ca

        def blueprint(C1, C2, C3, C4, C5, v0):
            return (C1 * np.exp(C2 * (V - v0)) + C3 * (V - v0)) / (
                1 + C4 * np.exp(C5 * (V - v0))
            )

        dmdt = lambda m: (
            blueprint(0.0, 0.0, 1.0, -1.0, -0.1, -47.0) * (1 - m)
            - blueprint(40.0, -0.056, 0.0, 0.0, 0.0, -72.0) * m
        )
        dhdt = lambda h: (
            blueprint(0.126, -0.25, 0.0, 0.0, 0.0, -77.0) * (1 - h)
            - blueprint(1.7, 0.0, 0.0, 1.0, -0.082, -22.5) * h
        )
        djdt = lambda j: (
            blueprint(0.055, -0.25, 0.0, 1.0, -0.2, -78.0) * (1 - j)
            - blueprint(0.3, 0.0, 0.0, 1.0, -0.1, -32.0) * j
        )
        dddt = lambda d: (
            blueprint(0.095, -0.01, 0.0, 1.0, -0.072, 5.0) * (1 - d)
            - blueprint(0.07, -0.017, 0.0, 1.0, 0.05, -44.0) * d
        )
        dfdt = lambda f: (
            blueprint(0.012, -0.008, 0.0, 1.0, 0.15, -28.0) * (1 - f)
            - blueprint(0.0065, -0.02, 0.0, 1.0, -0.2, -30.0) * f
        )
        dx1dt = lambda x1: (
            blueprint(0.0005, 0.083, 0.0, 1.0, 0.057, -50.0) * (1 - x1)
            - blueprint(0.0013, -0.06, 0.0, 1.0, -0.04, -20.0) * x1
        )

        self.m.x.array[:] = RK4_step(dmdt, dt, m)
        self.h.x.array[:] = RK4_step(dhdt, dt, h)
        self.j.x.array[:] = RK4_step(djdt, dt, j)
        self.d.x.array[:] = RK4_step(dddt, dt, d)
        self.f.x.array[:] = RK4_step(dfdt, dt, f)
        self.x1.x.array[:] = RK4_step(dx1dt, dt, x1)
        self.c.x.array[:] = RK4_step(dcdt, dt, c)

        return RK4_step(dVdt, dt, V)

    def visualize(
        self,
        T: float,
        V_0: float,
        c_0: float = 1,
        m_0: float = 0.011,
        h_0: float = 0.99,
        j_0: float = 0.97,
        d_0: float = 0.003,
        f_0: float = 1,
        x_0: float = 0.0074,
    ):
        def fun(t, z):
            V, c, m, h, j, d, f, x1 = z

            I_K1 = 0.35 * (
                4
                * (np.exp(0.04 * (V + 85)) - 1)
                / (np.exp(0.08 * (V + 53)) + np.exp(0.04 * (V + 53)))
                + 0.2 * (V + 23) / (1 - np.exp(-0.04 * (V + 23)))
            )
            I_x1 = 0.8 * x1 * (np.exp(0.04 * (V + 77)) - 1) / np.exp(0.04 * (V + 35))
            I_Na = (4 * m**3 * h * j + 0.003) * (V - 50)
            I_Ca = 0.09 * d * f * (V - 127.698 + 13.0287 * np.log(c))

            dVdt = -1 / self.C_m * (I_K1 + I_x1 + I_Na + I_Ca - self.I_app)

            dcdt = 0.07 * (1 - c) - I_Ca

            def blueprint(C1, C2, C3, C4, C5, v0):
                return (C1 * np.exp(C2 * (V - v0)) + C3 * (V - v0)) / (
                    1 + C4 * np.exp(C5 * (V - v0))
                )

            dmdt = (
                blueprint(0.0, 0.0, 1.0, -1.0, -0.1, -47.0) * (1 - m)
                - blueprint(40.0, -0.056, 0.0, 0.0, 0.0, -72.0) * m
            )
            dhdt = (
                blueprint(0.126, -0.25, 0.0, 0.0, 0.0, -77.0) * (1 - h)
                - blueprint(1.7, 0.0, 0.0, 1.0, -0.082, -22.5) * h
            )
            djdt = (
                blueprint(0.055, -0.25, 0.0, 1.0, -0.2, -78.0) * (1 - j)
                - blueprint(0.3, 0.0, 0.0, 1.0, -0.1, -32.0) * j
            )
            dddt = (
                blueprint(0.095, -0.01, 0.0, 1.0, -0.072, 5.0) * (1 - d)
                - blueprint(0.07, -0.017, 0.0, 1.0, 0.05, -44.0) * d
            )
            dfdt = (
                blueprint(0.012, -0.008, 0.0, 1.0, 0.15, -28.0) * (1 - f)
                - blueprint(0.0065, -0.02, 0.0, 1.0, -0.2, -30.0) * f
            )
            dxdt = (
                blueprint(0.0005, 0.083, 0.0, 1.0, 0.057, -50.0) * (1 - x1)
                - blueprint(0.0013, -0.06, 0.0, 1.0, -0.04, -20.0) * x1
            )
            return [dVdt, dcdt, dmdt, dhdt, djdt, dddt, dfdt, dxdt]

        time = np.linspace(0, T, 3000)
        sol = solve_ivp(
            fun,
            [0, T],
            [V_0, c_0, m_0, h_0, j_0, d_0, f_0, x_0],
            # method="DOP853",
            t_eval=time,
        )

        plt.plot(sol.t, sol.y[0])
        plt.xlabel("$t$")
        plt.ylabel("$V_m$")
        plt.title("Action potential")
        plt.show()

        print(
            "U trenutku T:",
            "V=",
            sol.y[0][-1],
            "c=",
            sol.y[1][-1],
            "m=",
            sol.y[2][-1],
            "h=",
            sol.y[3][-1],
            "j=",
            sol.y[4][-1],
            "d=",
            sol.y[5][-1],
            "f=",
            sol.y[6][-1],
            "x=",
            sol.y[7][-1],
        )
