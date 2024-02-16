import streamlit as st
from presentation.utils import *

# Configuring the app
st.set_page_config(
    page_title="Diplomski rad",
    page_icon="🎓",
    layout="wide",
)

# Title
st.title("Numerički model")
st.divider()
# Body
st.subheader("Diskretizacija vremenskog intervala")
st.divider()
latex_equation(
    r"[0,T]\rightarrow[0,\, \Delta t\,, 2\Delta t\,, ...\,, T-\Delta t\,, T]"
)
st.divider()
bullet("Podjela operatora")
latex_equation(
    r"\frac{\partial V_m}{\partial t} = \nabla\cdot (M_i\nabla(V_m + U_e)) - I_{ion}"
)
st.divider()
latex_equation(r"\frac{\partial V_m}{\partial t} = - I_{ion}")
latex_equation(r"\frac{\partial V_m}{\partial t} = \nabla\cdot (M_i\nabla(V_m + U_e))")
st.divider()
bullet("Iteracija između dva susjedna trenutka")
latex_equation(
    r"""\begin{align}
    \frac{\partial V_m}{\partial t} = - I_{ion}\\
    \frac{\partial V_m}{\partial t} = \nabla\cdot (M_i\nabla(V_m + U_e))\\\nonumber
    0 = \nabla\cdot (M_i\nabla V_m) + \nabla\cdot ((M_i+M_e)\nabla U_e)\\
    \frac{\partial V_m}{\partial t} = - I_{ion}
        \end{align}"""
)
bullet("Crank-Nicolson, Runge-Kutta")
st.divider()
st.subheader("Diskretizacija prostorne domene", "Metoda konačnih elemenata")
st.divider()
centered_image("nestrukturirana_mreza.jpg", "Nestrukturirana mreža srčanog mišića (preuzeto iz [10]).", 0.7)
st.divider()
bullet("Diskretizacija Crank-Nicolson metodom")
latex_equation(
    r"\frac{V_m^{n+1}-V_m^n}{\Delta t} = \nabla\cdot \left(M_i\nabla\left(\frac{V_m^{n+1}+V_m^n}{2} + U_e\right)\right)"
)
latex_equation(
    r"0 = \nabla\cdot \left(M_i\nabla\left( \frac{V_m^{n+1}+V_m^n}{2}\right) + \nabla\cdot ((M_i+M_e)\nabla U_e\right)"
)
st.divider()
st.subheader("Slaba/varijacijska forma")
st.divider()
bullet("Testne funkcije")
latex_equation(
    r"\int_\mathbb{H}\frac{V_m^{n+1}-V_m^n}{\Delta t}\phi\,\mathrm{d}V = \int_\mathbb{H}\nabla\cdot \left(M_i\nabla\left(\frac{V_m^{n+1}+V_m^n}{2} + U_e\right)\right)\phi\,\mathrm{d}V,  \qquad \forall \phi\in V"
)
latex_equation(
    r"0 = \int_\mathbb{H}\nabla\cdot \left(M_i\nabla \frac{V_m^{n+1}+V_m^n}{2}\right)\psi + \nabla\cdot ((M_i+M_e)\nabla U_e)\psi\,\mathrm{d}V,  \qquad \forall \psi\in V"
)
st.divider()
bullet("Identitet i Gaussov teorem")
latex_equation(
    r"\nabla\cdot(\psi\bm{A}) = \psi\nabla\cdot\bm{A} + (\nabla\psi)\cdot\bm{A}"
)
latex_equation(
    r"\int_\mathbb{H}\nabla\cdot \bm{F}\,\mathrm{d}V = \int_{\partial \mathbb{H}}\bm{F}\cdot\bm{\hat{n}}\,\mathrm{d}S"
)
st.divider()
latex_equation(
    r"""
    \int_\mathbb{H}\frac{V_m^{n+1}-V_m^n}{\Delta t}\phi\,\mathrm{d}V = \int_\mathbb{\partial H}\phi M_i\nabla\left(\frac{V_m^{n+1}+V_m^n}{2} + U_e\right)\cdot\bm{\hat{n}}\,\mathrm{d}S\\
      - \int_\mathbb{H} M_i\nabla\left(\frac{V_m^{n+1}+V_m^n}{2} + U_e\right)\cdot\nabla\phi\,\mathrm{d}V, \qquad \forall \phi\in V
"""
)
latex_equation(
    r"""0 = \int_\mathbb{\partial H}\psi\left(M_i\nabla \frac{V_m^{n+1}+V_m^n}{2}\right)\cdot\bm{\hat{n}}\,\mathrm{d}S - \int_\mathbb{H}\left(M_i\nabla \frac{V_m^{n+1}+V_m^n}{2}\right)\cdot\nabla\psi\,\mathrm{d}V\\
          + \int_\mathbb{\partial H}\psi (M_i+M_e)\nabla U_e\cdot\bm{\hat{n}}\,\mathrm{d}S - \int_\mathbb{H}(M_i+M_e)\nabla U_e\cdot\nabla\psi\,\mathrm{d}V, \qquad \forall \psi\in V"""
)
st.divider()
bullet("Rubni uvjeti", "Slaba forma")
latex_equation(
    r"\int_\mathbb{H}\frac{V_m^{n+1}-V_m^n}{\Delta t}\phi\,\mathrm{d}V + \int_\mathbb{H} M_i\nabla\left(\frac{V_m^{n+1}+V_m^n}{2} + U_e\right)\cdot\nabla\phi\,\mathrm{d}V = 0, \qquad \forall \phi\in V"
)
latex_equation(
    r"\int_\mathbb{H}M_i\nabla\left(\frac{V_m^{n+1}+V_m^n}{2}\right)\cdot\nabla\psi\,\mathrm{d}V + \int_\mathbb{H}(M_i+M_e)\nabla U_e\cdot\nabla\psi\,\mathrm{d}V = 0, \qquad \forall \psi\in V"
)
bullet("Ekstracelularni potencijal")
latex_equation(r"\int_\mathbb{H}U_e\,\mathrm{d}V= 0")
st.divider()
st.subheader("Jednodomenski model")
st.divider()
bullet("Crank-Nicolsonova metoda")
latex_equation(
    r"\frac{V_m^{n+1}-V_m^n}{\Delta t} = \frac{\lambda}{\lambda + 1}\nabla\cdot M_i\nabla\left(\frac{V_m^{n+1}+V_m^n}{2}\right)"
)
st.divider()
bullet("Slaba forma")
latex_equation(
    r"\int_\mathbb{H}\frac{V_m^{n+1}-V_m^n}{\Delta t}\phi\,\mathrm{d}V + \int_\mathbb{H} \frac{\lambda}{\lambda + 1}M_i\nabla\left(\frac{V_m^{n+1}+V_m^n}{2}\right)\cdot\nabla\phi\,\mathrm{d}V = 0, \qquad \forall \phi\in V"
)
