import streamlit as st
from presentation.utils import *

# Configuring the app
st.set_page_config(
    page_title="Diplomski rad",
    page_icon="🎓",
    layout="wide",
)

# Title
st.title("Matematički model")
st.divider()

# Body
bullet(
    "1 milijarda do 10 milijardi stanica<sup>[8]</sup>",
    "Homogenizacija",
    "Usrednjene veličine",
)
st.divider()
centered_image(
    "bidomain-cells.png",
    width=4,
    caption="Ilustracija intracelularne i ekstracelularne domene(preuzeto iz [8]).",
)
st.divider()
st.header("Dvodomenski model")
st.divider()
latex_equation(
    r"\bar{f}(\bm{x},t)\rightarrow f(\bm{x},t),\quad \mathbb{H},\quad \mathbb{\partial H}"
)
st.divider()
bullet(
    "Intracelularna domena",
    "Ekstracelularna domena",
    "Rascjepni kanali",
    "Maxwellova jednadžba",
)
latex_equation(
    r"\nabla\times\bm{E}(\bm{x},t) = -\frac{\partial\bm{B}(\bm{x},t)}{\partial t}"
)
st.divider()
bullet("Kvazistacionarnost")
latex_equation(r"\nabla\times\bm{E}(\bm{x},t) = 0 \quad\Rightarrow\quad \bm{E}(\bm{x},t) = -\nabla U(\bm{x},t)")
latex_equation(
    r"\bm{J}(\bm{x},t) = M(\bm{x},t)\bm{E}(\bm{x},t) \quad\Rightarrow\quad \bm{J}(\bm{x},t) = -M(\bm{x},t)\nabla U(\bm{x},t)"
)
st.divider()
bullet("Intracelularni potencijal", "Ekstracelularni potencijal")
latex_equation(
    r"U_i,\quad U_e,\quad V_m \coloneqq U_i - U_e"
)
st.divider()
latex_equation(r"\bm{J_i} = -M_i\nabla U_i")
latex_equation(r"\bm{J_e} = -M_e\nabla U_e")
st.divider()
bullet("Naboj je konstantan u vremenu")
latex_equation(r"\frac{\partial}{\partial t}(q_i(\bm{x},t)+q_e(\bm{x},t)) = 0")
st.divider()
bullet("Jednadžba kontinuiteta")
latex_equation(
    r"\nabla\cdot\bm{J_i} + \frac{\partial q_i}{\partial t} = -\chi I_{ion}(\bm{x},t)"
)
latex_equation(
    r"\nabla\cdot\bm{J_e} + \frac{\partial q_e}{\partial t} = \chi I_{ion}(\bm{x},t),"
)
st.divider()
bullet("Očuvanje struje")
latex_equation(r"\nabla\cdot(\bm{J_i} + \bm{J_e}) = 0")
bullet("Prva jednadžba modela")
latex_equation(r"    \nabla\cdot(M_i\nabla U_i + M_e\nabla U_e) = 0")
st.divider()
bullet("Kapacitet")
latex_equation(r"q = \chi C_m V_m")
latex_equation(r"q \coloneqq \frac{q_i - q_e}{2}")
st.divider()
latex_equation(
    r"\chi C_m \frac{\partial V_m}{\partial t} = \frac{1}{2}\frac{\partial(q_i-q_e)}{\partial t}"
)
latex_equation(
    r"\frac{\partial q_i}{\partial t}= -\frac{\partial q_e}{\partial t} = \chi C_m \frac{\partial V_m}{\partial t}"
)
st.divider()
bullet("Druga jednadžba modela")
latex_equation(
    r"\nabla\cdot (M_i\nabla U_i) = \chi C_m \frac{\partial V_m}{\partial t} +\chi I_{ion}"
)
st.divider()
latex_equation(
    r"\nabla\cdot (M_i\nabla(V_m + U_e)) = \chi C_m \frac{\partial V_m}{\partial t} +\chi I_{ion}"
)
latex_equation(r"\nabla\cdot(M_i\nabla V_m + (M_i + M_e)\nabla U_e) = 0")
st.divider()
bullet("Rubni uvjeti")
latex_equation(r"\bm{\hat{n}}\cdot\bm{J_i} = 0 \Rightarrow  \bm{\hat{n}}\cdot (M_i\nabla(V_m + U_e)) = 0")
latex_equation(r"\bm{\hat{n}}\cdot\bm{J_e} = 0 \Rightarrow \bm{\hat{n}}\cdot (M_e\nabla U_e) = 0")
st.divider()
bullet("Proširenje modela na torzo", "Generalizirana  Laplaceova jednadžba")
latex_equation(r"-\nabla\cdot(M_T\nabla V_T) = 0")
bullet("Rubni uvjet")
latex_equation(r"\bm{\hat{n}_0}\cdot(M_T\nabla V_T) = 0")
st.divider()
centered_image(
    "torso.jpg",
    width=2,
    caption="Primjer domene srca i torza. Torzo je prikazan plavom bojom, a srce crvenom.",
)
st.divider()
st.subheader("Vodljivost")
st.divider()
bullet("Srčana vlakna i snopovi")
centered_image(
    "vlakna.jpg",
    width=2,
    caption="Slika srčanih vlakana pod mikroskopom (preuzeto iz [9]).",
)
st.divider()
latex_equation(
    r"M_{ij} = a_{i,l} a_{j,l} \sigma^l + a_{i,t} a_{j,t} \sigma^t + a_{i,n} a_{j,n} \sigma^n"
)
latex_equation(
    r"M_i = \sigma^t_i \cdot \mathbb{1} + (\sigma^l_i - \sigma^t_i) \cdot\bm{a_l}\otimes\bm{a_l}+ (\sigma^n_i - \sigma^t_i) \cdot\bm{a_n}\otimes\bm{a_n}"
)
latex_equation(
    r"M_e = \sigma^t_e \cdot \mathbb{1} + (\sigma^l_e - \sigma^t_e) \cdot\bm{a_l}\otimes\bm{a_l}+ (\sigma^n_e - \sigma^t_e) \cdot\bm{a_n}\otimes\bm{a_n}"
)
st.divider()
bullet("Skalirani dvodomenski model")
latex_equation(
    r"""\begin{aligned}
  \nabla\cdot (M^*_i\nabla(V_m + U_e)) = \frac{\partial V_m}{\partial t} + I^*_{ion},  \qquad \forall \bm{x}\in \mathbb{H}\nonumber\\
  \nabla\cdot(M^*_i\nabla V_m + (M^*_i + M^*_e)\nabla U_e) = 0,  \qquad \forall \bm{x}\in \mathbb{H}\nonumber\\
  \bm{n}\cdot M^*_i\nabla(V_m + U_e) = 0,  \qquad \forall \bm{x}\in \partial\mathbb{H} \nonumber\\
  \bm{n}\cdot (M^*_e\nabla U_e) = 0,  \qquad \forall \bm{x}\in \partial\mathbb{H} \nonumber
\end{aligned}"""
)
bullet("Skalirane vrijednosti")
latex_equation(
    r"M^*_i \coloneqq \frac{M_i}{\chi C_m},\quad M^*_i \coloneqq \frac{M_i}{\chi C_m},\quad I^*_{ion} \coloneqq \frac{I_{ion}}{C_m}"
)
st.divider()
st.subheader("Jednodomenski model")
st.divider()
bullet("Jednodomenska aproksimacija", "Proporcionalnost vodljivosti")
latex_equation(r"M_e = \lambda M_i")
latex_equation(
    r"\nabla\cdot (M_i\nabla(V_m + U_e)) = \chi C_m \frac{\partial V_m}{\partial t} +\chi I_{ion}"
)
latex_equation(r"\nabla\cdot(M_i\nabla V_m + (1 + \lambda)M_i\nabla U_e) = 0")
st.divider()
bullet("Rubni uvjeti")
latex_equation(r"\bm{\hat{n}}\cdot (M_i\nabla(V_m + U_e)) = 0")
latex_equation(r"\bm{\hat{n}}\cdot(\lambda M_i\nabla U_e) = 0")
st.divider()
bullet("Skalirani jednodomenski model")
latex_equation(
    r"""\begin{aligned}
  \nabla\cdot (M^*\nabla V_m) = \frac{\partial V_m}{\partial t} +  I^*_{ion},  \qquad \forall \bm{x}\in \mathbb{H}\\
  \bm{\hat{n}}\cdot (M^*\nabla V_m) = 0,  \qquad \forall \bm{x}\in \partial\mathbb{H}
\end{aligned}"""
)
bullet("Skalirane vrijednosti")
latex_equation(
    r"M^* \coloneqq \frac{\lambda}{1+\lambda} \frac{M_i}{\chi C_m}\text{,\quad}I^*_{ion} \coloneqq \frac{I_{ion}}{C_m}"
)
