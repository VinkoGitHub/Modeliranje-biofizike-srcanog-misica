import streamlit as st
from presentation.utils import *

# Configuring the app
st.set_page_config(
    page_title="Diplomski rad",
    page_icon="🎓",
    layout="wide",
)

# Title
st.title("Matematika")

# Body
st.divider()
bullet(
    "1 do 10 milijardi stanica<sup>[cite]</sup>",
    "Homogenizacija<sup>[cite]</sup>",
    "Dvodomenski model",
    "Usrednjene veličine",
)
st.divider()
centered_image(
    "presentation/data/bidomain-cells.png",
    width=4,
    caption="https://carmen.gitlabpages.inria.fr/ceps/volumeFraction.html",
)
st.divider()
latex_equation(r"\bar{f}\rightarrow f,\quad \mathbb{H},\quad \mathbb{\partial H}")
st.divider()
bullet(
    "Intracelularna domena",
    "Ekstracelularna domena",
    "Rascjepni kanali",
    "Maxwellova jednadžba",
)
latex_equation(r'\nabla\times\bm{E} = -\frac{\partial\bm{B}}{\partial t}')
st.divider()
bullet('Kvazistacionarnost')
latex_equation(r'\nabla\times\bm{E} = 0 \quad\Rightarrow\quad \bm{E} = -\nabla U')
latex_equation(r'\bm{J} = M\bm{E} \quad\Rightarrow\quad \bm{J} = -M\nabla U')
st.divider()
bullet('Intracelularni potencijal', 'Ekstracelularni potencijal')
latex_equation(r'U_i,\quad U_e,\quad V_m \coloneqq U_i - U_e')
st.divider()
