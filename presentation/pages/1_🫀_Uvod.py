import streamlit as st
from presentation.utils import *

# Configuring the app
st.set_page_config(
    page_title="Diplomski rad",
    page_icon="🎓",
    layout="wide",
)

# Title
st.title("Uvod")

# Body
st.divider()
bullet(
    "2.5 milijardi otkucaja<sup>[1]</sup>",
    "200 000 tona krvi<sup>[1]</sup>",
    "160 000 kilometara<sup>[1]</sup>",
    "Ishemije srca - primaran uzrok smrti 2019. godine<sup>[2]</sup>",
)
st.divider()
centered_image(
    "presentation/data/figures/provodni-sustav.jpg", "Provodni sustav srčanog mišića."
)
st.divider()
bullet(
    "Elektrokardiogram - EKG",
    "W. Einthoven - 1901. godine",
    "Nobelova nagrada",
    "10 elektroda i 12 odvoda",
)
st.divider()
centered_image(
    "presentation/data/figures/EKG-slika.jpg",
    "Primjer elektrokardiograma (preuzeto iz [3]).",
    4,
)
st.divider()
bullet("Akcijski potencijal", "Polarizacija", "Depolarizacija", "Repolarizacija")
st.divider()
centered_image(
    "presentation/data/figures/akcijski-potencijal-slika.jpg",
    """Lijevo: izmjereni akcijski potencijal.
    Desno: modelirani akcijski potencijal(preuzeto iz [4]).""",
    4,
)
st.divider()
bullet(
    "[forward problem of electrocardilogy]",
    "Ionski modeli (Hodgkin-Huxley, FitzHugh-Nagumo, Noble, Beeler-Reuter, ...)",
    "Model dinamike (pasivni vodič, jednodomenski, dvodomensi, ...)",
    "Domena kao mreža (mesh)",
)
st.divider()
centered_image(
    "presentation/data/figures/akcijski_potencijali.jpg", width=4, caption="caption"
)
st.divider()
video()
