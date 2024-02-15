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
    "2.5 milijardi otkucaja<sup>[Sundnes]</sup>",
    "200 000 tona krvi<sup>[Sundnes]</sup>",
    "160 000 kilometara<sup>[Sundnes]</sup>",
    "Ishemije srca - primaran uzrok smrti 2019. godine<sup>[WHO-statistika]</sup>",
)
st.divider()
centered_image("provodni-sustav.jpg", width=1.5, caption="caption")
st.divider()
bullet(
    "Elektrokardiogram - EKG",
    "W. Einthoven - 1901. godine",
    "Nobelova nagrada",
    "12 odvoda",
)
st.divider()
centered_image("EKG-slika.jpg", width=5, caption="caption")
st.divider()
bullet("Akcijski potencijal", "Polarizacija", "Depolarizacija", "Repolarizacija")
st.divider()
centered_image(
    "akcijski-potencijal-slika.jpg", width=5, caption="caption"
)
st.divider()
bullet(
    "[forward problem of electrocardilogy]",
    "Ionski modeli (Hodgkin-Huxley, FitzHugh-Nagumo, Noble, Beeler-Reuter, ...)",
    "Model dinamike (pasivni vodič, jednodomenski, dvodomensi, ...)",
    "Domena kao mreža (mesh)",
)
st.divider()
centered_image("akcijski_potencijali.jpg", width=4, caption="caption")