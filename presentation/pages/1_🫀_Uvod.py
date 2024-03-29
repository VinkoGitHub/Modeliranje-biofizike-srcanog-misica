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
    "Ishemijske bolesti srca - primaran uzrok smrti 2019. godine<sup>[2]</sup>",
)
st.divider()
centered_image(
    "provodni-sustav.jpg", "Provodni sustav srčanog mišića (preuzeto iz [3])."
)
st.divider()
bullet(
    "Akcijski potencijal",
    "Natrijevi i kalijevi kanali",
    "Depolarizacija",
    "Repolarizacija",
)
st.divider()
centered_image(
    "akcijski-potencijal-slika.jpg",
    """Lijevo: izmjereni akcijski potencijal.
    Desno: modelirani akcijski potencijal(preuzeto iz [4]).""",
    4,
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
    "EKG-slika.jpg",
    "Primjer elektrokardiograma (preuzeto iz [5]).",
    4,
)
st.divider()
bullet(
    "Problem matematičke elektrofiziologije",
    "Ionski modeli (Hodgkin-Huxley, FitzHugh-Nagumo, Noble, Beeler-Reuter, ...)",
    "Model dinamike (pasivni vodič, jednodomenski, dvodomenski, ...)",
    "Domena kao mreža (mesh)",
)
st.divider()
centered_image(
    "akcijski_potencijali.jpg",
    width=4,
    caption="Ilustracija akcijskih potencijala pojedinih dijelova srčanog mišića (preuzeto iz [6]).",
)
st.divider()
st.subheader("Kardiomiocit")
st.divider()
video(
    "presentation/data/cardiomyocite.mp4",
    caption="Promjene na kardiomiocitu uslijed djelovanja akcijskog potencijala (preuzeto iz [7]).",
    relative_path=False,
)
