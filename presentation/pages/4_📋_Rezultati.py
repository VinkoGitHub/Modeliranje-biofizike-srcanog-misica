import streamlit as st
from presentation.utils import *

# Configuring the app
st.set_page_config(
    page_title="Diplomski rad",
    page_icon="🎓",
    layout="wide",
)

# Title
st.title("Rezultati")

# Body
bullet(
    "FEM mreže",
    "Pravokutna domena - 2352 elementa",
    "Domena presjeka srčanog mišića - 6760 elemenata",
    "Domena lijeve klijetke - 19781 element",
)
double_image("slice_mesh.jpg", "ventricle_mesh.jpg", 10, 11.8, "caption_1", "caption_2")
st.divider()
st.header("Pravokutna domena")
st.divider()
double_image(
    "rectangle_fibers.jpg",
    "rectangle_initial_V_m.jpg",
    10,
    10,
    "caption_1",
    "caption_2",
)

st.divider()
st.subheader("Reparametrizirani FitzHugh-Nagumo model")
st.divider()
centered_image("MFN_actionpotential.jpg", 5)
bullet("Rješenje")
video("rectangle_MFN.mp4")

st.divider()
st.subheader("Nobleov model")
centered_image("N_actionpotential.jpg", 5)
st.divider()
bullet("Rješenje")
video("rectangle_N.mp4")

st.divider()
st.subheader("Beeler-Reuter model")
st.divider()
centered_image("BR_actionpotential.jpg", 5)
bullet("Rješenje")
video("rectangle_BR.mp4")
st.divider()

st.subheader("Usporedba rezultata")
st.divider()
triple_image(
    "rectangle_MFN_50.jpg",
    "rectangle_N_50.jpg",
    "rectangle_BR_50.jpg",
    caption_1="caption_1",
    caption_2="caption_2",
    caption_3="caption_3",
)
triple_image(
    "rectangle_MFN_100.jpg",
    "rectangle_N_100.jpg",
    "rectangle_BR_100.jpg",
    caption_1="caption_1",
    caption_2="caption_2",
    caption_3="caption_3",
)
triple_image(
    "rectangle_MFN_300.jpg",
    "rectangle_N_300.jpg",
    "rectangle_BR_300.jpg",
    caption_1="caption_1",
    caption_2="caption_2",
    caption_3="caption_3",
)
triple_image(
    "rectangle_MFN_500.jpg",
    "rectangle_N_500.jpg",
    "rectangle_BR_500.jpg",
    caption_1="caption_1",
    caption_2="caption_2",
    caption_3="caption_3",
)
st.divider()

bullet()
centered_image("/")
centered_image("presentation/data/images/figures/")
centered_image("presentation/data/images/figures/")
centered_image("presentation/data/images/figures/")
centered_image("presentation/data/images/figures/")
centered_image("presentation/data/images/figures/")
centered_image("presentation/data/images/figures/")
