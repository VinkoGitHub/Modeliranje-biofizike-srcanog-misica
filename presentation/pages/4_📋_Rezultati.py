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
st.divider()

# Body
bullet(
    "Python, dolfinx, FEM mreže",
    "Pravokutna domena - 2352 elementa",
    "Domena presjeka srčanog mišića - 6760 elemenata",
    "Domena lijeve klijetke - 19781 element",
)
double_HTML("heart_slice.html", "heart_ventricle.html", separation=0.1)
st.divider()
st.header("Pravokutna domena")
st.divider()
double_image(
    "rectangle_fibers.jpg",
    "rectangle_initial_V_m.jpg",
    "Smjer srčanih vlakana.",
    "Početni transmembranski potencijal.",
)

st.divider()
st.subheader("Reparametrizirani FitzHugh-Nagumo model")
st.divider()
centered_image(
    "MFN_actionpotential.jpg",
    "Akcijski potencijal reparametriziranog FitzHugh-Nagumo modela.",
)
st.divider()
st.subheader("Rješenje")
st.divider()
video("rectangle_MFN.mp4")

st.divider()
st.subheader("Nobleov model")
st.divider()
centered_image("N_actionpotential.jpg", "Akcijski potencijal Nobleovog modela.")
st.divider()
st.subheader("Rješenje")
st.divider()
video("rectangle_N.mp4")

st.divider()
st.subheader("Beeler-Reuter model")
st.divider()
centered_image("BR_actionpotential.jpg", "Akcijski potencijal Beeler-Reuter modela.")
st.divider()
st.subheader("Rješenje")
st.divider()
video("rectangle_BR.mp4")
st.divider()

st.subheader("Usporedba rezultata")
st.divider()
triple_image(
    "rectangle_MFN_50.jpg",
    "rectangle_N_50.jpg",
    "rectangle_BR_50.jpg",
    caption_1="t=50 ms, MFN model",
    caption_2="t=50 ms, N model",
    caption_3="t=50 ms, BR model",
)
triple_image(
    "rectangle_MFN_100.jpg",
    "rectangle_N_100.jpg",
    "rectangle_BR_100.jpg",
    caption_1="t=100 ms, MFN model",
    caption_2="t=100 ms, N model",
    caption_3="t=100 ms, BR model",
)
triple_image(
    "rectangle_MFN_300.jpg",
    "rectangle_N_300.jpg",
    "rectangle_BR_300.jpg",
    caption_1="t=300 ms, MFN model",
    caption_2="t=300 ms, N model",
    caption_3="t=300 ms, BR model",
)
triple_image(
    "rectangle_MFN_500.jpg",
    "rectangle_N_500.jpg",
    "rectangle_BR_500.jpg",
    caption_1="t=500 ms, MFN model",
    caption_2="t=500 ms, N model",
    caption_3="t=500 ms, BR model",
)
st.divider()
st.header("Usporedba s literaturom")
st.divider()
bullet("Domena presjeka srca")
double_image(
    "comparison_fibers.jpg",
    "comparison_applied_current.jpg",
    "Smjer srčanih vlakana.",
    "Iznos i lokacija stimulacije.",
)
st.divider()
st.subheader("Usporedba")
st.divider()
double_image(
    "test25.jpg",
    "comparison_35.jpg",
    "t=25 ms, literatura",
    "t=25 ms, diplomski rad",
)
double_image(
    "test75.jpg",
    "comparison_85.jpg",
    "t=75 ms, literatura",
    "t=75 ms, diplomski rad",
)
double_image(
    "test220.jpg",
    "comparison_230.jpg",
    "t=220 ms, literatura",
    "t=220 ms, diplomski rad",
)
double_image(
    "test290.jpg",
    "comparison_300.jpg",
    "t=290 ms, literatura",
    "t=290 ms, diplomski rad",
)
st.divider()
st.subheader("Rješenje")
st.divider()
video("ischemia.mp4")
st.divider()
st.header("Ožiljno tkivo")
st.divider()
bullet("Ishemija", "Smanjena vodljivost")
centered_image(
    "ischemia_ischemia_location.jpg", "Lokacija ožiljka i normirana vodljivost."
)
double_image(
    "comparison_fibers.jpg",
    "comparison_applied_current.jpg",
    "Smjer srčanih vlakana.",
    "Iznos i lokacija stimulacije.",
)
st.divider()
st.subheader("Rješenje")
st.divider()
double_image(
    "ischemia_100.jpg",
    "comparison_100.jpg",
    "t=100 ms, signal tkiva s ožiljkom.",
    "t=100 ms, signal zdravog tkiva.",
)

st.divider()
st.subheader("Signali u točki")
st.divider()
centered_image(
    "ischemia_signals.jpg",
    "Usporedba transmembranskog potencijala zdravog tkiva i tkiva s ožiljkom u jednoj točki.",
)
st.divider()
st.header("Usporedba s jednodomenskim modelom")
st.divider()
double_image(
    "comparison_fibers.jpg",
    "comparison_applied_current.jpg",
    "Smjer srčanih vlakana.",
    "Iznos i lokacija stimulacije.",
)

st.divider()
bullet("Minimizacija izraza")
latex_equation(
    r"""\begin{bmatrix}
           \sigma_{e}^l \\
           \sigma_{e}^t \\
           \sigma_{e}^n
         \end{bmatrix} 
         -\lambda
         \begin{bmatrix}
           \sigma_{i}^l \\
           \sigma_{i}^t \\
           \sigma_{i}^n
         \end{bmatrix}"""
)

st.divider()
st.subheader("Signali u točki")
st.divider()
centered_image(
    "monodomain_signals.jpg",
    "Usporedba transmembranskog potencijala jednodomenskog i dvodomenskog modela u jednoj točki.",
)
st.divider()
st.header("Rješenje na 3D mreži")
st.divider()
bullet("Model lijeve klijetke")
double_image(
    "ventricle_fibers.jpg",
    "ventricle_initial_V_m.png",
    "Smjer srčanih vlakana.",
    "Početni transmembranski potencijal.",
)
st.divider()
st.subheader("Rješenje")
st.divider()
bullet("Modificirani FitzHugh-Nagumo ionski model")
double_image("ventricle_500.png", "ventricle_1000.png", "t=500 ms", "t=1000 ms")
double_image("ventricle_1500.png", "ventricle_2000.png", "t=1500 ms", "t=2000 ms")
