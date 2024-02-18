import streamlit as st
from presentation.utils import *

# Configuring the app
st.set_page_config(
    page_title="Diplomski rad",
    page_icon="🎓",
    layout="wide",
)

# Title
st.title("Zaključak")
st.divider()

# Body
bullet(
    "Dvodomenski i jednodomenski model rješivi su numerički",
    "Dvodomenski model daje smislene rezultate za više ionskih modela",
    "Dvodomenski model može simulirati ožiljno tkivo",
    "Jednodomenski model dobra je aproksimacija dvodomenskog modela",
    "Dvodomenski model dobra je početna osnova za simulacije propagacije srčanih signala",
)
