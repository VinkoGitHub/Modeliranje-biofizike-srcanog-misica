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
text(
    r"""Srce je mišićni organ kod ljudi i većine životinja koji pumpa krv kroz 
    krvne žile i time tijelo opskrbljuje kisikom. Funkcija srca u tom je smislu 
    ključna za rad organizma u cijelosti jer srce šalje kisik i hranjive tvari do 
    individualnih stanica a odnosi ugljikov dioksid do pluća kako bi ga tijelo moglo izbaciti.\\
    Kod ljudi, srce je otprilike veličine zatvorene šake i nalazi se u prsima između dva 
    plućna krila. Zbog njegove važnosti, veliki znanstveni, financijski i medicinski napori 
    uloženi su u razumijevanje ovog organa. Dovoljno je navesti da su prema Svjetskoj 
    zdravstvenoj organizaciji bolesti srca (ishemije srca) primaran uzrok smrti u 2019. 
    godini\cite{WHO-statistika}."""
)

image("presentation/data/slika.png")
