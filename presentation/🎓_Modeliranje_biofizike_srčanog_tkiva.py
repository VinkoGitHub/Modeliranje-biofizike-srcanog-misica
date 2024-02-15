import streamlit as st
import datetime
import pandas as pd
import numpy as np

# Configuring the app
st.set_page_config(
    page_title="Diplomski rad",
    page_icon="🎓",
    layout="centered",
)

st.markdown(
    """<h1 style='text-align: center; color: white;'>Modeliranje biofizike srčanog mišića</h1>""",
    unsafe_allow_html=True,
)
st.divider()
st.markdown(
    """<h3 style='text-align: center; color: white;'>Vinko Dragušica</h3>
    <h5 style='text-align: center; color: white;'>Mentor: doc.dr.sc. Andrej Novak</h5>""",
    unsafe_allow_html=True,
)
st.divider()
st.markdown("#\n\n")
col1, col2, col3 = st.columns(3)

with col1:
    st.image("presentation/data/figures/PMF.png", use_column_width=True)

with col2:
    st.markdown("#\n#")
    st.markdown(
        """<h5 style='text-align: center; color: white;'>PMF Zagreb</h5>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """<h5 style='text-align: center; color: white;'>"""
        + str(datetime.date.today().strftime("%d.%m.%Y."))
        + """</h5>""",
        unsafe_allow_html=True,
    )
with col3:
    st.image("presentation/data/figures/UNIZG.png", use_column_width=True)
