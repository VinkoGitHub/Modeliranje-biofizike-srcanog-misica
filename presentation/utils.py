import streamlit as st


def HTML(link: str, height: int = 600):
    HtmlFile = open(link, "r", encoding="utf-8")
    source_code = HtmlFile.read()
    st.components.v1.html(source_code, height=height)


def video(link: str):
    video_file = open(link, "rb")
    video_bytes = video_file.read()
    st.video(video_bytes)


def centered_image(link: str, caption: str | None = None, width: int = 1.5):
    col1, col2, col3 = st.columns([1, width, 1])
    with col1:
        st.text("")
    with col2:
        st.image(link, use_column_width=True, caption=caption)
    with col3:
        st.text("")


def image(link: str, width: int | None = None, caption: str | None = None):
    st.image(link, caption=caption, width=width)


def text(html_string: str, font_size: int = 17):
    st.markdown(
        '<span style="font-family: computer-modern; font-size: '
        + str(font_size)
        + 'pt;">'
        + html_string
        + "</span>",
        unsafe_allow_html=True,
    )


def bullet(*html_strings: str, font_size: int = 17):
    items = [
        '<li style="font-family: serif; font-size:'
        + str(font_size)
        + 'pt;">'
        + string
        + "</li>"
        for string in html_strings
    ]
    ils = ""
    for it in items:
        ils += it
    st.markdown(
        "<ul>" + ils + "</ul>",
        unsafe_allow_html=True,
    )


def latex_equation(string: str, font_size: int = "Large"):
    st.latex("\\" + font_size + " " + string)
