import streamlit as st


def double_HTML(
    link_1: str,
    link_2: str,
    height_1: float = 600,
    height_2: float = 600,
    separation: float = 1,
    relative_path: bool = True,
):
    if relative_path:
        link_1 = f"presentation/data/html/{link_1}"
        link_2 = f"presentation/data/html/{link_2}"
    HtmlFile_1 = open(link_1, "r", encoding="utf-8")
    HtmlFile_2 = open(link_2, "r", encoding="utf-8")
    source_code_1 = HtmlFile_1.read()
    source_code_2 = HtmlFile_2.read()
    col1, col2, col3 = st.columns([1, separation, 1])
    with col1:
        st.components.v1.html(source_code_1, height=height_1)
    with col2:
        st.text("")
    with col3:
        st.components.v1.html(source_code_2, height=height_2)


def video(
    link: str,
    width: float = 2.5,
    caption: str | None = None,
    relative_path: bool = True,
):
    if relative_path:
        link = f"animations/{link}"
    video_file = open(link, "rb")
    video_bytes = video_file.read()
    col1, col2, col3 = st.columns([1, width, 1])
    with col1:
        st.text("")
    with col2:
        st.video(video_bytes)
        if caption is not None:
            st.caption(
                f"<center>{caption}</center>",
                unsafe_allow_html=True,
            )

    with col3:
        st.text("")


def centered_image(
    link: str,
    caption: str | None = None,
    relative_path: bool = True,
    width: float = 2.5,
):
    if relative_path:
        link = f"presentation/data/figures/{link}"
    col1, col2, col3 = st.columns([1, width, 1])
    with col1:
        st.text("")
    with col2:
        st.image(link, use_column_width=True, caption=caption)
    with col3:
        st.text("")


def double_image(
    link_1: str,
    link_2: str,
    caption_1: str | None = None,
    caption_2: str | None = None,
    width_1: float = 5,
    width_2: float = 5,
    relative_path: bool = True,
):
    if relative_path:
        link_1 = f"presentation/data/figures/{link_1}"
        link_2 = f"presentation/data/figures/{link_2}"
    col1, col2, col3 = st.columns([width_1, 1, width_2])
    with col1:
        st.image(
            link_1,
            use_column_width=True,
            caption=caption_1,
        )
    with col2:
        st.text("")
    with col3:
        st.image(
            link_2,
            use_column_width=True,
            caption=caption_2,
        )


def triple_image(
    link_1: str,
    link_2: str,
    link_3: str,
    width_1: float = 1,
    width_2: float = 1,
    width_3: float = 1,
    caption_1: str | None = None,
    caption_2: str | None = None,
    caption_3: str | None = None,
    relative_path: bool = True,
):
    if relative_path:
        link_1 = f"presentation/data/figures/{link_1}"
        link_2 = f"presentation/data/figures/{link_2}"
        link_3 = f"presentation/data/figures/{link_3}"
    col1, col2, col3 = st.columns([width_1, width_2, width_3])
    with col1:
        st.image(
            link_1,
            use_column_width=True,
            caption=caption_1,
        )
    with col2:
        st.image(
            link_2,
            use_column_width=True,
            caption=caption_2,
        )
    with col3:
        st.image(
            link_3,
            use_column_width=True,
            caption=caption_3,
        )


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
