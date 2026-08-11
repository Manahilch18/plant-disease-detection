"""
Streamlit Grad-CAM component.
"""

import streamlit as st


def display_gradcam(
    original_image,
    gradcam_image,
):
    """
    Display Grad-CAM visualization in Streamlit.
    """
    col1, col2 = st.columns(2)

    with col1:

        st.image(
            original_image,
            caption="Original Image",
            use_container_width=True,
        )

    with col2:

        st.image(
            gradcam_image,
            caption="Grad-CAM Visualization",
            use_container_width=True,
        )