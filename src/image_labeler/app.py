import sys
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import image_manager
import numpy as np
import cv2
import os

st.set_page_config(page_title="Image Labeler", layout="wide")

@st.cache_data
def load_and_process_image(path: str):
    arr = image_manager.create_max_projection(path, image_index=1)[0]

    red_arr = np.stack([arr, np.zeros_like(arr), np.zeros_like(arr)], axis=-1)
    red_image = Image.fromarray(red_arr.astype(np.uint8))

    return red_image, arr.shape[0], arr.shape[1]


image_name = sys.argv[1]
image_folder = image_manager.get_image_folder()
lif_path = os.path.join(image_folder, image_name)

red_image, height, width = load_and_process_image(lif_path)

st.sidebar.header("🧰 Drawing Tools")

tool = st.sidebar.radio("Select tool", ["Stroke", "Lasso"])

stroke_color = st.sidebar.color_picker("Stroke color", "#0000FF")
fill_color = st.sidebar.color_picker("Fill color (for lasso)", "#FF0000") + "33"
stroke_width = st.sidebar.slider("Stroke width", 1, 30, 5)
update_streamlit = st.sidebar.checkbox("Live update", True)
drawing_mode = "freedraw" if tool == "Stroke" else "polygon"

#Image
canvas_result = st_canvas(
    fill_color=fill_color,
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_image=red_image,
    update_streamlit=update_streamlit,
    height=height,
    width=width,
    drawing_mode=drawing_mode,
    key="canvas",
)

if canvas_result.image_data is not None:
    st.image(canvas_result.image_data, use_container_width=True)