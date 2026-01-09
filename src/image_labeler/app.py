import sys
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import image_manager
import numpy as np
import cv2
import os
import argparse

#Page config
st.set_page_config(page_title="Image Labeler", layout="wide")

@st.cache_data
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image",
        type=str,
        nargs="?",
        default=None,
        help="Path to image"
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional path to label PNG"
    )
    return parser.parse_known_args()[0]

def label_to_rgba(label, color=(0, 0, 255), alpha=120):
    h, w = label.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    mask = label > 0
    rgba[mask, :3] = color
    rgba[mask, 3] = alpha

    return Image.fromarray(rgba)

@st.cache_data
def load_and_process_image(path: str):
    arr = image_manager.create_max_projection(path, image_index=1)[0]

    #Deal with st caching
    arr = np.array(arr, copy=True)

    grey_arr = np.stack([arr, arr, arr], axis=-1)
    grey_image = Image.fromarray(grey_arr.astype(np.uint8))
    image = grey_image.convert("RGBA")

    return np.array(image), arr.shape[0], arr.shape[1]

@st.cache_data
def load_label(file_name: str):
    #If existing image then load it, otherwise create a new blank one
    if label_name is not None:
        label_path = os.path.join(label_folder, str(file_name))
        label = image_manager.load_label_png(label_path)
    else:
        label = np.zeros((height, width), dtype=np.uint8)
    return label

#Get cmd arguments
args = parse_args()

image_name = args.image
label_name = args.label

#Get folders
image_folder = image_manager.get_image_folder()
label_folder = image_manager.get_label_folder()

#Get image
lif_path = os.path.join(image_folder, str(image_name))
base_image_arr, height, width = load_and_process_image(lif_path)
base_image = Image.fromarray(base_image_arr)

#Load or initialize label
if "label" not in st.session_state:
    st.session_state.label = load_label(label_name)

#Setup image stack
overlay = label_to_rgba(st.session_state.label)
composite = Image.alpha_composite(base_image, overlay)

#Overlay
st.sidebar.header("🧰 Drawing Tools")
tool = st.sidebar.radio("Select tool", ["Stroke", "Lasso"])
stroke_width = st.sidebar.slider("Stroke width", 1, 30, 5)
action = st.sidebar.radio("Select action", ["Add", "Remove"])

if action == "Add":
    stroke_color = "#0000FF"
else:
    stroke_color = "#FF0000"

# Will save the label to the file with the name <image_name>_label_.png in the label folder
if st.sidebar.button("Save Label (Beware will overwrite)"):
    if "label" not in st.session_state:
        st.warning("No label to save.")
    else:
        label_file_name = image_name.split(".")[0] + "_label" + ".png"
        label_path = os.path.join(label_folder, label_file_name)
        Image.fromarray(st.session_state["label"]).save(label_path)
        st.sidebar.write("Label saved")
update_streamlit = st.sidebar.checkbox("Live update", True)
drawing_mode = "freedraw" if tool == "Stroke" else "polygon"

#Initialize canvas counter
if "canvas_id" not in st.session_state:
    st.session_state.canvas_id = 0

#Canvas
canvas_result = st_canvas(
    fill_color=stroke_color,
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_image=composite,
    update_streamlit=update_streamlit,
    height=height,
    width=width,
    drawing_mode=drawing_mode,
    key=f"canvas_{st.session_state.canvas_id}"
)

#Update label
if canvas_result.image_data is not None:
    img = canvas_result.image_data

    # Get drawn
    alpha = img[:, :, 3]

    if np.any(alpha > 0):
        drawn_mask = alpha > 0

        if action == "Add":
            st.session_state.label[drawn_mask] = 1
        else:
            st.session_state.label[drawn_mask] = 0

        st.session_state.canvas_id += 1
        st.rerun()