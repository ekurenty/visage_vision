
import streamlit as st
# Local Modules
import functions

# Setting page layout
st.set_page_config(
    page_title="Facial Expression Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Facial Expression Detection using YOLOv8")

# Sidebar
st.sidebar.header("Confidence Level")

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

model_path = 'best.pt'

# Load the model
try:
    model = functions.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

functions.play_webcam(confidence, model)
