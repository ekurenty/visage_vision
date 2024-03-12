import streamlit as st
# Local Modules
import functions
import requests

# Setting page layout
st.set_page_config(
    page_title="Facial Expression Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("VISage VISion")

# Sidebar
st.sidebar.header("Confidence Level")

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

#model_path = 'best.pt'
model_path = 'best_face.pt'

# Load the model
try:
    model = functions.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

functions.play_webcam(confidence, model)

#params = {'confidence': confidence}


# url = 'http://localhost:8501/images'
# response = requests.get(url, params=params)

# prediction = response.json()

# pred = prediction['pred']

# st.header(f'This person is: {pred}')


# result = model.predict('20240209_LeWagon__0021.jpg')
# Image.fromarray(result[0].plot()[:,:,::-1])
