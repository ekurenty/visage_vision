import streamlit as st
# Local Modules
import functions
import requests
import numpy as np

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
# try:
#     model = functions.load_model(model_path)
# except Exception as ex:
#     st.error(f"Unable to load model. Check the specified path: {model_path}")
#     st.error(ex)


######################
# params = {'confidence': confidence}

# url = 'https://api-jx57tfny7a-km.a.run.app/images'
# files = {'file': open('/Users/ekurenty/Desktop/Team Photo/edi_pred.png', 'rb')}
# response = requests.post(url, files=files)

# prediction = eval(f'np.array({response.json()["Prediction"]})')


# st.image(prediction)
functions.play_webcam()   #confidence, model

####################



# Image.fromarray(result[0].plot()[:,:,::-1])
