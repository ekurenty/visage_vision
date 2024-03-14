from ultralytics import YOLO
from PIL import Image
import cv2
import io
import numpy as np
import requests
import streamlit as st

URL_API = 'https://api-jx57tfny7a-km.a.run.app/images'
def load_model(model_path):
    model = YOLO(model_path)
    return model


def _display_detected_frames(st_frame, image):  #conf, model
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A trained YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (240, int(240*(9/16))))

    # PASS TO API
    # Predict the objects in the image using the YOLOv8 model

    bytes_io = io.BytesIO()
    Image.fromarray(image).save(bytes_io, format="png")

    files = {'file': bytes_io.getvalue()}
    response = requests.post(URL_API, files=files)


    #res = model.predict(image, conf=conf,iou=0.6,)

    # Plot the detected objects on the video frame

    #res_plotted = res[0].plot()

    res_plotted = eval(f'np.array({response.json()["Prediction"]})')


    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )



def play_webcam():   #conf, model)
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.
    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    """
    source_webcam = 0
    if st.sidebar.button('Start'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(st_frame,
                                             image,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
