from fastapi import FastAPI
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO
import numpy as np
from ultralytics import YOLO
import cv2
import streamlit as st

# Initialize FastAPI app
app = FastAPI()
@app.get("/")
def read_root():
    return {}

def load_image_into_numpy_array(data):
    return np.array(Image.open(BytesIO(data)))

@app.post("/images")
async def read_root(file: UploadFile = File(...)):
    image = load_image_into_numpy_array(await file.read())
    return {"Hello": "World"}

@app.get("/model")
def load_model(model_path):
    model = YOLO(model_path)
    return model

@app.post("/predict")
def _display_detected_frames(conf, model, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A trained YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # PASS TO API
    # Predict the objects in the image using the YOLOv8 model
    res = model.predict(image, conf=conf)

    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


@app.post("/webcam")
def play_webcam(conf, model):
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
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))













# Define preprocessing function
# def preprocess_data(db):
#      # Implement your preprocessing steps here
#      # For example, scaling or encoding data
#      preprocess_features = data  # Placeholder for actual preprocessing
#      return preprocess_features

# @app.get("/predict")
# def predict(labels: dict):
#     app.state.model = load_model("CNN_vanilla2024-03-08 02_21_51.593838")
#     model = app.state.model
#     assert model is not None

#     # Extract the features from the incoming data
#     features = labels['label']

#     # Make prediction directly using the model
#     prediction = model.predict(features)

#     # Generate class labels
#     class_labels = ['anger','contempt','disgust','fear','happy','neutral','sad','surprise']  # Example labels



#     X_processed = preprocess_features(X_pred)
#     y_pred = model.predict(X_processed)
#     # Return prediction and class label
#     return {"prediction": prediction.tolist(), "class_label": class_labels[np.argmax(prediction)]}




    # Load model
    # Preporcess data
    # Predict
    # Generate the class label
    # Load the model and weights of me and edi
    # change np array in json file

# class NumpyArrayEncoder(JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, numpy.ndarray):
#             return obj.tolist()
#         return JSONEncoder.default(self, obj)
# numpyArray = #numpy.array([[11 ,22, 33], [44, 55, 66], [77, 88, 99]])
# # Serialization
# numpyData = {"Array": numpyArray}
# with open("numpyData.json", "w") as write_file:
#     json.dump(numpyData, write_file, cls=NumpyArrayEncoder)
