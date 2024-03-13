from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
#import requests

# Initialize FastAPI app
app = FastAPI()
@app.get("/")
def read_root():
    return {}

def load_image_into_numpy_array(data):
    return np.array(Image.open(BytesIO(data)))

model_path = 'best.pt'
@app.post("/model")
def load_model(model_path):
    model = YOLO(model_path)
    return model

@app.post("/images")
async def read_root(file: UploadFile = File(...)):
    image = load_image_into_numpy_array(await file.read())
    model = load_model(model_path)
    r = model.predict(image)
    #r = model.predict(image)[0].names[(model.predict(image)[0].boxes).cls[0].item()]
    return {"label": "world"}


# params = {'confidence': confidence}
#url = 'https:/'
#response = requests.get(url)

# prediction = response.json()

# pred = prediction['pred']

# st.header(f'This person is: {pred}')

# @app.post("/create_file")
# async def create_file(file: UploadFile = File(...)):
#       file2store = await file.read()
      # some code to store the BytesIO(file2store) to the other database


# result = model.predict('20240209_LeWagon__0021.jpg')
# Image.fromarray(result[0].plot()[:,:,::-1])
