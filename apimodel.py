from fastapi import FastAPI
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
from ultralytics import YOLO

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
    #r = model.predict(image)
    r = model.predict(image)[0].names[(model.predict(image)[0].boxes).cls[0].item()]

    return {"This person is": r}
