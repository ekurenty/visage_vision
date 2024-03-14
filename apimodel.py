from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
import numpy as np
from ultralytics import YOLO

# Initialize FastAPI app
app = FastAPI()
@app.get("/")
def read_root():
    return {"bonjour Harris"}

def load_image_into_numpy_array(data):
    return np.array(Image.open(BytesIO(data)))

model_path = 'best_face.pt'
@app.post("/model")
def load_model(model_path):
    model = YOLO(model_path)
    return model

@app.post("/images")
async def read_root(file: UploadFile = File(...)):
    image = load_image_into_numpy_array(await file.read())
    model = load_model(model_path)
    res_plotted = model.predict(image)[0].plot()
    print(np.array2string(res_plotted))
    return {"Prediction": np.array2string(res_plotted) }
