from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from src.inference import InferenceModel, Prediction

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app):
    # Load once at startup and reuse the model for all requests.
    app.state.model = InferenceModel()
    yield


app = FastAPI(title="Brain MRI Classifier", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "classes": app.state.model.classes}


@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    # Read one extra byte to detect uploads above the 10 MB limit.
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large")
    # Decode the actual image content rather than relying on its extension.
    try:
        with Image.open(BytesIO(data)) as image:
            image = image.convert("RGB")
    except OSError:
        raise HTTPException(status_code=400, detail="Invalid or unreadable image")
    return app.state.model.predict(image)
