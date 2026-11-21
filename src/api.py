"""
AIPI 510: Module 3 Project: Data/ML Engineering
- src/api.py
Hasan Al-Quaid
=================================================
FastAPI app that serves the trained model:
  GET  /health   -> checks the API is running
  POST /predict  -> takes one diamond as JSON, returns the predicted price

The model is loaded once when the API starts:
  - On Cloud Run: from the bucket (MODEL_URI = gs://diamonds-hasan-aipi510/models/model.joblib)
  - Locally: from models/model.joblib (made by train.py)

Run locally (from the repository's main folder):
    uvicorn src.api:app --port 8080
Then open http://localhost:8080/docs to try it
"""
import os
import tempfile
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI
from google.cloud import storage
from pydantic import BaseModel, Field

# global values
# MODEL_URI is set on Cloud Run; locally it falls back to the model train.py saved
MODEL_URI = os.getenv("MODEL_URI", "models/model.joblib")
PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "aipi510-diamonds-hasan-mod3")


def load_model(uri):
    """
    Load the trained model from the bucket (gs://...) or from a local file

    Parameters:
        uri (str): gs://bucket/path/model.joblib or a local file path

    Returns:
        Pipeline: The trained pipeline (encoding + random forest)
    """
    print("-------------------------- Start: Load model --------------------------")
    if uri.startswith("gs://"):
        # split "gs://bucket/models/model.joblib" into the bucket and the file path
        bucket_name, blob_path = uri.replace("gs://", "").split("/", 1)

        # download the model to a temporary file (works on Windows and Linux)
        local_file = os.path.join(tempfile.gettempdir(), "model.joblib")
        client = storage.Client(project=PROJECT)
        blob = client.bucket(bucket_name).blob(blob_path)
        blob.download_to_filename(local_file)
        print(f"Downloaded {uri}")
        uri = local_file

    model = joblib.load(uri)
    print(f"Loaded model from {uri}")
    print("-------------------------- End: Load model ----------------------------\n")
    return model


class Diamond(BaseModel):
    """
    One diamond sent to /predict. FastAPI checks every value,
    and anything invalid (like cut = "Amazing") is rejected with a 422 error.
    The allowed values match the ordinal lists in config.yaml.
    """
    carat: float = Field(gt=0, le=6)                    # cleaned data goes up to about 5 carats
    cut: Literal["Fair", "Good", "Very Good", "Premium", "Ideal"]
    color: Literal["J", "I", "H", "G", "F", "E", "D"]
    clarity: Literal["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
    depth: float = Field(default=61.8, ge=40, le=80)    # default = typical depth in the data
    table: float = Field(default=57.0, ge=40, le=100)   # default = typical table in the data


# load the model once when the API starts, not on every request
model = load_model(MODEL_URI)
app = FastAPI(title="Diamond Price API")


@app.get("/health")
def health():
    """Check the API is running"""
    return {"status": "ok"}


@app.post("/predict")
def predict(diamond: Diamond):
    """
    Predict the price of one diamond

    Parameters:
        diamond (Diamond): The diamond's carat, cut, color, clarity, depth, and table

    Returns:
        dict: {"predicted_price": price in USD}
    """
    # the pipeline expects a table with the same column names it was trained on
    row = pd.DataFrame([diamond.model_dump()])
    price = model.predict(row)[0]
    return {"predicted_price": round(float(price), 2)}
