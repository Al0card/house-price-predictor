import sys
from pathlib import Path
file_path = Path(__file__).resolve()
project_root = file_path.parent.parent

sys.path.append(str(project_root))
from fastapi import FastAPI
import pandas as pd
from src.predict import load_model, predict
from pydantic import BaseModel




app = FastAPI()
model = load_model()


class HouseFeatures(BaseModel):
    input_data: dict




@app.get("/")
def read_root():
    return{"message": "House Price Predictor API is running!"}


@app.post("/predict")
async def predict_endpoint(data: HouseFeatures):
    input_data =  data.input_data
    X = pd.DataFrame([input_data])
    
    # Retrieve the model from request.state
    prediction = predict(model, X)
    final_result = int(prediction[0])
    return {
        "predicted_price": final_result,
        "currency": "USD",
        "model": "ridge_regression"
    }
