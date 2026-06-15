import sys
from pathlib import Path
file_path = Path(__file__).resolve()
project_root = file_path.parent.parent

sys.path.append(str(project_root))
from fastapi import FastAPI, Body
import pandas as pd
from src.predict import load_model, predict



""" POST /predict
input: raw house features
output: predicted price """


app = FastAPI()

@app.get("/")
def read_root():
    return{"message": "House Price Predictor API is running!"}


@app.post("/predict")
async def predict_endpoint(data: dict = Body(...)):
    X = pd.DataFrame([data])
    model = load_model()
    prediction = predict(model, X)
    final_result = int(prediction[0])
    return {"prediction": final_result}
# receive JSON
# convert JSON to pandas DataFrame
# load model
# predict
# return predicted_price