import joblib
import numpy as np
from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parent.parent
MODEL_PATH =PROJECT_PATH / "models" / "ridge_pipeline.joblib"
def load_model():
    return joblib.load(MODEL_PATH)


def predict(model, X):
    return np.exp(model.predict(X))