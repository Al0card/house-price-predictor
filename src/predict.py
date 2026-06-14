import joblib

MODEL_PATH = "../models/ridge_pipeline.joblib"
def load_model():
    return joblib.load(MODEL_PATH)


def predict(model, X):
    return model.predict(X)