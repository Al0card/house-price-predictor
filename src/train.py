"""
File responsabilities
train model
evaluate RMSE
run experiment
"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import numpy as np


def run_experiment(X_train, X_val, y_train, y_val):
    y_train_log = np.log(y_train)
def train_model(x, y):
    model = LinearRegression()
    model.fit(x, y)
    return model
def predict(x, model):
    y = model.predict(x)
    return y
def evaluate_rmse(y_val, y_predict):
    rmse = root_mean_squared_error(y_val, y_predict)
    return rmse