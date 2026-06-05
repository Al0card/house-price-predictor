"""
File responsabilities
train model
"""

from sklearn.linear_model import LinearRegression




def train_model(x, y):
    model = LinearRegression()
    model.fit(x, y)
    return model
