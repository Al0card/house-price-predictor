"""
File responsibilities 
Load the csv
return a dataframe
"""
import pandas as pd

def load_data(path):
    df = pd.read_csv("../data/raw/train.csv")
    return  df