"""
File responsabilities
missing values
X / y split
feature type selection
preprocessing / encoding
"""
import pandas as pd
def preprocess_data(df):
   df_clean = fill_missing_values(df)
   df_encoded = encode_features(df_clean)
   x, y = split_features_target(df_encoded) 
   return x, y 

def fill_missing_values(df):
    cat_nulls_columns = [
    "PoolQC",
    "MiscFeature",
    "Alley",
    "Fence",
    "MasVnrType",
    "FireplaceQu",
    "GarageQual",
    "GarageFinish",
    "GarageType",
    "GarageCond",
    "BsmtFinType2",
    "BsmtExposure",
    "BsmtCond",
    "BsmtQual",
    "BsmtFinType1",
    "Electrical"
]
    df[cat_nulls_columns] = df[cat_nulls_columns].fillna("None")
    num_nulls_columns = [
    "LotFrontage",
    "GarageYrBlt",
    "MasVnrArea"
]
    for column in num_nulls_columns:
        mean = df[column].mean()
        df[column] = df[column].fillna(mean)
    
    return df
def split_features_target(df):
    x = df.drop(columns=["SalePrice", "Id"])
    y = df["SalePrice"]
    return x, y
def encode_features(df):
    cat_columns = [
    "MSSubClass",
    "MSZoning",
    "Street",
    "Alley",
    "LotShape",
    "LandContour",
    "Utilities",
    "LotConfig",
    "LandSlope",
    "Neighborhood",
    "Condition1",
    "Condition2",
    "BldgType",
    "HouseStyle",
    "RoofStyle",
    "RoofMatl",
    "Exterior1st",
    "Exterior2nd",
    "MasVnrType",
    "ExterQual",
    "ExterCond",
    "Foundation",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "Heating",
    "HeatingQC",
    "CentralAir",
    "Electrical",
    "KitchenQual",
    "Functional",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PavedDrive",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "MoSold",
    "SaleType",
    "SaleCondition",
]
    df = pd.get_dummies(df, columns=cat_columns)
    return df