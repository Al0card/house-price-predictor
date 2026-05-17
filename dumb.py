# %% [markdown]
# # 02 — Modeling

# %%
import sys
from pathlib import Path
sys.path.append(str(Path().resolve().parent))
from src.data import load_data
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

# %% [markdown]
# ## Load Data

# %%
df_raw = pd.read_csv("../data/raw/train.csv")

# %%
df_raw.shape

# %% [markdown]
# ## Preprocessing

# %%
df_clean = df_raw.drop(columns=["Id"])

# %%
columns_with_nulls = df_clean.isnull().sum().sort_values(ascending=False)[df_clean.isnull().sum()>0]
columns_with_nulls

# %%
columns_with_nulls.index

# %%
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
df_clean[cat_nulls_columns] = df_clean[cat_nulls_columns].fillna("None")

num_nulls_columns = [
    "LotFrontage",
    "GarageYrBlt",
    "MasVnrArea"
]
for column in num_nulls_columns:
    mean = df_clean[column].mean()
    df_clean[column] = df_clean[column].fillna(mean)

# %%
df_clean.isnull().sum()[df_clean.isnull().sum()>0].sort_values(ascending=False)

# %%
df_clean.columns

# %%
x = df_clean.drop(columns=["SalePrice"])
print(f"x shape: {x.shape}")
y = df_clean["SalePrice"]
print(f"y shape: {y.shape}")

# %% [markdown]
# ## Feature Encoding

# %%
x.columns

# %%
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

x[cat_columns].nunique()

# %%
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
print("number of categroical columns: ", len(cat_columns))
x = pd.get_dummies(x, columns=cat_columns)

# %%
x.shape

# %%
pd.set_option('display.max_seq_items', None)
x_columns = x.columns.tolist()
for col in x_columns:
    print(col, end=", ")

# %% [markdown]
# ## Train / Validation Split

# %%
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
print("X_train shape: ", X_train.shape)
print("X_val shape: ", X_val.shape)
print("y_train shape: ", y_train.shape)
print("y_val shape: ", y_val.shape)

# %%
y_train_log = np.log(y_train)

# %% [markdown]
# ## Train Model

# %%
model = LinearRegression()
model.fit(X_train, y_train_log)

# %%
y_predict_log = model.predict(X_val)
y_predict = np.exp(y_predict_log)

# %% [markdown]
# ## Evaluation

# %%
rmse = root_mean_squared_error(y_val, y_predict)
print(f"rmse = {rmse}")


