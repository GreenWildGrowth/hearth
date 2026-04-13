import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_parquet("data/processed/dataset.parquet")

X = df[["ndvi_mean", "ndbi_mean"]]
y = df["lst_mean"]

model = LinearRegression().fit(X, y)

print("coef_ndvi:", model.coef_[0])
print("coef_ndbi:", model.coef_[1])
print("R2:", model.score(X, y))