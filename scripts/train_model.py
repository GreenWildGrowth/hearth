# scripts/train_model.py

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to aggregated parquet dataset")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_parquet(dataset_path)

    required_cols = ["ndvi_mean", "ndbi_mean", "lst_mean"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    total_rows = len(df)
    df_model = df[required_cols].dropna().copy()
    used_rows = len(df_model)

    if used_rows == 0:
        raise RuntimeError("No valid rows left after dropping NaNs")

    print(f"Rows total: {total_rows}")
    print(f"Rows usable: {used_rows}")
    print(f"Rows dropped: {total_rows - used_rows}")

    X = df_model[["ndvi_mean", "ndbi_mean"]]
    y = df_model["lst_mean"]

    if used_rows < 10:
        raise RuntimeError("Too few valid rows to train a meaningful model")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print()
    print("Model:")
    print(f"Intercept : {model.intercept_:.6f}")
    print(f"coef_ndvi : {model.coef_[0]:.6f}")
    print(f"coef_ndbi : {model.coef_[1]:.6f}")

    print()
    print("Train metrics:")
    print(f"R2  : {r2_score(y_train, y_train_pred):.6f}")
    print(f"MAE : {mean_absolute_error(y_train, y_train_pred):.6f}")

    print()
    print("Test metrics:")
    print(f"R2  : {r2_score(y_test, y_test_pred):.6f}")
    print(f"MAE : {mean_absolute_error(y_test, y_test_pred):.6f}")


if __name__ == "__main__":
    main()