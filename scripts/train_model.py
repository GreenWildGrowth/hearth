# scripts/train_model.py

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def print_metrics(title: str, y_true, y_pred) -> dict:
    metrics = {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
    }
    print(title)
    print(f"R2   : {metrics['r2']:.6f}")
    print(f"MAE  : {metrics['mae']:.6f}")
    print(f"RMSE : {metrics['rmse']:.6f}")
    return metrics


def summarize_corr(df_model: pd.DataFrame) -> dict:
    corr = df_model[["ndvi_mean", "ndbi_mean", "lst_mean"]].corr(numeric_only=True)
    print()
    print("Correlations:")
    print(corr.round(6).to_string())
    return {
        "ndvi_ndbi": float(corr.loc["ndvi_mean", "ndbi_mean"]),
        "ndvi_lst": float(corr.loc["ndvi_mean", "lst_mean"]),
        "ndbi_lst": float(corr.loc["ndbi_mean", "lst_mean"]),
    }


def fit_plain_model(X_train, X_test, y_train, y_test) -> dict:
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print()
    print("Plain linear model:")
    print(f"Intercept : {model.intercept_:.6f}")
    print(f"coef_ndvi : {model.coef_[0]:.6f}")
    print(f"coef_ndbi : {model.coef_[1]:.6f}")

    train_metrics = print_metrics("Train metrics:", y_train, y_train_pred)
    print()
    test_metrics = print_metrics("Test metrics:", y_test, y_test_pred)

    return {
        "intercept": float(model.intercept_),
        "coef_ndvi": float(model.coef_[0]),
        "coef_ndbi": float(model.coef_[1]),
        "train": train_metrics,
        "test": test_metrics,
    }


def fit_standardized_model(X_train, X_test, y_train, y_test) -> dict:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])
    pipe.fit(X_train, y_train)

    model = pipe.named_steps["model"]
    y_train_pred = pipe.predict(X_train)
    y_test_pred = pipe.predict(X_test)

    print()
    print("Standardized linear model:")
    print("Coefficients are comparable in relative importance.")
    print(f"coef_ndvi_std : {model.coef_[0]:.6f}")
    print(f"coef_ndbi_std : {model.coef_[1]:.6f}")

    train_metrics = print_metrics("Train metrics:", y_train, y_train_pred)
    print()
    test_metrics = print_metrics("Test metrics:", y_test, y_test_pred)

    return {
        "coef_ndvi_std": float(model.coef_[0]),
        "coef_ndbi_std": float(model.coef_[1]),
        "train": train_metrics,
        "test": test_metrics,
    }


def fit_interaction_model(df_train: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    train = df_train.copy()
    test = df_test.copy()

    train["interaction"] = train["ndvi_mean"] * train["ndbi_mean"]
    test["interaction"] = test["ndvi_mean"] * test["ndbi_mean"]

    X_train = train[["ndvi_mean", "ndbi_mean", "interaction"]]
    X_test = test[["ndvi_mean", "ndbi_mean", "interaction"]]
    y_train = train["lst_mean"]
    y_test = test["lst_mean"]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])
    pipe.fit(X_train, y_train)

    model = pipe.named_steps["model"]
    y_train_pred = pipe.predict(X_train)
    y_test_pred = pipe.predict(X_test)

    print()
    print("Standardized interaction model:")
    print("Features: ndvi_mean, ndbi_mean, ndvi_mean*ndbi_mean")
    print(f"coef_ndvi_std        : {model.coef_[0]:.6f}")
    print(f"coef_ndbi_std        : {model.coef_[1]:.6f}")
    print(f"coef_interaction_std : {model.coef_[2]:.6f}")

    train_metrics = print_metrics("Train metrics:", y_train, y_train_pred)
    print()
    test_metrics = print_metrics("Test metrics:", y_test, y_test_pred)

    return {
        "coef_ndvi_std": float(model.coef_[0]),
        "coef_ndbi_std": float(model.coef_[1]),
        "coef_interaction_std": float(model.coef_[2]),
        "train": train_metrics,
        "test": test_metrics,
    }


def infer_location_name(dataset_path: Path) -> str | None:
    # e.g. data/processed/barcelona/dataset.parquet -> "barcelona"
    parts = dataset_path.parts
    try:
        idx = parts.index("processed")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    except ValueError:
        pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to aggregated parquet dataset")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--report-json", default=None, help="Optional path to save a JSON report")
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
    if used_rows < 10:
        raise RuntimeError("Too few valid rows to train a meaningful model")

    print(f"Rows total: {total_rows}")
    print(f"Rows usable: {used_rows}")
    print(f"Rows dropped: {total_rows - used_rows}")

    location = infer_location_name(dataset_path)
    if location:
        print(f"Location inferred from dataset path: {location}")

    corr_summary = summarize_corr(df_model)

    train_df, test_df = train_test_split(
        df_model,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    plain_result = fit_plain_model(
        train_df[["ndvi_mean", "ndbi_mean"]],
        test_df[["ndvi_mean", "ndbi_mean"]],
        train_df["lst_mean"],
        test_df["lst_mean"],
    )

    standardized_result = fit_standardized_model(
        train_df[["ndvi_mean", "ndbi_mean"]],
        test_df[["ndvi_mean", "ndbi_mean"]],
        train_df["lst_mean"],
        test_df["lst_mean"],
    )

    interaction_result = fit_interaction_model(train_df, test_df)

    summary = {
        "dataset": str(dataset_path),
        "location": location,
        "rows_total": int(total_rows),
        "rows_usable": int(used_rows),
        "rows_dropped": int(total_rows - used_rows),
        "correlations": corr_summary,
        "plain_linear_model": plain_result,
        "standardized_linear_model": standardized_result,
        "standardized_interaction_model": interaction_result,
    }

    if args.report_json:
        out_path = Path(args.report_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print()
        print(f"JSON report saved to {out_path}")


if __name__ == "__main__":
    main()