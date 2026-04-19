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


def summarize_corr(df_model: pd.DataFrame, features: list[str], target: str) -> dict:
    cols = [c for c in features + [target] if c in df_model.columns]
    corr = df_model[cols].corr(numeric_only=True)
    print()
    print("Correlations:")
    print(corr.round(6).to_string())
    return corr.to_dict()


def infer_location_name(dataset_path: Path) -> str | None:
    parts = dataset_path.parts
    try:
        idx = parts.index("processed")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    except ValueError:
        pass
    return None


def fit_plain_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list[str],
) -> dict:
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print()
    print("Plain linear model:")
    print(f"Intercept : {model.intercept_:.6f}")
    for name, coef in zip(feature_names, model.coef_):
        print(f"coef_{name} : {coef:.6f}")

    train_metrics = print_metrics("Train metrics:", y_train, y_train_pred)
    print()
    test_metrics = print_metrics("Test metrics:", y_test, y_test_pred)

    return {
        "intercept": float(model.intercept_),
        "coefficients": {name: float(coef) for name, coef in zip(feature_names, model.coef_)},
        "train": train_metrics,
        "test": test_metrics,
    }


def fit_standardized_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list[str],
) -> dict:
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
    for name, coef in zip(feature_names, model.coef_):
        print(f"coef_{name}_std : {coef:.6f}")

    train_metrics = print_metrics("Train metrics:", y_train, y_train_pred)
    print()
    test_metrics = print_metrics("Test metrics:", y_test, y_test_pred)

    return {
        "coefficients_std": {name: float(coef) for name, coef in zip(feature_names, model.coef_)},
        "train": train_metrics,
        "test": test_metrics,
    }


def maybe_add_interaction(df: pd.DataFrame, features: list[str], enable_interaction: bool) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    feature_names = list(features)

    if enable_interaction and "ndvi_mean" in feature_names and "ndbi_mean" in feature_names:
        interaction_name = "ndvi_x_ndbi"
        out[interaction_name] = out["ndvi_mean"] * out["ndbi_mean"]
        feature_names.append(interaction_name)

    return out, feature_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to aggregated parquet dataset")
    parser.add_argument("--features", nargs="+", required=True, help="Feature columns to use")
    parser.add_argument("--target", default="lst_mean", help="Target column")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--with-interaction", action="store_true", help="Add ndvi_mean*ndbi_mean if both exist")
    parser.add_argument("--report-json", default=None, help="Optional path to save a JSON report")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_parquet(dataset_path)

    required_cols = list(args.features) + [args.target]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    total_rows = len(df)
    df_model = df[required_cols].dropna().copy()
    df_model, feature_names = maybe_add_interaction(df_model, list(args.features), args.with_interaction)
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

    corr_summary = summarize_corr(df_model, feature_names, args.target)

    train_df, test_df = train_test_split(
        df_model,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    X_train = train_df[feature_names]
    X_test = test_df[feature_names]
    y_train = train_df[args.target]
    y_test = test_df[args.target]

    plain_result = fit_plain_model(X_train, X_test, y_train, y_test, feature_names)
    standardized_result = fit_standardized_model(X_train, X_test, y_train, y_test, feature_names)

    summary = {
        "dataset": str(dataset_path),
        "location": location,
        "features": feature_names,
        "target": args.target,
        "rows_total": int(total_rows),
        "rows_usable": int(used_rows),
        "rows_dropped": int(total_rows - used_rows),
        "correlations": corr_summary,
        "plain_linear_model": plain_result,
        "standardized_linear_model": standardized_result,
    }

    if args.report_json:
        out_path = Path(args.report_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print()
        print(f"JSON report saved to {out_path}")


if __name__ == "__main__":
    main()