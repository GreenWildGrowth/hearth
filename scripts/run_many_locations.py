from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
import yaml


def run_cmd(cmd: list[str]) -> None:
    print()
    print(">>>", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def flatten_report(report: dict) -> dict:
    row = {
        "location": report.get("location"),
        "dataset": report.get("dataset"),
        "target": report.get("target"),
        "features": ",".join(report.get("features", [])),
        "rows_total": report.get("rows_total"),
        "rows_usable": report.get("rows_usable"),
        "rows_dropped": report.get("rows_dropped"),
    }

    corr = report.get("correlations", {})
    if isinstance(corr, dict):
        for a, sub in corr.items():
            if isinstance(sub, dict):
                for b, val in sub.items():
                    row[f"corr__{a}__{b}"] = val

    plain = report.get("plain_linear_model", {})
    row["plain_intercept"] = plain.get("intercept")
    for name, val in plain.get("coefficients", {}).items():
        row[f"coef__{name}"] = val
    for split in ("train", "test"):
        metrics = plain.get(split, {})
        for metric_name, val in metrics.items():
            row[f"plain_{split}_{metric_name}"] = val

    std = report.get("standardized_linear_model", {})
    for name, val in std.get("coefficients_std", {}).items():
        row[f"coef_std__{name}"] = val
    for split in ("train", "test"):
        metrics = std.get(split, {})
        for metric_name, val in metrics.items():
            row[f"std_{split}_{metric_name}"] = val

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the urban microclimate pipeline for multiple locations.")
    parser.add_argument("--locations", nargs="+", help="Location keys from locations.yaml")
    parser.add_argument("--locations-yaml", default="locations.yaml", help="Path to locations.yaml")
    parser.add_argument("--features", nargs="+", required=True, help="Feature columns passed to train_model.py")
    parser.add_argument("--target", default="lst_mean")
    parser.add_argument("--out-csv", required=True, help="Path to comparative CSV")
    parser.add_argument("--reports-dir", default="reports/multi_city", help="Directory for per-city JSON reports")
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--skip-train-pipeline", action="store_true", help="Skip train step inside run_location.py")
    parser.add_argument("--force-fetch", action="store_true")
    parser.add_argument("--with-interaction", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    
    if not args.locations:
      yaml_path = Path(args.locations_yaml)
      if not yaml_path.exists():
          raise RuntimeError(f"locations.yaml not found: {yaml_path}")

      with yaml_path.open("r", encoding="utf-8") as f:
          data = yaml.safe_load(f)

      # assume structure: { "paris": {...}, "madrid": {...}, ... }
      args.locations = list(data.keys())

      print(f"[info] Using all locations from {yaml_path}:")
      print(", ".join(args.locations))

    py = sys.executable
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for location in args.locations:
        location = location.strip().lower()
        report_path = reports_dir / f"{location}.json"
        dataset_path = Path("data/processed") / location / "dataset.parquet"

        try:
            run_location_cmd = [py, "scripts/run_location.py", "--location", location]
            if args.skip_fetch:
                run_location_cmd.append("--skip-fetch")
            if args.skip_train_pipeline:
                run_location_cmd.append("--skip-train")
            if args.force_fetch:
                run_location_cmd.append("--force-fetch")

            run_cmd(run_location_cmd)

            if not dataset_path.exists():
                raise RuntimeError(f"Dataset not found after pipeline: {dataset_path}")

            train_cmd = [
                py,
                "scripts/train_model.py",
                "--dataset", str(dataset_path),
                "--features", *args.features,
                "--target", args.target,
                "--test-size", str(args.test_size),
                "--random-state", str(args.random_state),
                "--report-json", str(report_path),
            ]
            if args.with_interaction:
                train_cmd.append("--with-interaction")

            run_cmd(train_cmd)

            report = json.loads(report_path.read_text(encoding="utf-8"))
            row = flatten_report(report)
            row["status"] = "ok"
            row["error"] = ""
            rows.append(row)

        except Exception as e:
            rows.append({
                "location": location,
                "status": "failed",
                "error": str(e),
            })
            print(f"[warn] {location} failed: {e}")

    if not rows:
        raise RuntimeError("No rows generated")

    all_keys = sorted({k for row in rows for k in row.keys()})
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"[ok] Comparative CSV written to: {out_csv}")


if __name__ == "__main__":
    main()