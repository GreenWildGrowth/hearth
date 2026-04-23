from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path.cwd()
SCRIPTS_DIR = ROOT / "scripts"
DATA_RAW = ROOT / "data" / "raw"


def run_step(label: str, script_name: str, extra_args: list[str] | None = None) -> None:
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n=== {label} ===")
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def copy_city_catalog(source_name: str, target_name: str = "candidate_cities.csv") -> None:
    src = DATA_RAW / source_name
    dst = DATA_RAW / target_name
    if not src.exists():
        raise FileNotFoundError(f"Expected city catalog not found: {src}")
    if src.resolve() == dst.resolve():
        return
    shutil.copy2(src, dst)
    print(f"[info] Copied {src.name} -> {dst.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Starter pack for the climate analog pipeline: downloads data, builds the city "
            "catalog, extracts bioclim variables, and computes analog cities."
        )
    )
    parser.add_argument(
        "--skip-climate-download",
        action="store_true",
        help="Skip WorldClim/CMIP download step.",
    )
    parser.add_argument(
        "--skip-geonames-download",
        action="store_true",
        help="Skip GeoNames download step.",
    )
    parser.add_argument(
        "--skip-city-build",
        action="store_true",
        help="Skip city catalog build step.",
    )
    parser.add_argument(
        "--skip-dedup",
        action="store_true",
        help="Skip spatial deduplication even if deduplicate_candidate_cities.py exists.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip climate extraction step.",
    )
    parser.add_argument(
        "--skip-analogues",
        action="store_true",
        help="Skip analogue computation step.",
    )
    parser.add_argument(
        "--prefer-dedup",
        action="store_true",
        help=(
            "If data/raw/candidate_cities_dedup.csv exists after deduplication, copy it over "
            "data/raw/candidate_cities.csv before extraction."
        ),
    )
    args = parser.parse_args()

    if not args.skip_climate_download:
        run_step("Download climate rasters", "download_worldclim_data.py")

    if not args.skip_geonames_download:
        run_step("Download GeoNames", "download_geonames.py")

    if not args.skip_city_build:
        run_step("Build candidate cities from GeoNames", "build_candidate_cities_from_geonames.py")

    dedup_script = SCRIPTS_DIR / "deduplicate_candidate_cities.py"
    if dedup_script.exists() and not args.skip_dedup:
        run_step("Spatially deduplicate candidate cities", dedup_script.name)
        if args.prefer_dedup:
            copy_city_catalog("candidate_cities_dedup.csv", "candidate_cities.csv")
    elif args.prefer_dedup:
        print("[warn] --prefer-dedup ignored because deduplicate_candidate_cities.py is missing or dedup was skipped.")

    if not args.skip_extract:
        run_step("Extract BIOCLIM values for cities", "extract_bioclim_for_cities.py")

    if not args.skip_analogues:
        run_step("Compute climate analogues", "compute_city_analogues.py")

    print("\nPipeline completed.")
    print("Key outputs:")
    print("- data/interim/cities_bioclim.csv")
    print("- data/processed/city_analogues_weighted_normalized.csv")
    print("- data/processed/scaler.joblib")
    print("- data/processed/pca_model.joblib")


if __name__ == "__main__":
    main()
