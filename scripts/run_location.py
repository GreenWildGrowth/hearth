# scripts/run_location.py

from __future__ import annotations

import argparse
import subprocess
import sys

from hearth.locations import get_location_config
from hearth.paths import get_location_paths


def run_cmd(cmd: list[str]) -> None:
    print()
    print(">>>", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full urban microclimate pipeline for one location.")
    parser.add_argument("--location", required=True, help="Location key from locations.yaml")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip STAC fetch / manifest generation")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    parser.add_argument("--force-fetch", action="store_true", help="Force re-selection of scenes")
    args = parser.parse_args()

    location = args.location.strip().lower()

    # Validate location early
    cfg = get_location_config(location)
    paths = get_location_paths(location)
    paths.ensure_dirs()

    print(f"Location: {cfg.name} ({cfg.label})")
    print(f"BBox: {cfg.bbox_tuple}")
    print(f"Grid size: {cfg.grid_size_m} m")
    print(f"Processed dir: {paths.processed_dir}")

    py = sys.executable

    if not args.skip_fetch:
        cmd = [py, "scripts/fetch_location_data.py", "--location", location]
        if args.force_fetch:
            cmd.append("--force")
        run_cmd(cmd)

    run_cmd([py, "scripts/build_grid.py", "--location", location])
    run_cmd([py, "scripts/compute_ndvi.py", "--location", location])
    run_cmd([py, "scripts/compute_ndbi.py", "--location", location])
    run_cmd([py, "scripts/compute_lst.py", "--location", location])

    run_cmd([
        py,
        "scripts/aggregate_raster_to_grid.py",
        "--grid", str(paths.grid_path),
        "--ndvi", str(paths.processed_dir / "ndvi.tif"),
        "--ndbi", str(paths.processed_dir / "ndbi.tif"),
        "--lst", str(paths.processed_dir / "lst.tif"),
        "--out", str(paths.processed_dir / "dataset.parquet"),
    ])

    if not args.skip_train:
        run_cmd([
            py,
            "scripts/train_model.py",
            "--dataset", str(paths.processed_dir / "dataset.parquet"),
        ])

    print()
    print("[ok] Full pipeline completed successfully.")


if __name__ == "__main__":
    main()