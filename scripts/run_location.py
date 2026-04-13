from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from hearth.locations import get_location_config
from hearth.paths import (
    get_location_paths,
    find_sentinel_red,
    find_sentinel_nir,
    find_landsat_st_b10,
    find_landsat_sr_b5,
    find_landsat_sr_b6,
)


def run_cmd(args: list[str]) -> None:
    print(">>>", " ".join(args))
    subprocess.run(args, check=True)


def build_grid(location: str) -> None:
    """
    Assumes build_grid.py is upgraded to accept:
      --location <name>
    """
    run_cmd(
        [
            sys.executable,
            "scripts/build_grid.py",
            "--location",
            location,
        ]
    )


def compute_indices(location: str) -> None:
    paths = get_location_paths(location)

    red = find_sentinel_red(location)
    nir = find_sentinel_nir(location)

    st_b10 = find_landsat_st_b10(location)
    sr_b5 = find_landsat_sr_b5(location)
    sr_b6 = find_landsat_sr_b6(location)

    run_cmd(
        [
            sys.executable,
            "scripts/compute_ndvi.py",
            "--red",
            str(red),
            "--nir",
            str(nir),
            "--out",
            str(paths.ndvi_raster_path),
        ]
    )

    run_cmd(
        [
            sys.executable,
            "scripts/compute_lst.py",
            "--st",
            str(st_b10),
            "--out",
            str(paths.lst_raster_path),
        ]
    )

    run_cmd(
        [
            sys.executable,
            "scripts/compute_ndbi.py",
            "--b5",
            str(sr_b5),
            "--b6",
            str(sr_b6),
            "--out",
            str(paths.ndbi_raster_path),
        ]
    )


def aggregate(location: str) -> None:
    paths = get_location_paths(location)

    run_cmd(
        [
            sys.executable,
            "scripts/aggregate_raster_to_grid.py",
            "--grid",
            str(paths.grid_path),
            "--raster",
            str(paths.ndvi_raster_path),
            "--value-name",
            "ndvi_mean",
            "--out",
            str(paths.grid_ndvi_path),
        ]
    )

    run_cmd(
        [
            sys.executable,
            "scripts/aggregate_raster_to_grid.py",
            "--grid",
            str(paths.grid_path),
            "--raster",
            str(paths.ndbi_raster_path),
            "--value-name",
            "ndbi_mean",
            "--out",
            str(paths.grid_ndbi_path),
        ]
    )

    run_cmd(
        [
            sys.executable,
            "scripts/aggregate_raster_to_grid.py",
            "--grid",
            str(paths.grid_path),
            "--raster",
            str(paths.lst_raster_path),
            "--value-name",
            "lst_mean",
            "--out",
            str(paths.grid_lst_path),
        ]
    )


def build_dataset(location: str) -> None:
    """
    Assumes build_dataset.py is upgraded to accept:
      --location <name>
    """
    run_cmd(
        [
            sys.executable,
            "scripts/build_dataset.py",
            "--location",
            location,
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the microclimate pipeline for one location."
    )
    parser.add_argument("--location", required=True, help="Location name, e.g. toulouse")

    parser.add_argument("--build-grid", action="store_true")
    parser.add_argument("--compute-indices", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--build-dataset", action="store_true")

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all steps: build grid, compute indices, aggregate, build dataset",
    )

    args = parser.parse_args()

    location = args.location.strip().lower()

    # Validate config exists
    cfg = get_location_config(location)
    paths = get_location_paths(location)
    paths.ensure_dirs()

    print(f"Location: {cfg.name} ({cfg.label})")
    print(f"BBox: {cfg.bbox_tuple}")
    print(f"Grid size: {cfg.grid_size_m} m")
    print(f"Processed dir: {paths.processed_dir}")

    do_build_grid = args.all or args.build_grid
    do_compute_indices = args.all or args.compute_indices
    do_aggregate = args.all or args.aggregate
    do_build_dataset = args.all or args.build_dataset

    if not any([do_build_grid, do_compute_indices, do_aggregate, do_build_dataset]):
        parser.error("No step selected. Use --all or one of --build-grid, --compute-indices, --aggregate, --build-dataset")

    if do_build_grid:
        build_grid(location)

    if do_compute_indices:
        compute_indices(location)

    if do_aggregate:
        aggregate(location)

    if do_build_dataset:
        build_dataset(location)

    print("Pipeline finished successfully.")


if __name__ == "__main__":
    main()