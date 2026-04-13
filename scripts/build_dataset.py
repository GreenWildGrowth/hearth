from __future__ import annotations

import argparse

import geopandas as gpd

from hearth.paths import get_location_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", required=True)
    args = parser.parse_args()

    paths = get_location_paths(args.location)

    ndvi = gpd.read_parquet(paths.grid_ndvi_path)
    ndbi = gpd.read_parquet(paths.grid_ndbi_path)
    lst = gpd.read_parquet(paths.grid_lst_path)

    df = ndvi.merge(ndbi[["cell_id", "ndbi_mean"]], on="cell_id")
    df = df.merge(lst[["cell_id", "lst_mean"]], on="cell_id")

    df = df.dropna(subset=["ndvi_mean", "ndbi_mean", "lst_mean"])

    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(paths.dataset_path)

    print(f"Dataset saved to: {paths.dataset_path}")
    print(df[["ndvi_mean", "ndbi_mean", "lst_mean"]].describe())


if __name__ == "__main__":
    main()