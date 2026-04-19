# scripts/aggregate_raster_to_grid.py

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask


def aggregate_one_raster(grid: gpd.GeoDataFrame, raster_path: Path, value_name: str) -> gpd.GeoDataFrame:
    results_mean = []
    results_count = []

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        nodata = src.nodata

        grid_proj = grid.to_crs(raster_crs)

        for _, row in grid_proj.iterrows():
            geom = [row.geometry]

            try:
                out_image, _ = mask(src, geom, crop=True)
                data = out_image[0].astype("float32")

                if nodata is not None:
                    data = data[data != nodata]

                data = data[~np.isnan(data)]

                if data.size == 0:
                    value = np.nan
                    count = 0
                else:
                    value = float(data.mean())
                    count = int(data.size)

            except Exception:
                value = np.nan
                count = 0

            results_mean.append(value)
            results_count.append(count)

    grid[value_name] = results_mean
    grid[f"{value_name}_count"] = results_count
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", required=True, help="Path to grid file (GeoJSON/Parquet supported by geopandas)")
    parser.add_argument("--ndvi", required=True)
    parser.add_argument("--ndbi", required=True)
    parser.add_argument("--lst", required=True)
    parser.add_argument("--out", required=True, help="Output parquet or geojson")
    args = parser.parse_args()

    grid_path = Path(args.grid)
    out_path = Path(args.out)

    grid = gpd.read_file(grid_path)

    # Keep a stable row id if none exists
    if "cell_id" not in grid.columns:
        grid["cell_id"] = np.arange(len(grid))

    grid = aggregate_one_raster(grid, Path(args.ndvi), "ndvi_mean")
    grid = aggregate_one_raster(grid, Path(args.ndbi), "ndbi_mean")
    grid = aggregate_one_raster(grid, Path(args.lst), "lst_mean")
    
    valid_ndvi = grid["ndvi_mean"].notna()
    valid_ndbi = grid["ndbi_mean"].notna()
    valid_lst = grid["lst_mean"].notna()

    print("valid ndvi:", int(valid_ndvi.sum()))
    print("valid ndbi:", int(valid_ndbi.sum()))
    print("valid lst :", int(valid_lst.sum()))
    print("valid all :", int((valid_ndvi & valid_ndbi & valid_lst).sum()))

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".geojson":
        grid.to_file(out_path, driver="GeoJSON")
    elif out_path.suffix.lower() == ".parquet":
        grid.to_parquet(out_path)
    else:
        raise ValueError("Output must end with .geojson or .parquet")

    print(f"[ok] Saved aggregated dataset to {out_path}")

    preview_cols = [c for c in ["cell_id", "ndvi_mean", "ndvi_mean_count", "ndbi_mean", "ndbi_mean_count", "lst_mean", "lst_mean_count"] if c in grid.columns]
    print(grid[preview_cols].head())


if __name__ == "__main__":
    main()