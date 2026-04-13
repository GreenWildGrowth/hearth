import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask


def aggregate_raster_to_grid(grid_path, raster_path, value_name):
    grid = gpd.read_file(grid_path)
    results = []

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        nodata = src.nodata

        grid = grid.to_crs(raster_crs)

        for _, row in grid.iterrows():
            geom = [row.geometry]

            try:
                out_image, _ = mask(src, geom, crop=True)
                data = out_image[0].astype("float32")

                if nodata is not None:
                    data = data[data != nodata]

                data = data[~np.isnan(data)]

                if data.size == 0:
                    value = np.nan
                else:
                    value = float(data.mean())

            except Exception:
                value = np.nan

            results.append(value)

    grid[value_name] = results
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", required=True)
    parser.add_argument("--raster", required=True)
    parser.add_argument("--value-name", required=True)
    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    grid = aggregate_raster_to_grid(
        args.grid,
        args.raster,
        args.value_name,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".geojson":
        grid.to_file(out_path, driver="GeoJSON")
    elif out_path.suffix.lower() == ".parquet":
        grid.to_parquet(out_path)
    else:
        raise ValueError("Output must end with .geojson or .parquet")

    print(f"Saved to {out_path}")
    print(grid[[c for c in grid.columns if c != 'geometry']].head())


if __name__ == "__main__":
    main()