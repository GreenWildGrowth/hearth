from __future__ import annotations

import argparse

import geopandas as gpd
from shapely.geometry import box

from hearth.locations import get_location_config
from hearth.paths import get_location_paths


WGS84 = "EPSG:4326"
WORK_CRS = "EPSG:2154"  # Lambert-93


def make_bbox_gdf(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> gpd.GeoDataFrame:
    geom = box(min_lon, min_lat, max_lon, max_lat)
    return gpd.GeoDataFrame({"name": ["bbox"]}, geometry=[geom], crs=WGS84)


def build_grid_over_bbox(bbox_gdf: gpd.GeoDataFrame, cell_size_m: int) -> gpd.GeoDataFrame:
    bbox_metric = bbox_gdf.to_crs(WORK_CRS)
    minx, miny, maxx, maxy = bbox_metric.total_bounds

    xs = list(range(int(minx), int(maxx) + cell_size_m, cell_size_m))
    ys = list(range(int(miny), int(maxy) + cell_size_m, cell_size_m))

    cells = []
    cell_ids = []
    idx = 0

    for x in xs[:-1]:
        for y in ys[:-1]:
            cells.append(box(x, y, x + cell_size_m, y + cell_size_m))
            cell_ids.append(f"cell_{idx:05d}")
            idx += 1

    grid = gpd.GeoDataFrame({"cell_id": cell_ids}, geometry=cells, crs=WORK_CRS)
    grid = gpd.overlay(grid, bbox_metric, how="intersection")
    grid = grid[["cell_id", "geometry"]].copy()

    centroids = grid.geometry.centroid
    grid["centroid_x"] = centroids.x
    grid["centroid_y"] = centroids.y

    centroids_wgs84 = gpd.GeoSeries(centroids, crs=WORK_CRS).to_crs(WGS84)
    grid["centroid_lon"] = centroids_wgs84.x
    grid["centroid_lat"] = centroids_wgs84.y

    return grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", required=True)
    args = parser.parse_args()

    cfg = get_location_config(args.location)
    paths = get_location_paths(args.location)
    paths.ensure_dirs()

    bbox_gdf = make_bbox_gdf(*cfg.bbox_tuple)
    grid = build_grid_over_bbox(bbox_gdf, cfg.grid_size_m)

    grid.to_crs(WGS84).to_file(paths.grid_path, driver="GeoJSON")

    print(f"Grid saved to: {paths.grid_path}")
    print(f"Number of cells: {len(grid)}")


if __name__ == "__main__":
    main()