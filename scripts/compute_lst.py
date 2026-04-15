# scripts/compute_lst.py

from pathlib import Path
import argparse
import json

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
import planetary_computer
from pystac_client import Client

from hearth.paths import get_location_paths

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
NODATA = -9999.0


def load_manifest(location: str) -> dict:
    paths = get_location_paths(location)
    manifest_path = paths.raw_dir / "scene_manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    return json.loads(manifest_path.read_text(encoding="utf-8"))


def resolve_signed_asset_href(collection: str, item_id: str, asset_key: str) -> str:
    catalog = Client.open(
        PC_STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(collections=[collection], ids=[item_id])
    items = list(search.items())

    if not items:
        raise RuntimeError(f"Could not find STAC item: collection={collection}, item_id={item_id}")

    item = items[0]

    if asset_key not in item.assets:
        available = ", ".join(sorted(item.assets.keys()))
        raise KeyError(
            f"Asset key '{asset_key}' not found in item '{item_id}'. Available: {available}"
        )

    return item.assets[asset_key].href

def read_band_window(src: rasterio.io.DatasetReader, bbox_wgs84: list[float]):
    """
    bbox_wgs84: [minx, miny, maxx, maxy] in EPSG:4326
    """
    if src.crs is None:
        raise RuntimeError("Source raster has no CRS")

    # Reproject bbox from WGS84 to raster CRS
    if str(src.crs) != "EPSG:4326":
        bbox_src = transform_bounds(
            "EPSG:4326",
            src.crs,
            *bbox_wgs84,
            densify_pts=21,
        )
    else:
        bbox_src = bbox_wgs84

    window = from_bounds(*bbox_src, transform=src.transform)

    # Clip to raster extent
    window = window.round_offsets().round_lengths()
    full_window = rasterio.windows.Window(0, 0, src.width, src.height)
    window = window.intersection(full_window)

    if window.width <= 0 or window.height <= 0:
        raise RuntimeError(
            f"Computed empty window for bbox={bbox_wgs84} in raster CRS={src.crs}"
        )

    arr = src.read(1, window=window, masked=True).astype("float32")
    out_transform = rasterio.windows.transform(window, src.transform)
    return arr, out_transform


def dn_to_lst_celsius(dn: np.ma.MaskedArray) -> np.ma.MaskedArray:
    kelvin = dn * 0.00341802 + 149.0
    celsius = kelvin - 273.15
    celsius = np.ma.masked_where(~np.isfinite(celsius), celsius)
    celsius = np.ma.clip(celsius, -80.0, 90.0)
    return celsius


def compute_lst(location: str) -> Path:
    paths = get_location_paths(location)
    manifest = load_manifest(location)

    landsat = manifest.get("landsat")
    if landsat is None:
        raise RuntimeError("No Landsat scene found in manifest")

    collection = landsat["collection"]
    item_id = landsat["item_id"]
    asset_key = landsat["asset_keys"]["lst"]
    bbox = manifest["bbox"]

    print("[lst] resolving fresh signed URL...")
    lst_url = resolve_signed_asset_href(collection, item_id, asset_key)

    print("[lst] reading Landsat LST asset...")
    with rasterio.open(lst_url) as src:
        dn, transform = read_band_window(src, bbox)

        if src.nodata is not None:
            dn = np.ma.masked_where(dn == src.nodata, dn)

        lst_c = dn_to_lst_celsius(dn)
        lst_filled = lst_c.filled(NODATA).astype("float32")

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            height=lst_filled.shape[0],
            width=lst_filled.shape[1],
            transform=transform,
            dtype="float32",
            count=1,
            compress="lzw",
            nodata=NODATA,
        )

        out_path = paths.processed_dir / "lst.tif"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(lst_filled, 1)

    print(f"[ok] LST saved to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", required=True)
    args = parser.parse_args()

    compute_lst(args.location.strip().lower())


if __name__ == "__main__":
    main()