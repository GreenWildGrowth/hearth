# scripts/compute_ndbi.py

from pathlib import Path
import argparse
import json

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds, reproject, Resampling
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
    if src.crs is None:
        raise RuntimeError("Source raster has no CRS")

    if str(src.crs) != "EPSG:4326":
        bbox_src = transform_bounds("EPSG:4326", src.crs, *bbox_wgs84, densify_pts=21)
    else:
        bbox_src = bbox_wgs84

    window = from_bounds(*bbox_src, transform=src.transform)
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


def reproject_to_match(
    src_arr: np.ma.MaskedArray,
    src_transform,
    src_crs,
    dst_shape,
    dst_transform,
    dst_crs,
) -> np.ma.MaskedArray:
    src_filled = src_arr.filled(NODATA).astype("float32")
    dst = np.full(dst_shape, NODATA, dtype="float32")

    reproject(
        source=src_filled,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=NODATA,
        dst_nodata=NODATA,
        resampling=Resampling.bilinear,
    )

    return np.ma.masked_equal(dst, NODATA)


def compute_ndbi(location: str) -> Path:
    paths = get_location_paths(location)
    manifest = load_manifest(location)

    sentinel = manifest.get("sentinel")
    if sentinel is None:
        raise RuntimeError("No Sentinel scene in manifest")

    collection = sentinel["collection"]
    item_id = sentinel["item_id"]
    nir_asset_key = sentinel["asset_keys"]["nir"]
    swir_asset_key = sentinel["asset_keys"]["swir"]
    bbox = manifest["bbox"]

    print("[ndbi] resolving fresh signed URLs...")
    nir_url = resolve_signed_asset_href(collection, item_id, nir_asset_key)
    swir_url = resolve_signed_asset_href(collection, item_id, swir_asset_key)

    print("[ndbi] reading Sentinel assets...")
    with rasterio.open(nir_url) as nir_src, rasterio.open(swir_url) as swir_src:
        nir, nir_transform = read_band_window(nir_src, bbox)
        swir, swir_transform = read_band_window(swir_src, bbox)

        print(f"[ndbi] nir shape: {nir.shape}, res: {nir_src.res}")
        print(f"[ndbi] swir shape: {swir.shape}, res: {swir_src.res}")

        # Reproject SWIR onto NIR grid if needed
        if (
            swir.shape != nir.shape
            or swir_src.transform != nir_src.transform
            or swir_src.crs != nir_src.crs
            or swir_transform != nir_transform
        ):
            print("[ndbi] resampling SWIR onto NIR grid...")
            swir = reproject_to_match(
                src_arr=swir,
                src_transform=swir_transform,
                src_crs=swir_src.crs,
                dst_shape=nir.shape,
                dst_transform=nir_transform,
                dst_crs=nir_src.crs,
            )

        denom = swir + nir
        ndbi = np.ma.divide(swir - nir, denom)
        ndbi = np.ma.clip(ndbi, -1.0, 1.0)

        ndbi_filled = ndbi.filled(NODATA).astype("float32")

        profile = nir_src.profile.copy()
        profile.update(
            driver="GTiff",
            height=ndbi_filled.shape[0],
            width=ndbi_filled.shape[1],
            transform=nir_transform,
            dtype="float32",
            count=1,
            compress="lzw",
            nodata=NODATA,
        )

        out_path = paths.processed_dir / "ndbi.tif"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(ndbi_filled, 1)

    print(f"[ok] NDBI saved to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", required=True)
    args = parser.parse_args()

    compute_ndbi(args.location.strip().lower())


if __name__ == "__main__":
    main()