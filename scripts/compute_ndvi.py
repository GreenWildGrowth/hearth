from pathlib import Path
import argparse
import numpy as np
import rasterio

NODATA = -9999.0

def compute_ndvi(red_path, nir_path, out_path):
    with rasterio.open(red_path) as red_src, rasterio.open(nir_path) as nir_src:
        red = red_src.read(1, masked=True).astype("float32")
        nir = nir_src.read(1, masked=True).astype("float32")

        denom = nir + red
        ndvi = np.ma.divide(nir - red, denom)
        ndvi = np.ma.clip(ndvi, -1.0, 1.0)

        ndvi_filled = ndvi.filled(NODATA).astype("float32")

        profile = red_src.profile.copy()
        profile.update(
            driver="GTiff",
            dtype="float32",
            count=1,
            compress="lzw",
            nodata=NODATA,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(ndvi_filled, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--red", required=True)
    parser.add_argument("--nir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    compute_ndvi(Path(args.red), Path(args.nir), Path(args.out))
    print(f"NDVI saved to {args.out}")

if __name__ == "__main__":
    main()