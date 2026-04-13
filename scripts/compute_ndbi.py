# scripts/compute_ndbi.py

import argparse
import numpy as np
import rasterio


def compute_ndbi(b5_path, b6_path, out_path):
    with rasterio.open(b5_path) as nir_src, rasterio.open(b6_path) as swir_src:
        nir = nir_src.read(1).astype("float32")
        swir = swir_src.read(1).astype("float32")

        # masque nodata
        if nir_src.nodata is not None:
            nir = np.where(nir == nir_src.nodata, np.nan, nir)
        if swir_src.nodata is not None:
            swir = np.where(swir == swir_src.nodata, np.nan, swir)

        # conversion reflectance (important)
        nir = nir * 2.75e-05 - 0.2
        swir = swir * 2.75e-05 - 0.2

        denom = swir + nir
        ndbi = (swir - nir) / denom

        ndbi = np.where(np.isfinite(ndbi), ndbi, -9999.0).astype("float32")

        profile = nir_src.profile.copy()
        profile.update(dtype="float32", nodata=-9999.0, compress="lzw")

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(ndbi, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--b5", required=True)
    parser.add_argument("--b6", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    compute_ndbi(args.b5, args.b6, args.out)