from pathlib import Path
import argparse

import numpy as np
import rasterio

NODATA = -9999.0


def compute_lst(st_path, out_path):
    with rasterio.open(st_path) as src:
        dn = src.read(1).astype("float32")

        # masque nodata
        if src.nodata is not None:
            dn = np.where(dn == src.nodata, np.nan, dn)

        # conversion Kelvin
        kelvin = dn * 0.00341802 + 149.0

        # conversion Celsius
        celsius = kelvin - 273.15

        celsius = np.where(
            (celsius > 10) & (celsius < 65),
            celsius,
            np.nan
        )

        # clamp réaliste
        celsius = np.clip(celsius, -50, 70)

        # remplir nodata
        celsius = np.where(np.isfinite(celsius), celsius, NODATA).astype("float32")

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            dtype="float32",
            count=1,
            compress="lzw",
            nodata=NODATA,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(celsius, 1)

    print(f"LST saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--st", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    compute_lst(Path(args.st), Path(args.out))


if __name__ == "__main__":
    main()