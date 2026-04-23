from pathlib import Path
import re
import pandas as pd
import rasterio

CITIES_CSV = Path("data/raw/candidate_cities.csv")
OUT_CSV = Path("data/interim/cities_bioclim.csv")

CURRENT_DIR = Path("data/climate/current")
FUTURE_FILES = {
    "BCC_CSM2_MR": Path(
        "data/climate/future/BCC_CSM2_MR_ssp245_2041_2060/wc2.1_10m_bioc_BCC-CSM2-MR_ssp245_2041-2060.tif"
    ),
    "CNRM_CM6_1": Path(
        "data/climate/future/CNRM_CM6_1_ssp245_2041_2060/wc2.1_10m_bioc_CNRM-CM6-1_ssp245_2041-2060.tif"
    ),
    "IPSL_CM6A_LR": Path(
        "data/climate/future/IPSL_CM6A_LR_ssp245_2041_2060/wc2.1_10m_bioc_IPSL-CM6A-LR_ssp245_2041-2060.tif"
    ),
}

BIO_IDS = list(range(1, 20))


def sample_single_band_tif(tif_path: Path, lon: float, lat: float) -> float:
    with rasterio.open(tif_path) as src:
        value = next(src.sample([(lon, lat)]))[0]
        nodata = src.nodata
    if nodata is not None and value == nodata:
        return float("nan")
    return float(value)


def sample_multiband_tif(tif_path: Path, lon: float, lat: float) -> dict[int, float]:
    with rasterio.open(tif_path) as src:
        values = next(src.sample([(lon, lat)]))
        nodata = src.nodata

    out = {}
    for i, v in enumerate(values, start=1):
        if i > 19:
            break
        if nodata is not None and v == nodata:
            out[i] = float("nan")
        else:
            out[i] = float(v)
    return out


def normalize_name(name: str) -> str:
    return name.lower().replace("-", "_")


def build_current_bio_file_index(bio_dir: Path) -> dict[int, Path]:
    tif_files = sorted(bio_dir.rglob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"Aucun fichier .tif trouvé dans {bio_dir}")

    index = {}
    for tif in tif_files:
        stem = normalize_name(tif.stem)
        m = re.search(r"(?:^|_)bio_?([0-9]{1,2})(?:_|$)", stem)
        if m:
            bio_id = int(m.group(1))
            if bio_id in BIO_IDS and bio_id not in index:
                index[bio_id] = tif

    missing = [bio_id for bio_id in BIO_IDS if bio_id not in index]
    if missing:
        raise RuntimeError(
            f"BIO manquants dans {bio_dir}: {missing}\n"
            f"Fichiers vus: {[p.name for p in tif_files[:50]]}"
        )
    return index


def sample_current_bio(current_index: dict[int, Path], lon: float, lat: float, prefix: str) -> dict:
    out = {}
    for bio_id in BIO_IDS:
        out[f"{prefix}_bio{bio_id}"] = sample_single_band_tif(current_index[bio_id], lon, lat)
    return out


def sample_future_bio_multiband(tif_path: Path, lon: float, lat: float, prefix: str) -> dict:
    if not tif_path.exists():
        raise FileNotFoundError(f"Fichier futur introuvable: {tif_path}")

    vals = sample_multiband_tif(tif_path, lon, lat)

    missing = [bio_id for bio_id in BIO_IDS if bio_id not in vals]
    if missing:
        raise RuntimeError(f"Bandes BIO manquantes dans {tif_path}: {missing}")

    return {f"{prefix}_bio{bio_id}": vals[bio_id] for bio_id in BIO_IDS}


def main():
    df = pd.read_csv(CITIES_CSV)

    current_index = build_current_bio_file_index(CURRENT_DIR)

    rows = []
    for _, row in df.iterrows():
        lon = float(row["lon"])
        lat = float(row["lat"])

        rec = row.to_dict()

        # Présent: 19 fichiers séparés
        rec.update(sample_current_bio(current_index, lon, lat, "current"))

        # Futur: 1 TIFF multibande par GCM
        for gcm_name, tif_path in FUTURE_FILES.items():
            rec.update(sample_future_bio_multiband(tif_path, lon, lat, f"future_{gcm_name}"))

        # Moyenne des 3 GCM
        for bio_id in BIO_IDS:
            cols = [
                f"future_BCC_CSM2_MR_bio{bio_id}",
                f"future_CNRM_CM6_1_bio{bio_id}",
                f"future_IPSL_CM6A_LR_bio{bio_id}",
            ]
            rec[f"future_mean_bio{bio_id}"] = sum(rec[c] for c in cols) / len(cols)

        rows.append(rec)

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(out)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()