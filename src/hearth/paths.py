from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"


@dataclass(frozen=True)
class LocationPaths:
    location: str

    @property
    def raw_dir(self) -> Path:
        return RAW_DIR / self.location

    @property
    def processed_dir(self) -> Path:
        return PROCESSED_DIR / self.location

    @property
    def sentinel_raw_dir(self) -> Path:
        return self.raw_dir / "sentinel"

    @property
    def landsat_raw_dir(self) -> Path:
        return self.raw_dir / "landsat"

    @property
    def grid_path(self) -> Path:
        return self.processed_dir / "grid_500m.geojson"

    @property
    def ndvi_raster_path(self) -> Path:
        return self.processed_dir / "ndvi.tif"

    @property
    def ndbi_raster_path(self) -> Path:
        return self.processed_dir / "ndbi.tif"

    @property
    def lst_raster_path(self) -> Path:
        return self.processed_dir / "lst.tif"

    @property
    def grid_ndvi_path(self) -> Path:
        return self.processed_dir / "grid_ndvi.parquet"

    @property
    def grid_ndbi_path(self) -> Path:
        return self.processed_dir / "grid_ndbi.parquet"

    @property
    def grid_lst_path(self) -> Path:
        return self.processed_dir / "grid_lst.parquet"

    @property
    def dataset_path(self) -> Path:
        return self.processed_dir / "dataset.parquet"

    def ensure_dirs(self) -> None:
        self.sentinel_raw_dir.mkdir(parents=True, exist_ok=True)
        self.landsat_raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)


def get_location_paths(location: str) -> LocationPaths:
    if not location or not location.strip():
        raise ValueError("location must be a non-empty string")
    return LocationPaths(location=location.strip().lower())


def find_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching '{pattern}' in {directory}")
    if len(matches) > 1:
        pretty = "\n".join(str(p) for p in matches)
        raise RuntimeError(
            f"Multiple files matching '{pattern}' in {directory}:\n{pretty}"
        )
    return matches[0]


def find_optional_one(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        pretty = "\n".join(str(p) for p in matches)
        raise RuntimeError(
            f"Multiple files matching '{pattern}' in {directory}:\n{pretty}"
        )
    return matches[0]


def find_sentinel_band(location: str, band_suffix: str) -> Path:
    """
    Examples:
        find_sentinel_band("toulouse", "B04_10m.jp2")
        find_sentinel_band("toulouse", "B08_10m.jp2")
    """
    paths = get_location_paths(location)
    return find_one(paths.sentinel_raw_dir, f"*_{band_suffix}")


def find_landsat_file(location: str, suffix: str) -> Path:
    """
    Examples:
        find_landsat_file("toulouse", "ST_B10.TIF")
        find_landsat_file("toulouse", "SR_B5.TIF")
        find_landsat_file("toulouse", "SR_B6.TIF")
        find_landsat_file("toulouse", "MTL.txt")
    """
    paths = get_location_paths(location)
    return find_one(paths.landsat_raw_dir, f"*_{suffix}")


def find_sentinel_red(location: str) -> Path:
    return find_sentinel_band(location, "B04_10m.jp2")


def find_sentinel_nir(location: str) -> Path:
    return find_sentinel_band(location, "B08_10m.jp2")


def find_landsat_st_b10(location: str) -> Path:
    return find_landsat_file(location, "ST_B10.TIF")


def find_landsat_sr_b5(location: str) -> Path:
    return find_landsat_file(location, "SR_B5.TIF")


def find_landsat_sr_b6(location: str) -> Path:
    return find_landsat_file(location, "SR_B6.TIF")


def find_landsat_mtl(location: str) -> Path:
    return find_landsat_file(location, "MTL.txt")