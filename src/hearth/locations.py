from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from hearth.paths import METADATA_DIR


@dataclass(frozen=True)
class LocationConfig:
    name: str
    label: str
    bbox_wgs84: dict[str, float]
    center_wgs84: dict[str, float]
    grid_size_m: int = 500
    timezone: str = "UTC"
    preferred_dates: dict[str, str] | None = None
    max_cloud_cover: dict[str, float] | None = None

    @property
    def min_lon(self) -> float:
        return float(self.bbox_wgs84["min_lon"])

    @property
    def min_lat(self) -> float:
        return float(self.bbox_wgs84["min_lat"])

    @property
    def max_lon(self) -> float:
        return float(self.bbox_wgs84["max_lon"])

    @property
    def max_lat(self) -> float:
        return float(self.bbox_wgs84["max_lat"])

    @property
    def bbox_tuple(self) -> tuple[float, float, float, float]:
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)

    @property
    def center_lon(self) -> float:
        return float(self.center_wgs84["lon"])

    @property
    def center_lat(self) -> float:
        return float(self.center_wgs84["lat"])


def _validate_bbox(name: str, bbox: dict[str, Any]) -> None:
    required = {"min_lon", "min_lat", "max_lon", "max_lat"}
    missing = required - set(bbox.keys())
    if missing:
        raise ValueError(f"Location '{name}' is missing bbox keys: {sorted(missing)}")

    min_lon = float(bbox["min_lon"])
    min_lat = float(bbox["min_lat"])
    max_lon = float(bbox["max_lon"])
    max_lat = float(bbox["max_lat"])

    if min_lon >= max_lon:
        raise ValueError(f"Location '{name}' has invalid longitude bounds")
    if min_lat >= max_lat:
        raise ValueError(f"Location '{name}' has invalid latitude bounds")


def _validate_center(name: str, center: dict[str, Any]) -> None:
    required = {"lon", "lat"}
    missing = required - set(center.keys())
    if missing:
        raise ValueError(f"Location '{name}' is missing center keys: {sorted(missing)}")


def _default_locations_path() -> Path:
    return METADATA_DIR / "locations.yaml"


def load_locations(path: str | Path | None = None) -> dict[str, LocationConfig]:
    cfg_path = Path(path) if path is not None else _default_locations_path()

    if not cfg_path.exists():
        raise FileNotFoundError(f"Locations file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Invalid locations file format in {cfg_path}")

    out: dict[str, LocationConfig] = {}

    for name, cfg in raw.items():
        if not isinstance(cfg, dict):
            raise ValueError(f"Location '{name}' must map to a dictionary")

        if "label" not in cfg:
            raise ValueError(f"Location '{name}' is missing 'label'")
        if "bbox_wgs84" not in cfg:
            raise ValueError(f"Location '{name}' is missing 'bbox_wgs84'")
        if "center_wgs84" not in cfg:
            raise ValueError(f"Location '{name}' is missing 'center_wgs84'")

        _validate_bbox(name, cfg["bbox_wgs84"])
        _validate_center(name, cfg["center_wgs84"])

        out[name.lower()] = LocationConfig(
            name=name.lower(),
            label=str(cfg["label"]),
            bbox_wgs84=cfg["bbox_wgs84"],
            center_wgs84=cfg["center_wgs84"],
            grid_size_m=int(cfg.get("grid_size_m", 500)),
            timezone=str(cfg.get("timezone", "UTC")),
            preferred_dates=cfg.get("preferred_dates"),
            max_cloud_cover=cfg.get("max_cloud_cover"),
        )

    return out


def get_location_config(location: str, path: str | Path | None = None) -> LocationConfig:
    key = location.strip().lower()
    locations = load_locations(path)

    if key not in locations:
        available = ", ".join(sorted(locations.keys()))
        raise KeyError(f"Unknown location '{location}'. Available: {available}")

    return locations[key]