from __future__ import annotations

import argparse
import calendar
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import planetary_computer
import requests
from pystac_client import Client
from shapely.geometry import box, shape

from hearth.locations import get_location_config
from hearth.paths import get_location_paths


PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


@dataclass
class SceneSelection:
    collection: str
    item_id: str
    datetime: str
    cloud_cover: float | None
    platform: str | None
    coverage_ratio: float | None
    assets: dict[str, str]          # optional convenience only
    asset_keys: dict[str, str]      # stable source of truth
    properties: dict[str, Any]


@dataclass
class Manifest:
    location: str
    label: str
    bbox: list[float]
    generated_at: str
    sentinel: SceneSelection | None
    landsat: SceneSelection | None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _month_interval(date_pref: str) -> str:
    year, month = date_pref.split("-")
    year_i = int(year)
    month_i = int(month)
    last_day = calendar.monthrange(year_i, month_i)[1]
    return f"{year_i:04d}-{month_i:02d}-01/{year_i:04d}-{month_i:02d}-{last_day:02d}"


def _resolve_datetime_interval(date_pref: str | None) -> str | None:
    if not date_pref:
        return None
    date_pref = date_pref.strip()
    if "/" in date_pref:
        return date_pref
    if len(date_pref) == 7:
        return _month_interval(date_pref)
    if len(date_pref) == 10:
        return f"{date_pref}/{date_pref}"
    raise ValueError(f"Unsupported date format '{date_pref}'")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _cloud_cover(item) -> float | None:
    props = item.properties or {}
    for key in (
        "eo:cloud_cover",
        "s2:cloud_cover",
        "landsat:cloud_cover_land",
        "landsat:cloud_cover",
    ):
        val = _safe_float(props.get(key))
        if val is not None:
            return val
    return None


def _platform(item) -> str | None:
    props = item.properties or {}
    for key in ("platform", "constellation", "mission", "instruments"):
        value = props.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            return ",".join(map(str, value))
        return str(value)
    return None


def _download_file(url: str, dst: Path, chunk_size: int = 1024 * 1024) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        print(f"[skip] already exists: {dst}")
        return

    print(f"[download] {dst.name}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)


def _open_catalog() -> Client:
    return Client.open(
        PC_STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )


def _search_items(
    *,
    collection: str,
    bbox: list[float],
    dt: str | None,
    max_cloud: float | None,
) -> list[Any]:
    catalog = _open_catalog()

    query: dict[str, Any] = {}
    if max_cloud is not None:
        query["eo:cloud_cover"] = {"lte": float(max_cloud)}

    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=dt,
        query=query or None,
    )
    return list(search.items())


def _find_first_available_asset_key(item, candidates: list[str]) -> str | None:
    for key in candidates:
        if key in item.assets:
            return key
    return None


def _extract_asset_map(item, logical_to_candidates: dict[str, list[str]]) -> tuple[dict[str, str], dict[str, str]] | None:
    resolved_asset_keys: dict[str, str] = {}
    resolved_assets: dict[str, str] = {}

    for logical_name, candidates in logical_to_candidates.items():
        asset_key = _find_first_available_asset_key(item, candidates)
        if asset_key is None:
            return None
        resolved_asset_keys[logical_name] = asset_key
        resolved_assets[logical_name] = item.assets[asset_key].href

    return resolved_assets, resolved_asset_keys


def _coverage_ratio(item, bbox_wgs84: list[float]) -> float | None:
    if item.geometry is None:
        return None

    try:
        item_geom = shape(item.geometry)
        aoi_geom = box(*bbox_wgs84)

        if item_geom.is_empty or aoi_geom.is_empty:
            return None

        inter = item_geom.intersection(aoi_geom)
        if inter.is_empty:
            return 0.0

        aoi_area = aoi_geom.area
        if aoi_area <= 0:
            return None

        return float(inter.area / aoi_area)
    except Exception:
        return None


def _sort_items_by_cloud(items: Iterable[Any]) -> list[Any]:
    return sorted(
        items,
        key=lambda item: (
            9999.0 if _cloud_cover(item) is None else _cloud_cover(item),
            str(item.datetime) if item.datetime is not None else "",
            item.id,
        ),
    )


def _select_best_item(
    *,
    items: list[Any],
    logical_to_candidates: dict[str, list[str]],
    label: str,
) -> SceneSelection:
    if not items:
        raise RuntimeError(f"No {label} item found for the requested location/date constraints")

    ranked = _sort_items_by_cloud(items)

    for item in ranked:
        extracted = _extract_asset_map(item, logical_to_candidates)
        if extracted is None:
            continue

        assets, asset_keys = extracted
        props = item.properties or {}

        return SceneSelection(
            collection=str(item.collection_id),
            item_id=item.id,
            datetime=str(item.datetime),
            cloud_cover=_cloud_cover(item),
            platform=_platform(item),
            coverage_ratio=None,
            assets=assets,
            asset_keys=asset_keys,
            properties={
                "eo:cloud_cover": props.get("eo:cloud_cover"),
                "s2:cloud_cover": props.get("s2:cloud_cover"),
                "landsat:cloud_cover_land": props.get("landsat:cloud_cover_land"),
                "landsat:cloud_cover": props.get("landsat:cloud_cover"),
            },
        )

    raise RuntimeError(f"No {label} item with all required assets")


def _select_best_sentinel_item(
    *,
    items: list[Any],
    bbox_wgs84: list[float],
    logical_to_candidates: dict[str, list[str]],
) -> SceneSelection:
    if not items:
        raise RuntimeError("No Sentinel-2 item found for the requested location/date constraints")

    candidates: list[tuple[float, float, Any, dict[str, str], dict[str, str]]] = []

    for item in items:
        extracted = _extract_asset_map(item, logical_to_candidates)
        if extracted is None:
            continue

        assets, asset_keys = extracted
        coverage = _coverage_ratio(item, bbox_wgs84)
        cloud = _cloud_cover(item)

        coverage_score = -1.0 if coverage is None else coverage
        cloud_score = 9999.0 if cloud is None else cloud

        candidates.append((coverage_score, cloud_score, item, assets, asset_keys))

    if not candidates:
        raise RuntimeError("No Sentinel-2 item with all required assets")

    # Coverage first, then clouds
    candidates.sort(key=lambda x: (-x[0], x[1], str(x[2].datetime), x[2].id))
    best_cov, _, best_item, best_assets, best_asset_keys = candidates[0]

    props = best_item.properties or {}

    print(f"[sentinel] candidates with assets: {len(candidates)}")
    print(f"[sentinel] selected coverage_ratio={best_cov:.4f}, cloud={_cloud_cover(best_item)}")

    return SceneSelection(
        collection=str(best_item.collection_id),
        item_id=best_item.id,
        datetime=str(best_item.datetime),
        cloud_cover=_cloud_cover(best_item),
        platform=_platform(best_item),
        coverage_ratio=None if best_cov < 0 else best_cov,
        assets=best_assets,
        asset_keys=best_asset_keys,
        properties={
            "eo:cloud_cover": props.get("eo:cloud_cover"),
            "s2:cloud_cover": props.get("s2:cloud_cover"),
        },
    )


def fetch_best_sentinel(location: str) -> SceneSelection:
    cfg = get_location_config(location)

    date_pref = None
    max_cloud = None
    if cfg.preferred_dates:
        date_pref = cfg.preferred_dates.get("sentinel")
    if cfg.max_cloud_cover:
        max_cloud = cfg.max_cloud_cover.get("sentinel")

    dt = _resolve_datetime_interval(date_pref)

    items = _search_items(
        collection="sentinel-2-l2a",
        bbox=list(cfg.bbox_tuple),
        dt=dt,
        max_cloud=max_cloud,
    )

    logical_to_candidates = {
        "red": ["B04"],
        "nir": ["B08"],
        "swir": ["B11"],
        "qa": ["SCL"],
    }

    return _select_best_sentinel_item(
        items=items,
        bbox_wgs84=list(cfg.bbox_tuple),
        logical_to_candidates=logical_to_candidates,
    )


def fetch_best_landsat(location: str) -> SceneSelection:
    cfg = get_location_config(location)

    date_pref = None
    max_cloud = None
    if cfg.preferred_dates:
        date_pref = cfg.preferred_dates.get("landsat")
    if cfg.max_cloud_cover:
        max_cloud = cfg.max_cloud_cover.get("landsat")

    dt = _resolve_datetime_interval(date_pref)

    items = _search_items(
        collection="landsat-c2-l2",
        bbox=list(cfg.bbox_tuple),
        dt=dt,
        max_cloud=max_cloud,
    )

    logical_to_candidates = {
        "lst": ["ST_B10", "ST_B6", "lwir11", "lwir", "thermal", "thermal_infrared"],
        "qa": ["QA_PIXEL", "qa_pixel", "pixel_qa"],
        "red": ["SR_B4", "SR_B3", "red", "red-band"],
        "nir": ["SR_B5", "SR_B4", "nir08", "nir", "nir-band"],
        "swir": ["SR_B6", "SR_B5", "swir16", "swir", "swir1"],
    }

    return _select_best_item(
        items=items,
        logical_to_candidates=logical_to_candidates,
        label="Landsat",
    )


def _manifest_path(location: str) -> Path:
    paths = get_location_paths(location)
    return paths.raw_dir / "scene_manifest.json"


def _write_manifest(location: str, sentinel: SceneSelection | None, landsat: SceneSelection | None) -> Path:
    cfg = get_location_config(location)
    manifest = Manifest(
        location=cfg.name,
        label=cfg.label,
        bbox=list(cfg.bbox_tuple),
        generated_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        sentinel=sentinel,
        landsat=landsat,
    )

    out_path = _manifest_path(location)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    return out_path


def _filename_for_asset(scene: SceneSelection, logical_name: str, suffix: str = ".tif") -> str:
    stem = scene.item_id.replace("/", "_")
    return f"{stem}_{logical_name}{suffix}"


def download_sentinel_assets(location: str, scene: SceneSelection) -> None:
    paths = get_location_paths(location)
    out_dir = paths.sentinel_raw_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for logical_name, url in scene.assets.items():
        dst = out_dir / _filename_for_asset(scene, logical_name)
        _download_file(url, dst)


def download_landsat_assets(location: str, scene: SceneSelection) -> None:
    paths = get_location_paths(location)
    out_dir = paths.landsat_raw_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for logical_name, url in scene.assets.items():
        dst = out_dir / _filename_for_asset(scene, logical_name)
        _download_file(url, dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select and optionally download satellite assets for a configured location."
    )
    parser.add_argument("--location", required=True)
    parser.add_argument("--sentinel-only", action="store_true")
    parser.add_argument("--landsat-only", action="store_true")
    parser.add_argument("--download-assets", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    location = args.location.strip().lower()

    paths = get_location_paths(location)
    paths.ensure_dirs()
    cfg = get_location_config(location)

    manifest_path = _manifest_path(location)

    print(f"Location: {cfg.name} ({cfg.label})")
    print(f"BBox: {cfg.bbox_tuple}")
    print(f"Raw dir: {paths.raw_dir}")

    if manifest_path.exists() and not args.force:
        print(f"[skip] manifest already exists: {manifest_path}")
        print("Use --force to re-run selection.")
        return

    do_sentinel = not args.landsat_only
    do_landsat = not args.sentinel_only

    sentinel_scene: SceneSelection | None = None
    landsat_scene: SceneSelection | None = None

    if do_sentinel:
        sentinel_scene = fetch_best_sentinel(location)
        print(
            f"[sentinel] item={sentinel_scene.item_id} "
            f"datetime={sentinel_scene.datetime} "
            f"cloud={sentinel_scene.cloud_cover} "
            f"coverage={sentinel_scene.coverage_ratio} "
            f"assets={sentinel_scene.asset_keys}"
        )

    if do_landsat:
        landsat_scene = fetch_best_landsat(location)
        print(
            f"[landsat] item={landsat_scene.item_id} "
            f"datetime={landsat_scene.datetime} "
            f"cloud={landsat_scene.cloud_cover} "
            f"assets={landsat_scene.asset_keys}"
        )

    out_manifest = _write_manifest(location, sentinel_scene, landsat_scene)
    print(f"[ok] manifest written: {out_manifest}")

    if args.download_assets:
        if sentinel_scene is not None:
            download_sentinel_assets(location, sentinel_scene)
        if landsat_scene is not None:
            download_landsat_assets(location, landsat_scene)

    print("Done.")


if __name__ == "__main__":
    main()