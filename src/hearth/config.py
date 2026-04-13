from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Toulouse area, volontairement un peu large pour le MVP
TOULOUSE_BBOX_WGS84 = {
    "min_lon": 1.20,
    "min_lat": 43.50,
    "max_lon": 1.60,
    "max_lat": 43.72,
}

# Projection métrique pratique pour Toulouse
WORK_CRS = "EPSG:2154"   # Lambert-93
WGS84 = "EPSG:4326"

GRID_SIZE_M = 500