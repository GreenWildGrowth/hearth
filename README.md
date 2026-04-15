# Hearth

**Hearth** is a work-in-progress pipeline for exploring urban microclimate patterns from satellite data.

It estimates relationships between:

* vegetation (`NDVI`)
* built environment (`NDBI`)
* land surface temperature (`LST`)

for a given city, using publicly available satellite imagery.

---

## Overview

For each configured location, the pipeline:

1. selects satellite scenes (Sentinel-2, Landsat)
2. builds a spatial grid
3. computes:

   * NDVI (vegetation)
   * NDBI (built-up index)
   * LST (surface temperature)
4. aggregates raster values to grid cells
5. fits simple statistical models to explore relationships

The goal is **exploration and comparison across cities**, not production modeling.

---

## Status

> ⚠️ **Work in progress**

This repository is an evolving research prototype.

### What works

* end-to-end pipeline per location
* STAC-based scene selection (Planetary Computer)
* NDVI / NDBI / LST computation
* grid aggregation
* basic statistical modeling
* multi-city comparisons

### What is still evolving

* scene selection heuristics
* masking (clouds, water, etc.)
* interpretation of coefficients
* additional explanatory variables (e.g. altitude, distance to coast)
* model robustness

---

## Quick start

### 1. Configure a location

Edit `locations.yaml` and add a new entry:

```yaml
paris:
  label: "Paris"
  bbox_wgs84:
    min_lon: 2.2
    min_lat: 48.75
    max_lon: 2.5
    max_lat: 48.95
  center_wgs84:
    lon: 2.3522
    lat: 48.8566
  grid_size_m: 500
  timezone: "Europe/Paris"
  preferred_dates:
    sentinel: "2025-07"
    landsat: "2025-07"
  max_cloud_cover:
    sentinel: 10
    landsat: 10
```

---

### 2. Run the full pipeline

```bash
python scripts/run_location.py --location paris --force-fetch
```

This executes:

* data fetching
* grid generation
* NDVI / NDBI / LST computation
* aggregation
* model training

---

## Outputs

For each location:

```text
data/
  raw/<location>/
    scene_manifest.json

  processed/<location>/
    grid_500m.geojson
    ndvi.tif
    ndbi.tif
    lst.tif
    dataset.parquet
    model_report.json
```

---

## Modeling

The current modeling step includes:

* linear regression
* train/test split
* correlation analysis
* standardized coefficients
* optional interaction term (`NDVI × NDBI`)

This is designed for **interpretability**, not predictive performance.

---

## Interpretation

Results can vary significantly across cities.

### Typical observations

**Inland cities:**

* `NDBI ↑ → LST ↑`
* `NDVI ↑ → LST ↓`

**Coastal or complex urban environments:**

* relationships can invert
* indices may act as spatial proxies rather than causal drivers

> ⚠️ A coefficient should **not** be interpreted as causal without additional variables.

---

## Limitations

* strong correlation between NDVI and NDBI
* no full cloud / QA masking yet
* no terrain / coastal effects modeled
* single-scene selection (no compositing)
* linear models only

---

## Future work

Planned improvements:

* better scene compositing
* cloud / water masking
* additional features:

  * elevation
  * distance to coast
  * land use
* multi-city modeling
* non-linear models

---

## Requirements

Typical dependencies:

* geopandas
* rasterio
* numpy
* pandas
* scikit-learn
* shapely
* pystac-client
* planetary-computer

---

## Disclaimer

This project is experimental.

Results should be interpreted cautiously, especially across different urban contexts.
