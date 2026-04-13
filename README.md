# Urban Microclimate MVP

Estimate the relationship between vegetation, built-up intensity and surface temperature using satellite data.

## Pipeline

1. Sentinel-2 → NDVI
2. Landsat → LST + NDBI
3. Spatial aggregation (500m grid)
4. Linear model

## Result

LST ≈ -13.6 * NDVI + 31.3 * NDBI + 46.9  
R² ≈ 0.48

## Usage

```bash
python scripts/run_pipeline.py
python scripts/build_dataset.py
python scripts/train_model.py