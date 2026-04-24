# Climate analog cities

A small research-oriented repository to explore **city climate analogues**.

The core question is:

> For a given city and a future climate scenario, which current city has the most similar climate?

The project started as a practical reimplementation inspired by the paper **“Understanding climate change from a global analysis of city analogues”**, then evolved toward a slightly more product-oriented version with a larger city catalog and a small urban-similarity correction.

## Current objective

The repository builds a pipeline that:

1. downloads climate rasters,
2. builds a global candidate city catalog,
3. extracts climate descriptors for each city,
4. projects climates into a PCA space,
5. computes present-day analogues for future city climates.

The current focus is still **methodology + data pipeline**, not yet a polished end-user application.

## Working assumptions

### Climate side

- Present climate is represented with the **19 BIOCLIM variables**.
- Future climate is represented with a **future scenario raster stack**, currently averaged across multiple GCMs in the pipeline already prepared in the repo.
- PCA is fitted on the **current** climate space and future cities are projected into that same PCA space.
- Distance in PCA space is the main climate similarity metric.

### City side

- A larger world city catalog is built from **GeoNames**.
- Candidate cities can optionally be spatially deduplicated to avoid overly local matches such as suburb-to-core-city matches.
- A small **urban similarity** component is included in the matching score, using:
  - `population`
  - `is_country_capital`

### Final matching choice

The current recommended matching mode is:

- **mode**: `weighted_normalized`
- **urban lambda**: `0.25`

This was chosen because it produced the best compromise during the parameter sweep:
- still strongly climate-driven,
- but less naive than pure climate-only matching,
- and more useful in practice than reranking-only variants.

## Repository logic

Typical data flow:

1. `download_worldclim_data.py`
2. `download_geonames.py`
3. `build_candidate_cities_from_geonames.py`
4. `deduplicate_candidate_cities.py` *(optional but recommended when available)*
5. `extract_bioclim_for_cities.py`
6. `compute_city_analogues.py`

## Starter pack

A convenience script is provided to launch the full pipeline.

### Run everything

```bash
python3 run_pipeline.py --prefer-dedup
```

### Useful variants

Skip already completed downloads:

```bash
python3 run_pipeline.py --skip-climate-download --skip-geonames-download --prefer-dedup
```

Recompute only the final analogue step:

```bash
python3 run_pipeline.py \
  --skip-climate-download \
  --skip-geonames-download \
  --skip-city-build \
  --skip-dedup \
  --skip-extract
```

## Main outputs

After a normal run, the most useful files are typically:

- `data/interim/cities_bioclim.csv`
- `data/processed/city_analogues_weighted_normalized.csv`
- `data/processed/scaler.joblib`
- `data/processed/pca_model.joblib`

## Tuning guide

### 1. Change the city catalog

The strongest practical effect often comes from the candidate city catalog.

You can tune:

- GeoNames population threshold,
- whether national capitals are always kept,
- spatial deduplication distance,
- whether to keep only large cities or a denser urban catalog.

This dramatically changes the geographic distance of analogues.

### 2. Change the matching mode

In `compute_city_analogues.py`:

- `MATCH_MODE = "climate_only"`
  - pure climate nearest neighbor,
  - simplest scientific baseline,
  - often returns very local analogues when the candidate catalog is dense.

- `MATCH_MODE = "weighted"`
  - adds raw urban distance,
  - useful mostly for debugging or quick experiments.

- `MATCH_MODE = "weighted_normalized"`
  - recommended mode,
  - balances normalized climate distance and normalized urban distance.

### 3. Change the urban penalty

In `compute_city_analogues.py`:

- `URBAN_LAMBDA`
  - global strength of the urban correction.
  - current default: `0.25`

- `URBAN_WEIGHT_POP`
  - internal weight of log-population gap.

- `URBAN_WEIGHT_CAPITAL`
  - internal penalty for mismatched capital status.

Rule of thumb:

- lower lambda → more climate-pure,
- higher lambda → more city-profile-aware,
- too high → risk of degrading climate plausibility.

### 4. Change the number of PCA components

`N_PCS = 4` is currently aligned with the working approach used here.
You can test other values, but it is usually better to keep this stable unless there is a clear methodological reason.

### 5. Improve robustness near coasts / nodata areas

Some cities can fail climate extraction because of missing raster values.
Typical fixes are:

- nearest valid pixel fallback,
- small local search window,
- excluding problematic cities from the final catalogue.

## Current limitations

- City matching still depends heavily on candidate catalog design.
- Some analogues can remain unintuitive without further spatial or semantic constraints.
- Urban similarity is intentionally simple for now.
- The project is not yet packaged as a clean installable library.
- Frontend / visualization is not done yet.

## Work in progress

### In progress

- **Web frontend** for exploring analog cities interactively.

Planned frontend goals:

- city search,
- map interaction,
- display of top analogue cities,
- climate and geographic distance display,
- possibly PCA-space and scenario comparison views.

### Likely next backend improvements

- better city deduplication logic,
- optional geographic penalty to avoid hyper-local analogues,
- support for multiple future horizons,
- support for multiple catalog profiles (paper-like vs dense practical catalog).

## Recommended usage philosophy

It is useful to think of the project in two modes:

1. **paper-like mode**
   - fewer, larger cities,
   - easier to compare with published analog-city narratives.

2. **practical mode**
   - denser city catalog,
   - more realistic for an exploratory product,
   - can yield closer analogues because the search space is much denser.

Both are valid, but they answer slightly different questions.

## Suggested repo entry point for newcomers

If someone clones the repo and wants a first successful run:

```bash
python3 run_pipeline.py --prefer-dedup
```

Then inspect:

- `data/processed/city_analogues_weighted_normalized.csv`

That file is currently the best default output of the project.
