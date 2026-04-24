from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Climate Analog Explorer", layout="wide")

# Defaults matching compute_city_analogues_updated.py with:
# MATCH_MODE="weighted_normalized", DIVERSIFICATION_MODE="hard", MIN_ANALOG_SPACING_KM=750
DEFAULT_RESULTS = Path("data/processed/city_analogues_weighted_normalized_div-hard-750km.csv")
DEFAULT_VALID = Path("data/processed/city_analogues_valid_weighted_normalized_div-hard-750km.csv")

LEGACY_RESULTS = Path("data/processed/city_analogues_weighted_normalized.csv")
LEGACY_VALID = Path("data/processed/city_analogues_valid_weighted_normalized.csv")


@st.cache_data
def load_results(results_path: str, valid_path: str):
    results = pd.read_csv(results_path)
    valid = pd.read_csv(valid_path)
    return results, valid


def selected_row_as_display_df(row: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "field": [str(k) for k in row.index],
        "value": ["" if pd.isna(v) else str(v) for v in row.values],
    })


def build_city_key(df: pd.DataFrame, city_col: str, country_col: str):
    return df[city_col].astype(str) + " — " + df[country_col].astype(str)


def safe_metric(value, digits=2, suffix=""):
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}{suffix}"
    return str(value)


def get_equator_shift(row_or_series, rank=1):
    new_col = f"analog_{rank}_equator_shift_km_signed"
    old_col = f"analog_{rank}_lat_shift_km_signed"
    if new_col in row_or_series.index:
        return row_or_series.get(new_col)
    return row_or_series.get(old_col)


def format_shift(value):
    if value is None or pd.isna(value):
        return "NA"
    v = float(value)
    if abs(v) < 1e-9:
        return "0 km"
    direction = "towards equator" if v > 0 else "away from equator"
    return f"{abs(v):.0f} km {direction}"


def get_top_k(row: pd.Series) -> int:
    ranks = []
    for c in row.index:
        if c.startswith("analog_") and c.endswith("_city"):
            try:
                ranks.append(int(c.split("_")[1]))
            except Exception:
                pass
    return max(ranks) if ranks else 0


def get_valid_lookup(valid_df: pd.DataFrame):
    required = {"city", "country", "lat", "lon"}
    if not required.issubset(set(valid_df.columns)):
        return {}

    lookup = {}
    for _, row in valid_df.iterrows():
        lookup[(str(row["city"]), str(row["country"]))] = {
            "lat": row["lat"],
            "lon": row["lon"],
            "population": row["population"] if "population" in valid_df.columns else None,
            "is_country_capital": row["is_country_capital"] if "is_country_capital" in valid_df.columns else None,
        }
    return lookup


def analog_table_for_row(row: pd.Series):
    rows = []
    for rank in range(1, get_top_k(row) + 1):
        if f"analog_{rank}_city" not in row.index:
            continue
        rows.append({
            "rank": rank,
            "city": row.get(f"analog_{rank}_city"),
            "country": row.get(f"analog_{rank}_country"),
            "score": row.get(f"analog_{rank}_score"),
            "climate_distance": row.get(f"analog_{rank}_climate_distance"),
            "urban_distance": row.get(f"analog_{rank}_urban_distance"),
            "geo_distance_km": row.get(f"analog_{rank}_geo_distance_km"),
            "equator_shift_km_signed": row.get(f"analog_{rank}_equator_shift_km_signed", row.get(f"analog_{rank}_lat_shift_km_signed")),
            "lat_distance_km_abs": row.get(f"analog_{rank}_lat_distance_km_abs"),
            "lon_distance_km_abs": row.get(f"analog_{rank}_lon_distance_km_abs"),
            "towards_equator": row.get(f"analog_{rank}_is_towards_equator"),
            "min_dist_to_previous_km": row.get(f"analog_{rank}_min_distance_to_previous_analogs_km"),
            "population": row.get(f"analog_{rank}_population"),
            "is_country_capital": row.get(f"analog_{rank}_is_country_capital"),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        numeric_cols = [
            "score", "climate_distance", "urban_distance", "geo_distance_km",
            "equator_shift_km_signed", "lat_distance_km_abs", "lon_distance_km_abs",
            "min_dist_to_previous_km", "population",
        ]
        for c in numeric_cols:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def coords_for_city(city: str, country: str, rank: int, target_row: pd.Series, valid_lookup: dict):
    if rank == 0:
        info = valid_lookup.get((city, country))
        if info:
            return info.get("lat"), info.get("lon")
        return None, None

    lat = target_row.get(f"analog_{rank}_lat")
    lon = target_row.get(f"analog_{rank}_lon")
    if pd.notna(lat) and pd.notna(lon):
        return lat, lon

    info = valid_lookup.get((city, country))
    if info:
        return info.get("lat"), info.get("lon")
    return None, None


def build_map_df(target_row: pd.Series, analog_df: pd.DataFrame, valid_lookup: dict):
    points = []

    target_city = str(target_row["future_city"])
    target_country = str(target_row["future_country"])
    target_lat, target_lon = coords_for_city(target_city, target_country, 0, target_row, valid_lookup)

    if pd.notna(target_lat) and pd.notna(target_lon):
        points.append({
            "label": f"{target_city} ({target_country})",
            "kind": "Target",
            "lat": float(target_lat),
            "lon": float(target_lon),
            "rank": 0,
            "equator_shift_km_signed": 0.0,
        })

    for _, analog in analog_df.iterrows():
        rank = int(analog["rank"])
        city = str(analog["city"])
        country = str(analog["country"])
        lat, lon = coords_for_city(city, country, rank, target_row, valid_lookup)
        if pd.notna(lat) and pd.notna(lon):
            points.append({
                "label": f"{city} ({country})",
                "kind": f"Analog #{rank}",
                "lat": float(lat),
                "lon": float(lon),
                "rank": rank,
                "equator_shift_km_signed": analog.get("equator_shift_km_signed"),
            })

    return pd.DataFrame(points)


def render_map(map_df: pd.DataFrame):
    if map_df.empty:
        st.info("Map unavailable: missing lat/lon.")
        return

    display_df = map_df.copy()
    display_df["lat"] = pd.to_numeric(display_df["lat"], errors="coerce")
    display_df["lon"] = pd.to_numeric(display_df["lon"], errors="coerce")
    display_df = display_df.dropna(subset=["lat", "lon"])

    if display_df.empty:
        st.info("Map unavailable: all map points have invalid coordinates.")
        return

    display_df["size"] = display_df["rank"].map(lambda r: 900 if r == 0 else 450)

    st.map(
        display_df,
        latitude="lat",
        longitude="lon",
        size="size",
        zoom=1,
    )

    st.dataframe(
        display_df[["kind", "label", "lat", "lon", "equator_shift_km_signed"]],
        width="stretch",
        hide_index=True,
    )


st.title("Climate Analog Explorer")
st.caption(
    "Explore future climate analogs using PCA climate similarity, light urban matching, "
    "spatial diversification, and interpretable equator-shift metrics."
)

with st.sidebar:
    st.header("Data")

    default_results = DEFAULT_RESULTS if DEFAULT_RESULTS.exists() else LEGACY_RESULTS
    default_valid = DEFAULT_VALID if DEFAULT_VALID.exists() else LEGACY_VALID

    results_path = st.text_input("Results CSV", str(default_results))
    valid_path = st.text_input("Valid cities CSV", str(default_valid))

    st.markdown(
        """
        **Expected files**
        - results: output of `compute_city_analogues_updated.py`
        - valid cities: table with `city`, `country`, `lat`, `lon`

        The app remains compatible with the older CSV, but the north/south metrics only appear with the updated script output.
        """
    )

if not Path(results_path).exists():
    st.error(f"Results file not found: {results_path}")
    st.stop()

if not Path(valid_path).exists():
    st.warning(f"Valid cities file not found: {valid_path}. Map and some metadata may be unavailable.")

results_df, valid_df = load_results(results_path, valid_path if Path(valid_path).exists() else results_path)

future_city_col = "future_city"
future_country_col = "future_country"

if future_city_col not in results_df.columns or future_country_col not in results_df.columns:
    st.error("Results CSV must contain `future_city` and `future_country` columns.")
    st.stop()

results_df = results_df.copy()
results_df["city_key"] = build_city_key(results_df, future_city_col, future_country_col)
results_df = results_df.sort_values("city_key").reset_index(drop=True)

valid_lookup = get_valid_lookup(valid_df) if Path(valid_path).exists() else {}

with st.sidebar:
    query = st.text_input("Search city", "")
    filtered_keys = results_df["city_key"].tolist()
    if query.strip():
        q = query.strip().lower()
        filtered_keys = [k for k in filtered_keys if q in k.lower()]
    selected_key = st.selectbox("Target city", filtered_keys, index=0 if filtered_keys else None)

if not filtered_keys:
    st.warning("No city matches your search.")
    st.stop()

selected_row = results_df.loc[results_df["city_key"] == selected_key].iloc[0]
analog_df = analog_table_for_row(selected_row)

left, right = st.columns([1.25, 1])

with left:
    st.subheader(f"Target: {selected_row['future_city']} — {selected_row['future_country']}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Self distance", safe_metric(selected_row.get("self_distance")))
    c2.metric("Top-1 climate", safe_metric(selected_row.get("analog_1_climate_distance")))
    c3.metric("Top-1 geo", safe_metric(selected_row.get("analog_1_geo_distance_km"), digits=0, suffix=" km"))
    c4.metric("Top-1 equator shift", format_shift(get_equator_shift(selected_row, 1)))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Top-1 lon shift", safe_metric(selected_row.get("analog_1_lon_distance_km_abs"), digits=0, suffix=" km"))
    c6.metric("Top-1 urban", safe_metric(selected_row.get("analog_1_urban_distance")))
    c7.metric("Towards equator", str(selected_row.get("analog_1_is_towards_equator", "NA")))
    c8.metric("Analogs found", str(len(analog_df)))

    st.markdown("### Top analogs")
    if analog_df.empty:
        st.warning("No analog columns found in the results CSV.")
    else:
        display_cols = [
            "rank", "city", "country", "score", "climate_distance", "geo_distance_km",
            "equator_shift_km_signed", "lon_distance_km_abs", "towards_equator",
            "min_dist_to_previous_km", "population", "is_country_capital",
        ]
        display_cols = [c for c in display_cols if c in analog_df.columns]
        st.dataframe(
            analog_df[display_cols],
            width="stretch",
            hide_index=True,
        )

    with st.expander("Raw selected row"):
        st.dataframe(
            selected_row_as_display_df(selected_row),
            width="stretch",
            hide_index=True,
        )

with right:
    st.markdown("### Map")
    map_df = build_map_df(selected_row, analog_df, valid_lookup)
    render_map(map_df)

st.markdown("---")
st.markdown("### Notes")
st.markdown(
    """
    - `self_distance` tells how far the future city drifts from its current climate position.
    - `climate_distance` is the main PCA-space climate distance.
    - `urban_distance` is the auxiliary population/capital penalty used in matching.
    - `geo_distance_km` is the geographic distance between target city and analog.
    - `equator_shift_km_signed` is positive when the analog is closer to the equator than the target. It remains interpretable even if target and analog are in different hemispheres.
    - `min_dist_to_previous_km` helps verify that diversification is actually spacing selected analogs apart.
    """
)
