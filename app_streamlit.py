
from pathlib import Path
import pandas as pd
import streamlit as st
import pydeck as pdk

st.set_page_config(page_title="Climate Analog Explorer", layout="wide")

DEFAULT_RESULTS = Path("data/processed/city_analogues_weighted_normalized.csv")
DEFAULT_VALID = Path("data/processed/city_analogues_valid_weighted_normalized.csv")


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
    
def dataframe_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c) for c in out.columns]
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].astype(str)
    return out

def find_col(df: pd.DataFrame, preferred: str, fallback_contains: str | None = None):
    if preferred in df.columns:
        return preferred
    if fallback_contains:
        for c in df.columns:
            if fallback_contains.lower() in c.lower():
                return c
    return None


def build_city_key(df: pd.DataFrame, city_col: str, country_col: str):
    return df[city_col].astype(str) + " — " + df[country_col].astype(str)


def safe_metric(value, digits=2):
    if pd.isna(value):
        return "NA"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def get_valid_lookup(valid_df: pd.DataFrame):
    city_col = "city" if "city" in valid_df.columns else None
    country_col = "country" if "country" in valid_df.columns else None
    lat_col = "lat" if "lat" in valid_df.columns else None
    lon_col = "lon" if "lon" in valid_df.columns else None

    if not all([city_col, country_col, lat_col, lon_col]):
        return {}

    lookup = {}
    for _, row in valid_df.iterrows():
        lookup[(str(row[city_col]), str(row[country_col]))] = {
            "lat": row[lat_col],
            "lon": row[lon_col],
            "population": row["population"] if "population" in valid_df.columns else None,
            "is_country_capital": row["is_country_capital"] if "is_country_capital" in valid_df.columns else None,
        }
    return lookup


def analog_table_for_row(row: pd.Series):
    rows = []
    for rank in [1, 2, 3]:
        city_col = f"analog_{rank}_city"
        if city_col not in row.index:
            continue
        rows.append({
            "rank": rank,
            "city": row.get(f"analog_{rank}_city"),
            "country": row.get(f"analog_{rank}_country"),
            "score": row.get(f"analog_{rank}_score"),
            "climate_distance": row.get(f"analog_{rank}_climate_distance"),
            "urban_distance": row.get(f"analog_{rank}_urban_distance"),
            "geo_distance_km": row.get(f"analog_{rank}_geo_distance_km"),
            "population": row.get(f"analog_{rank}_population"),
            "is_country_capital": row.get(f"analog_{rank}_is_country_capital"),
        })
    return pd.DataFrame(rows)


def build_map_df(target_row: pd.Series, analog_df: pd.DataFrame, valid_lookup: dict):
    points = []

    target_city = str(target_row["future_city"])
    target_country = str(target_row["future_country"])
    target_info = valid_lookup.get((target_city, target_country))

    if target_info and pd.notna(target_info["lat"]) and pd.notna(target_info["lon"]):
        points.append({
            "label": f"{target_city} ({target_country})",
            "kind": "Target",
            "lat": float(target_info["lat"]),
            "lon": float(target_info["lon"]),
            "rank": 0,
        })

    for _, row in analog_df.iterrows():
        info = valid_lookup.get((str(row["city"]), str(row["country"])))
        if info and pd.notna(info["lat"]) and pd.notna(info["lon"]):
            points.append({
                "label": f"{row['city']} ({row['country']})",
                "kind": f"Analog #{int(row['rank'])}",
                "lat": float(info["lat"]),
                "lon": float(info["lon"]),
                "rank": int(row["rank"]),
            })

    return pd.DataFrame(points)


def render_map(map_df: pd.DataFrame):
    if map_df.empty:
        st.info("Map unavailable: missing lat/lon in valid city table.")
        return

    display_df = map_df.copy()
    display_df["lat"] = pd.to_numeric(display_df["lat"], errors="coerce")
    display_df["lon"] = pd.to_numeric(display_df["lon"], errors="coerce")
    display_df = display_df.dropna(subset=["lat", "lon"])

    if display_df.empty:
        st.info("Map unavailable: all map points have invalid coordinates.")
        return

    display_df["size"] = display_df["rank"].map(lambda r: 800 if r == 0 else 400)

    st.map(
        display_df,
        latitude="lat",
        longitude="lon",
        size="size",
        zoom=1,
    )

    st.dataframe(
        display_df[["kind", "label", "lat", "lon"]],
        width="stretch",
        hide_index=True,
    )


st.title("Climate Analog Explorer")
st.caption("Explore future climate analogs for cities using PCA-based climate similarity plus light urban matching.")

with st.sidebar:
    st.header("Data")
    results_path = st.text_input("Results CSV", str(DEFAULT_RESULTS))
    valid_path = st.text_input("Valid cities CSV", str(DEFAULT_VALID))
    st.markdown(
        """
        **Expected files**
        - results: output of `compute_city_analogues.py`
        - valid cities: table with `city`, `country`, `lat`, `lon`
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

left, right = st.columns([1.2, 1])

with left:
    st.subheader(f"Target: {selected_row['future_city']} — {selected_row['future_country']}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Self distance", safe_metric(selected_row.get("self_distance")))
    c2.metric("Top-1 climate dist", safe_metric(selected_row.get("analog_1_climate_distance")))
    c3.metric("Top-1 geo dist (km)", safe_metric(selected_row.get("analog_1_geo_distance_km")))
    c4.metric("Top-1 urban dist", safe_metric(selected_row.get("analog_1_urban_distance")))

    st.markdown("### Top analogs")
    st.dataframe(
        analog_df,
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
    st.write("Map points:", len(map_df))
    st.dataframe(map_df, width="stretch", hide_index=True)
    render_map(map_df)

st.markdown("---")
st.markdown("### Notes")
st.markdown(
    """
    - `self_distance` tells how far the future city drifts from its current climate position.
    - `climate_distance` is the main PCA-space climate distance.
    - `urban_distance` is the auxiliary population/capital penalty used in matching.
    - `geo_distance_km` is the geographic distance between target city and analog.
    """
)
