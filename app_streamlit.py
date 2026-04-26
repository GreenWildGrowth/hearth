from pathlib import Path
import pandas as pd
import streamlit as st
import pydeck as pdk

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


def build_city_key(df: pd.DataFrame, city_col: str, country_col: str):
    return df[city_col].astype(str) + " — " + df[country_col].astype(str)


def selected_row_as_display_df(row: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "field": [str(k) for k in row.index],
        "value": ["" if pd.isna(v) else str(v) for v in row.values],
    })


def safe_float(value):
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def format_number(value, digits=0, suffix=""):
    v = safe_float(value)
    if v is None:
        return "NA"
    return f"{v:.{digits}f}{suffix}"


def format_large_number(value):
    v = safe_float(value)
    if v is None:
        return "NA"
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f} M"
    if v >= 1_000:
        return f"{v / 1_000:.0f} k"
    return f"{v:.0f}"


def get_equator_shift(row_or_series, rank=1):
    new_col = f"analog_{rank}_equator_shift_km_signed"
    old_col = f"analog_{rank}_lat_shift_km_signed"
    if new_col in row_or_series.index:
        return row_or_series.get(new_col)
    return row_or_series.get(old_col)


def format_equator_shift(value):
    v = safe_float(value)
    if v is None:
        return "NA"
    if abs(v) < 1e-9:
        return "0 km"
    direction = "towards equator" if v > 0 else "away from equator"
    return f"{abs(v):.0f} km {direction}"


def climate_quality_label(value):
    """
    Heuristic label only. The absolute PCA distance has no universal physical unit,
    so this should stay descriptive rather than definitive.
    """
    v = safe_float(value)
    if v is None:
        return "Unknown"
    if v < 1.0:
        return "Very close"
    if v < 2.0:
        return "Close"
    if v < 3.5:
        return "Moderate"
    return "Distant"


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
            "climate_distance": row.get(f"analog_{rank}_climate_distance"),
            "geo_distance_km": row.get(f"analog_{rank}_geo_distance_km"),
            "equator_shift_km_signed": get_equator_shift(row, rank),
            "lon_distance_km_abs": row.get(f"analog_{rank}_lon_distance_km_abs"),
            "population": row.get(f"analog_{rank}_population"),
            "is_country_capital": row.get(f"analog_{rank}_is_country_capital"),
            # Technical columns, hidden by default.
            "score": row.get(f"analog_{rank}_score"),
            "urban_distance": row.get(f"analog_{rank}_urban_distance"),
            "lat_distance_km_abs": row.get(f"analog_{rank}_lat_distance_km_abs"),
            "towards_equator": row.get(f"analog_{rank}_is_towards_equator"),
            "min_dist_to_previous_km": row.get(f"analog_{rank}_min_distance_to_previous_analogs_km"),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        for c in out.columns:
            if c not in {"city", "country", "towards_equator"}:
                out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def user_friendly_analog_table(analog_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in analog_df.iterrows():
        rows.append({
            "#": int(r["rank"]),
            "Climate analogue": f"{r['city']} — {r['country']}",
            "Climate match": climate_quality_label(r.get("climate_distance")),
            "Equator shift": format_equator_shift(r.get("equator_shift_km_signed")),
            "Geographic distance": format_number(r.get("geo_distance_km"), digits=0, suffix=" km"),
            "East/west shift": format_number(r.get("lon_distance_km_abs"), digits=0, suffix=" km"),
            "Population": format_large_number(r.get("population")),
        })
    return pd.DataFrame(rows)


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
            "kind": "Selected city",
            "lat": float(target_lat),
            "lon": float(target_lon),
            "rank": 0,
        })

    for _, analog in analog_df.iterrows():
        rank = int(analog["rank"])
        city = str(analog["city"])
        country = str(analog["country"])
        lat, lon = coords_for_city(city, country, rank, target_row, valid_lookup)
        if pd.notna(lat) and pd.notna(lon):
            points.append({
                "label": f"{city} ({country})",
                "kind": f"Analogue #{rank}",
                "lat": float(lat),
                "lon": float(lon),
                "rank": rank,
            })

    return pd.DataFrame(points)


def map_color_for_rank(rank: int):
    """Soft, coherent map palette.

    The selected city is neutral/dark. Each analogue has its own color,
    reused for both the point and its arc.
    """
    palette = {
        0: [38, 50, 56, 230],      # selected city: blue-grey
        1: [0, 121, 107, 230],     # analogue #1: teal
        2: [245, 124, 0, 230],     # analogue #2: amber/orange
        3: [123, 31, 162, 230],    # analogue #3: purple
        4: [25, 118, 210, 230],    # optional: blue
        5: [198, 40, 40, 230],     # optional: muted red
    }
    return palette.get(int(rank), [97, 97, 97, 220])


def render_map(map_df):
    if map_df.empty:
        st.info("No map data")
        return

    map_df = map_df.copy()
    map_df["lat"] = pd.to_numeric(map_df["lat"], errors="coerce")
    map_df["lon"] = pd.to_numeric(map_df["lon"], errors="coerce")
    map_df["rank"] = pd.to_numeric(map_df["rank"], errors="coerce").fillna(0).astype(int)
    map_df = map_df.dropna(subset=["lat", "lon"])

    if map_df.empty or (map_df["rank"] == 0).sum() == 0:
        st.info("No valid map data")
        return

    target = map_df[map_df["rank"] == 0].iloc[0]
    analogs = map_df[map_df["rank"] > 0].copy()

    map_df["color"] = map_df["rank"].apply(map_color_for_rank)
    map_df["radius"] = map_df["rank"].apply(lambda r: 170000 if r == 0 else 125000)

    scatter = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_line_color=[255, 255, 255, 220],
        get_line_width=2,
        get_radius="radius",
        radius_min_pixels=6,
        radius_max_pixels=18,
        stroked=True,
        filled=True,
        pickable=True,
    )

    arcs_data = []
    for _, row in analogs.iterrows():
        rank = int(row["rank"])
        color = map_color_for_rank(rank)
        arcs_data.append({
            "from_lon": float(target["lon"]),
            "from_lat": float(target["lat"]),
            "to_lon": float(row["lon"]),
            "to_lat": float(row["lat"]),
            "rank": rank,
            "label": row.get("label", f"Analogue #{rank}"),
            "color": color,
            "width": max(2, 6 - rank),
        })

    arc_layer = pdk.Layer(
        "ArcLayer",
        data=arcs_data,
        get_source_position="[from_lon, from_lat]",
        get_target_position="[to_lon, to_lat]",
        get_source_color="color",
        get_target_color="color",
        get_width="width",
        pickable=True,
    )

    deck = pdk.Deck(
        layers=[arc_layer, scatter],
        initial_view_state=pdk.ViewState(
            latitude=float(target["lat"]),
            longitude=float(target["lon"]),
            zoom=2,
            pitch=35,
        ),
        tooltip={
            "html": "<b>{label}</b><br/>{kind}",
            "style": {"backgroundColor": "rgba(33, 33, 33, 0.85)", "color": "white"},
        },
        map_style="light",
    )

    st.pydeck_chart(deck, use_container_width=True)

    legend_items = []
    for _, row in map_df.sort_values("rank").iterrows():
        color = map_color_for_rank(int(row["rank"]))[:3]
        rgb = f"rgb({color[0]}, {color[1]}, {color[2]})"
        legend_items.append(
            f"<span style='display:inline-flex;align-items:center;margin-right:14px;'>"
            f"<span style='width:11px;height:11px;border-radius:50%;background:{rgb};display:inline-block;margin-right:6px;'></span>"
            f"{row['kind']}"
            f"</span>"
        )
    st.markdown("".join(legend_items), unsafe_allow_html=True)


st.title("Climate Analog Explorer")
st.caption("Find present-day cities whose climate resembles a selected city's projected future climate.")

with st.sidebar:
    st.header("City selection")

    default_results = DEFAULT_RESULTS if DEFAULT_RESULTS.exists() else LEGACY_RESULTS
    default_valid = DEFAULT_VALID if DEFAULT_VALID.exists() else LEGACY_VALID

    with st.expander("Data files", expanded=False):
        results_path = st.text_input("Results CSV", str(default_results))
        valid_path = st.text_input("Valid cities CSV", str(default_valid))
        st.caption("Use the diversified CSV generated by compute_city_analogues.py when available.")

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

st.subheader(f"{selected_row['future_city']} — {selected_row['future_country']}")

if analog_df.empty:
    st.warning("No analog columns found in the results CSV.")
    st.stop()

top1 = analog_df.iloc[0]

c1, c2, c3 = st.columns(3)
c1.metric("Best climate match", f"{top1['city']} — {top1['country']}")
c2.metric("Shift towards equator", format_equator_shift(top1.get("equator_shift_km_signed")))
c3.metric("Geographic distance", format_number(top1.get("geo_distance_km"), digits=0, suffix=" km"))

left, right = st.columns([1.15, 1])

with left:
    st.markdown("### Climate analogues")
    st.dataframe(
        user_friendly_analog_table(analog_df),
        width="stretch",
        hide_index=True,
    )

    st.markdown(
        """
        **How to read this:** the first city is the closest current-climate analogue.  
        The equator shift is positive when the analogue is closer to the equator than the selected city.
        """
    )

with right:
    st.markdown("### Map")
    map_df = build_map_df(selected_row, analog_df, valid_lookup)
    render_map(map_df)

with st.expander("Technical details", expanded=False):
    st.markdown("### Raw analogue metrics")
    technical_cols = [
        "rank", "city", "country", "score", "climate_distance", "urban_distance",
        "geo_distance_km", "equator_shift_km_signed", "lat_distance_km_abs",
        "lon_distance_km_abs", "towards_equator", "min_dist_to_previous_km",
        "population", "is_country_capital",
    ]
    technical_cols = [c for c in technical_cols if c in analog_df.columns]
    st.dataframe(analog_df[technical_cols], width="stretch", hide_index=True)

    st.markdown("### Selected city raw row")
    st.dataframe(selected_row_as_display_df(selected_row), width="stretch", hide_index=True)

    st.markdown(
        """
        - `climate_distance`: PCA-space climate distance. Lower is better.
        - `score`: final internal matching score after optional urban weighting.
        - `urban_distance`: population/capital mismatch penalty.
        - `min_dist_to_previous_km`: spacing from previously selected analogues, useful for checking diversification.
        - `equator_shift_km_signed`: positive when the analogue is closer to the equator than the target.
        """
    )
