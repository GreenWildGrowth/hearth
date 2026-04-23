from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

IN_CSV = Path("data/interim/cities_bioclim.csv")

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCALER_PATH = OUT_DIR / "scaler.joblib"
PCA_PATH = OUT_DIR / "pca_model.joblib"

BIO_IDS = list(range(1, 20))
N_PCS = 4
TOP_K = 3

# Modes disponibles:
# - "climate_only"
# - "weighted"
# - "rerank"
MATCH_MODE = "rerank"

# Paramètres urbains
URBAN_WEIGHT_POP = 0.10
URBAN_WEIGHT_CAPITAL = 0.05

# Pour le reranking
RERANK_CANDIDATES = 20


def safe_bool(x) -> int:
    if pd.isna(x):
        return 0
    s = str(x).strip().lower()
    return int(s in {"1", "true", "yes", "y"})


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a))


def compute_pairwise_euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a: (n, d)
    b: (m, d)
    returns: (n, m)
    """
    aa = np.sum(a * a, axis=1, keepdims=True)
    bb = np.sum(b * b, axis=1)[None, :]
    ab = a @ b.T
    d2 = np.maximum(aa + bb - 2.0 * ab, 0.0)
    return np.sqrt(d2)


def main():
    df = pd.read_csv(IN_CSV)

    current_cols = [f"current_bio{i}" for i in BIO_IDS]
    future_cols = [f"future_mean_bio{i}" for i in BIO_IDS]

    # Champs urbains attendus
    if "population" not in df.columns:
        df["population"] = np.nan
    if "is_country_capital" not in df.columns:
        df["is_country_capital"] = 0

    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df["is_country_capital"] = df["is_country_capital"].map(safe_bool)

    # Diagnostic NaN sur climat
    df["n_missing_current"] = df[current_cols].isna().sum(axis=1)
    df["n_missing_future"] = df[future_cols].isna().sum(axis=1)
    df["is_valid_for_analog"] = (
        (df["n_missing_current"] == 0) &
        (df["n_missing_future"] == 0)
    )

    invalid_df = df.loc[~df["is_valid_for_analog"]].copy()
    valid_df = df.loc[df["is_valid_for_analog"]].copy().reset_index(drop=True)

    if len(valid_df) < max(N_PCS, TOP_K + 1):
        raise RuntimeError(
            f"Pas assez de villes valides. valid={len(valid_df)}, "
            f"required>={max(N_PCS, TOP_K + 1)}"
        )

    # Matrices climatiques
    X_current = valid_df[current_cols].to_numpy(dtype=float)
    X_future = valid_df[future_cols].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_current_scaled = scaler.fit_transform(X_current)
    X_future_scaled = scaler.transform(X_future)

    pca = PCA(n_components=N_PCS)
    X_current_pca = pca.fit_transform(X_current_scaled)
    X_future_pca = pca.transform(X_future_scaled)

    # Distances climatiques
    climate_dist = compute_pairwise_euclidean(X_future_pca, X_current_pca)

    # Features urbaines
    pop = valid_df["population"].to_numpy(dtype=float)
    pop = np.where(np.isfinite(pop) & (pop > 0), pop, np.nan)
    log_pop = np.log10(pop)

    capital = valid_df["is_country_capital"].to_numpy(dtype=int)

    # Distance urbaine pairwise
    d_pop = np.abs(log_pop[:, None] - log_pop[None, :])
    d_pop = np.where(np.isnan(d_pop), 0.0, d_pop)

    d_cap = (capital[:, None] != capital[None, :]).astype(float)

    urban_dist = URBAN_WEIGHT_POP * d_pop + URBAN_WEIGHT_CAPITAL * d_cap

    # On exclut le self-match exact (même ville + même pays)
    same_city_mask = np.zeros((len(valid_df), len(valid_df)), dtype=bool)
    for i in range(len(valid_df)):
        same_city_mask[i, :] = (
            (valid_df.loc[i, "city"] == valid_df["city"]) &
            (valid_df.loc[i, "country"] == valid_df["country"])
        ).to_numpy()

    # self distance climatique utile pour debug
    self_distance = np.full(len(valid_df), np.nan)
    for i in range(len(valid_df)):
        self_idxs = np.where(same_city_mask[i])[0]
        if len(self_idxs) > 0:
            self_distance[i] = float(climate_dist[i, self_idxs[0]])

    rows = []

    for i in range(len(valid_df)):
        row = {
            "future_city": valid_df.loc[i, "city"],
            "future_country": valid_df.loc[i, "country"],
            "self_distance": self_distance[i],
        }

        climate_row = climate_dist[i].copy()
        urban_row = urban_dist[i].copy()

        # Exclure self
        climate_row[same_city_mask[i]] = np.inf
        urban_row[same_city_mask[i]] = np.inf

        if MATCH_MODE == "climate_only":
            final_score = climate_row
        elif MATCH_MODE == "weighted":
            final_score = climate_row + urban_row
        elif MATCH_MODE == "rerank":
            shortlist_idx = np.argsort(climate_row)[:RERANK_CANDIDATES]
            rerank_score = np.full_like(climate_row, np.inf)
            rerank_score[shortlist_idx] = climate_row[shortlist_idx] + urban_row[shortlist_idx]
            final_score = rerank_score
        else:
            raise ValueError(f"Unknown MATCH_MODE={MATCH_MODE}")

        best_idx = np.argsort(final_score)[:TOP_K]

        for rank, j in enumerate(best_idx, start=1):
            row[f"analog_{rank}_city"] = valid_df.loc[j, "city"]
            row[f"analog_{rank}_country"] = valid_df.loc[j, "country"]
            row[f"analog_{rank}_score"] = float(final_score[j])
            row[f"analog_{rank}_climate_distance"] = float(climate_row[j])
            row[f"analog_{rank}_urban_distance"] = float(urban_row[j])

            row[f"analog_{rank}_population"] = valid_df.loc[j, "population"]
            row[f"analog_{rank}_is_country_capital"] = valid_df.loc[j, "is_country_capital"]

            row[f"analog_{rank}_geo_distance_km"] = float(
                haversine_km(
                    valid_df.loc[i, "lat"], valid_df.loc[i, "lon"],
                    valid_df.loc[j, "lat"], valid_df.loc[j, "lon"],
                )
            )

        for pc in range(N_PCS):
            row[f"current_pc{pc+1}"] = float(X_current_pca[i, pc])
            row[f"future_pc{pc+1}"] = float(X_future_pca[i, pc])

        rows.append(row)

    out = pd.DataFrame(rows)

    suffix = MATCH_MODE
    out_csv = OUT_DIR / f"city_analogues_{suffix}.csv"
    valid_csv = OUT_DIR / f"city_analogues_valid_{suffix}.csv"
    invalid_csv = OUT_DIR / f"city_analogues_invalid_{suffix}.csv"

    out.to_csv(out_csv, index=False)
    valid_df.to_csv(valid_csv, index=False)
    invalid_df.to_csv(invalid_csv, index=False)

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(pca, PCA_PATH)

    print(f"Wrote analogues to {out_csv}")
    print(f"Wrote valid cities to {valid_csv}")
    print(f"Wrote invalid cities to {invalid_csv}")
    print()
    print(f"Mode           : {MATCH_MODE}")
    print(f"Total cities   : {len(df)}")
    print(f"Valid cities   : {len(valid_df)}")
    print(f"Invalid cities : {len(invalid_df)}")
    print()
    if len(invalid_df) > 0:
        print("Invalid cities summary:")
        print(
            invalid_df[["city", "country", "n_missing_current", "n_missing_future"]]
            .to_string(index=False)
        )
        print()

    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Cumulative explained variance:", np.cumsum(pca.explained_variance_ratio_))
    print()
    print("Top-1 geo distance stats (km):")
    print(out["analog_1_geo_distance_km"].describe())
    print()
    print("Top-1 climate distance stats:")
    print(out["analog_1_climate_distance"].describe())


if __name__ == "__main__":
    main()