from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import joblib

IN_CSV = Path("data/interim/cities_bioclim.csv")
OUT_CSV = Path("data/processed/city_analogues.csv")
SCALER_PATH = Path("data/processed/scaler.joblib")
PCA_PATH = Path("data/processed/pca_model.joblib")
VALID_CITIES_PATH = Path("data/processed/city_analogues_valid.csv")
INVALID_CITIES_PATH = Path("data/processed/city_analogues_invalid.csv")

BIO_IDS = list(range(1, 20))
N_PCS = 4
TOP_K = 3


def main():
    df = pd.read_csv(IN_CSV)

    current_cols = [f"current_bio{i}" for i in BIO_IDS]
    future_cols = [f"future_mean_bio{i}" for i in BIO_IDS]

    # Diagnostic NaN
    df["n_missing_current"] = df[current_cols].isna().sum(axis=1)
    df["n_missing_future"] = df[future_cols].isna().sum(axis=1)
    df["is_valid_for_analog"] = (
        (df["n_missing_current"] == 0) &
        (df["n_missing_future"] == 0)
    )

    invalid_df = df.loc[~df["is_valid_for_analog"]].copy()
    valid_df = df.loc[df["is_valid_for_analog"]].copy()

    if len(valid_df) < max(N_PCS, TOP_K):
        raise RuntimeError(
            f"Pas assez de villes valides pour calculer PCA/analogues. "
            f"valid={len(valid_df)}, required>={max(N_PCS, TOP_K)}"
        )

    # On travaille seulement sur les villes complètes
    X_current = valid_df[current_cols].to_numpy(dtype=float)
    X_future = valid_df[future_cols].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_current_scaled = scaler.fit_transform(X_current)
    X_future_scaled = scaler.transform(X_future)

    pca = PCA(n_components=N_PCS)
    X_current_pca = pca.fit_transform(X_current_scaled)
    X_future_pca = pca.transform(X_future_scaled)

    nn = NearestNeighbors(n_neighbors=min(len(valid_df), TOP_K + 5), metric="euclidean")
    nn.fit(X_current_pca)

    distances, indices = nn.kneighbors(X_future_pca)

    rows = []
    for i in range(len(valid_df)):
        future_city = valid_df.iloc[i]["city"]
        future_country = valid_df.iloc[i]["country"]

        row = {
            "future_city": future_city,
            "future_country": future_country,
        }

        kept = 0
        self_distance = None

        for rank in range(indices.shape[1]):
            j = indices[i, rank]
            cand_city = valid_df.iloc[j]["city"]
            cand_country = valid_df.iloc[j]["country"]
            cand_distance = float(distances[i, rank])

            is_self = (cand_city == future_city) and (cand_country == future_country)

            if is_self:
                self_distance = cand_distance
                continue

            kept += 1
            row[f"analog_{kept}_city"] = cand_city
            row[f"analog_{kept}_country"] = cand_country
            row[f"analog_{kept}_distance"] = cand_distance

            if kept == TOP_K:
                break

        row["self_distance"] = self_distance

        for pc in range(N_PCS):
            row[f"current_pc{pc+1}"] = float(X_current_pca[i, pc])
            row[f"future_pc{pc+1}"] = float(X_future_pca[i, pc])

        rows.append(row)

    out = pd.DataFrame(rows)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    valid_df.to_csv(VALID_CITIES_PATH, index=False)
    invalid_df.to_csv(INVALID_CITIES_PATH, index=False)

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(pca, PCA_PATH)

    print(f"Wrote analogues to {OUT_CSV}")
    print(f"Wrote valid cities to {VALID_CITIES_PATH}")
    print(f"Wrote invalid cities to {INVALID_CITIES_PATH}")
    print()
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


if __name__ == "__main__":
    main()