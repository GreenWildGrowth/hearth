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

# Mode recommandé après sweep :
# - "climate_only"
# - "weighted"
# - "weighted_normalized"
MATCH_MODE = "weighted_normalized"

# Poids internes de la distance urbaine brute
URBAN_WEIGHT_POP = 1.0
URBAN_WEIGHT_CAPITAL = 0.5

# Pondération globale de la composante urbaine
URBAN_LAMBDA = 0.25

# Diversification spatiale des analogues.
# L'objectif est d'éviter que analog_1, analog_2, analog_3 soient trois villes voisines.
# Mode :
# - "none" : comportement historique, prend les TOP_K meilleurs scores
# - "hard" : greedy avec contrainte stricte MIN_ANALOG_SPACING_KM entre analogues retenus
# - "soft" : ajoute une pénalité si un candidat est proche des analogues déjà retenus
DIVERSIFICATION_MODE = "hard"
MIN_ANALOG_SPACING_KM = 750.0
SOFT_DIVERSITY_LAMBDA = 0.50

# Filtre directionnel optionnel.
# Pour l'hémisphère nord, on peut imposer ou favoriser les analogues situés plus au sud.
# - "none" : pas de filtre
# - "strict" : exclut les analogues qui ne vont pas vers le sud
# - "soft" : pénalise les analogues qui ne vont pas vers le sud
DIRECTIONAL_FILTER = "none"
DIRECTIONAL_PENALTY = 1.0

# Pour le filtre directionnel, on évite de raisonner trop près de l'équateur.
# Si abs(lat) < seuil, on ne force pas la direction.
DIRECTION_MIN_ABS_LAT = 5.0

# Nombre de candidats pré-triés qu'on inspecte pour la diversification.
# Augmenter si la base contient beaucoup de villes très proches.
CANDIDATE_POOL_SIZE = 250



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



def equator_shift_km_signed(lat_target, lat_analog):
    """
    Shift latitudinal signé *vers l'équateur*, en km.

    Convention :
    - valeur positive : l'analogue est plus proche de l'équateur que la cible
    - valeur négative : l'analogue est plus éloigné de l'équateur que la cible

    Contrairement à une simple différence de latitude, cette métrique reste lisible
    lors d'un changement d'hémisphère.

    Exemple :
    - cible 45°N, analogue 35°N => +1110 km vers l'équateur
    - cible 45°N, analogue 35°S => +1110 km vers l'équateur aussi,
      pas ~8880 km, car on compare la distance à l'équateur.
    """
    lat_target = float(lat_target)
    lat_analog = float(lat_analog)
    return (abs(lat_target) - abs(lat_analog)) * 111.0


def signed_lat_shift_km(lat_target, lat_analog):
    """
    Alias de compatibilité : ancienne colonne, nouvelle sémantique.
    Voir equator_shift_km_signed().
    """
    return equator_shift_km_signed(lat_target, lat_analog)



def delta_lon_km(lat_target, lon_target, lon_analog):
    """
    Approximation est/ouest à latitude constante.
    Utile pour expliquer si l'analogue est surtout un déplacement longitudinal.
    """
    return abs(float(lon_analog) - float(lon_target)) * 111.0 * np.cos(np.radians(float(lat_target)))



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



def compute_pairwise_haversine(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Matrice n x n des distances géographiques entre villes candidates.
    Les lignes et colonnes correspondent à valid_df.
    """
    lat1 = lat[:, None]
    lon1 = lon[:, None]
    lat2 = lat[None, :]
    lon2 = lon[None, :]
    return haversine_km(lat1, lon1, lat2, lon2)



def apply_directional_filter(score_row: np.ndarray, lat_target: float, lat_candidates: np.ndarray) -> np.ndarray:
    """
    Optionnel : impose/favorise un analogue situé vers des latitudes plus chaudes.
    - Nord : analog_lat < target_lat
    - Sud  : analog_lat > target_lat
    """
    if DIRECTIONAL_FILTER == "none" or abs(lat_target) < DIRECTION_MIN_ABS_LAT:
        return score_row

    out = score_row.copy()
    if lat_target >= 0:
        wrong_direction = lat_candidates >= lat_target
    else:
        wrong_direction = lat_candidates <= lat_target

    if DIRECTIONAL_FILTER == "strict":
        out[wrong_direction] = np.inf
    elif DIRECTIONAL_FILTER == "soft":
        out[wrong_direction] = out[wrong_direction] + DIRECTIONAL_PENALTY
    else:
        raise ValueError(f"Unknown DIRECTIONAL_FILTER={DIRECTIONAL_FILTER}")

    return out



def select_diversified_indices(
    base_score: np.ndarray,
    analog_geo_dist: np.ndarray,
    top_k: int,
    candidate_pool_size: int,
) -> list[int]:
    """
    Sélection greedy des analogues.

    base_score[j] : score final candidat j pour la ville cible courante.
    analog_geo_dist[j, k] : distance géographique entre deux candidats analogues j et k.

    En mode "hard" :
      on prend le meilleur candidat qui reste à >= MIN_ANALOG_SPACING_KM de tous les analogues déjà choisis.
      Si on n'arrive pas à remplir TOP_K, on complète avec les meilleurs scores restants.

    En mode "soft" :
      à chaque rang, on ajoute une pénalité dépendant de la proximité aux analogues déjà retenus.
    """
    if DIVERSIFICATION_MODE == "none":
        return list(np.argsort(base_score)[:top_k])

    finite = np.where(np.isfinite(base_score))[0]
    if len(finite) == 0:
        return []

    order = finite[np.argsort(base_score[finite])]
    pool = order[: min(candidate_pool_size, len(order))]

    selected: list[int] = []

    if DIVERSIFICATION_MODE == "hard":
        for j in pool:
            if len(selected) == 0:
                selected.append(int(j))
            else:
                d_to_selected = analog_geo_dist[j, selected]
                if np.all(d_to_selected >= MIN_ANALOG_SPACING_KM):
                    selected.append(int(j))
            if len(selected) == top_k:
                break

        # Fallback : si la contrainte est trop stricte, on complète avec les meilleurs restants.
        if len(selected) < top_k:
            selected_set = set(selected)
            for j in order:
                if int(j) not in selected_set:
                    selected.append(int(j))
                    selected_set.add(int(j))
                if len(selected) == top_k:
                    break

        return selected

    if DIVERSIFICATION_MODE == "soft":
        available = list(pool)
        while available and len(selected) < top_k:
            if len(selected) == 0:
                best = min(available, key=lambda j: base_score[j])
            else:
                adjusted_scores = []
                for j in available:
                    d_min = float(np.min(analog_geo_dist[j, selected]))
                    closeness = max(0.0, MIN_ANALOG_SPACING_KM - d_min) / MIN_ANALOG_SPACING_KM
                    adjusted_scores.append(base_score[j] + SOFT_DIVERSITY_LAMBDA * closeness)
                best = available[int(np.argmin(adjusted_scores))]
            selected.append(int(best))
            available.remove(best)

        return selected

    raise ValueError(f"Unknown DIVERSIFICATION_MODE={DIVERSIFICATION_MODE}")



def main():
    df = pd.read_csv(IN_CSV)

    current_cols = [f"current_bio{i}" for i in BIO_IDS]
    future_cols = [f"future_mean_bio{i}" for i in BIO_IDS]

    if "population" not in df.columns:
        df["population"] = np.nan
    if "is_country_capital" not in df.columns:
        df["is_country_capital"] = 0

    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df["is_country_capital"] = df["is_country_capital"].map(safe_bool)

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

    X_current = valid_df[current_cols].to_numpy(dtype=float)
    X_future = valid_df[future_cols].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_current_scaled = scaler.fit_transform(X_current)
    X_future_scaled = scaler.transform(X_future)

    pca = PCA(n_components=N_PCS)
    X_current_pca = pca.fit_transform(X_current_scaled)
    X_future_pca = pca.transform(X_future_scaled)

    climate_dist = compute_pairwise_euclidean(X_future_pca, X_current_pca)

    lat = valid_df["lat"].to_numpy(dtype=float)
    lon = valid_df["lon"].to_numpy(dtype=float)
    geo_dist = compute_pairwise_haversine(lat, lon)

    pop = valid_df["population"].to_numpy(dtype=float)
    pop = np.where(np.isfinite(pop) & (pop > 0), pop, np.nan)
    log_pop = np.log10(pop)

    capital = valid_df["is_country_capital"].to_numpy(dtype=int)

    d_pop = np.abs(log_pop[:, None] - log_pop[None, :])
    d_pop = np.where(np.isnan(d_pop), 0.0, d_pop)

    d_cap = (capital[:, None] != capital[None, :]).astype(float)

    urban_dist = URBAN_WEIGHT_POP * d_pop + URBAN_WEIGHT_CAPITAL * d_cap

    climate_scale = float(np.nanstd(climate_dist[np.isfinite(climate_dist)]))
    urban_scale = float(np.nanstd(urban_dist[np.isfinite(urban_dist)]))

    if climate_scale == 0:
        climate_scale = 1.0
    if urban_scale == 0:
        urban_scale = 1.0

    same_city_mask = np.zeros((len(valid_df), len(valid_df)), dtype=bool)
    for i in range(len(valid_df)):
        same_city_mask[i, :] = (
            (valid_df.loc[i, "city"] == valid_df["city"]) &
            (valid_df.loc[i, "country"] == valid_df["country"])
        ).to_numpy()

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
            "future_lat": float(valid_df.loc[i, "lat"]),
            "future_lon": float(valid_df.loc[i, "lon"]),
            "self_distance": self_distance[i],
        }

        climate_row = climate_dist[i].copy()
        urban_row = urban_dist[i].copy()

        climate_row[same_city_mask[i]] = np.inf
        urban_row[same_city_mask[i]] = np.inf

        if MATCH_MODE == "climate_only":
            final_score = climate_row

        elif MATCH_MODE == "weighted":
            final_score = climate_row + URBAN_LAMBDA * urban_row

        elif MATCH_MODE == "weighted_normalized":
            final_score = (
                climate_row / climate_scale
                + URBAN_LAMBDA * (urban_row / urban_scale)
            )

        else:
            raise ValueError(f"Unknown MATCH_MODE={MATCH_MODE}")

        final_score = apply_directional_filter(final_score, lat_target=lat[i], lat_candidates=lat)

        best_idx = select_diversified_indices(
            base_score=final_score,
            analog_geo_dist=geo_dist,
            top_k=TOP_K,
            candidate_pool_size=CANDIDATE_POOL_SIZE,
        )

        for rank, j in enumerate(best_idx, start=1):
            geo_km = float(geo_dist[i, j])
            equator_shift_km = float(equator_shift_km_signed(lat[i], lat[j]))
            abs_lat_km = abs(float(lat[i] - lat[j])) * 111.0
            lon_km = float(delta_lon_km(lat[i], lon[i], lon[j]))

            row[f"analog_{rank}_city"] = valid_df.loc[j, "city"]
            row[f"analog_{rank}_country"] = valid_df.loc[j, "country"]
            row[f"analog_{rank}_lat"] = float(valid_df.loc[j, "lat"])
            row[f"analog_{rank}_lon"] = float(valid_df.loc[j, "lon"])
            row[f"analog_{rank}_score"] = float(final_score[j])
            row[f"analog_{rank}_climate_distance"] = float(climate_row[j])
            row[f"analog_{rank}_urban_distance"] = float(urban_row[j])
            row[f"analog_{rank}_population"] = valid_df.loc[j, "population"]
            row[f"analog_{rank}_is_country_capital"] = valid_df.loc[j, "is_country_capital"]
            row[f"analog_{rank}_geo_distance_km"] = geo_km
            row[f"analog_{rank}_equator_shift_km_signed"] = equator_shift_km
            # Backward-compatible alias: same value, but older column name.
            row[f"analog_{rank}_lat_shift_km_signed"] = equator_shift_km
            row[f"analog_{rank}_lat_distance_km_abs"] = abs_lat_km
            row[f"analog_{rank}_lon_distance_km_abs"] = lon_km
            row[f"analog_{rank}_is_towards_equator"] = bool(equator_shift_km > 0)

            if rank > 1:
                prev = best_idx[: rank - 1]
                row[f"analog_{rank}_min_distance_to_previous_analogs_km"] = float(np.min(geo_dist[j, prev]))
            else:
                row[f"analog_{rank}_min_distance_to_previous_analogs_km"] = np.nan

        for pc in range(N_PCS):
            row[f"current_pc{pc+1}"] = float(X_current_pca[i, pc])
            row[f"future_pc{pc+1}"] = float(X_future_pca[i, pc])

        rows.append(row)

    out = pd.DataFrame(rows)

    suffix_parts = [MATCH_MODE]
    if DIVERSIFICATION_MODE != "none":
        suffix_parts.append(f"div-{DIVERSIFICATION_MODE}-{int(MIN_ANALOG_SPACING_KM)}km")
    if DIRECTIONAL_FILTER != "none":
        suffix_parts.append(f"dir-{DIRECTIONAL_FILTER}")
    suffix = "_".join(suffix_parts)

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
    print(f"Mode                  : {MATCH_MODE}")
    print(f"Urban lambda          : {URBAN_LAMBDA}")
    print(f"Diversification mode  : {DIVERSIFICATION_MODE}")
    print(f"Min analog spacing km : {MIN_ANALOG_SPACING_KM}")
    print(f"Directional filter    : {DIRECTIONAL_FILTER}")
    print(f"Total cities          : {len(df)}")
    print(f"Valid cities          : {len(valid_df)}")
    print(f"Invalid cities        : {len(invalid_df)}")
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
    print()
    print("Top-1 equator shift stats (km):")
    print(out["analog_1_equator_shift_km_signed"].describe())
    print()
    print("Top-1 same-country rate:")
    print(
        (
            out["future_country"].astype(str).values
            == out["analog_1_country"].astype(str).values
        ).mean()
    )

    if TOP_K >= 2 and "analog_2_min_distance_to_previous_analogs_km" in out.columns:
        print()
        print("Analog diversification stats:")
        for rank in range(2, TOP_K + 1):
            col = f"analog_{rank}_min_distance_to_previous_analogs_km"
            if col in out.columns:
                print(f"Rank {rank} min distance to previous analogues (km):")
                print(out[col].describe())


if __name__ == "__main__":
    main()
