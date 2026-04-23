# scripts/build_candidate_cities_from_geonames.py
from pathlib import Path
import pandas as pd

GEONAMES_TXT = Path("data/raw/geonames/allCountries.txt")
COUNTRY_INFO_TXT = Path("data/raw/geonames/countryInfo.txt")
OUT_CSV = Path("data/raw/candidate_cities.csv")

GEONAMES_COLUMNS = [
    "geonameid",
    "name",
    "asciiname",
    "alternatenames",
    "latitude",
    "longitude",
    "feature_class",
    "feature_code",
    "country_code",
    "cc2",
    "admin1_code",
    "admin2_code",
    "admin3_code",
    "admin4_code",
    "population",
    "elevation",
    "dem",
    "timezone",
    "modification_date",
]


def load_country_info(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            rows.append(line.rstrip("\n").split("\t"))

    # countryInfo format: ISO, ISO3, ISO-Numeric, fips, Country, Capital, Area, Population, ...
    df = pd.DataFrame(rows)
    df = df.rename(columns={
        0: "country_code",
        4: "country",
        5: "capital",
    })
    return df[["country_code", "country", "capital"]].copy()


def main():
    df = pd.read_csv(
        GEONAMES_TXT,
        sep="\t",
        names=GEONAMES_COLUMNS,
        header=None,
        dtype={
            "geonameid": "Int64",
            "name": str,
            "asciiname": str,
            "latitude": float,
            "longitude": float,
            "feature_class": str,
            "feature_code": str,
            "country_code": str,
            "admin1_code": str,
            "population": "Int64",
        },
        low_memory=False,
    )

    country_df = load_country_info(COUNTRY_INFO_TXT)

    # lieux habités uniquement
    df = df[df["feature_class"] == "P"].copy()

    # population numérique
    df["population"] = pd.to_numeric(df["population"], errors="coerce").fillna(0)

    # merge pays
    df = df.merge(country_df, on="country_code", how="left")

    # capitale nationale selon countryInfo
    df["is_country_capital"] = (
        df["asciiname"].fillna("").str.lower() ==
        df["capital"].fillna("").str.lower()
    )

    # filtre V1
    keep = (
        (df["population"] >= 500_000) |
        (df["is_country_capital"])
    )
    out = df.loc[keep].copy()

    # score simple pour dédoublonnage
    out["sort_capital"] = out["is_country_capital"].astype(int)
    out = out.sort_values(
        ["sort_capital", "population"],
        ascending=[False, False]
    )

    out = out.drop_duplicates(
        subset=["asciiname", "country_code"],
        keep="first"
    )

    out = out.rename(columns={
        "asciiname": "city",
        "latitude": "lat",
        "longitude": "lon",
    })

    out = out[[
        "city",
        "country",
        "country_code",
        "lat",
        "lon",
        "population",
        "feature_code",
        "is_country_capital",
        "geonameid",
    ]].copy()

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    print(f"Wrote {len(out)} rows to {OUT_CSV}")
    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()