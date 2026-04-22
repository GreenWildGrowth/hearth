# scripts/build_cities_csv.py
from pathlib import Path
import pandas as pd

OUT = Path("data/raw/cities.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

# Subset MVP — à remplacer plus tard par la vraie liste des 520 villes
rows = [
    {"city": "Paris", "country": "France", "lat": 48.8566, "lon": 2.3522},
    {"city": "London", "country": "United Kingdom", "lat": 51.5074, "lon": -0.1278},
    {"city": "Madrid", "country": "Spain", "lat": 40.4168, "lon": -3.7038},
    {"city": "Rome", "country": "Italy", "lat": 41.9028, "lon": 12.4964},
    {"city": "Berlin", "country": "Germany", "lat": 52.5200, "lon": 13.4050},
    {"city": "Moscow", "country": "Russia", "lat": 55.7558, "lon": 37.6173},
    {"city": "New York", "country": "United States", "lat": 40.7128, "lon": -74.0060},
    {"city": "Los Angeles", "country": "United States", "lat": 34.0522, "lon": -118.2437},
    {"city": "Mexico City", "country": "Mexico", "lat": 19.4326, "lon": -99.1332},
    {"city": "Bogota", "country": "Colombia", "lat": 4.7110, "lon": -74.0721},
    {"city": "Cairo", "country": "Egypt", "lat": 30.0444, "lon": 31.2357},
    {"city": "Lagos", "country": "Nigeria", "lat": 6.5244, "lon": 3.3792},
    {"city": "Nairobi", "country": "Kenya", "lat": -1.2864, "lon": 36.8172},
    {"city": "Johannesburg", "country": "South Africa", "lat": -26.2041, "lon": 28.0473},
    {"city": "Mumbai", "country": "India", "lat": 19.0760, "lon": 72.8777},
    {"city": "Bangkok", "country": "Thailand", "lat": 13.7563, "lon": 100.5018},
    {"city": "Beijing", "country": "China", "lat": 39.9042, "lon": 116.4074},
    {"city": "Tokyo", "country": "Japan", "lat": 35.6762, "lon": 139.6503},
    {"city": "Sydney", "country": "Australia", "lat": -33.8688, "lon": 151.2093},
    {"city": "Canberra", "country": "Australia", "lat": -35.2809, "lon": 149.1300},
]

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f"Wrote {len(df)} cities to {OUT}")