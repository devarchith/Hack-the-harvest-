"""Quick end-to-end check of the loader on two small squares near BARC (no evaluation data needed).

Run from the repository root:  python preprocessing/smoke_test.py
"""
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing.loader import fetch_series, load_series  # noqa: E402

squares = gpd.GeoDataFrame(
    {"field_id": [1, 2]},
    geometry=[box(-76.872, 39.024, -76.869, 39.026), box(-76.900, 39.030, -76.897, 39.032)],
    crs=4326,
)
with tempfile.TemporaryDirectory() as tmp:
    df = fetch_series(squares, years=[2021, 2022], months=(8, 9), out_dir=tmp, tag="smoke", workers=8)
    back = load_series(Path(tmp) / "series_smoke.parquet", usable_only=False)
    assert len(back) == len(df) > 0
    print(df.groupby(["year", "sensor"]).agg(rows=("date", "size"), usable=("usable", "mean"),
                                             red=("red", "median"), nir=("nir", "median"),
                                             ndvi=("ndvi", "median"), ndti=("ndti", "median")).round(3))
    seen = df[df["n_valid"] > 0]
    assert seen["ndvi"].between(-1, 1).all() and seen["red"].between(0, 1).all(), "values out of range"
    scenes = df.loc[df["sensor"] == "S2", "items"].map(lambda s: [i.rpartition("_")[0] for i in s.split(";")])
    assert scenes.map(lambda s: len(s) == len(set(s))).all(), "a day mixes two copies of one S2 scene"
    s2 = df[df["usable"] & (df["sensor"] == "S2")]
    print("S2 baselines by year:", s2.groupby("year")["baseline"].unique().to_dict())
    print("columns:", list(df.columns))
print("smoke test passed")
