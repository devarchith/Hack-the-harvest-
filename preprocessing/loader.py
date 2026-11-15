"""Per-field satellite time series from Microsoft Planetary Computer.

fetch_series() searches Sentinel-2 and Landsat, masks clouds per pixel, converts to surface reflectance,
computes indices and writes one row per field, date and sensor to data/series_<tag>.parquet, with a
provenance JSON beside it. load_series() reads the parquet back. Columns are listed in README.md.

Labels never pass through fetch or load; barc_labels() is for validation only.

Usage (from the repository root):
    python preprocessing/loader.py --gpkg PATH/BARC_Corn.2021-2023.gpkg --years 2021 2022 2023
    python preprocessing/loader.py --fields data/fields_kraatz_2017_2020.gpkg --years 2017 2018 2019 2020 \
        --tag kraatz_2017_2020 --cell-m 50000 --weather
"""

from __future__ import annotations

import argparse
import calendar
import io
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer
from odc.geo import xy_
from odc.geo.geobox import GeoBox
from odc.stac import configure_rio, load
from pystac_client import Client
from rasterio.features import rasterize
from shapely.geometry import box

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
BARC_LAYER = "barc_silage_grain_corn20212023"
BARC_LABEL_COLUMNS = {"Corn_2021": 2021, "Corn_2022": 2022, "Corn_2023": 2023}
LOADER_VERSION = "0.1.3"

# Per sensor: collection, asset keys mapped to common names, resolution and cloud rule.
SENSORS = {
    "s2": {
        "collection": "sentinel-2-l2a",
        "bands": {"B02": "blue", "B03": "green", "B04": "red", "B05": "re1", "B08": "nir",
                  "B8A": "nir_narrow", "B11": "swir1", "B12": "swir2"},
        "qa": "SCL",
        "resolution": 10,
        "grid_anchor": 0.0,  # native pixel edges at multiples of 10 m in UTM
        "query": {},
    },
    "landsat": {
        "collection": "landsat-c2-l2",
        "bands": {"blue": "blue", "green": "green", "red": "red", "nir08": "nir",
                  "swir16": "swir1", "swir22": "swir2"},
        "qa": "qa_pixel",
        "resolution": 30,
        "grid_anchor": 0.5,  # native pixel edges at 15 m past multiples of 30 m; a 0 anchor shifts fields 15 m
        "query": {"platform": {"in": ["landsat-8", "landsat-9"]}},
    },
}
SCL_KEEP = (4, 5, 6, 7)  # vegetation, bare, water, unclassified: the starter notebook's choice
LANDSAT_QA_DROP_BITS = (0, 1, 2, 3, 4, 5)  # fill, dilated cloud, cirrus, cloud, shadow, snow
REFLECTANCE = ["blue", "green", "red", "nir", "swir1", "swir2", "re1", "nir_narrow"]
INDICES = ["ndvi", "ndvi_p25", "ndvi_p75", "ndti", "ndmi", "ndre"]


# ---------------------------------------------------------------- BARC evaluation file

def barc_fields(gpkg: str | Path) -> gpd.GeoDataFrame:
    """BARC polygons with field_id and farm metadata, label columns removed."""
    gdf = gpd.read_file(gpkg, layer=BARC_LAYER)
    return gdf.drop(columns=list(BARC_LABEL_COLUMNS)).rename(columns={"OBJECTID": "field_id"})


def barc_labels(gpkg: str | Path) -> pd.DataFrame:
    """Validation only: one row per labeled field-year (field_id, year, harvest_type)."""
    gdf = gpd.read_file(gpkg, layer=BARC_LAYER, ignore_geometry=True)
    long = gdf.melt(id_vars=["OBJECTID"], value_vars=list(BARC_LABEL_COLUMNS),
                    var_name="col", value_name="harvest_type")
    long = long[long["harvest_type"].isin(["Grain", "Silage"])]
    long["year"] = long["col"].map(BARC_LABEL_COLUMNS)
    return (long.rename(columns={"OBJECTID": "field_id"})[["field_id", "year", "harvest_type"]]
            .sort_values(["year", "field_id"]).reset_index(drop=True))


# ---------------------------------------------------------------- fetch

def _season_windows(years, months):
    m0, m1 = months
    return [(f"{y}-{m0:02d}-01", f"{y}-{m1:02d}-{calendar.monthrange(y, m1)[1]:02d}") for y in years]


def _solar_day(item, lon):
    return (item.datetime + timedelta(hours=lon / 15.0)).date().isoformat()


def _newest_copies(items):
    """Planetary Computer lists some Sentinel-2 scenes twice: as first processed and as reprocessed later
    (IDs differ only in the last, processing-time part). Keep the newest copy, so one day never mixes two
    processing baselines and therefore two scalings."""
    newest = {}
    for it in items:
        scene, _, processed = it.id.rpartition("_")
        if scene not in newest or processed > newest[scene][0]:
            newest[scene] = (processed, it)
    return [it for _, it in newest.values()]


def _to_reflectance(sensor, arr, baseline):
    arr = arr.astype("float32")
    arr[arr == 0] = np.nan
    if sensor == "s2":
        # Processing baseline 04.00 (Jan 2022 on) stores reflectance with a +1000 offset.
        offset = -1000.0 if baseline and float(baseline) >= 4.0 else 0.0
        return (arr + offset) / 10000.0
    return arr * 2.75e-5 - 0.2


def _valid_mask(sensor, qa):
    if sensor == "s2":
        return np.isin(qa, SCL_KEEP)
    qa = qa.astype("uint16")
    bad = np.zeros(qa.shape, dtype=bool)
    for bit in LANDSAT_QA_DROP_BITS:
        bad |= (qa >> bit) & 1 == 1
    return ~bad


def _nd(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a - b) / (a + b)


def _cells(gdf_proj, cell_m):
    """Group fields into square cells so a statewide sample loads many small boxes, not one huge one."""
    c = gdf_proj.geometry.centroid
    key = list(zip((c.x // cell_m).astype(int), (c.y // cell_m).astype(int)))
    return gdf_proj.assign(_cell=key).groupby("_cell")


def _field_pixels(geoms, ids, gbox, buffer_m, min_pixels=2):
    """Flat pixel indices per field on the geobox, after an inward buffer (fallback: no buffer)."""
    out, fallback = {}, 0
    for fid, geom in zip(ids, geoms):
        g = geom.buffer(-buffer_m) if buffer_m else geom
        idx = None
        if not g.is_empty:
            r = rasterize([(g, 1)], out_shape=gbox.shape, transform=gbox.transform, fill=0, dtype="uint8")
            idx = np.flatnonzero(r)
        if idx is None or idx.size < min_pixels:
            r = rasterize([(geom, 1)], out_shape=gbox.shape, transform=gbox.transform, fill=0,
                          dtype="uint8", all_touched=True)
            idx = np.flatnonzero(r)
            fallback += 1
        out[fid] = idx
    return out, fallback


def _process_day(sensor, items, gbox, pixels, lon, retries=3):
    cfg = SENSORS[sensor]
    assets = list(cfg["bands"]) + [cfg["qa"]]
    for attempt in range(retries):
        try:
            ds = load(items, bands=assets, geobox=gbox, groupby="solar_day", chunks=None,
                      patch_url=planetary_computer.sign, resampling="nearest")
            break
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    baseline = items[0].properties.get("s2:processing_baseline", "")
    platform = items[0].properties.get("platform", "")
    bands = {name: _to_reflectance(sensor, ds[asset].values[0], baseline).ravel()
             for asset, name in cfg["bands"].items()}
    # Very dark pixels can come out of atmospheric correction slightly below zero, which pushes NDVI past 1.
    valid = _valid_mask(sensor, ds[cfg["qa"]].values[0]).ravel() & (bands["red"] > 0) & (bands["nir"] > 0)
    ndvi = _nd(bands["nir"], bands["red"])
    idx = {"ndvi": ndvi, "ndti": _nd(bands["swir1"], bands["swir2"]), "ndmi": _nd(bands["nir"], bands["swir1"])}
    if "re1" in bands:
        idx["ndre"] = _nd(bands["nir_narrow"], bands["re1"])
    date = _solar_day(items[0], lon)
    rows = []
    for fid, pix in pixels.items():
        v = pix[valid[pix]]
        row = {"field_id": fid, "date": date, "sensor": sensor.upper(), "platform": platform,
               "items": ";".join(i.id for i in items), "baseline": baseline,
               "n_pix": int(pix.size), "n_valid": int(v.size)}
        if v.size:
            for name, arr in bands.items():
                row[name] = float(np.nanmedian(arr[v]))
            for name, arr in idx.items():
                row[name] = float(np.nanmedian(arr[v]))
            row["ndvi_p25"], row["ndvi_p75"] = (float(x) for x in np.nanpercentile(ndvi[v], [25, 75]))
        rows.append(row)
    return rows


def fetch_series(fields: gpd.GeoDataFrame, years, months=(3, 12), sensors=("s2", "landsat"),
                 out_dir="data", tag="barc", id_col="field_id", buffer_m=10.0, cell_m=20000,
                 max_scene_cloud=95, workers=32, usable_frac=0.6, crs="auto",
                 retry_failed=False) -> pd.DataFrame:
    """Build the per-field time series and write data/series_<tag>.parquet plus provenance JSON.

    retry_failed=True re-reads only the scene-days listed as failures in an earlier run with the same tag and
    the same arguments, and merges them into that run's parquet (Planetary Computer throttles bursts)."""
    out = Path(out_dir)
    prev = None
    if retry_failed:
        prev = json.loads((out / f"series_{tag}.provenance.json").read_text())
        failed_ids = {tuple(f["items"]) for f in prev["failures"]}
    # Without a timeout a stalled blob read blocks its thread forever and the run never finishes.
    configure_rio(cloud_defaults=True, GDAL_HTTP_TIMEOUT=60, GDAL_HTTP_MAX_RETRY=3, GDAL_HTTP_RETRY_DELAY=2)
    fields = fields.to_crs(4326)
    target_crs = fields.estimate_utm_crs() if crs == "auto" else crs
    proj = fields.to_crs(target_crs)
    lon = float(fields.geometry.union_all().centroid.x)
    catalog = Client.open(PC_STAC)
    windows = _season_windows(years, months)

    tasks, prov_queries, fallback_total = [], [], 0
    for sensor in sensors:
        cfg = SENSORS[sensor]
        res = cfg["resolution"]
        for _, cell in _cells(proj, cell_m):
            minx, miny, maxx, maxy = cell.total_bounds
            pad = 3 * res
            gbox = GeoBox.from_bbox((minx - pad, miny - pad, maxx + pad, maxy + pad),
                                    crs=target_crs, resolution=res,
                                    anchor=xy_(cfg["grid_anchor"], cfg["grid_anchor"]))
            pixels, fb = _field_pixels(cell.geometry, cell[id_col], gbox, buffer_m)
            fallback_total += fb
            bbox_ll = tuple(gpd.GeoSeries([box(minx, miny, maxx, maxy)], crs=target_crs)
                            .to_crs(4326).total_bounds)
            for start, end in windows:
                query = dict(cfg["query"])
                if max_scene_cloud < 100:
                    query["eo:cloud_cover"] = {"lt": max_scene_cloud}
                items = list(catalog.search(collections=[cfg["collection"]], bbox=bbox_ll,
                                            datetime=f"{start}/{end}", query=query or None).items())
                n_found = len(items)
                if sensor == "s2":
                    items = _newest_copies(items)
                prov_queries.append({"sensor": sensor, "collection": cfg["collection"], "bbox": bbox_ll,
                                     "datetime": f"{start}/{end}", "query": query, "n_items": n_found,
                                     "n_older_copies_dropped": n_found - len(items)})
                by_day = {}
                for it in items:
                    by_day.setdefault(_solar_day(it, lon), []).append(it)
                tasks += [(sensor, grp, gbox, pixels) for grp in by_day.values()]

    if prev is not None:
        tasks = [t for t in tasks if tuple(i.id for i in t[1]) in failed_ids]
        print(f"retrying {len(tasks)} of {len(failed_ids)} failed scene-days", flush=True)

    rows, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_process_day, s, g, gb, px, lon): (s, g) for s, g, gb, px in tasks}
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                rows += fut.result()
            except Exception as exc:
                s, g = futs[fut]
                failures.append({"sensor": s, "items": [i.id for i in g], "error": repr(exc)[:300]})
            if n % 50 == 0 or n == len(futs):
                print(f"  {n}/{len(futs)} scene-days done, {len(failures)} failed", flush=True)

    if not rows and prev is None:
        raise RuntimeError(f"no rows produced; first failure: {failures[:1]}")
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["field_id", "date", "sensor", "platform", "items", "baseline", "n_pix", "n_valid"])
    for col in REFLECTANCE + INDICES:
        if col not in df:
            df[col] = np.nan
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["valid_frac"] = df["n_valid"] / df["n_pix"].where(df["n_pix"] > 0)
    df["usable"] = df["valid_frac"] >= usable_frac
    front = ["field_id", "year", "date", "sensor", "platform", "items", "baseline",
             "n_pix", "n_valid", "valid_frac", "usable"]
    df = df[front + REFLECTANCE + INDICES]
    if prev is not None:
        old = pd.read_parquet(out / f"series_{tag}.parquet")
        df = pd.concat([old, df]).drop_duplicates(["field_id", "date", "sensor"], keep="last")
    df = df.sort_values(["field_id", "date", "sensor"]).reset_index(drop=True)

    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / f"series_{tag}.parquet", index=False)
    prov = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "loader_version": LOADER_VERSION, "catalog": PC_STAC, "years": list(years), "months": list(months),
        "sensors": {s: {k: v for k, v in SENSORS[s].items()} for s in sensors},
        "cloud_rule": {"s2_scl_keep": SCL_KEEP, "landsat_qa_drop_bits": LANDSAT_QA_DROP_BITS,
                       "max_scene_cloud": max_scene_cloud, "usable_frac": usable_frac,
                       "pixel_rule": "red > 0 and nir > 0"},
        "duplicates": "Sentinel-2: newest processing of each scene kept, older copies dropped",
        "scaling": {"s2": "(DN - 1000)/10000 if processing baseline >= 04.00 else DN/10000; DN 0 = nodata",
                    "landsat": "DN * 2.75e-5 - 0.2; DN 0 = nodata"},
        "crs": str(target_crs), "buffer_m": buffer_m, "fields": int(len(fields)),
        "fields_without_inward_buffer": fallback_total, "queries": prov_queries,
        "scene_days": len(tasks), "failures": failures, "rows": int(len(df)),
    }
    if prev is not None:
        prov["scene_days"] = prev["scene_days"]
        prov["retries"] = prev.get("retries", []) + [{
            "created_utc": prov["created_utc"], "retried": len(tasks),
            "recovered": len(tasks) - len(failures), "still_failed": len(failures)}]
        prov["created_utc"] = prev["created_utc"]
    (out / f"series_{tag}.provenance.json").write_text(json.dumps(prov, indent=1, default=str))
    return df


# ---------------------------------------------------------------- weather (Daymet single-pixel API)

DAYMET_API = "https://daymet.ornl.gov/single-pixel/api/data"


def fetch_daymet(fields: gpd.GeoDataFrame, years, id_col="field_id", out_dir="data", tag="barc") -> pd.DataFrame:
    """Daily precipitation and temperature at each field centroid (one request per ~1 km point)."""
    pts = fields.to_crs(fields.estimate_utm_crs()).geometry.centroid.to_crs(4326)
    keys = pd.DataFrame({id_col: fields[id_col].values,
                         "lat": pts.y.round(2).values, "lon": pts.x.round(2).values})
    frames = []
    for (lat, lon), grp in keys.groupby(["lat", "lon"]):
        url = (f"{DAYMET_API}?lat={lat}&lon={lon}&vars=prcp,tmax,tmin"
               f"&start={min(years)}-01-01&end={max(years)}-12-31")
        text = urllib.request.urlopen(url, timeout=120).read().decode()
        body = text[text.index("year,yday"):]
        d = pd.read_csv(io.StringIO(body))
        d.columns = ["year", "yday", "prcp_mm", "tmax_c", "tmin_c"]
        d["date"] = pd.to_datetime(d["year"].astype(str), format="%Y") + pd.to_timedelta(d["yday"] - 1, unit="D")
        for fid in grp[id_col]:
            frames.append(d.assign(**{id_col: fid})[[id_col, "date", "prcp_mm", "tmax_c", "tmin_c"]])
    w = pd.concat(frames, ignore_index=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    w.to_parquet(out / f"weather_{tag}.parquet", index=False)
    return w


def add_rain_flag(series: pd.DataFrame, weather: pd.DataFrame, days=3, id_col="field_id") -> pd.DataFrame:
    """Adds rain_<days>d_mm: precipitation summed over the `days` days before each reading."""
    w = weather.sort_values([id_col, "date"]).copy()
    w["_prior"] = (w.groupby(id_col)["prcp_mm"]
                   .transform(lambda s: s.rolling(days, min_periods=1).sum().shift(1)))
    return series.merge(w[[id_col, "date", "_prior"]], on=[id_col, "date"], how="left").rename(
        columns={"_prior": f"rain_{days}d_mm"})


# ---------------------------------------------------------------- load

def load_series(path="data/series_barc.parquet", years=None, field_ids=None, sensors=None,
                usable_only=True) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if years is not None:
        df = df[df["year"].isin(list(years))]
    if field_ids is not None:
        df = df[df["field_id"].isin(list(field_ids))]
    if sensors is not None:
        df = df[df["sensor"].isin([s.upper() for s in sensors])]
    if usable_only:
        df = df[df["usable"]]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- command line

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--gpkg", help="BARC GeoPackage from the starter kit")
    src.add_argument("--fields", help="any polygon file with an integer id column (e.g. from kraatz_fields.py)")
    p.add_argument("--id-col", default="field_id", help="id column in --fields")
    p.add_argument("--cell-m", type=int, default=20000,
                   help="field grouping cell; 50000 reads BARC once per scene instead of four times")
    p.add_argument("--years", type=int, nargs="+", default=[2021, 2022, 2023])
    p.add_argument("--months", type=int, nargs=2, default=[3, 12])
    p.add_argument("--sensors", nargs="+", default=["s2", "landsat"], choices=list(SENSORS))
    p.add_argument("--out", default="data")
    p.add_argument("--tag", default="barc")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--weather", action="store_true", help="also fetch Daymet daily weather")
    p.add_argument("--retry-failed", action="store_true",
                   help="re-read only the scene-days that failed in the earlier run with this --tag and merge them")
    a = p.parse_args(argv)
    if a.gpkg:
        fields, id_col = barc_fields(a.gpkg), "field_id"
    else:
        fields, id_col = gpd.read_file(a.fields), a.id_col
    t0 = time.time()
    df = fetch_series(fields, a.years, months=tuple(a.months), sensors=a.sensors, out_dir=a.out,
                      tag=a.tag, id_col=id_col, cell_m=a.cell_m, workers=a.workers, retry_failed=a.retry_failed)
    print(f"{len(df)} rows, {df['field_id'].nunique()} fields, {df['usable'].mean():.0%} usable, "
          f"{time.time() - t0:.0f} s -> {a.out}/series_{a.tag}.parquet")
    if a.weather:
        w = fetch_daymet(fields, a.years, id_col=id_col, out_dir=a.out, tag=a.tag)
        print(f"weather: {len(w)} rows -> {a.out}/weather_{a.tag}.parquet")


if __name__ == "__main__":
    sys.exit(main())
