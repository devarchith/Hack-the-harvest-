"""Per-pixel monthly Sentinel-2 (10 bands) and Sentinel-1 (VV, VH) composites for every labeled field-year (CPU,
network). Input for presto_train.py.

  python classifier/presto/presto_pixels.py --out /home/keeboh/work/presto/pixels --years 2021 --workers 6

The field-years and polygons come from field_years.csv in this folder, a copy of the Prithvi classifier's table, so
both models see the same 127 labeled field-years. Imagery rules are imported from preprocessing/loader.py
so they cannot drift: Planetary Computer `sentinel-2-l2a`, newest copy of each scene, reflectance scaling with the
04.00 baseline offset, SCL classes 4-7 kept, red > 0 and nir > 0, 10 m UTM grid anchored on native Sentinel-2 pixel
edges, fields shrunk 10 m inward, and a scene-day counts for a field only when at least 60% of its pixels are clear.
Added here: bands B06 and B07 (Presto uses them) and Sentinel-1 from `sentinel-1-rtc` (terrain-corrected gamma0,
linear power; nodata -32768), read nearest onto the same 10 m grid; a Sentinel-1 scene-day counts for a field when
at least 60% of its pixels have data.

Monthly composites per pixel, January to December: Sentinel-2 median of clear readings on usable days; Sentinel-1
mean in linear power, then dB. NaN where a month has nothing.

Writes <out>/pixels_<year>.npz (field_id, offsets, s2 (n_pix, 12, 10) reflectance, s2_nobs, s1 (n_pix, 12, 2) dB,
s1_nobs) and <out>/pixels_<year>.provenance.json, with a copy of the provenance in results/pixels/.
"""
import argparse, datetime, hashlib, json, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np, pandas as pd, geopandas as gpd
import planetary_computer
from odc.geo import xy_
from odc.geo.geobox import GeoBox
from odc.stac import configure_rio, load
from pystac_client import Client

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from preprocessing.loader import (PC_STAC, SCL_KEEP, _field_pixels, _newest_copies, _solar_day,  # noqa: E402
                                  _to_reflectance, _valid_mask)

CRS = "EPSG:32618"
S2 = {"B02": "B2", "B03": "B3", "B04": "B4", "B05": "B5", "B06": "B6", "B07": "B7", "B08": "B8", "B8A": "B8A",
      "B11": "B11", "B12": "B12"}
S1 = ["vv", "vh"]
USABLE = 0.6


def read_day(sensor, items, gbox, pixels, retries=3):
    """{field_id: (n_pix, nb) float32, NaN where not clear} for one scene-day; fields under 60% clear are left out."""
    assets = list(S2) + ["SCL"] if sensor == "s2" else S1
    for attempt in range(retries):
        try:
            ds = load(items, bands=assets, geobox=gbox, groupby="solar_day", chunks=None,
                      patch_url=planetary_computer.sign, resampling="nearest")
            break
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(3 * (attempt + 1))
    if sensor == "s2":
        base = items[0].properties.get("s2:processing_baseline", "")
        arr = np.stack([_to_reflectance("s2", ds[a].values[0], base).ravel() for a in S2], 1)
        ok = _valid_mask("s2", ds["SCL"].values[0]).ravel() & (arr[:, 2] > 0) & (arr[:, 6] > 0)
    else:
        arr = np.stack([ds[a].values[0].astype(np.float32).ravel() for a in S1], 1)
        arr[(arr == -32768) | ~np.isfinite(arr) | (arr <= 0)] = np.nan
        ok = np.isfinite(arr).all(1)
    arr[~ok] = np.nan
    return {fid: arr[pix] for fid, pix in pixels.items() if ok[pix].mean() >= USABLE}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field-years", type=Path, default=HERE / "field_years.csv")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--years", type=int, nargs="+", default=list(range(2017, 2024)))
    ap.add_argument("--months", type=int, nargs=2, default=(1, 12), help="first and last month to read (smoke tests)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    t0 = time.time()
    configure_rio(cloud_defaults=True, GDAL_HTTP_TIMEOUT=60, GDAL_HTTP_MAX_RETRY=3, GDAL_HTTP_RETRY_DELAY=2)
    args.out.mkdir(parents=True, exist_ok=True)

    fy = pd.read_csv(args.field_years)
    fy = gpd.GeoDataFrame(fy, geometry=gpd.GeoSeries.from_wkt(fy.wkt), crs=CRS)
    cat = Client.open(PC_STAC)
    m0, m1 = args.months
    for yr in args.years:
        d = fy[fy.year == yr].sort_values("field_id")
        ids = d.field_id.tolist()
        x0, y0, x1, y1 = d.total_bounds
        gbox = GeoBox.from_bbox((x0 - 30, y0 - 30, x1 + 30, y1 + 30), crs=CRS, resolution=10, anchor=xy_(0, 0))
        bbox_ll = tuple(d.to_crs(4326).total_bounds)
        lon = float(np.mean(bbox_ll[0::2]))
        pixels, fb = _field_pixels(d.geometry, ids, gbox, 10.0)
        end = pd.Timestamp(yr, m1, 1) + pd.offsets.MonthEnd(0)
        dt = f"{yr}-{m0:02d}-01/{end.date()}"
        tasks, queries = [], []
        for sensor, coll in (("s2", "sentinel-2-l2a"), ("s1", "sentinel-1-rtc")):
            items = list(cat.search(collections=[coll], bbox=bbox_ll, datetime=dt).items())
            n = len(items)
            if sensor == "s2":
                items = _newest_copies(items)
            queries.append({"sensor": sensor, "collection": coll, "bbox": bbox_ll, "datetime": dt, "n_items": n,
                            "n_older_copies_dropped": n - len(items)})
            days = {}
            for it in items:
                days.setdefault(_solar_day(it, lon), []).append(it)
            tasks += [(sensor, day, g) for day, g in days.items()]
        print(f"{yr}: {len(ids)} field-years, {len(tasks)} scene-days to read", flush=True)

        acc, failures = {}, []  # (sensor, field_id, month) -> list of (n_pix, nb)
        with ThreadPoolExecutor(args.workers) as pool:
            futs = {pool.submit(read_day, s, g, gbox, pixels): (s, day, g) for s, day, g in tasks}
            for n, f in enumerate(as_completed(futs), 1):
                s, day, g = futs[f]
                try:
                    for fid, a in f.result().items():
                        acc.setdefault((s, fid, int(day[5:7])), []).append(a)
                except Exception as e:
                    failures.append({"sensor": s, "day": day, "items": [i.id for i in g], "error": repr(e)[:300]})
                if n % 50 == 0 or n == len(futs):
                    print(f"  {yr}: {n}/{len(futs)} scene-days, {len(failures)} failed, {time.time() - t0:.0f} s",
                          flush=True)

        off = np.cumsum([0] + [len(pixels[f]) for f in ids])
        npx = int(off[-1])
        s2 = np.full((npx, 12, len(S2)), np.nan, np.float32); s2n = np.zeros((npx, 12), np.uint8)
        s1 = np.full((npx, 12, 2), np.nan, np.float32); s1n = np.zeros((npx, 12), np.uint8)
        for i, fid in enumerate(ids):
            sl = slice(off[i], off[i + 1])
            for m in range(1, 13):
                a = acc.get(("s2", fid, m))
                if a:
                    st = np.stack(a)
                    s2n[sl, m - 1] = np.isfinite(st[..., 0]).sum(0)
                    with np.errstate(all="ignore"):
                        s2[sl, m - 1] = np.nanmedian(st, 0)
                b = acc.get(("s1", fid, m))
                if b:
                    st = np.stack(b)
                    s1n[sl, m - 1] = np.isfinite(st[..., 0]).sum(0)
                    with np.errstate(all="ignore"):
                        s1[sl, m - 1] = 10 * np.log10(np.nanmean(st, 0))
        npz = args.out / f"pixels_{yr}.npz"
        np.savez_compressed(npz, field_id=np.array(ids), offsets=off, s2=s2, s2_nobs=s2n, s1=s1, s1_nobs=s1n,
                            s2_bands=np.array(list(S2.values())), s1_bands=np.array(S1))
        prov = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "year": yr, "field_years": len(ids), "pixels": npx, "fields_without_inward_buffer": fb,
                "field_years_csv_sha256": hashlib.sha256(args.field_years.read_bytes()).hexdigest(),
                "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
                "catalog": PC_STAC, "crs": CRS, "resolution_m": 10, "grid_anchor": 0, "buffer_m": 10,
                "s2_bands": S2, "s1_bands": S1, "s2_scl_keep": list(SCL_KEEP), "usable_frac": USABLE,
                "s2_scaling": "loader._to_reflectance: (DN - 1000)/1e4 if baseline >= 04.00 else DN/1e4",
                "s1": "sentinel-1-rtc gamma0 linear; -32768 and <= 0 dropped; monthly mean in linear, then 10*log10",
                "months": [m0, m1], "queries": queries, "scene_days": len(tasks), "failures": failures,
                "runtime_s": round(time.time() - t0)}
        for p in (args.out / f"pixels_{yr}.provenance.json", HERE / "results" / "pixels" / f"{yr}.provenance.json"):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(prov, indent=1, default=str), encoding="utf-8")
        print(f"{yr}: {len(ids)} field-years, {npx} pixels, {fb} without inward buffer, {len(failures)} failed "
              f"scene-days, done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
