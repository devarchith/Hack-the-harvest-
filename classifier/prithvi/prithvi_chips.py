"""30 m, six-band image stacks over the BARC fields for the Prithvi classifier.

  python classifier/prithvi/prithvi_chips.py --out /home/keeboh/work/prithvi/chips

For every year in field_years.csv, searches Sentinel-2 L2A and Landsat 8/9 Collection 2 Level-2 on Microsoft
Planetary Computer over the BARC fields (June to November), loads each scene-day onto one 30 m grid in EPSG:32618
(the Landsat grid; Sentinel-2 is averaged from 10/20 m), masks clouds per pixel with the team loader's rules
(Sentinel-2 SCL classes 4-7, Landsat QA bits 0-5, both widened by one pixel), and writes <out>/<year>.npz (with <out>/<year>.provenance.json recording every query and failure):

  x          (days, 6, H, W) int16, surface reflectance x 10000, -32768 where masked
  dates      ISO dates (solar day), sensors ("S2" or "LANDSAT"), items (scene IDs)
  transform  the grid's affine transform (6 numbers)

Bands are in Prithvi-EO-2.0's order: blue, green, red, narrow NIR, SWIR 1, SWIR 2 (Sentinel-2 B02 B03 B04 B8A B11
B12; Landsat blue green red nir08 swir16 swir22). Prithvi was trained on NASA HLS, which Planetary Computer only
holds from 2020; the training years start in 2017, so the same six bands are built from the two source collections.
HLS's BRDF and bandpass adjustments are not applied.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer
from odc.geo import xy_
from odc.geo.geobox import GeoBox
from odc.stac import configure_rio, load
from pystac_client import Client
from rasterio.features import geometry_mask
from scipy.ndimage import binary_dilation
from shapely import make_valid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from preprocessing.loader import (LOADER_VERSION, PC_STAC, SENSORS, _newest_copies, _season_windows,  # noqa: E402
                                  _solar_day, _to_reflectance, _valid_mask)

BANDS = {"s2": ["B02", "B03", "B04", "B8A", "B11", "B12"],
         "landsat": ["blue", "green", "red", "nir08", "swir16", "swir22"]}
CRS, RES, NODATA = "EPSG:32618", 30, -32768


def grid(fields, pad_m):
    minx, miny, maxx, maxy = fields.total_bounds
    return GeoBox.from_bbox((minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m), crs=CRS, resolution=RES,
                            anchor=xy_(0.5, 0.5))  # Landsat pixel edges sit 15 m past multiples of 30 m


def read_day(sensor, items, gbox, field_px, retries=4):
    qa = SENSORS[sensor]["qa"]
    for attempt in range(retries):
        try:
            ds = load(items, bands=BANDS[sensor] + [qa], geobox=gbox, groupby="solar_day", chunks=None,
                      patch_url=planetary_computer.sign, resampling={"*": "average", qa: "nearest"})
            break
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(3 * (attempt + 1))
    baseline = items[0].properties.get("s2:processing_baseline", "")
    refl = np.stack([_to_reflectance(sensor, ds[b].values[0], baseline) for b in BANDS[sensor]])
    ok = _valid_mask(sensor, ds[qa].values[0]) & (refl[2] > 0) & (refl[3] > 0) & np.isfinite(refl).all(0)
    ok &= ~binary_dilation(~ok, iterations=1)
    x = np.where(ok, np.clip(np.round(refl * 10000), -32767, 32767), NODATA).astype(np.int16)
    return x, float(ok[field_px].mean())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field-years", default=str(HERE / "field_years.csv"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--years", type=int, nargs="*", help="default: every year in field_years.csv")
    ap.add_argument("--months", type=int, nargs=2, default=(6, 11))
    ap.add_argument("--max-scene-cloud", type=float, default=80)
    ap.add_argument("--min-field-valid", type=float, default=0.05,
                    help="keep a scene-day only if at least this share of field pixels is cloud-free")
    ap.add_argument("--pad-m", type=float, default=1500)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-days", type=int, help="smoke test: read only the first N scene-days per year")
    a = ap.parse_args(argv)

    configure_rio(cloud_defaults=True, GDAL_HTTP_TIMEOUT=60, GDAL_HTTP_MAX_RETRY=3, GDAL_HTTP_RETRY_DELAY=2)
    fy = pd.read_csv(a.field_years)
    fields = gpd.GeoDataFrame(fy, geometry=gpd.GeoSeries.from_wkt(fy.wkt).apply(make_valid), crs=CRS)
    gbox = grid(fields, a.pad_m)
    field_px = geometry_mask(list(fields.geometry), out_shape=gbox.shape, transform=gbox.transform, invert=True)
    lon = float(fields.to_crs(4326).geometry.union_all().centroid.x)
    bbox_ll = tuple(gpd.GeoSeries([fields.geometry.union_all().envelope], crs=CRS).to_crs(4326).total_bounds)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    catalog = Client.open(PC_STAC)
    years = a.years or sorted(fy.year.unique().tolist())
    prov = {"script": "prithvi_chips.py", "loader_version": LOADER_VERSION, "stac": PC_STAC,
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "args": vars(a),
            "grid": {"crs": CRS, "resolution": RES, "shape": list(gbox.shape), "transform": list(gbox.transform)[:6]},
            "bands": BANDS, "years": {}}
    t0 = time.time()
    for yr in years:
        tasks, queries = [], []
        for sensor in ("s2", "landsat"):
            cfg = SENSORS[sensor]
            query = dict(cfg["query"], **{"eo:cloud_cover": {"lt": a.max_scene_cloud}})
            dt = "/".join(_season_windows([yr], a.months)[0])
            items = list(catalog.search(collections=[cfg["collection"]], bbox=bbox_ll, datetime=dt,
                                        query=query).items())
            n = len(items)
            if sensor == "s2":
                items = _newest_copies(items)
            queries.append({"sensor": sensor, "collection": cfg["collection"], "bbox": bbox_ll, "datetime": dt,
                            "query": query, "n_items": n, "n_kept": len(items)})
            by_day = {}
            for it in items:
                by_day.setdefault(_solar_day(it, lon), []).append(it)
            tasks += [(sensor, d, g) for d, g in sorted(by_day.items())]
        tasks = tasks[:a.max_days] if a.max_days else tasks
        days, fails = [], []
        with ThreadPoolExecutor(a.workers) as ex:
            futs = {ex.submit(read_day, s, g, gbox, field_px): (s, d, g) for s, d, g in tasks}
            for f in as_completed(futs):
                s, d, g = futs[f]
                try:
                    x, frac = f.result()
                    days.append((d, s.upper(), ";".join(i.id for i in g), frac, x))
                except Exception as e:  # recorded, then retried once below
                    fails.append((s, d, g, repr(e)[:200]))
        for s, d, g, _ in list(fails):
            try:
                x, frac = read_day(s, g, gbox, field_px)
                days.append((d, s.upper(), ";".join(i.id for i in g), frac, x))
                fails.remove((s, d, g, _))
            except Exception:
                pass
        kept = sorted([r for r in days if r[3] >= a.min_field_valid], key=lambda r: (r[0], r[1]))
        np.savez_compressed(out / f"{yr}.npz", x=np.stack([r[4] for r in kept]),
                            dates=np.array([r[0] for r in kept]), sensors=np.array([r[1] for r in kept]),
                            items=np.array([r[2] for r in kept]), field_valid=np.array([r[3] for r in kept]),
                            transform=np.array(list(gbox.transform)[:6]))
        prov["years"][str(yr)] = {"queries": queries, "scene_days_read": len(days), "scene_days_kept": len(kept),
                                  "failures": [{"sensor": s, "date": d, "items": [i.id for i in g], "error": e}
                                               for s, d, g, e in fails]}
        prov["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        (out / f"{yr}.provenance.json").write_text(json.dumps({**prov, "years": {str(yr): prov["years"][str(yr)]}},
                                                              indent=1))
        print(f"{yr}: {len(tasks)} scene-days, kept {len(kept)} (>= {a.min_field_valid:.0%} of field pixels clear), "
              f"failed {len(fails)}, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
