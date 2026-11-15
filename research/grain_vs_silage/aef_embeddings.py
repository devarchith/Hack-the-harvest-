"""AlphaEarth Foundations annual embeddings, averaged over each labeled BARC field-year.

  python research/grain_vs_silage/aef_embeddings.py

Source: Google / Google DeepMind Satellite Embedding V1 (annual), as Cloud-Optimized GeoTIFFs on Source Cooperative
(https://source.coop/tge-labs/aef, CC-BY 4.0). Required attribution: "The AlphaEarth Foundations Satellite Embedding
dataset is produced by Google and Google DeepMind." Each 10 m pixel has 64 int8 channels; de-quantized as
(v / 127.5)^2 * sign(v), and -128 means no data. The files are stored bottom-up. The .vrt files provided to flip them
point at /vsis3/ paths that need S3 settings, so this script reads windows of the raw .tiff over HTTPS and flips them.

Writes data/aef/barc_<year>.npz (cached mosaics of the BARC area, git-ignored) and results/aef_embeddings.csv: one row
per labeled field-year (2017-2020 farm-record labels, 2021-2023 evaluation labels) holding the mean embedding over the
field's pixels after a 10 m inward buffer, rescaled to length 1 (the same rule the dataset uses for its overviews).
"""
import os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from rasterio.windows import Window

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from preprocessing.loader import barc_fields  # noqa: E402

BASE = "https://data.source.coop/tge-labs/aef/v1/annual"
CACHE = ROOT / "data" / "aef"
CRS, RES, NODATA = "EPSG:32618", 10.0, -128
YEARS = list(range(2017, 2024))
AEF = [f"A{i:02d}" for i in range(64)]
for k, v in dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tiff",
                 GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="3").items():
    os.environ.setdefault(k, v)


def tiles(ll):
    """Index rows for every file that overlaps the lon/lat box ll = (west, south, east, north)."""
    p = CACHE / "aef_index.parquet"
    if not p.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{BASE}/aef_index.parquet", p)
    d = pd.read_parquet(p, columns=["path", "year", "crs", "wgs84_west", "wgs84_south", "wgs84_east", "wgs84_north"])
    w, s, e, n = ll
    d = d[(d.wgs84_west < e) & (d.wgs84_east > w) & (d.wgs84_south < n) & (d.wgs84_north > s) & d.year.isin(YEARS)]
    if (d.crs != CRS).any():
        sys.exit(f"expected every file in {CRS}, got {sorted(d.crs.unique())}")
    d = d.copy()
    d["url"] = "/vsicurl/" + d.path.str.replace("s3://us-west-2.opendata.source.coop/", "https://data.source.coop/",
                                                regex=False)
    return d


def mosaic(year, urls, box):
    """North-up int8 array (64, H, W) covering box (UTM metres) on the dataset's 10 m grid, plus its transform."""
    out = CACHE / f"barc_{year}.npz"
    if out.exists():
        z = np.load(out)
        return z["a"], Affine(*z["t"])
    x0, y0 = np.floor(box[0] / RES) * RES, np.floor(box[1] / RES) * RES
    x1, y1 = np.ceil(box[2] / RES) * RES, np.ceil(box[3] / RES) * RES
    a = np.full((64, int(round((y1 - y0) / RES)), int(round((x1 - x0) / RES))), NODATA, np.int8)
    for u in urls:
        with rasterio.open(u) as r:
            t = r.transform  # bottom-up file: origin at the south-west corner, positive y step
            assert (t.a, t.e, t.b, t.d) == (RES, RES, 0, 0), t
            c0, c1 = max(0, round((x0 - t.c) / RES)), min(r.width, round((x1 - t.c) / RES))
            r0, r1 = max(0, round((y0 - t.f) / RES)), min(r.height, round((y1 - t.f) / RES))
            if c1 <= c0 or r1 <= r0:
                continue
            blk = r.read(window=Window(c0, r0, c1 - c0, r1 - r0))[:, ::-1, :]  # flip rows to north-up
            dc = round((t.c + c0 * RES - x0) / RES)
            dr = round((y1 - (t.f + r1 * RES)) / RES)
            dst = a[:, dr:dr + blk.shape[1], dc:dc + blk.shape[2]]
            ok = blk != NODATA
            dst[ok] = blk[ok]
    tr = Affine(RES, 0, x0, 0, -RES, y1)
    np.savez_compressed(out, a=a, t=np.array(tuple(tr)[:6]))
    return a, tr


def field_means(a, tr, polys):
    rows = []
    for fid, geom in zip(polys.field_id, polys.geometry):
        g = geom.buffer(-10)
        g = geom if g.is_empty else g
        m = geometry_mask([g], out_shape=a.shape[1:], transform=tr, invert=True)
        v = a[:, m]
        v = v[:, (v != NODATA).all(0)].astype(np.float32)
        r = {"field_id": fid, "aef_n_pix": v.shape[1]}
        if v.shape[1]:
            mu = ((v / 127.5) ** 2 * np.sign(v)).mean(1)
            r.update(zip(AEF, mu / np.linalg.norm(mu)))
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ev = barc_fields(ROOT / "04_Evaluation" / "BARC_Corn.2021-2023.gpkg")[["field_id", "geometry"]].to_crs(CRS)
    kz = gpd.read_file(ROOT / "data" / "fields_kraatz_2017_2020.gpkg", layer="fields")[["field_id", "geometry"]]
    kz = kz.to_crs(CRS)
    cols = ["field_id", "farm", "field", "year", "label"]
    lab21 = pd.read_csv(HERE / "results" / "features_all_field_years.csv", usecols=cols)
    lab21 = lab21[lab21.label.notna()]
    lab17 = pd.read_csv(HERE / "results" / "features_2017_2020.csv", usecols=cols)
    lab17 = lab17[lab17.label.notna()]

    allp = pd.concat([ev, kz], ignore_index=True)
    box = allp.total_bounds + np.array([-200, -200, 200, 200])
    t = tiles(gpd.GeoSeries(allp.geometry, crs=CRS).to_crs(4326).total_bounds)
    print(f"{len(t)} files for {YEARS[0]}-{YEARS[-1]}; mosaic box {box.round(0)} (UTM 18N)", flush=True)
    with ThreadPoolExecutor(4) as ex:
        mos = dict(zip(YEARS, ex.map(lambda yr: mosaic(yr, t[t.year == yr].url.tolist(), box), YEARS)))
    for yr, (a, _) in mos.items():
        print(yr, "mosaic", a.shape, f"no-data share {(a[0] == NODATA).mean():.3f}", flush=True)

    out = []
    for lab, polys in ((lab17, kz), (lab21, ev)):
        for yr, g in lab.groupby("year"):
            a, tr = mos[yr]
            out.append(g.merge(field_means(a, tr, polys[polys.field_id.isin(g.field_id)]), on="field_id", how="left"))
    res = pd.concat(out, ignore_index=True)
    res.round(5).to_csv(HERE / "results" / "aef_embeddings.csv", index=False)
    print(f"wrote results/aef_embeddings.csv: {len(res)} field-years, pixels per field min "
          f"{int(res.aef_n_pix.min())} median {int(res.aef_n_pix.median())}, missing {int(res.A00.isna().sum())}")


if __name__ == "__main__":
    main()
