"""Québec silage and grain corn field-years from the FADQ parcel declarations, 2021-2023, for the Presto pixel pull.

  python classifier/quebec/quebec_field_years.py --src /home/keeboh/work/quebec/src --out classifier/quebec

Source: Base de données des parcelles et productions agricoles déclarées (BDPPAD), La Financière agricole du Québec,
CC BY 4.0, one shapefile per year. A parcel is kept when its crop group is MAF (maïs fourrager, silage) or MGR
(maïs-grain) and it is at least 3 ha. The labels are insurance declarations, not harvest records: farmers can switch
between grain and silage until mid-September (FADQ user guide).

To keep the image reads small, all field-years come from one 30 km square in UTM zone 18N, the one where the rarer
class (silage) is most common in its worst year. Up to --per-class parcels per class and year are drawn at random
(seed 0). A parcel keeps one field_id in every year (from IDPAR), so splits by field never put the same ground in
training and test.

Writes field_years_quebec.csv (same columns as the BARC field_years.csv, polygon as WKT in EPSG:32618; the five hand
features are empty) and quebec_selection.txt (counts and the square).
"""
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio

CRS = "EPSG:32618"
GROUPS = {"MAF": "Silage", "MGR": "Grain"}
SMALL = ["drop_doy", "ndvi_pre_drop", "drop_size", "ndvi_sep15", "ndti_post_drop"]
COLS = ["AN", "IDANPAR", "IDPAR", "GROPRO", "NBPRO", "CODPRO1", "PRCSUPPR1", "SUPHEC"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="folder with the unzipped BDPPAD_*_<year> folders")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--years", type=int, nargs="+", default=[2021, 2022, 2023])
    ap.add_argument("--min-ha", type=float, default=3.0)
    ap.add_argument("--square-km", type=float, default=30.0)
    ap.add_argument("--per-class", type=int, default=400)
    a = ap.parse_args()
    log = []

    parts = []
    for yr in a.years:
        shp = sorted(a.src.glob(f"BDPPAD_*_{yr}/*.shp"))[0]
        d = pyogrio.read_dataframe(shp, columns=COLS, where="GROPRO IN ('MAF', 'MGR')")
        log.append(f"{yr}: {shp.name}: {len(d)} MAF/MGR parcels; NBPRO {d.NBPRO.value_counts().to_dict()}; "
                   f"CODPRO1 {d.CODPRO1.value_counts().to_dict()}")
        d = d[d.SUPHEC >= a.min_ha]
        b = d.geometry.bounds
        lon = (b.minx + b.maxx) / 2
        d = d[(lon > -78) & (lon < -72)].to_crs(CRS)  # UTM zone 18N
        d["year"], d["label"] = yr, d.GROPRO.map(GROUPS)
        parts.append(d)
    d = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=CRS)
    for (yr, lab), n in d.groupby(["year", "label"]).size().items():
        log.append(f"{yr} {lab}: {n} parcels >= {a.min_ha} ha in zone 18N")

    # Best square: slide a square_km window in 10 km steps; score = fewest silage parcels in any year.
    step, w = 10_000, int(a.square_km * 1000)
    c = d.geometry.centroid
    d["gx"], d["gy"] = (c.x // step).astype(int), (c.y // step).astype(int)
    k = w // step
    counts = d.groupby(["gx", "gy", "year", "label"]).size()
    best = None
    for cx0 in range(d.gx.min(), d.gx.max() + 1):
        for cy0 in range(d.gy.min(), d.gy.max() + 1):
            sel = counts[(counts.index.get_level_values(0) >= cx0) & (counts.index.get_level_values(0) < cx0 + k)
                         & (counts.index.get_level_values(1) >= cy0) & (counts.index.get_level_values(1) < cy0 + k)]
            if sel.empty:
                continue
            t = sel.groupby(level=[2, 3]).sum()
            score = min(t.get((yr, "Silage"), 0) for yr in a.years)
            grain = min(t.get((yr, "Grain"), 0) for yr in a.years)
            if best is None or (score, grain) > best[:2]:
                best = (score, grain, cx0, cy0)
    score, grain, cx0, cy0 = best
    x0, y0 = cx0 * step, cy0 * step
    log.append(f"square EPSG:32618 x {x0}-{x0 + w}, y {y0}-{y0 + w}: fewest silage in a year {score}, "
               f"fewest grain {grain}")
    d = d[(d.gx >= cx0) & (d.gx < cx0 + k) & (d.gy >= cy0) & (d.gy < cy0 + k)]  # centroid in the square

    rng = np.random.default_rng(0)
    keep = []
    for (yr, lab), g in d.groupby(["year", "label"]):
        idx = g.index.to_numpy()
        if len(idx) > a.per_class:
            idx = np.sort(rng.choice(idx, a.per_class, replace=False))
        keep.append(idx)
        log.append(f"{yr} {lab}: {len(g)} parcels in the square, {len(idx)} kept")
    d = d.loc[np.concatenate(keep)].copy()

    ids = {p: 1_000_000 + i for i, p in enumerate(sorted(d.IDPAR.astype(str).unique()))}
    d["field_id"] = d.IDPAR.astype(str).map(ids)
    d["farm"], d["field"] = "Quebec", d.IDPAR.astype(str)
    for col in SMALL:
        d[col] = np.nan
    d["label_source"] = "FADQ BDPPAD declaration " + d.GROPRO + "/" + d.CODPRO1.astype(str)
    ll = d.geometry.centroid.to_crs(4326)
    d["lon"], d["lat"] = ll.x.round(6), ll.y.round(6)
    d["wkt"] = d.geometry.simplify(2.0).to_wkt(rounding_precision=0)
    d["area_ha"] = d.SUPHEC.round(2)
    d["idanpar"] = d.IDANPAR.astype(str)
    out = pd.DataFrame(d[["field_id", "farm", "field", "year", "label"] + SMALL +
                         ["label_source", "lon", "lat", "wkt", "area_ha", "idanpar"]])
    out = out.sort_values(["year", "field_id"]).reset_index(drop=True)
    if out.duplicated(["field_id", "year"]).any():
        raise SystemExit("a parcel appears twice in one year")
    a.out.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out / "field_years_quebec.csv", index=False)
    multi = out.groupby("field_id").label.nunique().gt(1).sum()
    log.append(f"wrote {len(out)} field-years, {out.field_id.nunique()} parcels, {out.label.value_counts().to_dict()}; "
               f"{multi} parcels change class between years")
    (a.out / "quebec_selection.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))


if __name__ == "__main__":
    main()
