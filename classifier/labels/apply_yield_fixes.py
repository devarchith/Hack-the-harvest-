"""Label fixes from the BARC combine yield-monitor records, applied to copies of the label tables.

  python classifier/labels/apply_yield_fixes.py

Evidence: the raw combine yield-monitor files of the USDA BARC Gridded Yield Data Archive (Ag Data Commons,
10.15482/USDA.ADC/28950746, yearly zips 2014 and 2016-2021, U.S. Public Domain) and the farm reports in the same
record. A combine file for corn on a field means grain harvest; the archive has no chopper files. Coverage = share of
the outline within a few metres of a combine point that year. Every fix below was found twice, by two separate
checks (raw CSVs on a 10 m grid; raw shapefiles with dates, header width and grain moisture), and agrees with the
farm reports where they have a harvest record.

Reads (never changes): classifier/prithvi/field_years.csv, classifier/labels/results/extra_labels_outside_2021_2023.csv,
04_Evaluation/BARC_Corn.2021-2023.gpkg. Writes to classifier/labels/results/:
  label_fixes.csv                              every fix and flag with its evidence
  field_years_fixed.csv                        field_years.csv without the 2 contradicted rows
  field_years_combine_additions.csv            extra Grain field-years with combine evidence, same columns (optional)
  extra_labels_outside_2021_2023_fixed.csv     the 2014-2024 table with the 4 contradicted silage labels fixed
  7_yield_fixes.txt                            counts
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from preprocessing.loader import barc_fields  # noqa: E402

RES = HERE / "results"
CRS = "EPSG:32618"
SMALL = ["drop_doy", "ndvi_pre_drop", "drop_size", "ndvi_sep15", "ndti_post_drop"]

# (table, field_id, idn, year, label, action, new_label, evidence)
FIXES = [
    ("training", 235, "CentralFarm_1-14C", 2020, "Grain", "drop", "",
     "Soybeans on this outline: the combine soybean file the farm calls 1-14B (harvested 2020-10-09) covers 84-99% of "
     "it; corn files cover 3-8% at the edge. The farm redrew the 1-14 sub-fields in 2020, so its grain '1-14C' "
     "(11.53 ac) is other ground."),
    ("training", 278, "CentralFarm_2-32A", 2017, "Grain", "drop", "",
     "Not corn at harvest: corn planted 2017-05-04, the whole field redrilled to sorghum-sudangrass 2017-06-22 (farm "
     "report); combine sorghum file 2017-10-28 and 11-03 covers 87-100% of the outline."),
    ("extra labels", None, "CentralFarm_4-7", 2014, "Silage", "relabel", "Grain",
     "Silage planting record only. Combined 2014-11-04 over 83-89% of the outline, 16.4% grain moisture, median "
     "87 bu/ac; rye drilled 11-15."),
    ("extra labels", None, "EastFarm_5-12", 2014, "Silage", "relabel", "Grain",
     "Silage planting record only. Combined 2014-10-30/31 over 97-99%, median 164 bu/ac; rye drilled 11-09."),
    ("extra labels", None, "CentralFarm_1-37", 2016, "Silage", "relabel", "Grain",
     "Silage planting record only. Combined 2016-10-19, 11-02 and 11-03 over 100%, median 202 bu/ac."),
    ("extra labels", None, "CentralFarm_2-32B", 2014, "Silage", "drop", "",
     "Silage planting record only. One 4.9 ac block (30-33% of the outline) combined 2014-10-10 at 22.9% moisture; "
     "the rest is unknown, so the field-year is mixed."),
    ("evaluation", 327, "CentralFarm_4-7", 2021, "Silage", "flag", "",
     "Official label kept. Combined 2021-11-04 over 90-93% of the outline with a corn head, 19.6% moisture, median "
     "80 bu/ac; the farm report also has a combine harvest that day with crop code CSIL. The label follows the silage "
     "planting code; as a harvest, and to a satellite, it is grain. Report scores with and without it."),
    ("training", None, "CentralFarm_1-14B/C/D", None, "", "flag", "",
     "Rows for field_id 234, 235, 236 in 2017-2019 are matched by sub-field name. No combine data for them; the gpkg "
     "acreages equal the farm's pre-2020 acreages, so the old letters probably match. Kept."),
]

# Grain field-years with combine corn coverage >= 0.7 and no other crop >= 0.1, not labeled anywhere yet.
ADDITIONS = {
    2020: {236: 0.98, 392: 0.88, 393: 0.93, 394: 0.89, 395: 0.96, 34: 0.80, 230: 0.72, 347: 0.97, 434: 0.99,
           292: 0.96},
    2021: {22: 0.86, 23: 0.83, 25: 0.72, 235: 0.99, 251: 0.81, 416: 0.96, 602: 0.98},
}


def main():
    log = []
    fixes = pd.DataFrame(FIXES, columns=["table", "field_id", "idn", "year", "label", "action", "new_label", "evidence"])
    fixes = fixes.astype({"field_id": "Int64", "year": "Int64"})
    fixes.to_csv(RES / "label_fixes.csv", index=False)

    fy = pd.read_csv(HERE.parent / "prithvi" / "field_years.csv")
    drop = fixes[(fixes.table == "training") & (fixes.action == "drop")]
    gone = fy.set_index(["field_id", "year"]).index.isin(list(zip(drop.field_id.astype(int), drop.year.astype(int))))
    if gone.sum() != len(drop):
        raise SystemExit(f"expected to drop {len(drop)} training rows, found {gone.sum()}")
    fixed = fy[~gone]
    fixed.to_csv(RES / "field_years_fixed.csv", index=False)
    log.append(f"field_years_fixed.csv: {len(fy)} -> {len(fixed)} field-years, {fixed.label.value_counts().to_dict()}")

    ev = barc_fields(ROOT / "04_Evaluation" / "BARC_Corn.2021-2023.gpkg").to_crs(CRS).set_index("field_id")
    have = set(zip(fy.field_id, fy.year))
    rows = []
    for yr, cov in ADDITIONS.items():
        for fid, c in cov.items():
            if (fid, yr) in have:
                continue
            g = ev.loc[fid]
            rows.append({"field_id": fid, "farm": g.FARM.replace(" ", ""), "field": g.FIELD, "year": yr,
                         "label": "Grain", **{k: np.nan for k in SMALL},
                         "label_source": f"combine yield records {yr} (coverage {c:.2f})", "geometry": g.geometry})
    add = pd.DataFrame(rows)
    gs = gpd.GeoSeries(add.geometry, crs=CRS)
    ll = gs.centroid.to_crs(4326)
    add["lon"], add["lat"] = ll.x.round(6).values, ll.y.round(6).values
    add["wkt"] = gs.simplify(2.0).to_wkt(rounding_precision=0).values
    add = add.drop(columns="geometry")[fy.columns]
    add.to_csv(RES / "field_years_combine_additions.csv", index=False)
    log.append(f"field_years_combine_additions.csv: {len(add)} Grain field-years "
               f"({add.year.value_counts().sort_index().to_dict()}); not merged into any training table")

    ex = pd.read_csv(RES / "extra_labels_outside_2021_2023.csv")
    for f in fixes[fixes.table == "extra labels"].itertuples():
        m = (ex.idn == f.idn) & (ex.year == f.year)
        if m.sum() != 1 or ex.loc[m, "label"].item() != f.label:
            raise SystemExit(f"{f.idn} {f.year}: expected one {f.label} row")
        if f.action == "drop":
            ex = ex[~m]
        else:
            ex.loc[m, "label"] = f.new_label
            ex.loc[m, "basis"] = "planting record; combine yield record"
    ex.to_csv(RES / "extra_labels_outside_2021_2023_fixed.csv", index=False)
    log.append(f"extra_labels_outside_2021_2023_fixed.csv: {len(ex)} field-years, {ex.label.value_counts().to_dict()}")
    log.append(f"label_fixes.csv: {len(fixes)} rows, {fixes.action.value_counts().to_dict()}")
    (RES / "7_yield_fixes.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))


if __name__ == "__main__":
    main()
