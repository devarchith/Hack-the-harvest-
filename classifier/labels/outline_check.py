"""Check each 2017-2020 training field-year: what its label rests on, and whether its outline fits the field that year.

  python classifier/labels/outline_check.py --zenodo-dir /home/keeboh/work/labels/zenodo/farmdb \
      --zenodo-outlines /home/keeboh/work/labels/zenodo/outlines/<name>.shp

Reads classifier/prithvi/field_years.csv (the outline each field-year is trained with), results/zenodo_labels.csv
(kraatz_vs_eval.py), the Zenodo whole-field acres, the 2021-2023 evaluation outlines and the Zenodo study outlines.
Writes results/training_label_checks.csv, one row per farm-record field-year:
  label_basis   harvest record | planting record only
  outline_fit   ok | misfit | unchecked
An outline fits when the recorded whole-field acres are within 0.8-1.25 of its acres. Where no acres are recorded, it
fits when it overlaps the other source's outline of the same field name (intersection / union) by 0.9 or more.
prithvi_field_years.py copies both columns into field_years.csv; prithvi_train.py --exclude uses them.
"""
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import make_valid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CRS = "EPSG:32618"
ACRES_OK, IOU_OK = (0.8, 1.25), 0.9


def iou(a, b):
    return a.intersection(b).area / a.union(b).area


def outlines(path, name):
    """One geometry per farm_field name, in CRS."""
    g = gpd.read_file(path).to_crs(CRS)
    g["idn"] = name(g)
    g["geometry"] = g.geometry.apply(make_valid)
    return g.dissolve("idn").geometry


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zenodo-dir", required=True, help="folder with the Zenodo BARC_fields_*_filt.csv files")
    ap.add_argument("--zenodo-outlines", required=True, help="the Zenodo study outlines (.shp)")
    ap.add_argument("--zenodo-labels", default=str(HERE / "results" / "zenodo_labels.csv"))
    ap.add_argument("--field-years", default=str(ROOT / "classifier" / "prithvi" / "field_years.csv"))
    ap.add_argument("--eval-gpkg", default=str(ROOT / "04_Evaluation" / "BARC_Corn.2021-2023.gpkg"))
    ap.add_argument("--out", default=str(HERE / "results" / "training_label_checks.csv"))
    a = ap.parse_args()

    t = pd.read_csv(a.field_years)
    t = t[t.label_source.str.startswith("farm records")].reset_index(drop=True)
    t["idn"] = t.farm.str.replace(" ", "", regex=False) + "_" + t.field.astype(str)
    geom = gpd.GeoSeries.from_wkt(t.wkt, crs=CRS).apply(make_valid)

    # what the label rests on
    zen = pd.read_csv(a.zenodo_labels)
    t = t.merge(zen[["idn", "year", "hd"]], on=["idn", "year"], how="left", validate="1:1")
    t["label_basis"] = np.where(t.hd.notna(), "harvest record", "planting record only")

    # which outline the field-year is trained with, and its overlap with the other source's outline
    ev = outlines(a.eval_gpkg, lambda g: g["FARM"].str.replace(" ", "", regex=False) + "_" + g["FIELD"].astype(str))
    zo = outlines(a.zenodo_outlines, lambda g: g["id"].str.strip())
    t["outline_used"] = np.where(t.idn.isin(ev.index), "evaluation", "zenodo")
    other = [zo if u == "evaluation" else ev for u in t.outline_used]
    t["iou_other_outline"] = [iou(g, o[i]) if i in o.index else np.nan for g, i, o in zip(geom, t.idn, other)]

    # recorded whole-field acres against the acres of the outline
    rec = []
    for f in sorted(Path(a.zenodo_dir).glob("BARC_fields_*_filt.csv")):
        d = pd.read_csv(f)
        rec.append(d[d.split.str.strip() == "Whole Field"].assign(year=int(f.name.split("_")[3]))[["idn", "year", "acres"]])
    t = t.join(pd.concat(rec).groupby(["idn", "year"]).acres.max().rename("record_acres"), on=["idn", "year"])
    t["outline_acres"] = geom.area.to_numpy() / 4046.86
    t["acres_ratio"] = t.record_acres / t.outline_acres

    by_acres = t.acres_ratio.between(*ACRES_OK)
    by_overlap = t.iou_other_outline >= IOU_OK
    t["outline_fit"] = np.select([t.acres_ratio.notna() & by_acres, t.acres_ratio.notna(), by_overlap,
                                  t.iou_other_outline.notna()], ["ok", "misfit", "ok", "misfit"], "unchecked")
    t["outline_checked_by"] = np.select([t.acres_ratio.notna() & t.iou_other_outline.notna(), t.acres_ratio.notna(),
                                         t.iou_other_outline.notna()], ["acres and overlap", "acres", "overlap"], "nothing")

    cols = ["field_id", "idn", "year", "label", "label_basis", "outline_used", "record_acres", "outline_acres",
            "acres_ratio", "iou_other_outline", "outline_checked_by", "outline_fit"]
    t[cols].sort_values(["year", "field_id"]).round({"outline_acres": 2, "acres_ratio": 2, "iou_other_outline": 2}) \
        .to_csv(a.out, index=False, lineterminator="\n")

    print(f"{len(t)} farm-record field-years in the training table; outline used:",
          t.outline_used.value_counts().to_dict())
    print("\nLabel basis:")
    print(pd.crosstab(t.label_basis, t.label, margins=True).to_string())
    print("\nOutline fit, by what it was checked against:")
    print(pd.crosstab([t.outline_fit, t.outline_checked_by], t.label, margins=True).to_string())
    low = t[(t.outline_fit == "ok") & (t.iou_other_outline < IOU_OK)]
    print(f"\nAcres match but overlap with the other outline is below {IOU_OK} ({len(low)}):")
    print(low[["idn", "year", "label", "record_acres", "outline_acres", "iou_other_outline"]].round(2).to_string(index=False))
    print("\nOutline does not fit:")
    print(t[t.outline_fit == "misfit"].sort_values(["idn", "year"])
          [["idn", "year", "label", "label_basis", "record_acres", "outline_acres", "acres_ratio", "iou_other_outline"]]
          .round(2).to_string(index=False))
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
