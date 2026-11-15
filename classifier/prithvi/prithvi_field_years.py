"""One table of every labeled BARC field-year with its polygon, for the Prithvi classifier.

  python classifier/prithvi/prithvi_field_years.py

Reads the label lists and hand features from research/grain_vs_silage/results/ (2017-2020 farm records and the
2021-2023 evaluation labels), the 2021-2023 polygons from 04_Evaluation/ and the 2017-2020 polygons from
data/fields_kraatz_2017_2020.gpkg. Writes field_years.csv in this folder: one row per labeled field-year with the
five hand features (for the baseline on the same split), the centroid in lon/lat and the polygon as WKT in
EPSG:32618. The file is small and self-contained so the pod needs no other inputs.

The last two columns, label_basis and outline_fit, come from classifier/labels/results/training_label_checks.csv
(classifier/labels/outline_check.py, which reads the polygons of this table). To refresh them after the polygons
change, run this script, then outline_check.py, then this script again.
"""
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from preprocessing.loader import barc_fields  # noqa: E402

RES = ROOT / "research" / "grain_vs_silage" / "results"
SMALL = ["drop_doy", "ndvi_pre_drop", "drop_size", "ndvi_sep15", "ndti_post_drop"]
CRS = "EPSG:32618"
CHECKS = ROOT / "classifier" / "labels" / "results" / "training_label_checks.csv"


def add_checks(d, checks=CHECKS):
    """label_basis and outline_fit per field-year: 'official' for the evaluation labels, the outline_check.py result
    for the farm records, 'unchecked' for a farm-record field-year that outline_check.py has not seen."""
    c = pd.read_csv(checks, usecols=["field_id", "year", "label_basis", "outline_fit"])
    d = d.merge(c, on=["field_id", "year"], how="left", validate="1:1")
    official = d.label_source.str.startswith("evaluation")
    for col in ("label_basis", "outline_fit"):
        d[col] = d[col].mask(official, "official").fillna("unchecked")
    return d


def main():
    ev = barc_fields(ROOT / "04_Evaluation" / "BARC_Corn.2021-2023.gpkg")[["field_id", "geometry"]].to_crs(CRS)
    kz = gpd.read_file(ROOT / "data" / "fields_kraatz_2017_2020.gpkg", layer="fields")[["field_id", "geometry"]]
    kz = kz.to_crs(CRS)
    cols = ["field_id", "farm", "field", "year", "label"] + SMALL
    out = []
    for csv, polys, source in (("features_2017_2020.csv", kz, "farm records 2017-2020"),
                               ("features_all_field_years.csv", ev, "evaluation labels 2021-2023")):
        d = pd.read_csv(RES / csv, usecols=cols)
        d = d[d.label.notna()].merge(polys, on="field_id", how="left", validate="m:1")
        if d.geometry.isna().any():
            sys.exit(f"{csv}: {int(d.geometry.isna().sum())} labeled field-years have no polygon")
        out.append(d.assign(label_source=source))
    d = gpd.GeoDataFrame(pd.concat(out, ignore_index=True), geometry="geometry", crs=CRS)
    c = d.geometry.centroid.to_crs(4326)
    d["lon"], d["lat"] = c.x.round(6), c.y.round(6)
    # A 2 m simplification is far below the 30 m pixel and keeps the file small enough to carry to the pod.
    d["wkt"] = d.geometry.simplify(2.0).to_wkt(rounding_precision=0)
    d = pd.DataFrame(d.drop(columns="geometry")).sort_values(["year", "field_id"]).reset_index(drop=True)
    d = add_checks(d)
    d.to_csv(HERE / "field_years.csv", index=False)
    print(f"wrote field_years.csv: {len(d)} field-years, {d.field_id.nunique()} fields, "
          f"{d.label.value_counts().to_dict()}, years {sorted(d.year.unique())}, "
          f"outline_fit {d.outline_fit.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
