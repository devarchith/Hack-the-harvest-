"""Extra BARC Grain/Silage field-years (2017-2020) from the Kraatz farm records, with polygons.

Source: Kraatz, S. (2023). Zenodo record 8087301, https://doi.org/10.5281/zenodo.8087301
  farmdb.zip    BARC farm-database exports: planting (pd, 2017-2021) and harvest (hd, 2019-2021) tables
  outlines.zip  BARC field outlines (EPSG:5070)

Label rule per field-year: whole-field rows only; "Corn Field (FLDCRN)" = Grain, "Corn Silage (CSIL)" = Silage.
The harvest-table class wins when present (it records what was harvested); otherwise the planting-table class
(what was planned). Field-years with both Grain and Silage rows are dropped as ambiguous.
Polygon rule: the evaluation GeoPackage polygon when the field name matches (field_id = its OBJECTID, so a
field keeps one id across all years), else the Kraatz outline (field_id = 1000 + row). No polygon: dropped.

Run from the repo root (reproduce.py does this for you):
  python research/grain_vs_silage/kraatz_fields.py --kraatz-dir data/kraatz_zenodo_8087301
Writes data/fields_kraatz_2017_2020.gpkg, data/labels_kraatz_2017_2020.csv (training field-years) and
data/records_kraatz_2021.csv (2021 corn records with harvest dates; only for checking harvest-date detection).
"""
import argparse, re, zipfile
from pathlib import Path
import pandas as pd, geopandas as gpd

CLASSES = {"Corn Field (FLDCRN)": "Grain", "Corn Silage (CSIL)": "Silage"}
# Zenodo file names, and the shorter names used in an earlier local copy
FARMDB = ("farm_database_annual_planting_harvesting_after_some_screening..zip", "farmdb.zip")
OUTLINES = ("barc_field_outlines_for_study.zip", "outlines.zip")


def _find(kdir, names):
    for n in names:
        if (kdir / n).exists():
            return kdir / n
    raise FileNotFoundError(f"none of {names} in {kdir}")


def read_records(farmdb_zip, years):
    rows = []
    with zipfile.ZipFile(farmdb_zip) as z:
        for name in z.namelist():
            m = re.match(r"BARC_fields_(hd|pd)_(\d{4})_filt\.csv$", Path(name).name)
            if not m or int(m.group(2)) not in years:
                continue
            kind, year = m.group(1), int(m.group(2))
            d = pd.read_csv(z.open(name))
            d = d[d["split"].astype(str).str.strip() == "Whole Field"].copy()
            d["cls"] = d["crop"].astype(str).str.split(" /").str[0].str.strip()
            d = d[d["cls"].isin(CLASSES)]
            d["label"] = d["cls"].map(CLASSES)
            d["date"] = pd.to_datetime(d[kind]).dt.date
            for idn, g in d.groupby("idn"):
                rows.append(dict(idn=idn, year=year, kind=kind, label="/".join(sorted(set(g.label))),
                                 date=min(g.date)))
    r = pd.DataFrame(rows)
    wide = r.pivot_table(index=["idn", "year"], columns="kind", values=["label", "date"], aggfunc="first")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    for c in ("label_hd", "label_pd", "date_hd", "date_pd"):
        if c not in wide:
            wide[c] = None
    wide["label"] = wide["label_hd"].fillna(wide["label_pd"])
    wide["label_source"] = wide["label_hd"].notna().map({True: "harvest record", False: "planting record"})
    return wide.rename(columns={"date_hd": "harvest_date", "date_pd": "planting_date",
                                "label_hd": "harvest_label", "label_pd": "planting_label"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kraatz-dir", required=True, help="folder holding farmdb.zip and outlines.zip")
    ap.add_argument("--gpkg", default="04_Evaluation/BARC_Corn.2021-2023.gpkg")
    ap.add_argument("--years", type=int, nargs="+", default=[2017, 2018, 2019, 2020])
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    kdir, out = Path(a.kraatz_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rec21 = read_records(_find(kdir, FARMDB), {2021})
    rec21 = rec21.rename(columns={"label": "record_label"})
    rec21[["idn", "year", "record_label", "harvest_label", "planting_label", "harvest_date", "planting_date"]] \
        .to_csv(out / "records_kraatz_2021.csv", index=False)
    print(f"2021 corn records (harvest-date check only): {len(rec21)}, "
          f"with harvest date: {int(rec21.harvest_date.notna().sum())}")

    lab = read_records(_find(kdir, FARMDB), set(a.years))
    amb = lab.label.str.contains("/")
    disagree = lab.harvest_label.notna() & lab.planting_label.notna() & (lab.harvest_label != lab.planting_label)
    print(f"corn field-years in records: {len(lab)}; ambiguous (both classes) dropped: {int(amb.sum())}; "
          f"planting vs harvest class differ (harvest kept): {int(disagree.sum())}")
    lab = lab[~amb].copy()

    ev = gpd.read_file(a.gpkg).to_crs(4326)
    ev["idn"] = ev["FARM"].str.replace(" ", "", regex=False) + "_" + ev["FIELD"].astype(str)
    ev = ev.rename(columns={"OBJECTID": "field_id"})[["field_id", "idn", "geometry"]].assign(poly_source="eval gpkg")
    kr = gpd.read_file(f"zip://{_find(kdir, OUTLINES).as_posix()}").to_crs(4326)
    kr = kr.rename(columns={"id": "idn"})[["idn", "geometry"]].reset_index(drop=True)
    kr = kr[~kr.idn.isin(ev.idn)].assign(field_id=lambda d: 1000 + d.index, poly_source="Kraatz outline")
    polys = pd.concat([ev, kr[["field_id", "idn", "geometry", "poly_source"]]], ignore_index=True)
    polys = gpd.GeoDataFrame(polys, geometry="geometry", crs=4326)

    lab = lab.merge(polys[["idn", "field_id", "poly_source"]], on="idn", how="left")
    print(f"no polygon, dropped: {int(lab.field_id.isna().sum())} field-years "
          f"({', '.join(sorted(lab[lab.field_id.isna()].idn.unique())[:12])}...)")
    lab = lab[lab.field_id.notna()].astype({"field_id": int})
    used = polys[polys.field_id.isin(lab.field_id)]
    used.to_file(out / "fields_kraatz_2017_2020.gpkg", layer="fields", driver="GPKG")
    cols = ["field_id", "idn", "year", "label", "label_source", "planting_label", "harvest_label",
            "planting_date", "harvest_date", "poly_source"]
    lab[cols].sort_values(["year", "idn"]).to_csv(out / "labels_kraatz_2017_2020.csv", index=False)
    print(pd.crosstab([lab.year, lab.label_source], lab.label, margins=True).to_string())
    print(f"polygons: {len(used)} ({(used.poly_source == 'eval gpkg').sum()} from the eval gpkg, "
          f"{(used.poly_source == 'Kraatz outline').sum()} Kraatz outlines)")


if __name__ == "__main__":
    main()
