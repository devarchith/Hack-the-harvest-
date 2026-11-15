"""Combine the Zenodo and farm-report labels outside the evaluation years, and check how far they can be trusted.

  python classifier/labels/combine_check.py --zenodo-dir /home/keeboh/work/labels/zenodo/farmdb

Reads results/zenodo_labels.csv (kraatz_vs_eval.py), results/farm_report_field_years.csv and
results/farm_report_events.csv (mgmt_vs_eval.py, parse_farmmgmt.py). Writes results/extra_labels_outside_2021_2023.csv
and prints:
  A. field-years per source and label conflicts between the two sources
  B. whether each label rests on a harvest record or only on a planting record
  C. how often the class planned at planting changed by harvest
  D. recorded field acres vs the acres of the 2021-2023 evaluation outline used for that field
"""
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EVAL_YEARS = (2021, 2022, 2023)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zenodo-dir", required=True, help="folder with the Zenodo BARC_fields_*_filt.csv files")
    ap.add_argument("--zenodo-labels", default=str(HERE / "results" / "zenodo_labels.csv"))
    ap.add_argument("--report-labels", default=str(HERE / "results" / "farm_report_field_years.csv"))
    ap.add_argument("--events", default=str(HERE / "results" / "farm_report_events.csv"))
    ap.add_argument("--eval-gpkg", default=str(ROOT / "04_Evaluation" / "BARC_Corn.2021-2023.gpkg"))
    ap.add_argument("--out", default=str(HERE / "results" / "extra_labels_outside_2021_2023.csv"))
    a = ap.parse_args()
    zen, rep, ev = pd.read_csv(a.zenodo_labels), pd.read_csv(a.report_labels), pd.read_csv(a.events)
    b = gpd.read_file(a.eval_gpkg)
    b["idn"] = b["FARM"].str.replace(" ", "") + "_" + b["FIELD"].astype(str)

    # A. one table, harvest record first, then planting record
    z = zen[zen.year <= 2020].assign(label=lambda d: d.hd.fillna(d.pd),
                                     z_basis=lambda d: np.where(d.hd.notna(), "harvest", "planting"))
    r = rep[~rep.year.isin(EVAL_YEARS)].assign(r_basis=lambda d: np.where(d.harvest.notna(), "harvest", "planting"))
    m = z[["idn", "year", "label", "z_basis", "harvest_date"]].merge(
        r[["idn", "year", "label", "r_basis", "harvest_date", "harvest_machine"]], on=["idn", "year"], how="outer",
        suffixes=("_z", "_r"))
    both = m.label_z.notna() & m.label_r.notna()
    print(f"A. {len(m)} field-years outside {EVAL_YEARS[0]}-{EVAL_YEARS[-1]}: Zenodo {int(m.label_z.notna().sum())}, "
          f"farm reports {int(m.label_r.notna().sum())}, in both {int(both.sum())}; "
          f"label conflicts {int((both & (m.label_z != m.label_r)).sum())}")
    m["label"] = m.label_r.fillna(m.label_z)
    m["sources"] = np.select([both, m.label_r.notna()], ["reports+zenodo", "reports"], "zenodo")
    m["basis"] = np.where((m.z_basis == "harvest") | (m.r_basis == "harvest"), "harvest record", "planting record only")
    m["harvest_date"] = m.harvest_date_r.fillna(m.harvest_date_z)
    m["outline_in_eval_gpkg"] = m.idn.isin(set(b.idn))
    cols = ["idn", "year", "label", "sources", "basis", "harvest_date", "harvest_machine", "outline_in_eval_gpkg"]
    m[cols].sort_values(["idn", "year"]).to_csv(a.out, index=False)
    print(pd.crosstab(m.year, [m.outline_in_eval_gpkg, m.label], margins=True).to_string())

    # B. what each label rests on (field-years that have an outline, so can be used)
    o = m[m.outline_in_eval_gpkg]
    print("\nB. label basis, field-years with an outline in the evaluation file:")
    print(pd.crosstab(o.year, [o.basis, o.label], margins=True).to_string())

    # C. planned class vs harvested class, where a field-year has both whole-field records
    zb = zen[zen.pd.notna() & zen.hd.notna()]
    rb = rep[rep.plant.notna() & rep.harvest.notna()]
    print(f"\nC. planting class differs from harvest class: Zenodo {int((zb.pd != zb.hd).sum())} of {len(zb)}, "
          f"farm reports {int((rb.plant != rb.harvest).sum())} of {len(rb)}")
    print("Zenodo (rows planted, columns harvested):"); print(pd.crosstab(zb.pd, zb.hd).to_string())
    print("Farm reports:"); print(pd.crosstab(rb.plant, rb.harvest).to_string())

    # D. does the 2021-2023 outline fit the field in that year?
    w = ev[ev.split == "Whole Field"].copy()
    w["acres"] = pd.to_numeric(w.body.str.split(" \\| ").str[2], errors="coerce")
    rep_ac = w.groupby(["idn", "year"]).acres.max().rename("rep_acres")
    zz = []
    for f in sorted(Path(a.zenodo_dir).glob("BARC_fields_*_filt.csv")):
        d = pd.read_csv(f)
        d = d[d.split.str.strip() == "Whole Field"]
        zz.append(d.assign(year=int(f.name.split("_")[3]))[["idn", "year", "acres"]])
    zen_ac = pd.concat(zz).groupby(["idn", "year"]).acres.max().rename("zen_acres")
    o = o.join(rep_ac, on=["idn", "year"]).join(zen_ac, on=["idn", "year"])
    o["record_acres"] = o.rep_acres.fillna(o.zen_acres)
    o["outline_acres"] = o.idn.map(b.drop_duplicates("idn").set_index("idn")["ACRES"])
    o["ratio"] = o.record_acres / o.outline_acres
    off = o[(o.ratio < 0.8) | (o.ratio > 1.25)]
    print(f"\nD. record acres vs outline acres: {int(o.record_acres.notna().sum())} of {len(o)} field-years have recorded "
          f"acres; {len(off)} differ by more than 20-25%:")
    print(off.sort_values(["idn", "year"])[["idn", "year", "label", "record_acres", "outline_acres", "ratio"]]
          .round(2).to_string(index=False))
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
