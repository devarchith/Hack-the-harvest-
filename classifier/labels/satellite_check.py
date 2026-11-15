"""Satellite check of the Zenodo 2017-2020 training labels.

Fits the 5-feature logistic regression of research/grain_vs_silage (same features, same settings) on the 44 official
2021-2023 labels only, then classifies the Zenodo field-years used for training. A label the model contradicts is not
proven wrong, but a pattern by label basis (harvest record vs planting record only) points at the labels. For scale,
each official year is also predicted by a model fitted on the other two official years.

  python classifier/labels/satellite_check.py

Writes results/satellite_check.csv (one row per Zenodo training field-year, with the model's silage probability).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "research" / "grain_vs_silage"))
import grain_silage as gs  # noqa: E402

SMALL = ["drop_doy", "ndvi_pre_drop", "drop_size", "ndvi_sep15", "ndti_post_drop"]  # as in model_comparison.py


def fit_pred(tr, te):
    return gs.make_logreg().fit(tr[SMALL], tr.y).predict_proba(te[SMALL])[:, 1]


def show(name, d, p):
    ok = (p >= 0.5) == (d.y == 1)
    s, g = d.y == 1, d.y == 0
    print(f"  {name}: agree {int(ok.sum())}/{len(d)} ({ok.mean():.0%}) | silage {int((ok & s).sum())}/{int(s.sum())}"
          f" | grain {int((ok & g).sum())}/{int(g.sum())}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default=str(ROOT / "research" / "grain_vs_silage" / "results"))
    ap.add_argument("--zenodo-labels", default=str(HERE / "results" / "zenodo_labels.csv"))
    ap.add_argument("--out", default=str(HERE / "results" / "satellite_check.csv"))
    a = ap.parse_args()
    t = pd.read_csv(Path(a.features) / "features_all_field_years.csv")
    t = t[t.label.notna()].reset_index(drop=True)
    x = pd.read_csv(Path(a.features) / "features_2017_2020.csv")
    x = x[x.label.notna()].reset_index(drop=True)
    for d in (t, x):
        d["y"] = (d.label == "Silage").astype(int)
        d["idn"] = d.farm.str.replace(" ", "", regex=False) + "_" + d.field.astype(str)
    zen = pd.read_csv(a.zenodo_labels)
    x = x.merge(zen[["idn", "year", "hd"]], on=["idn", "year"], how="left", validate="1:1")
    x["basis"] = np.where(x.hd.notna(), "harvest record", "planting record only")

    print(f"Official labels ({len(t)}), each year predicted by a model fitted on the other official years:")
    p44 = np.empty(len(t))
    for yr in sorted(t.year.unique()):
        p44[t.year == yr] = fit_pred(t[t.year != yr], t[t.year == yr])
    show("official", t, p44)
    print(f"Zenodo training labels ({len(x)}), predicted by a model fitted on all {len(t)} official labels:")
    px = fit_pred(t, x)
    ok = show("all", x, px)
    for k, d in x.groupby("basis"):
        show(k, d, px[d.index])
    for yr, d in x.groupby("year"):
        show(str(yr), d, px[d.index])
    x["p_silage"], x["model_agrees"] = px.round(3), ok
    x[["field_id", "idn", "year", "label", "basis", "p_silage", "model_agrees"]].to_csv(a.out, index=False)
    print("Disagreements:")
    print(x[~ok][["idn", "year", "label", "basis", "p_silage"]].to_string(index=False))
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
