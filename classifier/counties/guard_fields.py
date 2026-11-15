"""BARC 2017-2020 held-out-year predictions from the BARC guard of a presto_national.py run (--parallel > 1 keeps
each fit in <results>/jobs/). Each training year is predicted by the model fit on the other three years, for every w.

  python guard_fields.py --results <results folder> --field-years classifier/labels/results/field_years_fixed.csv

Writes <results>/guard_field_predictions.csv (field_id, year, label, p_w<w> per w; BARC latitude/longitude, as in the
guard) and prints silage caught / grain called silage per year for w 0 and the chosen w.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "presto")]
from presto_standard_test import CHOICE_YEARS  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--results", required=True)
ap.add_argument("--field-years", required=True)
a = ap.parse_args()
res = Path(a.results)
run = json.loads((res / "run.json").read_text())
fy = pd.read_csv(a.field_years)
tr = np.flatnonzero(fy.year.isin(CHOICE_YEARS).to_numpy())  # BARC rows come first in the run, in this order
out = fy.loc[tr, ["field_id", "year", "label"]].reset_index(drop=True)
n_seeds = len(run["setting"]["choice_seeds"])
for f in sorted((res / "jobs").glob("fit_*.json")):
    spec = json.loads(f.read_text())
    if "drop_year" not in spec:
        continue
    p = np.load(f.with_suffix(".npz"))["fix"][-len(tr):]  # pred_rows = (other rows, tr)
    col = f"p_w{float(spec['w']):g}"
    if col not in out:
        out[col] = 0.0
    held = (out.year == spec["drop_year"]).to_numpy()
    out.loc[held, col] += p[held] / n_seeds
out.round(4).to_csv(res / "guard_field_predictions.csv", index=False)
w = float(run["setting"]["w_chosen"])
y = out.label == "Silage"
for c in dict.fromkeys(["p_w0", f"p_w{w:g}"]):
    s = out[c] >= 0.5
    per = [f"{yr}: silage {int((s & y)[out.year == yr].sum())}/{int(y[out.year == yr].sum())}, grain->silage "
           f"{int((s & ~y)[out.year == yr].sum())}/{int((~y)[out.year == yr].sum())}" for yr in CHOICE_YEARS]
    print(c, f"right {int((s == y).sum())}/{len(out)} |", " | ".join(per))
