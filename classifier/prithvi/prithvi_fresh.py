"""One test of the Prithvi classifier on field-years no model or model choice has seen: 2014, 2015, 2016 and 2024.

  bash classifier/prithvi/prithvi_fresh.sh

Fixed before any input for these years was built:
- Test set: every field-year in classifier/labels/results/extra_labels_outside_2021_2023.csv whose field has an
  outline in the 2021-2023 evaluation file and that is not one of the 127 labeled field-years in field_years.csv. That
  leaves 45 field-years from the farm reports (35 Grain, 10 Silage), outlined with the 2021-2023 polygons.
- Models, each fitted once on all 127 labeled field-years 2017-2023, with the head evaluate.py chose for that table
  on the 2017-2020 choice split (top row of its choice_grid.csv):
    prithvi_300M_s1_f8               the model under test: 8 two-week frames, 30 m chip
    prithvi_300M_s1_f4               the table evaluate.py chose: 4 frames, 30 m chip
    hand 5 / logistic (baseline)
- Inputs are built by the same code as for the training years: the team loader (preprocessing/loader.py) and
  grain_silage.py's feature functions for the 5 hand features; prithvi_chips.py and prithvi_embed.py for Prithvi.
  One change is forced: Landsat is put on the Sentinel-2 scale with a per-year offset from same-day pairs, and 2014
  has no Sentinel-2. A year without pairs gets the median of the other fetched years' offsets. 2021 is fetched only
  to check that the hand features rebuild the team's 2021 values.
- Primary result: correct field-years out of 45 at p >= 0.5, Prithvi 8-frame against the baseline, with an exact
  McNemar test. Also reported: silage caught, grain called silage, AUC, and the same per year and per label basis.
- Run once; the result is reported whatever it is.

Known before the run: every silage label here rests on a planting record only (2014, 2016 and 2024 have no harvest
records), and in the farm reports planned silage was harvested wholly or partly as grain 3 of 25 times. 2014 has
Landsat 8 only and 2015-2016 one Sentinel-2 satellite, so these years have fewer clear readings than 2017-2023.

  prepare  writes <fresh>/fresh_field_years.csv (the test set with polygons)
  score    builds the hand features, fits the three models, writes <out>/summary.md, scores.csv, predictions.csv,
           hand_features.csv, sensor_offsets.csv, check_2021.csv and run.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LABELS = ROOT / "classifier" / "labels" / "results" / "extra_labels_outside_2021_2023.csv"
CRS = "EPSG:32618"
TABLES = ["prithvi_300M_s1_f8", "prithvi_300M_s1_f4"]
BASE = "hand 5 / logistic (baseline)"


def idn(farm, field):
    return farm.str.replace(" ", "", regex=False) + "_" + field.astype(str)


def prepare(a):
    sys.path.insert(0, str(a.tree))
    from preprocessing.loader import barc_fields
    ex = pd.read_csv(LABELS)
    fy = pd.read_csv(HERE / "field_years.csv")
    used = set(zip(idn(fy.farm, fy.field), fy.year))
    ex = ex[ex.outline_in_eval_gpkg & ~pd.Series([k in used for k in zip(ex.idn, ex.year)], index=ex.index)]
    ev = barc_fields(a.tree / "04_Evaluation" / "BARC_Corn.2021-2023.gpkg").to_crs(CRS)
    ev["idn"] = idn(ev.FARM, ev.FIELD)
    d = ex.merge(ev[["idn", "field_id", "FARM", "FIELD", "geometry"]], on="idn", how="left", validate="m:1")
    if d.field_id.isna().any():
        sys.exit(f"{int(d.field_id.isna().sum())} test field-years have no outline")
    c = d.geometry.centroid.to_crs(4326)
    d["lon"], d["lat"] = c.x.round(6), c.y.round(6)
    d["wkt"] = d.geometry.simplify(2.0).to_wkt(rounding_precision=0)  # as in prithvi_field_years.py
    d = d.rename(columns={"FARM": "farm", "FIELD": "field"})
    d = d[["field_id", "farm", "field", "year", "label", "basis", "lon", "lat", "wkt"]]
    d = d.sort_values(["year", "field_id"]).reset_index(drop=True)
    a.fresh.mkdir(parents=True, exist_ok=True)
    d.to_csv(a.fresh / "fresh_field_years.csv", index=False)
    print(f"wrote fresh_field_years.csv: {len(d)} field-years, {d.field_id.nunique()} fields,",
          d.groupby(["year", "label"]).size().unstack(fill_value=0).to_dict("index"))


def hand_features(gs, series):
    """grain_silage.py's steps (sensor offsets, merged series, features_one) on the loader output."""
    s2, ls = gs.load_series(series, series)
    off = gs.sensor_offsets(s2, ls)
    measured = off.copy()
    for yr in sorted(set(ls.year) - set(off.index)):  # 2014: no Sentinel-2, so no same-day pairs
        off.loc[yr] = measured.median()
        off.loc[yr, "pairs"] = 0
    series = gs.merged_series(s2, ls, off)
    no_weather = pd.DataFrame({"doy": [], "tmax_c": [], "tmin_c": []})  # only the unused GDD features need it
    rows = [{"field_id": fid, "year": yr, **gs.features_one(g, no_weather)}
            for (fid, yr), g in series.groupby(["field_id", "year"])]
    return pd.DataFrame(rows), off.sort_index()


def score(a):
    t0 = time.time()
    sys.path.insert(0, str(a.tree / "research" / "pretrained_classifier"))
    import evaluate as ev  # also puts research/grain_vs_silage on the path
    gs, mc = ev.gs, ev.mc
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    runs = a.tree / "research" / "pretrained_classifier" / "results"

    hf, off = hand_features(gs, a.fresh / "series_fresh.parquet")
    off.round(4).to_csv(out / "sensor_offsets.csv")
    team = pd.read_csv(mc.OUT / "features_all_field_years.csv")
    chk = team[team.year == 2021][["field_id", "year"] + mc.SMALL].merge(
        hf[["field_id", "year"] + mc.SMALL], on=["field_id", "year"], suffixes=("_team", "_rebuilt"))
    diff = pd.DataFrame({c: (chk[c + "_team"] - chk[c + "_rebuilt"]).abs() for c in mc.SMALL})
    chk.to_csv(out / "check_2021.csv", index=False)
    same = int((diff.fillna(0) < 1e-6).all(1).sum())

    _, _, allv, _ = mc.load()
    te = pd.read_csv(a.fresh / "fresh_field_years.csv").drop(columns="wkt")
    te = te.merge(hf[["field_id", "year"] + mc.SMALL], on=["field_id", "year"], how="left", validate="1:1")
    te["y"] = (te.label == "Silage").astype(int)
    hf.merge(te[["field_id", "year", "label", "basis"]], on=["field_id", "year"]).to_csv(out / "hand_features.csv",
                                                                                       index=False)
    P, heads, inputs = {}, {}, [LABELS, a.fresh / "fresh_field_years.csv", a.fresh / "series_fresh.parquet"]
    P[BASE] = gs.make_logreg().fit(allv[mc.SMALL], allv.y).predict_proba(te[mc.SMALL])[:, 1]
    for name in TABLES:
        r = pd.read_csv(runs / name / "choice_grid.csv").iloc[0]
        hand = [] if r.family == "emb" else mc.SMALL
        k = int(r.k) if r.pca else None
        tr_e, te_e = pd.read_csv(runs / "prithvi_emb" / f"{name}.csv"), pd.read_csv(a.fresh / "emb" / f"{name}.csv")
        emb = [c for c in tr_e.columns if c.startswith("E")]
        tr = allv.merge(tr_e, on=["field_id", "year"], how="left", validate="1:1")
        tt = te.merge(te_e, on=["field_id", "year"], how="left", validate="1:1")
        if tr[emb[0]].isna().any() or tt[emb[0]].isna().any():
            sys.exit(f"{name}: a field-year has no embedding")
        m = ev.head(emb, hand, k, float(r.C)).fit(tr[emb + hand], tr.y.values)
        tag = f"{name} {r.family} k={r.k} C={r.C}"
        P[tag], heads[name] = m.predict_proba(tt[emb + hand])[:, 1], tag
        inputs += [runs / name / "choice_grid.csv", runs / "prithvi_emb" / f"{name}.csv", a.fresh / "emb" / f"{name}.csv"]

    y = te.y.values
    subsets = {"all": np.ones(len(te), bool)}
    subsets.update({str(yr): te.year.values == yr for yr in sorted(te.year.unique())})
    subsets.update({b: te.basis.values == b for b in sorted(te.basis.unique())})
    rows = [dict(subset=s, model=k, **mc.score(y[m], p[m])) for s, m in subsets.items() for k, p in P.items()]
    sc = pd.DataFrame(rows)
    sc.to_csv(out / "scores.csv", index=False)
    pr = te[["field_id", "farm", "field", "year", "label", "basis"]].copy()
    for k, p in P.items():
        pr[k] = p.round(4)
    pr.to_csv(out / "predictions.csv", index=False)

    test = heads[TABLES[0]]
    lines = ["# Prithvi-EO-2.0 on field-years no model has seen (2014, 2015, 2016, 2024)", "",
             f"{len(te)} field-years from the BARC farm reports ({int((y == 0).sum())} Grain, {int(y.sum())} Silage), "
             "none used for training or for any choice. Each model was fitted once on all 127 labeled field-years "
             "2017-2023 with the head `evaluate.py` had already chosen for it; the test and the models were fixed in "
             "`prithvi_fresh.py` before these years' inputs were built, and this is the only run.", "",
             "## All field-years", "",
             mc.md_table(sc[sc.subset == "all"].drop(columns="subset")), ""]
    for k in P:
        if k == BASE:
            continue
        b_only, k_only, pv = ev.mcnemar(y, P[BASE], P[k])
        lines.append(f"- {k}: right where the baseline is wrong {k_only}, the reverse {b_only}, "
                     f"exact McNemar p = {pv:.3f}")
    lines += ["", "## By year and by label basis", "",
              mc.md_table(sc[sc.subset != "all"][["subset", "model", "n", "correct", "silage_caught",
                                                   "grain_called_silage", "auc"]]), "",
              "## Checks", "",
              f"- The rebuilt hand features match the team's 2021 values for {same} of {len(chk)} field-years "
              f"(largest difference per feature: {', '.join(f'{c} {diff[c].max():.2g}' for c in mc.SMALL)}).",
              "- Landsat-to-Sentinel-2 offsets per year (`sensor_offsets.csv`; `pairs` 0 means the median of the "
              "measured years was used):", "", mc.md_table(off.round(4).reset_index()), "",
              "## Caveats", "",
              "- Every silage label here rests on a planting record only; planned silage was sometimes harvested "
              "as grain (3 of 25 times in the farm reports). Grain called silage may be a label error.",
              "- The outlines are the 2021-2023 polygons; a field's shape may have differed in 2014-2016.",
              "- 2014 has Landsat 8 only and 2015-2016 one Sentinel-2 satellite, so fewer clear readings than the "
              "training years.",
              f"- {len(te)} field-years with {int(y.sum())} silage: a difference of one or two fields is noise.", ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    info = ev.run_info([], t0)
    info.update(script_sha256=ev.sha256(__file__), heads=heads, test_model=test,
                inputs_sha256={str(p): ev.sha256(p) for p in inputs})
    (out / "run.json").write_text(json.dumps(info, indent=1))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prepare", "score"])
    ap.add_argument("--tree", type=Path, required=True, help="copy of the team repository's main branch")
    ap.add_argument("--fresh", type=Path, required=True, help="working folder for this test's inputs")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "fresh")
    a = ap.parse_args()
    prepare(a) if a.step == "prepare" else score(a)


if __name__ == "__main__":
    main()
