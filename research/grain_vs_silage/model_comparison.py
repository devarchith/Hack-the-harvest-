"""Logistic regression vs random forest vs XGBoost, on hand-built features and on AlphaEarth embeddings.

  python research/grain_vs_silage/model_comparison.py

Run after grain_silage.py (results/features_*.csv) and aef_embeddings.py (results/aef_embeddings.csv).
Uses the same splits as grain_silage.py plus one spatial holdout, and checks that the hand-feature logistic regression
and random forest reproduce the numbers in results/report.md before reporting anything new.

XGBoost settings are copied from the teammate's modeling/baseline_xgboost.ipynb (branch add-xgboost-baseline):
fixed before this comparison and not tuned on any split here.

Writes results/model_comparison.csv, results/model_comparison_predictions.csv and results/model_comparison.md.
"""
import sys
from pathlib import Path

import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grain_silage as gs  # noqa: E402

OUT = HERE / "results"
SMALL = ["drop_doy", "ndvi_pre_drop", "drop_size", "ndvi_sep15", "ndti_post_drop"]
META = {"field_id", "farm", "field", "acres", "year", "label", "n_obs", "n_obs_aug_nov", "label_source", "farmkey"}
AEF = [f"A{i:02d}" for i in range(64)]
CHOICE = "Hold out one year, 2017-20 only (used to choose the model)"


def make_xgb(y):
    pos = int(y.sum())
    return XGBClassifier(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=1, reg_lambda=1.0, scale_pos_weight=(len(y) - pos) / pos,
                         eval_metric="logloss", random_state=0, n_jobs=4)


MODELS = {"logistic": lambda y: gs.make_logreg(), "forest": lambda y: gs.make_forest(), "xgboost": make_xgb}


def load():
    t = pd.read_csv(OUT / "features_all_field_years.csv")
    hand_all = [c for c in t.columns if c not in META]  # same list and order as feat_cols in grain_silage.py
    t = t[t.label.notna()].reset_index(drop=True)
    x = pd.read_csv(OUT / "features_2017_2020.csv")
    x = x[x.label.notna()].reset_index(drop=True)
    emb = pd.read_csv(OUT / "aef_embeddings.csv", usecols=["field_id", "year", "aef_n_pix"] + AEF)
    t, x = (d.merge(emb, on=["field_id", "year"], how="left", validate="1:1") for d in (t, x))
    for d in (t, x):
        d["farmkey"] = d.farm.str.replace(" ", "", regex=False)
        d["y"] = (d.label == "Silage").astype(int)
    allv = pd.concat([x, t.assign(label_source="evaluation gpkg")], ignore_index=True)
    return t, x, allv, hand_all


def splits(t, x, allv):
    return {
        "Hold out one year, 2021-23": [(t[t.year != yr], t[t.year == yr]) for yr in (2021, 2022, 2023)],
        "Train 2017-20, test 2021-23": [(x, t)],
        "Hold out one year, 2017-23": [(allv[allv.year != yr], allv[allv.year == yr]) for yr in range(2017, 2024)],
        "Train 2017-20 other farms, test Central Farm 2021-23": [(x[x.farmkey != "CentralFarm"],
                                                                  t[t.farmkey == "CentralFarm"])],
        # AlphaEarth also describes the land itself (soil, shape, surroundings), so it could recognize a field it saw
        # in training rather than its harvest. This split keeps every 2021-23 field out of training.
        "Train 2017-20 unseen fields only, test 2021-23": [(x[~x.field_id.isin(t.field_id)], t)],
        # Model choice uses only this split, which never touches the 2021-23 labels.
        CHOICE: [(x[x.year != yr], x[x.year == yr]) for yr in (2017, 2018, 2019, 2020)],
    }


def md_table(df):
    cells = [[str(c) for c in df.columns]] + [["" if pd.isna(v) else str(v) for v in r] for r in df.to_numpy()]
    return "\n".join(["| " + " | ".join(cells[0]) + " |", "|" + "---|" * len(cells[0])] +
                     ["| " + " | ".join(r) + " |" for r in cells[1:]])


def score(y, p):
    pred = p >= 0.5
    tp, fn = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum())
    fp, tn = int((pred & (y == 0)).sum()), int((~pred & (y == 0)).sum())
    return dict(n=len(y), correct=tp + tn, silage_caught=f"{tp}/{tp + fn}", grain_called_silage=f"{fp}/{fp + tn}",
                silage_precision=round(tp / (tp + fp), 2) if tp + fp else np.nan,
                auc=round(roc_auc_score(y, p), 3) if 0 < y.sum() < len(y) else np.nan)


def main():
    t, x, allv, hand_all = load()
    missing = int(pd.concat([t, x]).A00.isna().sum())
    fsets = {"hand 5": SMALL, "hand all": hand_all, "AlphaEarth 64": AEF, "hand 5 + AlphaEarth": SMALL + AEF}
    rows, preds = [], []
    for split, folds in splits(t, x, allv).items():
        for fname, cols in fsets.items():
            for mname, mk in MODELS.items():
                p, ys, keys = [], [], []
                for tr, te in folds:
                    m = mk(tr.y.values).fit(tr[cols], tr.y.values)
                    p.append(m.predict_proba(te[cols])[:, 1]); ys.append(te.y.values)
                    keys.append(te[["field_id", "farm", "field", "year", "label"]])
                p, ys = np.concatenate(p), np.concatenate(ys)
                rows.append(dict(split=split, features=fname, model=mname, **score(ys, p)))
                preds.append(pd.concat(keys, ignore_index=True).assign(split=split, features=fname, model=mname,
                                                                       p_silage=p.round(3)))
    res = pd.DataFrame(rows)
    pr = pd.concat(preds, ignore_index=True)

    # The hand-feature baselines must match results/report.md before any new number is trusted.
    want = {("Hold out one year, 2021-23", "hand 5", "logistic"): 42,
            ("Hold out one year, 2021-23", "hand all", "forest"): 43,
            ("Train 2017-20, test 2021-23", "hand 5", "logistic"): 39,
            ("Train 2017-20, test 2021-23", "hand all", "forest"): 41,
            ("Hold out one year, 2017-23", "hand 5", "logistic"): 113,
            ("Train 2017-20 unseen fields only, test 2021-23", "hand 5", "logistic"): 40}
    got = res.set_index(["split", "features", "model"]).correct
    bad = {k: (v, int(got[k])) for k, v in want.items() if int(got[k]) != v}
    if bad:
        sys.exit(f"baseline mismatch (expected, got): {bad}")

    res.to_csv(OUT / "model_comparison.csv", index=False)
    pr.to_csv(OUT / "model_comparison_predictions.csv", index=False)

    lines = ["# Model comparison: logistic regression, random forest, XGBoost",
             "",
             "Generated by `model_comparison.py`. Every cell is correct / total at a 0.5 cut on the held-out field-years. "
             "The five hand-feature baselines were checked against `report.md` first and match.",
             "",
             f"AlphaEarth embeddings: annual 64-number summaries per 10 m pixel (Google / Google DeepMind, CC-BY 4.0, "
             f"via source.coop/tge-labs/aef), averaged over each field. Field-years missing an embedding: {missing}.",
             "XGBoost settings come from the teammate's `baseline_xgboost.ipynb` and were not tuned here.",
             ""]
    for split in res.split.unique():
        s = res[res.split == split]
        n = int(s.n.iloc[0]); ns = int(s.silage_caught.iloc[0].split("/")[1])
        lines += [f"## {split} ({n} field-years, {ns} Silage)", "",
                  md_table(s.drop(columns=["split", "n"])), ""]
    wide = res.pivot_table(index=["features", "model"], columns="split", values="correct", aggfunc="first")
    wide = wide[list(res.split.unique())]
    wide.columns = [f"{c} (/{int(res[res.split == c].n.iloc[0])})" for c in wide.columns]
    lines += ["## Summary: correct field-years", "", md_table(wide.reset_index()), ""]

    # Choose on 2017-20 only (most correct, then highest AUC), then read off its 2021-23 scores.
    ch = res[res.split == CHOICE].sort_values(["correct", "auc"], ascending=False)
    best = ch.iloc[0]
    lines += ["## Model choice without the test labels", "",
              f"Ranked on '{CHOICE}' ({int(best.n)} field-years). Top 5:", "",
              md_table(ch.head(5).drop(columns=["split", "n"])), "",
              f"Chosen: **{best.features} / {best.model}**. Its scores on the 2021-23 tests, against the original "
              "hand 5 / logistic baseline (exact McNemar test on the field-years where the two disagree):", ""]
    rows = []
    for split in [s for s in res.split.unique() if s != CHOICE]:
        q = pr[pr.split == split]
        a = q[(q.features == "hand 5") & (q.model == "logistic")].reset_index(drop=True)
        b = q[(q.features == best.features) & (q.model == best.model)].reset_index(drop=True)
        ya = (a.label == "Silage").values
        ca, cb = (a.p_silage >= 0.5).values == ya, (b.p_silage >= 0.5).values == ya
        only_b, only_a = int((cb & ~ca).sum()), int((ca & ~cb).sum())
        p = binomtest(only_b, only_a + only_b).pvalue if only_a + only_b else 1.0
        rows.append({"test": split, "baseline correct": f"{int(ca.sum())}/{len(ca)}",
                     "chosen correct": f"{int(cb.sum())}/{len(cb)}", "only chosen right": only_b,
                     "only baseline right": only_a, "McNemar p": round(p, 3)})
    lines += [md_table(pd.DataFrame(rows))]
    (OUT / "model_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
