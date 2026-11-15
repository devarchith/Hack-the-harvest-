"""Metrics report for saved test predictions: every metric with a confidence interval, a paired comparison with a
baseline, and a per-seed view when several splits are pooled.

  python classifier/metrics/report.py --predictions classifier/presto/results/presto/test_predictions.csv \
      --models p_presto=Presto p_baseline="5-feature baseline" --baseline p_baseline \
      --out classifier/presto/results/presto/metrics.md

The CSV needs a label column (Silage or Grain), field_id (intervals resample whole fields) and one probability column
per model. With a seed column, a per-seed table is added. --join adds a column from another run's predictions on the
same test rows (e.g. a run with extra training data), so the two can be compared pair by pair. Writes <out> (Markdown)
and <out> with .csv (one row per model: every metric, its interval and, against the baseline, the paired difference).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import RATES, THRESHOLD, cluster_bootstrap, confusion, mcnemar, point_metrics  # noqa: E402

KEYS = ("seed", "field_id", "year")


def join(pred, spec):
    """pred plus a probability column of another predictions file with the same test rows. spec: CSV:COLUMN=NEW."""
    path, rest = spec.rsplit(":", 1)
    col, new = rest.split("=", 1)
    other = pd.read_csv(path)
    keys = [k for k in KEYS if k in pred and k in other]
    j = pred.merge(other[keys + ["label", col]].rename(columns={col: new, "label": "_label"}), on=keys, how="left",
                   validate="1:1")
    if len(other) != len(pred) or j[new].isna().any() or (j["_label"] != j["label"]).any():
        raise ValueError(f"{path} does not hold the same test rows and labels as the predictions")
    return j.drop(columns="_label")


def fmt(v, ci=None, d=2):
    if v is None or not np.isfinite(v):
        return "n/a"
    s = f"{v:.{d}f}"
    return s if ci is None or not np.isfinite(ci).all() else f"{s} ({ci[0]:.{d}f} to {ci[1]:.{d}f})"


def md_table(rows):
    cols = list(rows[0])
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows])


def build(pred, models, baseline=None, n_boot=2000, seed=0):
    """Point metrics, intervals and paired comparisons for each model column of pred."""
    missing = [c for c in models if pred[c].isna().any()]
    if missing:
        raise ValueError(f"missing probabilities in {missing}")
    y = (pred.label == "Silage").to_numpy(int)
    probs = {c: pred[c].to_numpy(float) for c in models}
    point = {c: point_metrics(y, p) for c, p in probs.items()}
    ci = cluster_bootstrap(y, probs, pred.field_id.to_numpy(), base=baseline, n_boot=n_boot, seed=seed)
    paired = {c: mcnemar(y, probs[c], probs[baseline]) for c in models if baseline and c != baseline}
    return y, point, ci, paired


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="column=Display name, one per model")
    ap.add_argument("--baseline", default=None, help="column of the model the others are compared with")
    ap.add_argument("--join", nargs="+", default=[], metavar="CSV:COLUMN=NEW",
                    help="add COLUMN of another predictions file with the same test rows, named NEW")
    ap.add_argument("--out", required=True, help="Markdown file to write; a .csv with the same name is written too")
    ap.add_argument("--title", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    pred = pd.read_csv(a.predictions)
    for spec in a.join:
        pred = join(pred, spec)
    names = dict(m.split("=", 1) if "=" in m else (m, m) for m in a.models)
    y, point, ci, paired = build(pred, list(names), a.baseline, a.n_boot, a.seed)

    cls, prob, cmp_rows, csv_rows, cm = [], [], [], [], []
    for c, name in names.items():
        pm, cc = point[c], ci[c]
        tp, fn, fp, tn = confusion(y, pred[c].to_numpy(float))
        cm += [{"model": name, "true class": "Silage", "called Silage": tp, "called Grain": fn, "total": tp + fn},
               {"model": "", "true class": "Grain", "called Silage": fp, "called Grain": tn, "total": fp + tn}]
        cls.append({"model": name, "correct": f"{pm['correct']}/{pm['n']}",
                    "accuracy": fmt(pm["accuracy"], cc["accuracy"]),
                    "balanced accuracy": fmt(pm["balanced_accuracy"], cc["balanced_accuracy"]),
                    "silage caught (recall)":
                        f"{pm['silage_caught']} = {fmt(pm['silage_recall'], cc['silage_recall'])}",
                    "grain called silage": pm["grain_called_silage"],
                    "silage precision": fmt(pm["silage_precision"], cc["silage_precision"]),
                    "silage F1": fmt(pm["silage_f1"], cc["silage_f1"]),
                    "grain precision": fmt(pm["grain_precision"], cc["grain_precision"]),
                    "grain F1": fmt(pm["grain_f1"], cc["grain_f1"]), "macro F1": fmt(pm["macro_f1"], cc["macro_f1"]),
                    "MCC": fmt(pm["mcc"], cc["mcc"])})
        prob.append({"model": name, "AUC": fmt(pm["auc"], cc["auc"], 3),
                     "log loss": fmt(pm["log_loss"], cc["log_loss"], 3),
                     "Brier": fmt(pm["brier"], cc["brier"], 3),
                     "predicted silage share": fmt(pm["silage_share_pred"], cc["silage_share_pred"])})
        row = {"model": name, "column": c, **{k: v for k, v in pm.items()},
               **{f"{k}_ci_low": cc[k][0] for k in RATES}, **{f"{k}_ci_high": cc[k][1] for k in RATES}}
        if c in paired:
            oa, ob, pv = paired[c]
            d = {k: point[c][k] - point[a.baseline][k] for k in RATES}
            cmp_rows.append({"model": name, "only this model right": oa, "only baseline right": ob,
                             "McNemar p": fmt(pv, d=3),
                             "accuracy difference": fmt(d["accuracy"], cc["diff_accuracy"]),
                             "balanced accuracy difference": fmt(d["balanced_accuracy"], cc["diff_balanced_accuracy"]),
                             "AUC difference": fmt(d["auc"], cc["diff_auc"], 3),
                             "Brier difference": fmt(d["brier"], cc["diff_brier"], 3)})
            row.update({"only_model_right": oa, "only_baseline_right": ob, "mcnemar_p": pv,
                        **{f"diff_{k}": d[k] for k in RATES},
                        **{f"diff_{k}_ci_low": cc[f"diff_{k}"][0] for k in RATES},
                        **{f"diff_{k}_ci_high": cc[f"diff_{k}"][1] for k in RATES}})
        csv_rows.append(row)

    n_fy = len(pred[["field_id", "year"]].drop_duplicates()) if "year" in pred else len(pred)
    src = ", ".join(f"`{Path(s.rsplit(':', 1)[0]).parent.name}/{Path(s.rsplit(':', 1)[0]).name}`"
                    for s in [a.predictions + ":"] + a.join)
    lines = [f"# {a.title or 'Test metrics'}", "",
             f"Generated by `classifier/metrics/report.py` from {src}: {len(pred)} test "
             f"predictions ({n_fy} distinct field-years, {pred.field_id.nunique()} fields, {int(y.sum())} Silage). "
             f"A field-year is called Silage when its probability is at least {THRESHOLD}. Brackets: 95% interval from "
             f"{a.n_boot} resamples of whole fields (all rows of a field are drawn together). Definitions: "
             "`classifier/metrics/README.md`.", "",
             "## Calls at the 0.5 cut", "", md_table(cls), "",
             "## Confusion matrices", "", md_table(cm), "",
             "## Probabilities", "",
             "Lower log loss and Brier are better. True silage share: "
             f"{fmt(point[list(names)[0]]['silage_share_true'])}.",
             "", md_table(prob), ""]
    if cmp_rows:
        lines += [f"## Against {names[a.baseline]} (same rows, paired)", "",
                  "Differences are this model minus the baseline; an interval that excludes 0 means the difference "
                  "holds across the field resamples. McNemar: exact test on the field-years where only one of the two "
                  "is right." + (" Here a field-year can be tested in more than one split seed, and McNemar counts "
                                 "each row as independent, so its p is optimistic; the field-resampled intervals are "
                                 "not." if "seed" in pred else ""), "", md_table(cmp_rows), ""]
    if "seed" in pred:
        per = []
        for s, g in pred.groupby("seed"):
            ys = (g.label == "Silage").to_numpy(int)
            r = {"seed": s, "test field-years (silage)": f"{len(g)} ({int(ys.sum())})"}
            for c, name in names.items():
                pm = point_metrics(ys, g[c].to_numpy(float))
                r[f"{name}: correct"] = pm["correct"]
                r[f"{name}: balanced acc."] = fmt(pm["balanced_accuracy"])
                r[f"{name}: AUC"] = fmt(pm["auc"], d=3)
            per.append(r)
        lines += ["## By split seed", "", md_table(per), ""]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame(csv_rows).to_csv(out.with_suffix(".csv"), index=False, float_format="%.4f")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
