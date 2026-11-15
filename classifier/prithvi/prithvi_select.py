"""Pick one Prithvi embedding table on the 2017-2020 choice split only, then report its scores on every split.

  python classifier/prithvi/prithvi_select.py --runs TREE/research/pretrained_classifier/results \
      --out classifier/prithvi/results/evaluate

Reads the evaluate.py output folders <runs>/prithvi_*/. Each run has already chosen its own head setting on the
choice split ('Hold out one year, 2017-20 only'); the table whose chosen setting scores best there (most correct, then
AUC, then fewer inputs) is the result. The other tables' test scores are listed for reference only, since picking
between them by test score would use the test labels. Writes <out>/summary.md, choice.csv, scores.csv and a copy of
each run folder.
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

CHOICE = "Hold out one year, 2017-20 only (used to choose the model)"
BASE = "hand 5 / logistic (baseline)"


def md_table(df):
    cols = list(df.columns)
    rows = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="evaluate.py results folder holding prithvi_* runs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = sorted(p for p in Path(a.runs).glob("prithvi_*") if (p / "choice_grid.csv").exists())
    ch = pd.DataFrame([dict(table=r.name, **pd.read_csv(r / "choice_grid.csv").iloc[0].to_dict()) for r in runs])
    ch = ch.sort_values(["correct", "auc", "inputs"], ascending=[False, False, True]).reset_index(drop=True)
    chosen = ch.table.iloc[0]
    ch.to_csv(out / "choice.csv", index=False)

    sc = []
    for r in runs:
        s = pd.read_csv(r / "scores.csv")
        s = s[s.model.str.endswith("(chosen)") | (s.model == BASE)].copy()
        s["table"] = r.name
        sc.append(s)
        shutil.copytree(r, out / r.name, dirs_exist_ok=True)
    sc = pd.concat(sc, ignore_index=True)
    sc.to_csv(out / "scores.csv", index=False)
    splits = list(dict.fromkeys(sc.split))
    base = sc[(sc.model == BASE) & (sc.table == chosen)].set_index("split")
    wide = [{"model": BASE, **{s: f"{base.at[s, 'correct']}/{base.at[s, 'n']}" for s in splits}}]
    for t in [chosen] + [t for t in ch.table if t != chosen]:
        q = sc[(sc.table == t) & (sc.model != BASE)].set_index("split")
        wide.append({"model": t + (" (chosen)" if t == chosen else " (reference only)"),
                     **{s: f"{q.at[s, 'correct']}/{q.at[s, 'n']}" for s in splits}})
    cq = sc[(sc.table == chosen) & (sc.model != BASE)].set_index("split")
    detail = pd.DataFrame([{"split": s, "model": "chosen", **cq.loc[s, ["correct", "silage_caught",
                                                                           "grain_called_silage", "auc"]].to_dict()}
                           for s in splits] +
                          [{"split": s, "model": "baseline", **base.loc[s, ["correct", "silage_caught",
                                                                            "grain_called_silage", "auc"]].to_dict()}
                           for s in splits]).sort_values(["split", "model"])
    lines = ["# Prithvi-EO-2.0 frozen embeddings: scored with the shared evaluator", "",
             "Each table below is one frozen Prithvi embedding (`prithvi_embed.py`), scored by `evaluate.py` exactly "
             "as the Presto, AlphaEarth and TESSERA embeddings were: the hand 5 / logistic baseline must reproduce "
             "first, the head (embedding alone or with the 5 hand features, PCA size, C) is chosen on the 2017-2020 "
             "hold-out-one-year split, then run once on every split.", "",
             "## Choice between tables (2017-2020 choice split only, 83 field-years)", "",
             md_table(ch[["table", "family", "k", "C", "correct", "silage_caught", "grain_called_silage", "auc"]]), "",
             f"**Chosen: {chosen}.**", "",
             "## Correct field-years on every split", "", md_table(pd.DataFrame(wide)), "",
             "Only the chosen row was picked without the test labels; the reference rows are shown so nothing is "
             "hidden, not to be picked from.", "",
             "## Chosen table against the baseline", "", md_table(detail), "",
             f"McNemar tests and the full head grid: `{chosen}/summary.md`.", ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
