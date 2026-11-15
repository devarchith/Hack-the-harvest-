"""Score a per-field-year feature table (for example a pretrained embedding) against the hand-feature baseline.

  python research/pretrained_classifier/evaluate.py --features research/pretrained_classifier/results/tessera/tessera_embeddings.csv --prefix T --name tessera

The table needs `field_id`, `year` and numeric columns named <prefix><digits> (T000..T127, A00..A63, ...).
Splits and scoring come from research/grain_vs_silage/model_comparison.py (`load()`, `splits()`, `score()`), imported,
not copied, so every model is scored the same way. Before anything new is reported, the hand 5 / logistic baseline
must reproduce its six published numbers (42, 39, 113, 24, 40, 68).

Head: a frozen embedding, so only a small, strongly regularized logistic regression is fitted, inside each training
fold: impute, standardize, optional PCA to k components, then logistic (class_weight="balanced"). Two families:
the embedding alone, and the 5 hand features plus the embedding's components. The grid (family x k x C) is ranked on
the 2017-2020 hold-out-one-year split only (most correct, then AUC, then the simpler model); the chosen setting is
then run once on every split. The best setting of each family is also reported, labeled as such.

Writes results/<name>/: choice_grid.csv, scores.csv, predictions.csv, summary.md, run.json.
"""
import argparse, datetime, hashlib, json, platform, re, subprocess, sys, time
from pathlib import Path

import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "research" / "grain_vs_silage"))
import grain_silage as gs  # noqa: E402
import model_comparison as mc  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

BASELINE = {"Hold out one year, 2021-23": 42, "Train 2017-20, test 2021-23": 39, "Hold out one year, 2017-23": 113,
            "Train 2017-20 other farms, test Central Farm 2021-23": 24,
            "Train 2017-20 unseen fields only, test 2021-23": 40, mc.CHOICE: 68}
KS = [4, 8, 16, 32, None]  # None = every embedding column, no PCA
CS = [0.05, 0.5]
KEYS = ["field_id", "farm", "field", "year", "label"]


def head(emb, hand, k, c):
    """Impute + standardize; PCA(k) on the embedding columns only; logistic regression."""
    e = [SimpleImputer(strategy="median"), StandardScaler()] + ([PCA(k, svd_solver="full")] if k else [])
    parts = [("emb", make_pipeline(*e), emb)]
    if hand:
        parts.append(("hand", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), hand))
    return make_pipeline(ColumnTransformer(parts),
                         LogisticRegression(C=c, class_weight="balanced", max_iter=5000))


def run(folds, make, cols):
    p, ys, keys = [], [], []
    for tr, te in folds:
        m = make().fit(tr[cols], tr.y.values)
        p.append(m.predict_proba(te[cols])[:, 1]); ys.append(te.y.values); keys.append(te[KEYS])
    return np.concatenate(p), np.concatenate(ys), pd.concat(keys, ignore_index=True)


def mcnemar(y, pa, pb):
    ca, cb = (pa >= 0.5) == (y == 1), (pb >= 0.5) == (y == 1)
    only_b, only_a = int((cb & ~ca).sum()), int((ca & ~cb).sum())
    return only_b, only_a, (binomtest(only_b, only_a + only_b).pvalue if only_a + only_b else 1.0)


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_info(inputs, t0):
    """Small reproducibility record: commit, versions, machine, runtime, input hashes."""
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        except OSError:
            return None
    import sklearn, scipy
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
            "scipy": scipy.__version__, "runtime_s": round(time.time() - t0, 1),
            "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "inputs_sha256": {str(Path(p).resolve().relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in inputs}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features", required=True, help="CSV with field_id, year and <prefix><digits> columns")
    ap.add_argument("--prefix", required=True, help="embedding column prefix, e.g. T for T000..T127")
    ap.add_argument("--name", required=True, help="output folder under results/ and the label in tables")
    ap.add_argument("--title", default=None, help="human-readable model name for summary.md")
    args = ap.parse_args()
    t0 = time.time()
    out = HERE / "results" / args.name
    out.mkdir(parents=True, exist_ok=True)

    t, x, allv, _ = mc.load()
    sp = mc.splits(t, x, allv)

    # 1. The baseline must reproduce before anything new is trusted.
    base = {}
    for split, folds in sp.items():
        p, y, k = run(folds, gs.make_logreg, mc.SMALL)
        base[split] = (p, y, k)
        got = mc.score(y, p)["correct"]
        if got != BASELINE[split]:
            sys.exit(f"baseline mismatch on '{split}': expected {BASELINE[split]}, got {got}")
    print("baseline hand 5 / logistic reproduces on all six splits:", list(BASELINE.values()), flush=True)

    # 2. Attach the new features.
    f = pd.read_csv(args.features)
    pat = re.compile(rf"^{re.escape(args.prefix)}\d+$")
    emb = [c for c in f.columns if pat.match(c)]
    if not emb:
        sys.exit(f"no columns matching {pat.pattern} in {args.features}")
    f = f[["field_id", "year"] + emb]
    t, x, allv = (d.drop(columns=[c for c in emb if c in d.columns]).merge(f, on=["field_id", "year"], how="left",
                                                                         validate="1:1") for d in (t, x, allv))
    missing = int(allv[emb[0]].isna().sum())
    print(f"{len(emb)} embedding columns; labeled field-years missing an embedding: {missing} of {len(allv)}",
          flush=True)
    sp = mc.splits(t, x, allv)

    # 3. Rank the grid on the 2017-2020 choice split only.
    grid = []
    for fam, hand in (("emb", []), ("hand5+emb", mc.SMALL)):
        for k in KS:
            if k and k >= len(emb):
                continue
            for c in CS:
                p, y, _ = run(sp[mc.CHOICE], lambda: head(emb, hand, k, c), emb + hand)
                grid.append(dict(family=fam, k=k or len(emb), pca=bool(k), C=c, **mc.score(y, p)))
    g = pd.DataFrame(grid)
    g["inputs"] = g.k + np.where(g.family == "emb", 0, len(mc.SMALL))
    g = g.sort_values(["correct", "auc", "inputs", "C"], ascending=[False, False, True, True]).reset_index(drop=True)
    g.to_csv(out / "choice_grid.csv", index=False)
    chosen = g.iloc[0]
    fam_best = {fam: g[g.family == fam].iloc[0] for fam in ("emb", "hand5+emb")}
    print(f"chosen on the choice split: {chosen.family} k={chosen.k} C={chosen.C} -> {chosen.correct}/83", flush=True)

    # 4. Run the chosen setting (and each family's best) once on every split.
    setups = {"hand 5 / logistic (baseline)": None}
    for fam, r in fam_best.items():
        tag = f"{args.name} {fam} k={r.k} C={r.C}" + (" (chosen)" if r.equals(chosen) else "")
        setups[tag] = r
    rows, preds = [], []
    for split, folds in sp.items():
        for tag, r in setups.items():
            if r is None:
                p, y, k = base[split]
            else:
                hand = [] if r.family == "emb" else mc.SMALL
                kk = int(r.k) if r.pca else None
                p, y, k = run(folds, lambda: head(emb, hand, kk, float(r.C)), emb + hand)
            rows.append(dict(split=split, model=tag, **mc.score(y, p)))
            preds.append(k.assign(split=split, model=tag, p_silage=p.round(4)))
    res, pr = pd.DataFrame(rows), pd.concat(preds, ignore_index=True)
    res.to_csv(out / "scores.csv", index=False)
    pr.to_csv(out / "predictions.csv", index=False)

    # 5. McNemar against the baseline on the 2021-2023 tests.
    mrows = []
    for split in [s for s in sp if s != mc.CHOICE and s != "Hold out one year, 2017-23"]:
        q = pr[pr.split == split]
        a = q[q.model == "hand 5 / logistic (baseline)"].reset_index(drop=True)
        for tag in list(setups)[1:]:
            b = q[q.model == tag].reset_index(drop=True)
            assert (a.field_id.values == b.field_id.values).all() and (a.year.values == b.year.values).all()
            y = (a.label == "Silage").astype(int).values
            ob, oa, pv = mcnemar(y, a.p_silage.values, b.p_silage.values)
            mrows.append({"test": split, "model": tag, "only model right": ob, "only baseline right": oa,
                          "McNemar p": round(pv, 3)})
    mt = pd.DataFrame(mrows)

    wide = res.pivot_table(index="model", columns="split", values="correct", aggfunc="first")
    wide = wide.reindex(index=list(setups), columns=list(sp))
    wide.columns = [f"{c} (/{int(res[res.split == c].n.iloc[0])})" for c in wide.columns]
    title = args.title or args.name
    lines = [f"# {title}: frozen embedding + logistic head vs the hand-feature baseline", "",
             f"Generated by `evaluate.py --features {Path(args.features).name} --prefix {args.prefix} "
             f"--name {args.name}`. Correct field-years at a 0.5 cut. The baseline reproduced on all six splits "
             "before anything else ran.", "",
             f"Embedding columns: {len(emb)}. Labeled field-years missing an embedding: {missing} of {len(allv)} "
             "(median-imputed inside each training fold).", "",
             "## Choice on 2017-2020 only", "",
             f"Grid: family (embedding alone; 5 hand features + embedding) x PCA components {KS} x C {CS}, ranked on "
             f"'{mc.CHOICE}' (83 field-years, 19 Silage) by correct, then AUC, then fewer inputs. Top 8 of "
             f"{len(g)}:", "", mc.md_table(g.head(8)), "",
             f"**Chosen: {chosen.family}, k={chosen.k}, C={chosen.C}** ({chosen.correct}/83 vs baseline 68/83). "
             "Each family's best setting is also shown below; only the chosen one was picked without the test labels "
             "deciding between families.", "",
             "## Every split", "", mc.md_table(wide.reset_index()), "",
             "## Against the baseline on the 2021-2023 tests (exact McNemar on the field-years where they disagree)",
             "", mc.md_table(mt), "",
             "## Full scores", "", mc.md_table(res), ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    inputs = [args.features, mc.OUT / "features_all_field_years.csv", mc.OUT / "features_2017_2020.csv"]
    info = run_info(inputs, t0) | {"command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
                                   "chosen": {"family": chosen.family, "k": int(chosen.k), "C": float(chosen.C)}}
    (out / "run.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print("\n".join(lines[lines.index("## Every split"):]))


if __name__ == "__main__":
    main()
