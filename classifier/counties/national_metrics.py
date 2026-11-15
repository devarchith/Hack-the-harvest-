"""Challenge and USDA-style metrics for the national runs, from their result CSVs.

  python classifier/counties/national_metrics.py [<results folder> ...]   (default: national/results_final*)

Writes <results>/metrics_usda.md.
- Fields (BARC 2021-2023 test, Québec): fields by year and class, confusion matrix, per class precision (USDA: user's
  accuracy) and recall (producer's accuracy), F1, overall and balanced accuracy, Cohen's kappa, ROC AUC. 95%
  intervals: Wilson for accuracy, bootstrap over fields (2,000 draws, seed 0) for kappa and AUC.
- Counties (sampled counties; county-share model out of fold by state; called share = share of blocks with p >= 0.5):
  mean absolute error, root mean square error, bias, R^2, Spearman, 3-class accuracy and quadratic-weighted kappa,
  next to the last Census reused. 95% intervals by bootstrap over states (2,000 draws, seed 0).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).resolve().parent
B, SEED = 2000, 0
EDGES, CLASSES = [-1, 0.05, 0.40, 2], ["under 5%", "5-40%", "40% and over"]
MODELS = {"p_barc": "BARC-only", "p_llp": "County-share"}


def auc(y, p):
    r = rankdata(p)
    n1 = y.sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def kappa(y, s):
    po = (y == s).mean()
    pe = y.mean() * s.mean() + (1 - y.mean()) * (1 - s.mean())
    return (po - pe) / (1 - pe)


def wilson(k, n, z=norm.ppf(0.975)):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def ci(v):
    return f"{np.percentile(v, 2.5):.3f}-{np.percentile(v, 97.5):.3f}"


def field_section(f, name):
    y = (f.label == "Silage").to_numpy()
    out = [f"### {name}: {len(f)} field-years", "",
           "Fields by year and class: " + "; ".join(
               f"{yr} silage {int(((f.year == yr) & y).sum())}, grain {int(((f.year == yr) & ~y).sum())}"
               for yr in sorted(f.year.unique())), ""]
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(f), (B, len(f)))
    rows = []
    for col, m in MODELS.items():
        p = f[col].to_numpy()
        s = p >= 0.5
        tp, fn, fp, tn = (s & y).sum(), (~s & y).sum(), (s & ~y).sum(), (~s & ~y).sum()
        out += [f"{m}: rows are the label, columns the call.", "",
                "| | called Silage | called Grain |", "|---|---|---|",
                f"| Silage | {tp} | {fn} |", f"| Grain | {fp} | {tn} |", ""]
        ok = tp + tn
        lo, hi = wilson(ok, len(f))
        ps, rs = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        pg, rg = tn / max(tn + fn, 1), tn / max(tn + fp, 1)
        kb = [kappa(y[i], s[i]) for i in idx]
        ab = [auc(y[i], p[i]) for i in idx if 0 < y[i].sum() < len(i)]
        rows.append([m, f"{ok}/{len(f)} = {ok / len(f):.3f} ({lo:.3f}-{hi:.3f})", f"{(rs + rg) / 2:.3f}",
                     f"{ps:.3f} / {rs:.3f} / {2 * ps * rs / max(ps + rs, 1e-9):.3f}",
                     f"{pg:.3f} / {rg:.3f} / {2 * pg * rg / max(pg + rg, 1e-9):.3f}",
                     f"{kappa(y, s):.3f} ({ci(kb)})", f"{auc(y, p):.3f} ({ci(ab)})"])
        out += ["Per year, correct: " + "; ".join(
            f"{yr} {int((s == y)[(f.year == yr).to_numpy()].sum())}/{int((f.year == yr).sum())}"
            for yr in sorted(f.year.unique())), ""]
    out += ["| model | overall accuracy (95%) | balanced accuracy | Silage precision / recall / F1 | "
            "Grain precision / recall / F1 | Cohen's kappa (95%) | ROC AUC (95%) |", "|---|---|---|---|---|---|---|"]
    out += ["| " + " | ".join(r) + " |" for r in rows] + [""]
    return out


def county_stats(t, e):
    d = e - t
    cl = lambda v: pd.cut(v, EDGES, right=False, labels=False)  # noqa: E731
    return {"MAE": np.abs(d).mean(), "RMSE": np.sqrt((d ** 2).mean()), "bias": d.mean(),
            "R2": 1 - (d ** 2).sum() / ((t - t.mean()) ** 2).sum(), "Spearman": spearmanr(t, e)[0],
            "class right": (cl(t) == cl(e)).mean(),
            "weighted kappa": cohen_kappa_score(cl(t), cl(e), weights="quadratic")}


def county_section(res):
    cp = pd.read_csv(res / "county_predictions.csv", dtype={"fips": str})
    est = {"BARC-only": cp.set_index("fips").barc_called, "County-share (out of fold)": cp.set_index("fips").llp_oof_called}
    pr = res / "census_prior" / "census_prior_counties.csv"
    if pr.exists():
        c = pd.read_csv(pr, dtype={"fips": str}).set_index("fips")
        est["Last Census reused"] = c.last_census
        est["Last Census + satellite (other folds)"] = c.census_plus_satellite
    cp = cp.set_index("fips")
    t = cp.census_share
    states = cp.state.to_numpy()
    groups = {s: np.flatnonzero(states == s) for s in np.unique(states)}
    keys = list(groups)
    rng = np.random.default_rng(SEED)
    draws = [np.concatenate([groups[keys[j]] for j in rng.integers(0, len(keys), len(keys))]) for _ in range(B)]
    out = [f"### Counties: {len(cp)} sampled counties in {len(keys)} states, against the 2022 Census-only share", "",
           "| estimate | MAE (95%) | RMSE | bias | R^2 | Spearman | 3-class right | weighted kappa (95%) |",
           "|---|---|---|---|---|---|---|---|"]
    for name, e in est.items():
        e = e.reindex(t.index).to_numpy(float)
        tt = t.to_numpy(float)
        st = county_stats(tt, e)
        bs = [county_stats(tt[i], e[i]) for i in draws]
        out.append(f"| {name} | {st['MAE']:.3f} ({ci([b['MAE'] for b in bs])}) | {st['RMSE']:.3f} | {st['bias']:+.3f} | "
                   f"{st['R2']:.2f} | {st['Spearman']:.2f} | {st['class right'] * len(tt):.0f}/{len(tt)} | "
                   f"{st['weighted kappa']:.2f} ({ci([b['weighted kappa'] for b in bs])}) |")
    out.append("")
    tc = res / "test_county_predictions.csv"
    if tc.exists():
        x = pd.read_csv(tc, dtype={"fips": str})
        out += [f"Test counties never trained on ({len(x)}): mean absolute error BARC-only "
                f"{(x.barc_called - x.census_share).abs().mean():.3f}, county-share "
                f"{(x.llp_called - x.census_share).abs().mean():.3f}.", ""]
    sc = res / "state_check.csv"
    if sc.exists():
        us = pd.read_csv(sc).iloc[-1]
        out += ["National silage share of corn (band-stratified estimate): " + ", ".join(
            f"{k} {v:.3f}" for k, v in us.drop(["state", "counties"], errors="ignore").items()) + ".", ""]
    return out


def main(dirs):
    for res in dirs:
        f = pd.read_csv(res / "field_predictions.csv")
        out = [f"# Metrics: {res.name}", "", "Threshold 0.5 on the predicted silage probability. Silage is the "
               "positive class. Precision is USDA's user's accuracy, recall its producer's accuracy.", ""]
        out += field_section(f[f.kind == "barc"], "Maryland (BARC) 2021-2023, held-out years")
        out += field_section(f[f.kind == "quebec"], "Québec 2021-2023, no Québec label used in training")
        out += county_section(res)
        (res / "metrics_usda.md").write_text("\n".join(out), encoding="utf-8")
        print("wrote", res / "metrics_usda.md")


if __name__ == "__main__":
    main([Path(d) for d in sys.argv[1:]] or sorted((HERE / "national").glob("results_final*")))
