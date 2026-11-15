"""Recompute every number on the claims sheet from the committed result CSVs.

Run:  python classifier/atlas/claims.py
Writes classifier/atlas/claims.json and prints it.

Fixed in advance: silage is called at p >= 0.5; a call is "confident" when
p < 0.2 or p > 0.8; a county is "fast-changing" when its 2022 Census share
differs from its previous Census share by more than 0.10; county classes are
under 5%, 5-40%, 40% and over.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score

HERE = Path(__file__).resolve().parent
NAT = HERE.parent / "counties" / "national"
RES = NAT / "results_final"
JANOCT = NAT / "results_final_janoct"

MODELS = {"maryland_only": "p_barc", "county_share": "p_llp"}
LO, HI = 0.2, 0.8


def r(x, n=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def field_metrics(df, col):
    y = (df["label"] == "Silage").to_numpy()
    p = df[col].to_numpy()
    c = p >= 0.5
    tp, fn = int((y & c).sum()), int((y & ~c).sum())
    fp, tn = int((~y & c).sum()), int((~y & ~c).sum())
    per_year = {
        str(yr): f"{int((g['label'].eq('Silage') == (g[col] >= 0.5)).sum())}/{len(g)}"
        for yr, g in df.groupby("year")
    }
    return {
        "n": len(df),
        "n_silage": int(y.sum()),
        "n_grain": int((~y).sum()),
        "correct": tp + tn,
        "accuracy": r((tp + tn) / len(df)),
        "silage_called_silage": tp,
        "silage_called_grain": fn,
        "grain_called_silage": fp,
        "grain_called_grain": tn,
        "silage_precision": r(tp / (tp + fp)) if tp + fp else None,
        "silage_recall": r(tp / (tp + fn)),
        "grain_precision": r(tn / (tn + fn)) if tn + fn else None,
        "grain_recall": r(tn / (tn + fp)),
        "balanced_accuracy": r((tp / (tp + fn) + tn / (tn + fp)) / 2),
        "kappa": r(cohen_kappa_score(y, c)),
        "auc": r(roc_auc_score(y, p)),
        "per_year_correct": per_year,
    }


def confident(df, col):
    y = (df["label"] == "Silage").to_numpy()
    p = df[col].to_numpy()
    m = (p < LO) | (p > HI)
    ok = (p >= 0.5) == y
    return {
        "n": len(df),
        "n_confident": int(m.sum()),
        "coverage": r(m.mean()),
        "correct_on_confident": int(ok[m].sum()),
        "accuracy_on_confident": r(ok[m].mean()) if m.any() else None,
        "n_not_confident": int((~m).sum()),
        "accuracy_on_not_confident": r(ok[~m].mean()) if (~m).any() else None,
    }


def cls3(s):
    # under 5%, 5-40%, 40% and over
    return np.digitize(np.asarray(s, dtype=float), [0.05, 0.40])


def county_metrics(truth, est):
    truth, est = np.asarray(truth, float), np.asarray(est, float)
    gap = est - truth
    return {
        "n": len(truth),
        "mae": r(np.abs(gap).mean()),
        "bias": r(gap.mean()),
        "spearman": r(spearmanr(est, truth)[0]),
        "within_0.10": int((np.abs(gap) <= 0.10).sum()),
        "share_within_0.10": r((np.abs(gap) <= 0.10).mean()),
        "within_0.20": int((np.abs(gap) <= 0.20).sum()),
        "share_within_0.20": r((np.abs(gap) <= 0.20).mean()),
        "class3_right": int((cls3(est) == cls3(truth)).sum()),
    }


def main():
    out = {
        "definitions": {
            "silage_call": "p >= 0.5",
            "confident_call": f"p < {LO} or p > {HI}",
            "fast_changing_county": "|2022 Census share - previous Census share| > 0.10",
            "county_classes": "under 5%, 5-40%, 40% and over (boundaries belong to the upper class)",
            "county_share_model_on_counties": "out of fold by state (llp_oof_*)",
        }
    }

    fp = pd.read_csv(RES / "field_predictions.csv", dtype={"fips": str})
    md, qc = fp[fp.kind == "barc"], fp[fp.kind == "quebec"]

    # A, B
    out["A_maryland_44"] = {k: field_metrics(md, c) for k, c in MODELS.items()}
    out["B_quebec"] = {k: field_metrics(qc, c) for k, c in MODELS.items()}
    b = out["B_quebec"]
    out["B_quebec"]["silage_caught_ratio_county_share_over_maryland_only"] = r(
        b["county_share"]["silage_called_silage"] / b["maryland_only"]["silage_called_silage"], 2
    )

    # C
    cp = pd.read_csv(RES / "county_predictions.csv", dtype={"fips": str})
    t = cp["census_share"]
    out["C_counties_163"] = {
        "n_counties": len(cp),
        "n_states": int(cp["state"].nunique()),
        "maryland_only_called": county_metrics(t, cp["barc_called"]),
        "county_share_called": county_metrics(t, cp["llp_oof_called"]),
        "maryland_only_mean_p": county_metrics(t, cp["barc_mean_p"]),
        "county_share_mean_p": county_metrics(t, cp["llp_oof_mean_p"]),
        "note": "'called' = share of a county's blocks called silage (the definition metrics.md "
        "uses for the gap and the 3-class agreement); 'mean_p' = mean block probability.",
    }

    # D
    tc = pd.read_csv(RES / "test_county_predictions.csv", dtype={"fips": str})
    tt = tc["census_share"]
    out["D_test_counties_7"] = {
        "n": len(tc),
        "mae_maryland_only_called": r((tc["barc_called"] - tt).abs().mean()),
        "mae_county_share_called": r((tc["llp_called"] - tt).abs().mean()),
        "mae_maryland_only_mean_p": r((tc["barc_mean_p"] - tt).abs().mean()),
        "mae_county_share_mean_p": r((tc["llp_mean_p"] - tt).abs().mean()),
        "spearman_county_share_mean_p": r(spearmanr(tc["llp_mean_p"], tt)[0]),
        "spearman_county_share_mean_p_pvalue": r(spearmanr(tc["llp_mean_p"], tt)[1]),
        "spearman_maryland_only_mean_p": r(spearmanr(tc["barc_mean_p"], tt)[0]),
        "class3_right_county_share_called": int((cls3(tc["llp_called"]) == cls3(tt)).sum()),
        "class3_right_maryland_only_called": int((cls3(tc["barc_called"]) == cls3(tt)).sum()),
    }

    # E
    bp = pd.read_csv(RES / "block_predictions.csv", dtype={"fips": str})
    nat, tcb = bp[bp.kind == "national"], bp[bp.kind == "test_county"]

    def blocks_conf(p):
        p = p.to_numpy()
        m = (p < LO) | (p > HI)
        return {
            "n": len(p),
            "n_confident": int(m.sum()),
            "coverage": r(m.mean()),
            "n_confident_grain_side": int((p < LO).sum()),
            "n_confident_silage_side": int((p > HI).sum()),
        }

    out["E_confident_calls"] = {
        "computed_after_main_tests": True,
        "maryland_county_share": confident(md, "p_llp"),
        "maryland_maryland_only": confident(md, "p_barc"),
        "quebec_county_share": confident(qc, "p_llp"),
        "quebec_maryland_only": confident(qc, "p_barc"),
        "national_blocks_county_share_oof": blocks_conf(nat["p_llp_oof"]),
        "test_county_blocks_county_share": blocks_conf(tcb["p_llp"]),
    }

    # F
    p = qc["p_llp"].to_numpy()
    y = (qc["label"] == "Silage").to_numpy()
    idx = np.minimum((p * 10).astype(int), 9)
    bins, ece = [], 0.0
    for i in range(10):
        m = idx == i
        n = int(m.sum())
        row = {"bin": f"{i/10:.1f}-{(i+1)/10:.1f}", "n": n,
               "mean_p": r(p[m].mean()) if n else None,
               "observed_silage_rate": r(y[m].mean()) if n else None}
        if n:
            ece += n / len(p) * abs(p[m].mean() - y[m].mean())
        bins.append(row)
    out["F_quebec_calibration_county_share"] = {
        "computed_after_main_tests": True,
        "n": len(p),
        "bins": bins,
        "expected_calibration_error": r(ece),
        "mean_p": r(p.mean()),
        "observed_silage_rate": r(y.mean()),
    }

    # G
    pr = pd.read_csv(RES / "census_prior" / "census_prior_counties.csv", dtype={"fips": str, "prior_source": str})
    change = (pr["truth"] - pr["last_census"]).abs()
    fast = change > 0.10

    def grp(m):
        g = pr[m]
        e_last = (g["last_census"] - g["truth"]).abs()
        e_sat = (g["satellite"] - g["truth"]).abs()
        e_both = (g["census_plus_satellite"] - g["truth"]).abs()
        return {
            "n": len(g),
            "mae_last_census": r(e_last.mean()),
            "mae_satellite": r(e_sat.mean()),
            "mae_census_plus_satellite": r(e_both.mean()),
            "counties_satellite_closer": int((e_sat < e_last).sum()),
        }

    out["G_fast_changing_counties"] = {
        "computed_after_main_tests": True,
        "grouping_defined_after_results_seen": True,
        "satellite_column": "county-share model, out of fold, mean block probability",
        "fast_changing": grp(fast),
        "stable": grp(~fast),
        "overall": grp(np.ones(len(pr), bool)),
        "prior_source_counts": {str(k): int(v) for k, v in pr["prior_source"].value_counts().items()},
    }

    # H
    fj = pd.read_csv(JANOCT / "field_predictions.csv", dtype={"fips": str})
    out["H_in_season_jan_oct"] = {
        "maryland_44": {k: field_metrics(fj[fj.kind == "barc"], c) for k, c in MODELS.items()},
        "quebec": {k: field_metrics(fj[fj.kind == "quebec"], c) for k, c in MODELS.items()},
    }

    # I
    out["I_scale"] = {
        "counties_scored": len(cp),
        "states": int(cp["state"].nunique()),
        "national_blocks": len(nat),
        "blocks_per_county": r(len(nat) / len(cp), 1),
        "test_counties": len(tc),
        "test_county_blocks": len(tcb),
        "test_county_states": int(tc["state_name"].nunique()),
        "blocks_total": len(bp),
        "quebec_field_years": len(qc),
        "quebec_silage_field_years": int((qc.label == "Silage").sum()),
        "maryland_test_field_years": len(md),
        "years_field_tests": sorted(int(v) for v in fp["year"].unique()),
        "block_year": sorted(int(v) for v in bp["year"].unique()),
    }

    # Quoted, not recomputed here (no CSV in this folder tree holds it).
    out["quoted_not_recomputed"] = {
        "five_feature_baseline_maryland_correct": "39/44",
        "mcnemar_vs_baseline": "4 vs 2, p = 0.69",
        "source": "VALIDATION.md",
    }

    text = json.dumps(out, indent=2, ensure_ascii=False)
    (HERE / "claims.json").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
