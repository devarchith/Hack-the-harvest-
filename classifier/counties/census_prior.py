"""County silage share: satellite estimate vs the last Census, and the two combined, scored out of fold by state.

Reads each run's county_predictions.csv (presto_national.py) and the QuickStats Census file. The truth is the 2022
Census-only share (CENSUS rows only; SURVEY rows are never summed in), recomputed here for every county.

Estimates compared, each scored on the same counties against the same truth:
  - last Census: the county's 2017 Census share (2012 if 2017 is withheld, else the state's 5-year prior share);
  - state survey: the state's 2022 survey share for every county in the state;
  - satellite: the run's out-of-fold county estimate (mean probability), as is;
  - satellite, recalibrated: logit(truth) ~ logit(satellite), fitted on the other folds;
  - last Census + satellite: logit(truth) ~ logit(last Census) + logit(satellite), fitted on the other folds.
Fitted estimates for a fold use only the other folds' counties (cross-fitting by the run's state folds).
The question this answers: does the satellite add anything to the last Census, on states held out.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr, wilcoxon

EPS = 0.005
CLASS_EDGES = [0, 0.05, 0.40, 1.0001]
CLASS_NAMES = ["under 5%", "5-40%", "40% and over"]


def census_shares(path, year):
    c = pd.read_csv(path, dtype=str)
    c = c[(c.year == str(year)) & (c.aggregation_level == "COUNTY") & (c.domain == "TOTAL") & (c.source == "CENSUS")
          & c.short_description.isin(["CORN, GRAIN - ACRES HARVESTED", "CORN, SILAGE - ACRES HARVESTED"])]
    c = c.assign(fips=c.state_ansi + c.county_ansi,
                 acres=pd.to_numeric(c.value.str.replace(",", "").str.strip(), errors="coerce"))
    w = c.pivot_table(index="fips", columns="crop_use", values="acres", aggfunc="first")
    return w.silage / (w.silage + w.grain)


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def cross_fit(d, cols):
    """sigmoid(b0 + sum b_i logit(x_i)), fitted on the other folds by least squares on the share scale (the scale
    the gaps are scored on), starting from least squares on the logit scale; predicted for each fold."""
    out = pd.Series(np.nan, index=d.index)
    sig = lambda z: 1 / (1 + np.exp(-z))
    for k in sorted(d.fold.unique()):
        tr, te = d.fold != k, d.fold == k
        X = np.column_stack([np.ones(tr.sum())] + [logit(d.loc[tr, c]) for c in cols])
        y = d.loc[tr, "truth"].to_numpy(float)
        b0 = np.linalg.lstsq(X, logit(y), rcond=None)[0]
        b = least_squares(lambda b: sig(X @ b) - y, b0).x
        Xt = np.column_stack([np.ones(te.sum())] + [logit(d.loc[te, c]) for c in cols])
        out[te] = sig(Xt @ b)
    return out


def share_class(s):
    return pd.cut(pd.Series(s, dtype=float), CLASS_EDGES, right=False, labels=CLASS_NAMES).astype(str).to_numpy()


def score(est, truth):
    rho, pv = spearmanr(est, truth)
    return {"counties": len(truth), "mean abs gap": round(float((est - truth).abs().mean()), 4),
            "median abs gap": round(float((est - truth).abs().median()), 4),
            "Spearman": f"{rho:.2f} (p {pv:.1g})",
            "class right": f"{int((share_class(est) == share_class(truth)).sum())}/{len(truth)}"}


def md_table(df):
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="results folders holding county_predictions.csv")
    ap.add_argument("--census", required=True, help="02_QuickStats/corn_grain_silage_acres_2012_2025.csv")
    ap.add_argument("--state-ratios", required=True, help="02_QuickStats/state_silage_ratios.csv")
    ap.add_argument("--sat-col", default="llp_oof_mean_p", help="county estimate column (out of fold)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    s22, s17, s12 = (census_shares(a.census, y) for y in (2022, 2017, 2012))
    st = pd.read_csv(a.state_ratios, dtype={"state_ansi": str})
    st22 = st[st.year == 2022].set_index("state_ansi")

    lines, all_rows = ["# County silage share: satellite vs the last Census", ""], []
    for run in a.runs:
        ct = pd.read_csv(Path(run) / "county_predictions.csv", dtype={"fips": str}).set_index("fips")
        d = pd.DataFrame({"fold": ct.fold, "state": ct.state, "county": ct.county,
                          "share_in_run": ct.census_share, "truth": s22.reindex(ct.index)})
        n_all = len(d)
        d = d[d.truth.notna()].copy()
        prior17, prior12 = s17.reindex(d.index), s12.reindex(d.index)
        st_prior = pd.Series(st22.prior_silage_share_5yr.reindex(d.index.str[:2]).to_numpy(), index=d.index)
        d["last_census"] = prior17.fillna(prior12).fillna(st_prior)
        d["prior_source"] = np.where(prior17.notna(), "2017", np.where(prior12.notna(), "2012", "state 5-yr"))
        d["state_survey"] = st22.silage_share_harvested.reindex(d.index.str[:2]).to_numpy()
        d["satellite"] = ct.loc[d.index, a.sat_col]
        d["maryland_only"] = ct.loc[d.index, "barc_mean_p"]
        d = d[d[["last_census", "state_survey", "satellite"]].notna().all(axis=1)]
        d["satellite_recal"] = cross_fit(d, ["satellite"])
        d["census_plus_satellite"] = cross_fit(d, ["last_census", "satellite"])

        names = {"last_census": "Last Census (2017) alone", "state_survey": "State 2022 survey share",
                 "maryland_only": "Satellite, Maryland-only model", "satellite": "Satellite, county-share model",
                 "satellite_recal": "Satellite, recalibrated on other folds",
                 "census_plus_satellite": "Last Census + satellite (fitted on other folds)"}
        tab = pd.DataFrame([{"estimate": v, **score(d[k], d.truth)} for k, v in names.items()])
        err_c, err_cs = (d.last_census - d.truth).abs(), (d.census_plus_satellite - d.truth).abs()
        w = wilcoxon(err_cs, err_c)
        changed = int((d.share_in_run - d.truth).abs().gt(0.01).sum())
        lines += [f"## {run}", "",
                  f"{len(d)} of {n_all} counties scored (Census-only 2022 share available; folds as in the run). "
                  f"The run's own census_share differs from the Census-only share by more than 0.01 in {changed}. "
                  f"Last-Census source: {d.prior_source.value_counts().to_dict()}.", "",
                  md_table(tab), "",
                  f"Last Census + satellite vs last Census alone, per-county absolute gap: better in "
                  f"{int((err_cs < err_c).sum())}, worse in {int((err_cs > err_c).sum())}; mean "
                  f"{err_cs.mean():.4f} vs {err_c.mean():.4f}; Wilcoxon signed-rank p {w.pvalue:.2g}.", "",
                  "3-class matrix, last Census + satellite (rows: Census 2022 class):", "",
                  pd.crosstab(pd.Series(share_class(d.truth), name="Census"),
                              pd.Series(share_class(d.census_plus_satellite), name="estimate")).to_string(), ""]
        all_rows.append(d.assign(run=run))
    (out / "census_prior.md").write_text("\n".join(lines), encoding="utf-8")
    pd.concat(all_rows).round(4).to_csv(out / "census_prior_counties.csv")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
