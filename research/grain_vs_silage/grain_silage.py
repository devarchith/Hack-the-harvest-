"""Grain vs silage: features and validation for the BARC corn fields.

Builds one row of harvest-timing, greenness and residue features per field-year from the per-field satellite
series, then tests how well they separate Grain from Silage on the 44 labeled 2021-2023 field-years:
holding out one year at a time, and (when the 2017-2020 inputs exist) training on 2017-2020 only.

Inputs (all under the git-ignored data/ folder; reproduce.py creates them):
  data/series_barc.parquet        loader output for 2021-2023 (Sentinel-2 + Landsat), --s2-series/--landsat-series
                                  can point the two sensors at different files
  data/weather_barc.parquet       Daymet daily weather per field (--weather)
  04_Evaluation/BARC_Corn.2021-2023.gpkg   labels; blank = unlabeled, NOT a negative
Optional, used when present:
  data/labels_kraatz_2017_2020.csv, data/series_kraatz_2017_2020.parquet, data/weather_kraatz_2017_2020.parquet
      2017-2020 training field-years (kraatz_fields.py, then the loader with --fields)
  data/records_kraatz_2021.csv
      2021 farm-record harvest dates; only used to check the harvest-date detector, never for training

Run from the repo root:  python research/grain_vs_silage/grain_silage.py
Outputs go to research/grain_vs_silage/results/.
"""
import argparse, sqlite3, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "results"
YEARS = (2021, 2022, 2023)
FIXED_DOYS = {"aug01": 213, "aug15": 227, "sep01": 244, "sep15": 258, "oct01": 274, "oct15": 288, "nov01": 305}


# ---------------------------------------------------------------- inputs
def load_labels():
    con = sqlite3.connect(ROOT / "04_Evaluation" / "BARC_Corn.2021-2023.gpkg")
    g = pd.read_sql("select OBJECTID, FARM, FIELD, ACRES, Corn_2021, Corn_2022, Corn_2023 "
                    "from barc_silage_grain_corn20212023", con)
    rows = []
    for _, r in g.iterrows():
        for y in YEARS:
            lab = r[f"Corn_{y}"]
            rows.append(dict(field_id=int(r.OBJECTID), farm=r.FARM, field=r.FIELD, acres=r.ACRES, year=y,
                             label=lab if isinstance(lab, str) and lab else None))
    return pd.DataFrame(rows)


def _rows(path, sensor):
    df = pd.read_parquet(path)
    df = df[(df.sensor == sensor) & df.usable]
    return df[["field_id", "year", "date", "sensor", "ndvi", "ndti", "ndmi"]].copy()


def load_series(s2_path, landsat_path):
    return _rows(s2_path, "S2"), _rows(landsat_path, "LANDSAT")


def load_extra():
    """2017-2020 field-years from the Kraatz farm records (preprocessing/kraatz_fields.py + loader --fields)."""
    p = ROOT / "data" / "series_kraatz_2017_2020.parquet"
    if not p.exists():
        return None
    lab = pd.read_csv(ROOT / "data" / "labels_kraatz_2017_2020.csv")
    lab["farm"] = lab.idn.str.split("_").str[0]
    lab["field"] = lab.idn.str.split("_", n=1).str[1]
    lab["acres"] = np.nan
    return (_rows(p, "S2"), _rows(p, "LANDSAT"),
            pd.read_parquet(ROOT / "data" / "weather_kraatz_2017_2020.parquet"),
            lab[["field_id", "farm", "field", "acres", "year", "label", "label_source"]])


def sensor_offsets(s2, ls):
    """Median S2 minus Landsat on the same field and day, per year. Uses all 83 fields, no labels."""
    p = s2.merge(ls, on=["field_id", "year", "date"], suffixes=("_s2", "_ls"))
    off = p.groupby("year").apply(lambda d: pd.Series({
        "pairs": len(d), "ndvi": (d.ndvi_s2 - d.ndvi_ls).median(),
        "ndti": (d.ndti_s2 - d.ndti_ls).median(), "ndmi": (d.ndmi_s2 - d.ndmi_ls).median()}))
    return off


def merged_series(s2, ls, off):
    """Put Landsat on the Sentinel-2 scale, merge, average same-day readings, drop isolated cloud dips."""
    ls = ls.copy()
    for v in ("ndvi", "ndti", "ndmi"):
        ls[v] = ls[v] + ls.year.map(off[v])
    m = pd.concat([s2, ls]).groupby(["field_id", "year", "date"], as_index=False)[["ndvi", "ndti", "ndmi"]].mean()
    m["doy"] = m.date.dt.dayofyear
    keep = []
    for _, g in m.sort_values("date").groupby(["field_id", "year"]):
        g = g.copy()
        while True:  # isolated dip: >=0.15 below both neighbours, both within 10 days (preprocessing README)
            v, d = g.ndvi.values, g.doy.values
            dip = np.zeros(len(g), bool)
            for i in range(1, len(g) - 1):
                if d[i] - d[i - 1] <= 10 and d[i + 1] - d[i] <= 10 and v[i] <= min(v[i - 1], v[i + 1]) - 0.15:
                    dip[i] = True
            if not dip.any():
                break
            g = g[~dip]
        keep.append(g)
    out = pd.concat(keep)
    print(f"merged readings: {len(m)}; isolated dips removed: {len(m) - len(out)}")
    return out


# ---------------------------------------------------------------- features
def crossing_down(doy, v, start_idx, x):
    """Interpolated DOY where NDVI first falls below x after start_idx and the next reading stays below x+0.05."""
    for i in range(start_idx + 1, len(v)):
        if v[i] < x and v[i - 1] >= x and (i + 1 == len(v) or v[i + 1] < x + 0.05):
            return doy[i - 1] + (v[i - 1] - x) / (v[i - 1] - v[i]) * (doy[i] - doy[i - 1])
    return np.nan


def gdd_between(w, start, end):
    """Corn GDD (30/10 C caps) between two DOYs, from daily Daymet."""
    if np.isnan(start) or np.isnan(end) or end <= start:
        return np.nan
    ww = w[(w.doy >= start) & (w.doy <= end)]
    t = (ww.tmax_c.clip(upper=30) + ww.tmin_c.clip(lower=10)) / 2 - 10
    return float(t.clip(lower=0).sum())


def features_one(g, w):
    g = g.sort_values("doy")
    d, v = g.doy.values.astype(float), g.ndvi.values
    f = {"n_obs": len(g), "n_obs_aug_nov": int(((d >= 213) & (d <= 334)).sum())}
    sm = pd.Series(v).rolling(3, center=True, min_periods=1).median().values
    win = np.where((d >= 150) & (d <= 260))[0]
    if len(win) == 0:
        return f
    ip = win[np.argmax(sm[win])]
    f["peak_ndvi"], f["peak_doy"] = sm[ip], d[ip]
    # green-up: first upward crossing of 0.4 after the spring minimum (skips winter cover crops)
    spring = np.where((d >= 90) & (d <= 180))[0]
    i0 = spring[np.argmin(v[spring])] if len(spring) else 0
    f["greenup_doy"] = next((d[i - 1] + (0.4 - v[i - 1]) / (v[i] - v[i - 1]) * (d[i] - d[i - 1])
                             for i in range(i0 + 1, ip + 1) if v[i - 1] < 0.4 <= v[i]), np.nan)
    # largest drop between consecutive readings after the peak (gap <= 20 days, before Dec)
    best = None
    for i in range(ip, len(v) - 1):
        gap = d[i + 1] - d[i]
        if gap <= 20 and d[i + 1] <= 335:
            drop = v[i] - v[i + 1]
            if best is None or drop > best[0]:
                best = (drop, i, gap)
    if best:
        drop, i, gap = best
        f.update(drop_size=drop, drop_rate=drop / gap, drop_doy=(d[i] + d[i + 1]) / 2,
                 ndvi_pre_drop=v[i], ndvi_post_drop=v[i + 1])
        post = g[(g.doy >= f["drop_doy"] + 5) & (g.doy <= f["drop_doy"] + 40)]
        f["ndti_post_drop"] = post.ndti.median() if len(post) else np.nan
        late = g[g.doy >= f["drop_doy"] + 20]
        f["regrowth"] = (late.ndvi.max() - v[i + 1]) if len(late) else np.nan
    for x in (0.6, 0.5, 0.4, 0.3):
        f[f"t_below_{x}"] = crossing_down(d, v, ip, x)
    for name, doy in FIXED_DOYS.items():
        inside = d.min() <= doy <= d.max()
        f[f"ndvi_{name}"] = np.interp(doy, d, v) if inside else np.nan
        f[f"ndti_{name}"] = np.interp(doy, d, g.ndti.values) if inside else np.nan
        f[f"ndmi_{name}"] = np.interp(doy, d, g.ndmi.values) if inside else np.nan
    f["season_days"] = f.get("t_below_0.4", np.nan) - f["greenup_doy"]
    f["gdd_greenup_to_drop"] = gdd_between(w, f["greenup_doy"], f.get("drop_doy", np.nan))
    f["gdd_greenup_to_0.4"] = gdd_between(w, f["greenup_doy"], f.get("t_below_0.4", np.nan))
    return f


def build_features(series, weather, labels):
    weather = weather.assign(year=weather.date.dt.year, doy=weather.date.dt.dayofyear)
    wg = dict(tuple(weather.groupby(["field_id", "year"])))
    rows = []
    for (fid, y), g in series.groupby(["field_id", "year"]):
        rows.append({"field_id": fid, "year": y, **features_one(g, wg[(fid, y)])})
    return labels.merge(pd.DataFrame(rows), on=["field_id", "year"], how="left")


# ---------------------------------------------------------------- evaluation
def best_threshold(x, y):
    """Feature direction and cut maximising balanced accuracy on training data. y=1 is Silage."""
    ok = ~np.isnan(x)
    x, y = x[ok], y[ok]
    xs = np.unique(x)
    cuts = (xs[:-1] + xs[1:]) / 2 if len(xs) > 1 else xs
    best = (-1, None, None)
    for c in cuts:
        for sign in (1, -1):  # sign=1: silage if x > c ; sign=-1: silage if x < c
            pred = (sign * (x - c) > 0).astype(int)
            tpr = (pred[y == 1] == 1).mean(); tnr = (pred[y == 0] == 0).mean()
            if (tpr + tnr) / 2 > best[0]:
                best = ((tpr + tnr) / 2, c, sign)
    return best


def apply_threshold(x, c, sign, fallback):
    p = np.where(np.isnan(x), fallback, (sign * (x - c) > 0).astype(int))
    return p.astype(int)


def report(name, y, p, years, lines):
    cm = confusion_matrix(y, p, labels=[0, 1])
    def pr(cls):
        tp = ((p == cls) & (y == cls)).sum()
        prec = tp / max((p == cls).sum(), 1); rec = tp / max((y == cls).sum(), 1)
        return prec, rec
    pg, rg = pr(0); ps, rs = pr(1)
    lines.append(f"\n### {name}")
    lines.append(f"confusion (rows = true Grain, Silage; cols = predicted Grain, Silage): {cm.tolist()}")
    lines.append(f"Grain  precision {pg:.2f}  recall {rg:.2f} | Silage precision {ps:.2f}  recall {rs:.2f} | "
                 f"accuracy {(p == y).mean():.2f}  balanced acc {(rg + rs) / 2:.2f}")
    per = pd.DataFrame({"y": y, "ok": p == y, "year": years}).groupby("year").agg(
        n=("y", "size"), silage=("y", "sum"), correct=("ok", "sum"))
    lines.append("per held-out year: " + "; ".join(f"{yr}: {int(r.correct)}/{int(r.n)} correct "
                                                  f"({int(r.silage)} silage)" for yr, r in per.iterrows()))


def main():
    ap = argparse.ArgumentParser()
    data = ROOT / "data"
    ap.add_argument("--s2-series", default=data / "series_barc.parquet")
    ap.add_argument("--landsat-series", default=data / "series_barc.parquet")
    ap.add_argument("--weather", default=data / "weather_barc.parquet")
    ap.add_argument("--records-2021", default=data / "records_kraatz_2021.csv")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# Grain vs silage: full validation report", "",
             f"Inputs: S2 {Path(args.s2_series).name}, Landsat {Path(args.landsat_series).name}, "
             f"weather {Path(args.weather).name}", ""]

    labels = load_labels()
    s2, ls = load_series(args.s2_series, args.landsat_series)
    off = sensor_offsets(s2, ls)
    lines.append("Sensor offsets (median S2 minus Landsat, same field and day, all fields; added to Landsat):")
    lines.append(off.round(3).to_string())
    series = merged_series(s2, ls, off)
    weather = pd.read_parquet(args.weather)
    feats = build_features(series, weather, labels)
    feats.to_csv(OUT / "features_all_field_years.csv", index=False)

    lab = feats[feats.label.notna()].reset_index(drop=True)
    y = (lab.label == "Silage").astype(int).values
    lines.append(f"\nLabeled field-years: {len(lab)} (Grain {int((y == 0).sum())}, Silage {int(y.sum())}); "
                 "blank labels are unlabeled and excluded.")
    lines.append(lab.groupby(["year", "label"]).size().unstack(fill_value=0).to_string())
    lines.append(f"Min usable readings Aug-Nov per labeled field-year: {int(lab.n_obs_aug_nov.min())}")

    feat_cols = [c for c in feats.columns if c not in
                 ("field_id", "farm", "field", "acres", "year", "label", "n_obs", "n_obs_aug_nov")]
    # --- univariate separation
    uni = []
    central = (lab.farm == "Central Farm").values
    for c in feat_cols:
        x = lab[c].values.astype(float); ok = ~np.isnan(x)
        if ok.sum() < 10 or len(set(y[ok])) < 2:
            continue
        auc = roc_auc_score(y[ok], x[ok])
        okc = ok & central
        aucc = roc_auc_score(y[okc], x[okc]) if len(set(y[okc])) == 2 else np.nan
        per_year = []
        for yr in YEARS:
            m = ok & (lab.year.values == yr)
            per_year.append(roc_auc_score(y[m], x[m]) if len(set(y[m])) == 2 else np.nan)
        uni.append(dict(feature=c, n=int(ok.sum()), auc=auc, separation=max(auc, 1 - auc),
                        silage_is="higher" if auc > 0.5 else "lower",
                        auc_central_farm_only=aucc, auc_2021=per_year[0], auc_2022=per_year[1], auc_2023=per_year[2],
                        grain_median=np.nanmedian(x[y == 0]), silage_median=np.nanmedian(x[y == 1])))
    uni = pd.DataFrame(uni).sort_values("separation", ascending=False)
    uni.to_csv(OUT / "univariate_separation.csv", index=False)
    lines.append("\n## Univariate separation (AUC, Silage = positive; 0.5 = none, 1.0 or 0.0 = perfect)")
    lines.append(uni.head(20).round(3).to_string(index=False))

    # --- held-out-year evaluation
    lines.append("\n## Leave-one-year-out: train on two years, test on the third")
    years = lab.year.values
    farm_pred = central.astype(int)
    report("Farm only (Silage if Central Farm): confound check, not a method", y, farm_pred, years, lines)

    preds = pd.DataFrame({"field_id": lab.field_id, "farm": lab.farm, "field": lab.field, "year": years,
                          "label": lab.label})
    def loyo_threshold(cands, name):
        p = np.zeros(len(y), int); chosen = []
        for yr in YEARS:
            tr, te = years != yr, years == yr
            fits = [(best_threshold(lab.loc[tr, c].values.astype(float), y[tr]), c) for c in cands]
            (ba, cut, sign), c = max(fits, key=lambda t: t[0][0])
            p[te] = apply_threshold(lab.loc[te, c].values.astype(float), cut, sign, fallback=0)
            chosen.append(f"{yr}: {c} {'>' if sign == 1 else '<'} {cut:.3f} (train bal-acc {ba:.2f})")
        report(name, y, p, years, lines)
        lines.append("rule chosen on the training years: " + " | ".join(chosen))
        return p

    preds["timing_rule"] = loyo_threshold(["drop_doy"], "H1 timing: one cut on the date of the largest NDVI drop")
    preds["green_cut_rule"] = loyo_threshold(["ndvi_pre_drop"], "H2 cut green: one cut on NDVI just before the largest drop")
    preds["best_single_rule"] = loyo_threshold(
        feat_cols, "Best single feature, chosen inside each training fold (nested; no peeking at the test year)")

    small = ["drop_doy", "ndvi_pre_drop", "drop_size", "ndvi_sep15", "ndti_post_drop"]
    for name, cols, model in [
        ("Logistic regression on 5 hypothesis features", small,
         make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                       LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000))),
        ("Random forest on all features", feat_cols,
         make_pipeline(SimpleImputer(strategy="median"),
                       RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced",
                                              random_state=0)))]:
        prob = np.zeros(len(y))
        for yr in YEARS:
            tr, te = years != yr, years == yr
            model.fit(lab.loc[tr, cols], y[tr])
            prob[te] = model.predict_proba(lab.loc[te, cols])[:, 1]
        key = "logreg_p_silage" if "Logistic" in name else "forest_p_silage"
        preds[key] = prob
        report(name + " (threshold 0.5)", y, (prob >= 0.5).astype(int), years, lines)
        lines.append(f"held-out ROC-AUC {roc_auc_score(y, prob):.3f}")
    # strict variant: also drop from training every field that appears in the test year
    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                       LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000))
    prob = np.zeros(len(y))
    for yr in YEARS:
        te = years == yr
        tr = (years != yr) & ~lab.field_id.isin(lab.field_id[te]).values
        lr.fit(lab.loc[tr, small], y[tr])
        prob[te] = lr.predict_proba(lab.loc[te, small])[:, 1]
    report("Logistic regression, strict: held-out year AND its fields removed from training", y,
           (prob >= 0.5).astype(int), years, lines)
    lines.append(f"held-out ROC-AUC {roc_auc_score(y, prob):.3f}")
    def wilson(k, n, z=1.96):
        p = k / n; den = 1 + z * z / n
        c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return c - h, c + h
    pl = (preds.logreg_p_silage.values >= 0.5).astype(int)
    for cls, nm in ((1, "Silage"), (0, "Grain")):
        k, n = int(((pl == cls) & (y == cls)).sum()), int((y == cls).sum())
        lo, hi = wilson(k, n)
        lines.append(f"{nm} recall (logistic, LOYO) {k}/{n}, 95% Wilson interval {lo:.2f}-{hi:.2f}")

    lr.fit(lab[small], y)
    coef = pd.Series(lr[-1].coef_[0], index=small).sort_values(key=abs, ascending=False)
    lines.append("\n## Learned weights: logistic regression on all 44, standardized features (+ pushes toward Silage)")
    lines.append(coef.round(2).to_string())
    allf = feats.copy()
    allf["p_silage"] = lr.predict_proba(allf[small])[:, 1]
    allf[["field_id", "farm", "field", "year", "label", "p_silage"] + small + ["regrowth"]].to_csv(
        OUT / "scores_all_field_years.csv", index=False)
    lines.append("scores_all_field_years.csv: p_silage for all 249 field-years from the model fit on the 44. "
                 "Only meaningful where the field was corn; most unlabeled field-years are not corn.")
    preds = preds.merge(lab[["field_id", "year", "drop_doy", "ndvi_pre_drop", "drop_size", "ndvi_sep15",
                             "ndti_post_drop", "regrowth"]], on=["field_id", "year"])
    preds.to_csv(OUT / "loyo_predictions.csv", index=False)
    wrong = preds[(preds.logreg_p_silage >= 0.5) != (preds.label == "Silage")]
    lines.append("\n## Field-years the logistic model gets wrong on the held-out year")
    lines.append(wrong[["year", "farm", "field", "label", "logreg_p_silage", "drop_doy", "ndvi_pre_drop",
                        "drop_size", "ndti_post_drop"]].round(3).to_string(index=False) if len(wrong) else "none")

    # --- harvest-date check against farm records (never used for training)
    if Path(args.records_2021).exists():
        x = pd.read_csv(args.records_2021)
        x = x[x.harvest_date.notna() & x.year.isin(YEARS)].copy()
        x["harvest_doy"] = pd.to_datetime(x.harvest_date).dt.dayofyear
        feats["idn"] = feats.farm.str.replace(" ", "", regex=False) + "_" + feats.field.astype(str)
        h = x.merge(feats, on=["idn", "year"], how="inner")
        lines.append(f"\n## Harvest-date check: {len(x)} farm-record dates in {sorted(x.year.unique())}, "
                     f"{len(h)} matched to a BARC polygon")
        if len(h):
            h["err_drop"] = h.drop_doy - h.harvest_doy
            h["err_t04"] = h["t_below_0.4"] - h.harvest_doy
            lines.append(h[["idn", "year", "record_label", "label", "harvest_date", "harvest_doy", "drop_doy",
                            "err_drop", "t_below_0.4", "err_t04"]].round(1)
                         .sort_values(["record_label", "harvest_doy"]).to_string(index=False))
            for c in ("err_drop", "err_t04"):
                e = h[c].dropna()
                lines.append(f"{c}: n={len(e)} median {e.median():+.1f} d, median |error| {e.abs().median():.1f} d, "
                             f"within 10 days {int((e.abs() <= 10).sum())}/{len(e)}")
            h.to_csv(OUT / "harvest_date_check.csv", index=False)

    extra = load_extra()
    if extra is not None:
        extra_section(extra, lab, y, small, feat_cols, lines)

    txt = "\n".join(lines)
    (OUT / "report.md").write_text(txt, encoding="utf-8")
    print(txt)


def make_logreg():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000))


def make_forest():
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced",
                                                random_state=0))


def extra_section(extra, lab, y, small, feat_cols, lines):
    """Train on 2017-2020 farm-record labels, test on the 2021-2023 evaluation labels."""
    s2, ls, weather, xlab = extra
    off = sensor_offsets(s2, ls)
    lines.append("\n# Adding 2017-2020 (Kraatz farm records, Zenodo 8087301)")
    lines.append("Sensor offsets 2017-2020 (median S2 minus Landsat, added to Landsat):")
    lines.append(off.round(3).to_string())
    xf = build_features(merged_series(s2, ls, off), weather, xlab)
    xf.to_csv(OUT / "features_2017_2020.csv", index=False)
    xf = xf[xf.label.notna()].reset_index(drop=True)
    yx = (xf.label == "Silage").astype(int).values
    lines.append(f"\n2017-2020 labeled field-years: {len(xf)} (Grain {int((yx == 0).sum())}, Silage {int(yx.sum())})")
    lines.append(pd.crosstab([xf.year, xf.label_source], xf.label).to_string())
    lines.append(f"Min usable readings Aug-Nov: {int(xf.n_obs_aug_nov.min())}; field-years with fewer than 8: "
                 f"{int((xf.n_obs_aug_nov < 8).sum())}")

    lines.append("\n## Same features, 2017-2020 vs 2021-2023 (medians; AUC with Silage = positive)")
    rows = []
    for c in small + ["regrowth", "t_below_0.3"]:
        r = {"feature": c}
        for nm, df, yy in (("2017-20", xf, yx), ("2021-23", lab, y)):
            x = df[c].values.astype(float); ok = ~np.isnan(x)
            r[f"auc_{nm}"] = roc_auc_score(yy[ok], x[ok])
            r[f"grain_{nm}"] = np.nanmedian(x[yy == 0]); r[f"silage_{nm}"] = np.nanmedian(x[yy == 1])
        rows.append(r)
    lines.append(pd.DataFrame(rows).round(3).to_string(index=False))

    years_t = lab.year.values
    lines.append("\n## Train on 2017-2020 only, test on all 44 labeled field-years of 2021-2023")
    out = pd.DataFrame({"field_id": lab.field_id, "farm": lab.farm, "field": lab.field, "year": years_t,
                        "label": lab.label})
    for name, cols, mk in (("Logistic regression, 5 hypothesis features", small, make_logreg),
                           ("Random forest, all features", feat_cols, make_forest)):
        m = mk().fit(xf[cols], yx)
        prob = m.predict_proba(lab[cols])[:, 1]
        out[name.split(",")[0].lower().replace(" ", "_") + "_p_silage"] = prob
        report(name, y, (prob >= 0.5).astype(int), years_t, lines)
        lines.append(f"test ROC-AUC {roc_auc_score(y, prob):.3f}")
        if cols is small:
            coef = pd.Series(m[-1].coef_[0], index=small).sort_values(key=abs, ascending=False)
            lines.append("weights learned from 2017-2020 alone: " +
                         ", ".join(f"{k} {v:+.2f}" for k, v in coef.items()))
    # strict: no field may be in both training and test
    tr = ~xf.field_id.isin(lab.field_id)
    m = make_logreg().fit(xf.loc[tr, small], yx[tr])
    prob = m.predict_proba(lab[small])[:, 1]
    report(f"Logistic regression, strict: trained only on the {int(tr.sum())} 2017-2020 field-years whose fields "
           "never appear in 2021-2023", y, (prob >= 0.5).astype(int), years_t, lines)
    lines.append(f"test ROC-AUC {roc_auc_score(y, prob):.3f}")
    # one threshold, chosen on 2017-2020 only
    (ba, cut, sign), c = max(((best_threshold(xf[c].values.astype(float), yx), c) for c in feat_cols),
                             key=lambda t: t[0][0])
    p = apply_threshold(lab[c].values.astype(float), cut, sign, fallback=0)
    report(f"Single rule chosen on 2017-2020: {c} {'>' if sign == 1 else '<'} {cut:.3f} (train bal-acc {ba:.2f})",
           y, p, years_t, lines)
    out.to_csv(OUT / "train1720_test2123_predictions.csv", index=False)

    lines.append("\n## Leave one year out over all seven years (logistic regression, 5 features)")
    allv = pd.concat([xf.assign(label_source=xf.label_source),
                      lab.assign(label_source="evaluation gpkg")], ignore_index=True)
    ya, yrs = (allv.label == "Silage").astype(int).values, allv.year.values
    prob = np.zeros(len(allv))
    for yr in sorted(set(yrs)):
        te = yrs == yr
        prob[te] = make_logreg().fit(allv.loc[~te, small], ya[~te]).predict_proba(allv.loc[te, small])[:, 1]
    report("Train on six years, test on the seventh", ya, (prob >= 0.5).astype(int), yrs, lines)
    lines.append(f"held-out ROC-AUC {roc_auc_score(ya, prob):.3f}")
    allv["p_silage"] = prob
    allv["correct"] = (prob >= 0.5) == (ya == 1)
    lines.append("accuracy by label source: " + "; ".join(
        f"{k}: {int(g.correct.sum())}/{len(g)}" for k, g in allv.groupby("label_source")))
    wrong = allv[~allv.correct]
    lines.append("\nField-years wrong when their year is held out:")
    lines.append(wrong[["year", "farm", "field", "label", "label_source", "p_silage", "drop_doy", "ndvi_pre_drop",
                        "drop_size", "ndvi_sep15", "ndti_post_drop", "n_obs_aug_nov"]]
                 .sort_values(["year", "field"]).round(3).to_string(index=False))
    allv.to_csv(OUT / "loyo7_predictions.csv", index=False)


if __name__ == "__main__":
    main()
