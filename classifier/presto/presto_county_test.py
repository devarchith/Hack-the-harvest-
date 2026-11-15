"""Presto tested on county totals: predicted silage share of 2022 corn in seven US counties against the 2022 Census.

  python classifier/presto/presto_county_test.py --pixels /home/keeboh/work/presto/pixels \
      --county-pixels /home/keeboh/work/counties/pixels --presto-dir /home/keeboh/ext/presto \
      --setting-from classifier/presto/results/standard_test_fixed_labels --out classifier/presto/results/county_test

The test, fixed before any county prediction was made:
  - Model: the one chosen for the official test, as in presto_quebec_test.py: fine-tuned Presto with the setting stored
    in <--setting-from>/run.json, trained on the 2017-2020 BARC field-years only, probabilities averaged over the
    seeds. The same run predicts the 44 BARC 2021-2023 field-years; they must match that run's predictions.
  - Test: the 200 corn blocks per county of ../counties/county_squares.csv (90 m, drawn at random from 2022 CDL
    corn in a 20 km square of the county), each scored once. No county number is used to train or choose anything.
  - Reported per county: the Census 2022 silage share of harvested corn acres, the mean predicted probability and the
    share of blocks called silage at 0.5, with 95% intervals from resampling 1 km cells (blocks in one cell can
    share a field).
  - Across the seven counties: Spearman rank correlation of the mean probability with the Census share (main result;
    exact one-sided p-value over all 5,040 orderings), and the mean absolute gap between the share called at 0.5 and
    the Census share. Reference for the gap: calling every block Grain.
What it cannot show: which blocks are wrong. Over- and under-calls can cancel in a county total.
"""
import argparse
import itertools
import json
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from scipy.stats import pearsonr, spearmanr

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from presto_standard_test import CHOICE_YEARS, train_curve  # noqa: E402
from presto_train import (EXTRA_MAX_PIX, MONTHS, field_inputs, load_presto, md_table, sha256,  # noqa: E402
                          stack_inputs)

COUNTIES = HERE.parent / "counties"
CRS = "EPSG:32618"


def census(selection):
    """{fips: (county name, Census silage share)} from county_selection.txt."""
    return {f: (name, float(s)) for f, name, s in re.findall(
        r"^(\d{5}) (.+?): Census \d{4} harvested acres .*?silage share ([\d.]+);", Path(selection).read_text(), re.M)}


def cell_bootstrap(p, cells, n_boot=2000, seed=0, level=0.95):
    """Intervals for the mean probability and the share called silage, resampling whole 1 km cells."""
    uniq, inv = np.unique(cells, return_inverse=True)
    rows_of = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    mean, called = [], []
    for _ in range(n_boot):
        idx = np.concatenate([rows_of[g] for g in rng.integers(0, len(uniq), len(uniq))])
        mean.append(p[idx].mean()); called.append((p[idx] >= 0.5).mean())
    q = [100 * (1 - level) / 2, 100 * (1 + level) / 2]
    return np.percentile(mean, q), np.percentile(called, q)


def exact_spearman_p(x, y):
    """One-sided p-value: share of all orderings of y with a Spearman correlation at least the observed one."""
    rho = spearmanr(x, y).statistic
    perms = [spearmanr(x, np.array(v)).statistic for v in itertools.permutations(y)]
    return float(rho), float(np.mean(np.array(perms) >= rho - 1e-12))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field-years", default=str(HERE.parent / "labels" / "results" / "field_years_fixed.csv"))
    ap.add_argument("--counties", default=str(COUNTIES / "county_squares.csv"))
    ap.add_argument("--selection", default=str(COUNTIES / "county_selection.txt"))
    ap.add_argument("--pixels", required=True, help="BARC folder written by presto_pixels.py")
    ap.add_argument("--county-pixels", required=True, help="folder with one presto_pixels.py output folder per FIPS")
    ap.add_argument("--presto-dir", required=True)
    ap.add_argument("--setting-from", required=True, help="results folder of presto_standard_test.py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    src = Path(a.setting_from)
    run = json.loads((src / "run.json").read_text())
    if run["chosen"] != "finetune" or run["args"].get("extra_train"):
        sys.exit(f"{src} did not choose fine-tuning on BARC rows alone")
    if run["inputs"]["field_years.csv"] != sha256(a.field_years):
        sys.exit(f"{a.field_years} is not the table {src} was run on")
    lr, epochs = float(run["best"]["finetune"]["lr"]), int(run["best"]["finetune"]["epochs"])
    seeds, last_month = run["args"]["seeds"], int(run["args"]["last_month"])

    fy = pd.read_csv(a.field_years)
    y = (fy.label == "Silage").to_numpy(int)
    tr = np.flatnonzero(fy.year.isin(CHOICE_YEARS))
    te_barc = np.flatnonzero(fy.year >= 2021)
    enc, groups = load_presto(a.presto_dir, pretrained=True)
    inputs = field_inputs(fy, a.pixels, groups, last_month)
    stacked, hashes = inputs[:5], dict(inputs[6])

    cs = pd.read_csv(a.counties, dtype={"fips": str})
    if set(cs.field_id) & set(fy.field_id):
        sys.exit("county blocks share field_id values with the labeled field-years")
    kept = []
    for fips, d in cs.groupby("fips", sort=False):
        X, M, LL, offs, key, cov, h = field_inputs(d, Path(a.county_pixels) / fips, groups, last_month, EXTRA_MAX_PIX,
                                                  skip=True)
        stacked = stack_inputs(stacked, (X, M, LL, offs, key))
        kept.append(d.merge(cov[["field_id", "year", "n_pix", "s2_months"]], on=["field_id", "year"], how="inner"))
        hashes.update({f"{fips}/{k}": v for k, v in h.items()})
    q = pd.concat(kept, ignore_index=True)
    dropped = len(cs) - len(q)
    X, M, LL, offs, key = stacked
    te_c = np.arange(len(fy), len(fy) + len(q))
    y_fit = np.concatenate([y, np.zeros(len(q), int)])  # county blocks never reach the model as training rows
    data = (torch.from_numpy(X).to(device), torch.from_numpy(M).to(device), torch.from_numpy(LL).to(device),
            offs, np.repeat(key, np.diff(offs)))
    print(f"{len(tr)} BARC training field-years; {len(q)} county blocks ({dropped} without pixels or a usable "
          f"month); lr {lr:g}, {epochs} epochs, seeds {seeds}, January to {MONTHS[last_month - 1]}; device {device}",
          flush=True)

    ho = np.concatenate([te_barc, te_c])
    p = np.mean([train_curve(enc, data, y_fit, tr, ho, device, s, lr, epochs)[-1] for s in seeds], 0)
    p_barc, p_c = p[:len(te_barc)], p[len(te_barc):]

    ref = pd.read_csv(src / "test_predictions.csv")
    chk = fy.loc[te_barc, ["field_id", "year", "label"]].assign(p_here=p_barc).merge(
        ref[["field_id", "year", "p_finetune"]], on=["field_id", "year"], validate="1:1")
    diff = float((chk.p_here - chk.p_finetune).abs().max())
    same_calls = int(((chk.p_here >= 0.5) == (chk.p_finetune >= 0.5)).sum())
    print(f"BARC 2021-2023 check: max |p - official run| = {diff:.4f}, same call on {same_calls}/{len(chk)}", flush=True)

    cent = gpd.GeoSeries.from_wkt(q.wkt, crs=CRS).centroid
    q = q.assign(p_presto=p_c.round(4), cell=(cent.x // 1000).astype(int).astype(str) + "_"
                 + (cent.y // 1000).astype(int).astype(str))
    q[["field_id", "field", "fips", "year", "cdl_code", "lon", "lat", "n_pix", "s2_months", "p_presto"]].to_csv(
        out / "test_predictions.csv", index=False)

    cen = census(a.selection)
    rows, per = [], []
    for fips, d in q.groupby("fips", sort=False):
        pp = d.p_presto.to_numpy()
        ci_m, ci_c = cell_bootstrap(pp, d.cell.to_numpy())
        name, share = cen[fips]
        per.append({"fips": fips, "census": share, "mean_p": pp.mean(), "called": (pp >= 0.5).mean()})
        rows.append({"County": f"{name} ({fips})", "Census 2022 silage share": f"{share:.3f}",
                     "blocks (1 km cells)": f"{len(d)} ({d.cell.nunique()})",
                     "mean probability": f"{pp.mean():.3f} ({ci_m[0]:.3f} to {ci_m[1]:.3f})",
                     "called silage at 0.5": f"{(pp >= 0.5).mean():.3f} ({ci_c[0]:.3f} to {ci_c[1]:.3f})"})
    per = pd.DataFrame(per)
    rho, p_rho = exact_spearman_p(per.census.to_numpy(), per.mean_p.to_numpy())
    rho_c = float(spearmanr(per.census, per.called).statistic)
    r_m = float(pearsonr(per.census, per.mean_p).statistic)
    gap = float((per.called - per.census).abs().mean())
    gap_grain = float(per.census.abs().mean())

    runj = {"finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "runtime_s": round(time.time() - t0), "args": vars(a),
            "setting": {"from": str(src), "lr": lr, "epochs": epochs, "seeds": seeds, "last_month": last_month},
            "barc_check": {"max_abs_diff": diff, "same_calls": same_calls, "n": len(chk)},
            "county_dropped": dropped, "county_max_pix": EXTRA_MAX_PIX,
            "across_counties": {"spearman_mean_p": rho, "spearman_mean_p_exact_one_sided_p": p_rho,
                                "spearman_called": rho_c, "pearson_mean_p": r_m, "mean_abs_gap_called": gap,
                                "mean_abs_gap_all_grain": gap_grain},
            "per_county": per.round(4).to_dict("records"),
            "inputs": {"field_years.csv": sha256(a.field_years), "county_squares.csv": sha256(a.counties),
                       "county_selection.txt": sha256(a.selection), **hashes},
            "versions": {"python": platform.python_version(), "torch": torch.__version__,
                         "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None}}
    (out / "run.json").write_text(json.dumps(runj, indent=1, default=str))

    lines = [f"# Presto on county totals: 2022 corn in seven US counties against the 2022 Census ({MONTHS[0]} to "
             f"{MONTHS[last_month - 1]})", "",
             "Generated by `presto_county_test.py`. The model is the one chosen for the official test "
             f"(`{src.name}`: fine-tuned, lr {lr:g}, {epochs} epochs, seeds {', '.join(map(str, seeds))}), trained on "
             f"the {len(tr)} BARC 2017-2020 field-years; no county number was used to train or choose anything. "
             f"{len(q)} corn blocks scored ({dropped} left out for no usable month). Brackets: 95% interval from "
             "resampling 1 km cells.", "",
             md_table(pd.DataFrame(rows)), "",
             f"Across the {len(per)} counties: Spearman correlation of the mean probability with the Census share "
             f"{rho:.2f} (exact one-sided p {p_rho:.3f}); of the share called silage, {rho_c:.2f}. Pearson correlation "
             f"of the mean probability, {r_m:.2f}. Mean absolute gap between the share called silage and the Census "
             f"share: {gap:.3f}; calling every block Grain: {gap_grain:.3f}.", "",
             "The Census share counts harvested acres in the whole county; the blocks come from a 20 km square of "
             "it and sit inside fields, so very small fields are under-sampled.", "",
             f"Check: the same run predicts the 44 BARC 2021-2023 field-years; the largest difference from "
             f"`{src.name}` is {diff:.4f}, and the call is the same on {same_calls} of {len(chk)}.", ""]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
