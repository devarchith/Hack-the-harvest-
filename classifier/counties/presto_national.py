"""Presto across the US: the BARC-trained model scored on corn blocks in sampled counties against the 2022 Census, and
next to it Presto also trained on county silage shares (learning from label proportions).

  python classifier/counties/presto_national.py --field-years classifier/labels/results/field_years_fixed.csv \
      --pixels <BARC pixels> --national <national_counties.py output, with pixels/<fips>/> \
      --test-counties <county_squares.py output, with pixels/<fips>/> \
      --census 02_QuickStats/corn_grain_silage_acres_2012_2025.csv --quebec classifier/quebec/field_years_quebec.csv \
      --quebec-selection classifier/quebec/quebec_selection.txt --quebec-pixels <Québec pixels> \
      --presto-dir <nasaharvest/presto> --setting-from classifier/presto/results/standard_test_fixed_labels \
      --quebec-official classifier/presto/results/quebec_test --out <results folder>

Fixed before any county prediction was made:
  - Data. BARC: field_years_fixed.csv; training = 2017-2020 (81 field-years), test = 2021-2023 (44). Sampled counties:
    <national>/selection.csv (national_counties.py), 100 corn blocks each, 2022 pixels from national_pixels.py; a
    county whose pull did not finish is left out and listed. Test counties: the seven of county_squares.py, never
    used to train or choose. Québec: field_years_quebec.csv, at most 256 pixels per field-year (as presto_quebec_test.py).
  - BARC-only model: the setting chosen for the official test (run.json of --setting-from), trained on BARC 2017-2020,
    probabilities averaged over its seeds. It must reproduce the official BARC 2021-2023 predictions.
  - Proportion-trained model: the same fine-tuning plus, at every step, a county loss: K_COUNTIES counties, B_BLOCKS
    blocks x P_PIX pixels each; binary cross-entropy between a county's mean predicted probability and its Census
    silage share, times the weight w. Latitude and longitude of every non-BARC pixel are set to BARC's mean, in
    training and prediction, so a county's location cannot stand in for its share.
  - Choice of w from W_GRID (0 = BARC-only with BARC's latitude and longitude): counties in 5 groups by state
    (StratifiedGroupKFold on the share band, seed 0); each group is predicted by models trained without it (seeds
    CHOICE_SEEDS); lowest mean absolute gap between the share of blocks called silage and the Census share wins,
    among the w that lose at most GUARD_FIELDS of the BARC 2017-2020 field calls when each training year is predicted
    by a model trained on the other three (BARC guard: the scored evaluation data are BARC fields). The
    final model uses the chosen w, every sampled county and the official seeds, and is scored once on the BARC
    2021-2023 test, Québec and the test counties.
  - County predicted share: share of blocks called silage (p >= 0.5); the mean probability is shown too. County classes
    by silage share: under 5%, 5-40%, 40% and over. National share: per band, the corn-acre weighted mean predicted
    share of the sampled counties, weighted by the band's corn acres in the population (national_counties.py).
  - States (--state-ratios, 02_QuickStats/state_silage_ratios.csv): per state, the corn-acre weighted share of the
    sampled counties (Census and each model) next to the state's 2022 survey and Census shares. The sampled counties
    are drawn by share band, not in proportion, so the state's own share is context, not the target.
What county totals cannot show: which blocks are wrong (over- and under-calls cancel), and the sampled square is not
the whole county.
--parallel N runs the fits as separate processes of this script (--job), N at a time; the results are the same fits.
"""
import argparse
import copy
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedGroupKFold

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "presto"), str(HERE.parent / "metrics"), str(HERE)]
from metrics import point_metrics  # noqa: E402
from presto_quebec_test import square_weights, weighted  # noqa: E402
from presto_standard_test import CHOICE_YEARS  # noqa: E402
from presto_train import (EXTRA_MAX_PIX, PixelClassifier, batches, epoch_rows, field_inputs, load_presto,  # noqa: E402
                          md_table, pixel_weights, predict, sha256, stack_inputs)

W_GRID = (0.0, 0.3, 1.0, 3.0, 10.0)
CHOICE_SEEDS = (0,)
GUARD_FIELDS = 1  # w may lose at most this many BARC 2017-2020 held-out-year calls against w = 0
K_COUNTIES, B_BLOCKS, P_PIX = 8, 32, 4
CLASS_EDGES = [0, 0.05, 0.40, 1.0001]
CLASS_NAMES = ["under 5%", "5-40%", "40% and over"]


# ---------------------------------------------------------------- training

def county_loss(model, data, bags, rng, device):
    """Binary cross-entropy between each drawn county's mean predicted probability and its Census share."""
    X, M, LL, offs, key_pix = data
    pick = rng.choice(len(bags), min(K_COUNTIES, len(bags)), replace=False)
    idx, target = [], []
    for c in pick:
        share, rows = bags[c]
        r = rng.choice(rows, B_BLOCKS, replace=len(rows) < B_BLOCKS)
        idx.append(np.concatenate([rng.integers(offs[q], offs[q + 1], P_PIX) for q in r]))
        target.append(share)
    idx = np.concatenate(idx)
    logit = torch.zeros(len(idx), device=device)
    k = key_pix[idx]
    for v in np.unique(k):  # Presto needs one masked-token count per batch
        sel = np.flatnonzero(k == v)
        bt = torch.from_numpy(idx[sel]).to(device)
        logit = logit.index_put((torch.from_numpy(sel).to(device),), model(X[bt], M[bt], LL[bt]))
    p = torch.sigmoid(logit).view(len(pick), -1).mean(1)
    return F.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), torch.tensor(target, dtype=torch.float32, device=device))


def fit(enc, data, y, tr, device, seed, lr, epochs, bags=(), w=0.0, ppf=64, bs=256, wd=0.05):
    """Fine-tuning as presto_standard_test.train_curve (the same draws, so w = 0 gives the same model), plus w x the
    county loss at every step (its draws come from a separate generator)."""
    X, M, LL, offs, key_pix = data
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    brng = np.random.default_rng(10_000 + seed)
    model = PixelClassifier(copy.deepcopy(enc)).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    owner = np.repeat(np.arange(len(offs) - 1), np.diff(offs))
    ypix = torch.from_numpy(y[owner].astype(np.float32)).to(device)
    for ep in range(epochs):
        model.train()
        rows = epoch_rows(tr, (), 1.0, rng)
        wpix = pixel_weights(ypix, y, rows)
        idx = np.concatenate([rng.integers(offs[r], offs[r + 1], ppf) for r in rows])
        for b in batches(idx, key_pix, bs, rng):
            bt = torch.from_numpy(b).to(device)
            loss = (F.binary_cross_entropy_with_logits(model(X[bt], M[bt], LL[bt]), ypix[bt], reduction="none")
                    * wpix[bt]).mean()
            if w > 0:
                loss = loss + w * county_loss(model, data, bags, brng, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model


# ---------------------------------------------------------------- scoring

def share_class(s):
    return pd.cut(pd.Series(s, dtype=float), CLASS_EDGES, right=False, labels=CLASS_NAMES)


def county_scores(meta, p, name):
    """Per county: blocks scored, share called silage, mean probability."""
    d = meta.assign(p=p)
    g = d.groupby("fips")
    return pd.DataFrame({f"{name}_called": g.p.apply(lambda v: float((v >= 0.5).mean())),
                         f"{name}_mean_p": g.p.mean()})


def county_summary(ct, name, census="census_share"):
    called, meanp, cen = ct[f"{name}_called"], ct[f"{name}_mean_p"], ct[census]
    rho, pv = spearmanr(meanp, cen)
    cls_t, cls_p = share_class(cen.to_numpy()), share_class(called.to_numpy())
    return {"counties": len(ct), "gap called (mean abs)": round(float((called - cen).abs().mean()), 3),
            "gap mean p (mean abs)": round(float((meanp - cen).abs().mean()), 3),
            "Spearman (mean p vs Census)": f"{rho:.2f} (p {pv:.1g})",
            "class right": f"{int((cls_t.to_numpy() == cls_p.to_numpy()).sum())}/{len(ct)}"}


def class_matrix(ct, name, census="census_share"):
    t = share_class(ct[census].to_numpy()).rename("Census class")
    p = share_class(ct[f"{name}_called"].to_numpy()).rename("predicted class")
    m = pd.crosstab(t, p, dropna=False).reindex(index=CLASS_NAMES, columns=CLASS_NAMES, fill_value=0)
    return m.reset_index().rename(columns={c: f"called {c}" for c in CLASS_NAMES})


def national_share(ct, pop, name):
    """Stratified estimate: band corn acres x corn-acre weighted mean predicted share of the band's sampled counties."""
    est, tot = 0.0, 0.0
    for band, d in pop.groupby("band_name"):
        s = ct[ct.band == band]
        if not len(s):
            continue
        a = s.grain_acres + s.silage_acres
        A = float((d.grain + d.silage).sum())
        est += A * float((a * s[f"{name}_called"]).sum() / a.sum())
        tot += A
    return est / tot


def field_row(name, y, p):
    pm = point_metrics(y, p)
    return {"model": name, "n (silage)": f"{pm['n']} ({pm['silage']})", "correct": pm["correct"],
            "silage caught": pm["silage_caught"], "grain called silage": pm["grain_called_silage"],
            "AUC": f"{pm['auc']:.3f}", "balanced accuracy": f"{pm['balanced_accuracy']:.2f}"}


def confusion_rows(name, y, p):
    pred = np.asarray(p) >= 0.5
    y = np.asarray(y).astype(bool)
    return [{"model": name, "true class": "Silage", "called Silage": int((pred & y).sum()),
             "called Grain": int((~pred & y).sum()), "total": int(y.sum())},
            {"model": "", "true class": "Grain", "called Silage": int((pred & ~y).sum()),
             "called Grain": int((~pred & ~y).sum()), "total": int((~y).sum())}]


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field-years", required=True, help="classifier/labels/results/field_years_fixed.csv")
    ap.add_argument("--pixels", required=True, help="BARC folder written by presto_pixels.py")
    ap.add_argument("--national", required=True, help="folder of national_counties.py and national_pixels.py")
    ap.add_argument("--test-counties", default=None, help="folder of county_squares.py output and its pixels")
    ap.add_argument("--census", required=True, help="02_QuickStats/corn_grain_silage_acres_2012_2025.csv")
    ap.add_argument("--quebec", required=True, help="classifier/quebec/field_years_quebec.csv")
    ap.add_argument("--quebec-selection", required=True, help="classifier/quebec/quebec_selection.txt")
    ap.add_argument("--quebec-pixels", required=True)
    ap.add_argument("--presto-dir", required=True)
    ap.add_argument("--setting-from", required=True, help="results folder of presto_standard_test.py")
    ap.add_argument("--quebec-official", default=None, help="results folder of presto_quebec_test.py (check only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--state-ratios", default=None, help="02_QuickStats/state_silage_ratios.csv (state check)")
    ap.add_argument("--smoke", action="store_true", help="1 seed, 2 epochs, 2 folds: a test of the code, not a result")
    ap.add_argument("--parallel", type=int, default=1, help="fits run as this many processes at a time")
    ap.add_argument("--w-grid", default=None, help="weights to try, comma-separated, 0 first (default: W_GRID)")
    ap.add_argument("--job", default=None, help=argparse.SUPPRESS)  # one fit, written by --parallel
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    a = ap.parse_args(argv)
    job = json.loads(Path(a.job).read_text()) if a.job else None
    t0 = time.time()
    w_grid = tuple(float(v) for v in a.w_grid.split(",")) if a.w_grid else W_GRID
    if w_grid[0] != 0:
        sys.exit("--w-grid must start with 0 (the BARC-only baseline of the choice)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    nat_dir = Path(a.national)
    log = lambda *s: print(f"[{time.time() - t0:6.0f} s]", *s, flush=True)
    late = [p for p in (a.quebec_official, a.census, a.state_ratios, a.quebec_selection) if p and not Path(p).exists()]
    if late:  # read only after the fits
        sys.exit(f"not found: {late}")

    src = Path(a.setting_from)
    run = json.loads((src / "run.json").read_text())
    if run["chosen"] != "finetune" or run["args"].get("extra_train"):
        sys.exit(f"{src} did not choose fine-tuning on BARC rows alone")
    if run["inputs"]["field_years.csv"] != sha256(a.field_years):
        sys.exit(f"{a.field_years} is not the table {src} was run on")
    lr, epochs = float(run["best"]["finetune"]["lr"]), int(run["best"]["finetune"]["epochs"])
    seeds, last_month = list(run["args"]["seeds"]), int(run["args"]["last_month"])
    choice_seeds, n_folds = list(CHOICE_SEEDS), 5
    if a.smoke:
        seeds, choice_seeds, epochs, n_folds = [0], [0], 2, 2

    # ---- inputs: BARC, sampled counties, test counties, Québec (in that row order)
    fy = pd.read_csv(a.field_years)
    enc, groups = load_presto(a.presto_dir, pretrained=True)
    X, M, LL, offs, key, _, hashes = field_inputs(fy, a.pixels, groups, last_month)
    data = (X, M, LL, offs, key)
    meta = [pd.DataFrame({"kind": "barc", "fips": "", "field_id": fy.field_id, "year": fy.year, "label": fy.label})]
    used_hashes = {f"barc/{k}": v for k, v in hashes.items()}

    if not job:  # the copy this run and its jobs read; the sampler may rewrite selection.csv meanwhile
        (out / "selection_used.csv").write_bytes((nat_dir / "selection.csv").read_bytes())
    sel = pd.read_csv(out / "selection_used.csv", dtype={"fips": str})
    sel = sel[sel.status == "ok"].set_index("fips")
    pop = None if job else pd.read_csv(nat_dir / "population.csv", dtype={"fips": str}).set_index("fips")
    missing, nat_order = [], []
    for fips in (job["national"] if job else sel.index):  # a job reads the parent's counties, in its order
        pdir = nat_dir / "pixels" / fips
        # the provenance file is written after the npz, so a pull still writing is left out
        if not (pdir / "pixels_2022.npz").exists() or not (pdir / "pixels_2022.provenance.json").exists():
            missing.append(fips)
            continue
        c = pd.read_csv(nat_dir / "fy" / f"{fips}.csv")
        try:
            cx, cm, cl, co, ck, cov, h = field_inputs(c, pdir, groups, last_month, skip=True)
        except ValueError as e:  # no usable block
            log(f"county {fips}: {e}; left out")
            missing.append(fips)
            continue
        data = stack_inputs(data, (cx, cm, cl, co, ck))
        meta.append(pd.DataFrame({"kind": "national", "fips": fips, "field_id": cov.field_id, "year": cov.year,
                                  "label": ""}))
        used_hashes.update({f"national/{fips}/{k}": v for k, v in h.items()})
        nat_order.append(fips)
    if job and missing:
        sys.exit(f"job: counties without pixels {missing}")
    log(f"sampled counties with pixels: {len(sel) - len(missing)} of {len(sel)}; missing {missing}")

    test_order = []
    if a.test_counties:
        tdir = Path(a.test_counties)
        tq = pd.read_csv(tdir / "county_squares.csv", dtype={"fips": str})
        for fips, c in tq.groupby("fips"):
            pdir = tdir / "pixels" / fips
            if job and fips not in job["test"]:
                continue
            if not (pdir / "pixels_2022.npz").exists():
                log(f"test county {fips}: no pixels yet, left out")
                continue
            cx, cm, cl, co, ck, cov, h = field_inputs(c, pdir, groups, last_month, skip=True)
            data = stack_inputs(data, (cx, cm, cl, co, ck))
            meta.append(pd.DataFrame({"kind": "test_county", "fips": fips, "field_id": cov.field_id,
                                      "year": cov.year, "label": ""}))
            used_hashes.update({f"test_counties/{fips}/{k}": v for k, v in h.items()})
            test_order.append(fips)
    if job and test_order != job["test"]:
        sys.exit(f"job: test counties {test_order} differ from {job['test']}")

    q = pd.read_csv(a.quebec)
    qx, qm, ql, qo, qk, qcov, qh = field_inputs(q, a.quebec_pixels, groups, last_month, EXTRA_MAX_PIX, skip=True)
    q = q.merge(qcov[["field_id", "year"]], on=["field_id", "year"], how="inner")
    data = stack_inputs(data, (qx, qm, ql, qo, qk))
    meta.append(pd.DataFrame({"kind": "quebec", "fips": "", "field_id": q.field_id, "year": q.year, "label": q.label}))
    used_hashes.update({f"quebec/{k}": v for k, v in qh.items()})
    meta = pd.concat(meta, ignore_index=True)

    X, M, LL, offs, key = data
    n_barc = len(fy)
    barc_pix = offs[n_barc]
    ll_barc = LL[:barc_pix].mean(0)
    LL_fix = LL.copy()
    LL_fix[barc_pix:] = ll_barc
    key_pix = np.repeat(key, np.diff(offs))
    dev = {}

    def on_device():  # (d_real, d_fix) on the GPU; only processes that fit need them
        if not dev:
            Xd, Md = torch.from_numpy(X).to(device), torch.from_numpy(M).to(device)
            dev["real"] = (Xd, Md, torch.from_numpy(LL).to(device), offs, key_pix)
            dev["fix"] = (Xd, Md, torch.from_numpy(LL_fix).to(device), offs, key_pix)
        return dev["real"], dev["fix"]

    y = np.where(meta.kind == "barc", (meta.label == "Silage").astype(int), 0)  # only BARC labels reach the model
    tr = np.flatnonzero((meta.kind == "barc") & meta.year.isin(CHOICE_YEARS))
    te_barc = np.flatnonzero((meta.kind == "barc") & (meta.year >= 2021))
    nat_rows = np.flatnonzero(meta.kind == "national")
    test_rows = np.flatnonzero(meta.kind == "test_county")
    q_rows = np.flatnonzero(meta.kind == "quebec")
    other = np.concatenate([te_barc, nat_rows, test_rows, q_rows])
    log(f"rows: BARC train {len(tr)}, BARC test {len(te_barc)}, national blocks {len(nat_rows)}, test-county blocks "
        f"{len(test_rows)}, Québec {len(q_rows)}; pixels {len(key_pix)}; lr {lr:g}, {epochs} epochs, seeds {seeds}; "
        f"BARC mean lat/lon {ll_barc.round(4).tolist()}; device {device}")

    nat_meta = meta.loc[nat_rows]
    ct = sel.loc[sorted(nat_meta.fips.unique()), ["state", "county", "band", "census_share", "grain_acres",
                                                  "silage_acres", "crs"]].copy()
    rows_of = {f: nat_rows[(nat_meta.fips == f).to_numpy()] for f in ct.index}
    fold = np.zeros(len(ct), int)
    sgk = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=0)
    for k, (_, te) in enumerate(sgk.split(ct, ct.band, ct.state)):
        fold[te] = k
    ct["fold"] = fold

    pred_rows = np.concatenate([other, tr])  # tr: for the leave-one-year-out BARC guard

    def job_fit(seed, w, bag_fips, drop_year=None):
        """One fine-tuning (w = 0: BARC-only; BARC rows have the same latitude/longitude in d_real and d_fix, so the fit
        is the official one) on the BARC training rows, without drop_year if given; predicted on pred_rows with real
        and with BARC latitude/longitude."""
        d_real, d_fix = on_device()
        bags = [(float(ct.census_share[f]), rows_of[f]) for f in bag_fips]
        rows = tr if drop_year is None else tr[meta.year.to_numpy()[tr] != drop_year]
        m = fit(enc, d_fix, y, rows, device, seed, lr, epochs, bags, w)
        return {"real": predict(m, d_real, pred_rows), "fix": predict(m, d_fix, pred_rows)}

    if job:
        if [len(meta), int(offs[-1])] != job["shape"]:
            sys.exit(f"job: rows and pixels {[len(meta), int(offs[-1])]} differ from the parent's {job['shape']}")
        np.savez(Path(a.job).with_suffix(".npz"),
                 **job_fit(job["seed"], job["w"], job["bags"], job.get("drop_year")))
        log("job done")
        return

    def run_jobs(specs, tag):
        """job_fit for each spec, in this process or (--parallel > 1) in --parallel processes at a time."""
        if a.parallel <= 1:
            return [job_fit(s["seed"], s["w"], s["bags"], s.get("drop_year")) for s in specs]
        jd = out / "jobs"
        jd.mkdir(exist_ok=True)
        todo, running, res = list(range(len(specs))), {}, [None] * len(specs)
        while todo or running:
            while todo and len(running) < a.parallel:
                i = todo.pop(0)
                f = jd / f"{tag}_{i:03d}.json"
                spec = json.dumps({**specs[i], "national": nat_order, "test": test_order,
                                   "shape": [len(meta), int(offs[-1])], "selection": sha256(out / "selection_used.csv")})
                lg = f.with_suffix(".log")
                if f.exists() and f.read_text() == spec and lg.exists() and lg.read_text().rstrip().endswith("job done"):
                    z = np.load(f.with_suffix(".npz"))  # same job finished in an earlier run into this --out
                    res[i] = {"real": z["real"], "fix": z["fix"]}
                    continue
                f.write_text(spec)
                with open(lg, "w") as lf:
                    running[i] = (subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *raw_argv,
                                                    "--job", str(f)], stdout=lf, stderr=subprocess.STDOUT), f)
            time.sleep(1)
            for i, (p, f) in list(running.items()):
                if p.poll() is None:
                    continue
                if p.returncode:
                    for q_, _ in running.values():
                        q_.kill()
                    sys.exit(f"{f.name} failed:\n" + f.with_suffix(".log").read_text()[-3000:])
                z = np.load(f.with_suffix(".npz"))
                res[i] = {"real": z["real"], "fix": z["fix"]}
                del running[i]
                if sum(r is not None for r in res) % 5 == 0:
                    log(f"{tag}: {sum(r is not None for r in res)} of {len(specs)} fits done")
        return res

    def full(r, kind):
        v = np.zeros(len(meta))
        v[pred_rows] = r[kind]
        return v

    # ---- BARC-only model (the official seeds), the out-of-fold fits for the choice of w and the BARC guard, in one batch
    oof_specs = [(w, k, s) for w in w_grid[1:] for k in range(n_folds) for s in choice_seeds]
    tr_years = sorted(int(v) for v in meta.year[tr].unique())
    guard_specs = [(w, yr, s) for w in w_grid for yr in tr_years for s in choice_seeds]
    res = run_jobs([{"seed": s, "w": 0.0, "bags": []} for s in seeds] +
                   [{"seed": s, "w": w, "bags": list(ct.index[ct.fold != k])} for w, k, s in oof_specs] +
                   [{"seed": s, "w": w, "bags": list(ct.index) if w > 0 else [], "drop_year": yr}
                    for w, yr, s in guard_specs], "fit")
    p_real = np.mean([full(r, "real") for r in res[:len(seeds)]], 0)
    p_fix_seed = {s: full(r, "fix") for s, r in zip(seeds, res[:len(seeds)])}
    p_fix = np.mean([p_fix_seed[s] for s in seeds], 0)
    log("BARC-only and out-of-fold fits done")
    ref = pd.read_csv(src / "test_predictions.csv")
    chk = meta.loc[te_barc, ["field_id", "year"]].assign(p=p_real[te_barc]).merge(
        ref[["field_id", "year", "p_finetune"]], on=["field_id", "year"], validate="1:1")
    checks = {"barc_max_abs_diff": float((chk.p - chk.p_finetune).abs().max()),
              "barc_same_calls": f"{int(((chk.p >= 0.5) == (chk.p_finetune >= 0.5)).sum())}/{len(chk)}"}
    if a.quebec_official:
        qref = pd.read_csv(Path(a.quebec_official) / "test_predictions.csv")
        qc = meta.loc[q_rows, ["field_id", "year"]].assign(p=p_real[q_rows]).merge(
            qref[["field_id", "year", "p_presto"]], on=["field_id", "year"], validate="1:1")
        checks.update(quebec_max_abs_diff=float((qc.p - qc.p_presto).abs().max()),
                      quebec_same_calls=f"{int(((qc.p >= 0.5) == (qc.p_presto >= 0.5)).sum())}/{len(qc)}")
    log("checks", checks)
    early = ct.join(county_scores(nat_meta, p_real[nat_rows], "barc"))
    early.round(4).to_csv(out / "county_predictions_barc_only.csv")
    log("BARC-only on the sampled counties:", county_summary(early, "barc"),
        f"national share {national_share(early, pop, 'barc'):.3f}\n" + class_matrix(early, "barc").to_string())

    # ---- choice of w, out of fold by state
    p_oof = {0.0: np.mean([p_fix_seed[s] for s in choice_seeds], 0)}
    p_oof.update({w: np.zeros(len(meta)) for w in w_grid[1:]})
    for (w, k, s), r in zip(oof_specs, res[len(seeds):len(seeds) + len(oof_specs)]):
        ho = np.concatenate([rows_of[f] for f in ct.index[ct.fold == k]])
        p_oof[w][ho] += full(r, "fix")[ho] / len(choice_seeds)
    p_guard = {w: np.zeros(len(meta)) for w in w_grid}
    for (w, yr, s), r in zip(guard_specs, res[len(seeds) + len(oof_specs):]):
        ho = tr[meta.year.to_numpy()[tr] == yr]
        p_guard[w][ho] += full(r, "fix")[ho] / len(choice_seeds)
    for w in w_grid[1:]:  # out of fold exists for the sampled counties only
        p_oof[w][np.setdiff1d(np.arange(len(meta)), nat_rows)] = np.nan
    y_tr = y[tr]
    choice = []
    for w in w_grid:
        cs_ = county_scores(nat_meta, p_oof[w][nat_rows], "m").join(ct)
        pg = np.clip(p_guard[w][tr], 1e-6, 1 - 1e-6)
        choice.append({"w": w, **county_summary(cs_, "m"),
                       "BARC 2017-2020 held-out year, correct": int(((pg >= 0.5) == (y_tr == 1)).sum()),
                       "BARC held-out year, log loss": round(float(-np.mean(y_tr * np.log(pg) +
                                                                            (1 - y_tr) * np.log(1 - pg))), 4)})
    choice = pd.DataFrame(choice)
    guard_col = "BARC 2017-2020 held-out year, correct"
    ok = choice[choice[guard_col] >= int(choice[guard_col][choice.w == 0].iloc[0]) - GUARD_FIELDS]
    w_best = float(ok.w[ok["gap called (mean abs)"].astype(float).idxmin()])
    log("choice\n" + choice.to_string())

    # ---- final proportion-trained model
    if w_best == 0:
        p_llp = p_fix.copy()
    else:
        res = run_jobs([{"seed": s, "w": w_best, "bags": list(ct.index)} for s in seeds], "final")
        p_llp = np.mean([full(r, "fix") for r in res], 0)
    log(f"final w {w_best:g} fitted")

    # ---- results
    ct = ct.join(county_scores(nat_meta, p_real[nat_rows], "barc")) \
           .join(county_scores(nat_meta, p_fix[nat_rows], "barc_ll")) \
           .join(county_scores(nat_meta, p_oof[w_best][nat_rows], "llp_oof")) \
           .join(county_scores(nat_meta, p_llp[nat_rows], "llp_in_sample"))
    ct["census_class"] = share_class(ct.census_share.to_numpy()).to_numpy()
    ct.round(4).to_csv(out / "county_predictions.csv")
    blocks = meta.loc[np.concatenate([nat_rows, test_rows])].assign(
        p_barc=p_real[np.concatenate([nat_rows, test_rows])], p_barc_ll=p_fix[np.concatenate([nat_rows, test_rows])],
        p_llp_oof=p_oof[w_best][np.concatenate([nat_rows, test_rows])],
        p_llp=p_llp[np.concatenate([nat_rows, test_rows])])
    blocks.drop(columns=["label"]).round(4).to_csv(out / "block_predictions.csv", index=False)
    fields = meta.loc[np.concatenate([te_barc, q_rows])].assign(
        p_barc=p_real[np.concatenate([te_barc, q_rows])], p_llp=p_llp[np.concatenate([te_barc, q_rows])])
    fields.round(4).to_csv(out / "field_predictions.csv", index=False)

    models = [("barc", "BARC-only (official setting)"), ("barc_ll", "BARC-only, BARC lat/lon"),
              ("llp_oof", f"Proportion-trained, w {w_best:g}, out of fold")]
    nat_tab = pd.DataFrame([{"model": lbl, **county_summary(ct, k),
                             "national silage share": f"{national_share(ct, pop, k):.3f}"} for k, lbl in models])
    cen_nat = float(pop.silage.sum() / (pop.grain + pop.silage).sum())

    tc_tab, tct = None, None
    if len(test_rows):
        import county_squares as csq  # copy of classifier/counties/county_squares.py, next to this file
        cen = csq.census_shares(a.census)
        tm = meta.loc[test_rows]
        tct = cen.loc[sorted(tm.fips.unique()), ["state_name", "county_name", "share"]].rename(
            columns={"share": "census_share"})
        tct = tct.join(county_scores(tm, p_real[test_rows], "barc")).join(county_scores(tm, p_llp[test_rows], "llp"))
        tct.round(4).to_csv(out / "test_county_predictions.csv")
        tc_tab = pd.DataFrame([{"model": lbl, **county_summary(tct, k)} for k, lbl in
                               [("barc", "BARC-only"), ("llp", f"Proportion-trained, w {w_best:g}")]])

    st_tab = None
    if a.state_ratios:
        sr = pd.read_csv(a.state_ratios)
        sr = sr[sr.year == 2022].assign(state=lambda d: d.state.str.title()).set_index("state")
        st = []
        for name, d in ct.groupby("state"):
            acres = d.grain_acres + d.silage_acres
            wm = lambda col: round(float((acres * d[col]).sum() / acres.sum()), 3)  # noqa: E731
            r = sr.loc[name] if name in sr.index else None
            st.append({"state": name, "counties": len(d), "sampled, Census": wm("census_share"),
                       "sampled, BARC-only": wm("barc_called"), "sampled, proportion-trained (out of fold)":
                       wm("llp_oof_called"), "state 2022, survey": None if r is None else r.silage_share_harvested,
                       "state 2022, Census": None if r is None else r.census_silage_share_harvested})
        us = sr.loc[["Us Total"]] if "Us Total" in sr.index else sr[sr.silage_acres.notna() & sr.grain_acres.notna()]
        st.append({"state": "US (stratified estimate; file: US total)", "counties": len(ct),
                   "sampled, Census": round(cen_nat, 3), "sampled, BARC-only": round(national_share(ct, pop, "barc"), 3),
                   "sampled, proportion-trained (out of fold)": round(national_share(ct, pop, "llp_oof"), 3),
                   "state 2022, survey": round(float(us.silage_acres.sum() / (us.silage_acres + us.grain_acres).sum()), 3),
                   "state 2022, Census": round(float(us.census_silage_acres.sum() /
                                                     (us.census_silage_acres + us.census_grain_acres).sum()), 3)})
        st_tab = pd.DataFrame(st)
        st_tab.to_csv(out / "state_check.csv", index=False)

    yb = (meta.label[te_barc] == "Silage").to_numpy(int)
    yq = (meta.label[q_rows] == "Silage").to_numpy(int)
    qw = square_weights(q, a.quebec_selection)
    field_tab = pd.DataFrame([field_row("BARC-only", yb, p_real[te_barc]),
                              field_row(f"Proportion-trained, w {w_best:g}", yb, p_llp[te_barc])])
    q_tab = pd.DataFrame([{**field_row("BARC-only", yq, p_real[q_rows]),
                           "accuracy at the square's share": f"{weighted(yq, p_real[q_rows], qw)['accuracy']:.2f}"},
                          {**field_row(f"Proportion-trained, w {w_best:g}", yq, p_llp[q_rows]),
                           "accuracy at the square's share": f"{weighted(yq, p_llp[q_rows], qw)['accuracy']:.2f}"}])

    runj = {"finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "runtime_s": round(time.time() - t0), "smoke": a.smoke, "args": vars(a),
            "setting": {"lr": lr, "epochs": epochs, "seeds": seeds, "last_month": last_month,
                        "choice_seeds": choice_seeds, "folds": n_folds, "w_grid": w_grid, "w_chosen": w_best, "guard_fields": GUARD_FIELDS,
                        "county_loss": {"counties": K_COUNTIES, "blocks": B_BLOCKS, "pixels": P_PIX},
                        "barc_mean_latlon": ll_barc.tolist()},
            "checks": checks, "counties_used": len(ct), "counties_missing_pixels": missing,
            "choice": choice.to_dict("records"), "inputs": {"field_years.csv": sha256(a.field_years),
            "selection.csv": sha256(out / "selection_used.csv"), "quebec": sha256(a.quebec),
            **({"state_ratios": sha256(a.state_ratios)} if a.state_ratios else {}), **used_hashes},
            "versions": {"python": platform.python_version(), "torch": torch.__version__,
                         "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None}}
    (out / "run.json").write_text(json.dumps(runj, indent=1, default=str))

    L = ["# Presto across the US: county silage shares against the 2022 Census", "",
         ("**SMOKE RUN: 1 seed, 2 epochs, 2 folds. Not a result.**\n" if a.smoke else "") +
         f"Generated by `presto_national.py`. {len(ct)} sampled counties (100 corn blocks of 90 m each, 2022), "
         f"{len(missing)} left out for missing pixels. Classes by silage share of harvested corn acres: "
         f"{', '.join(CLASS_NAMES)}. A county's predicted share is the share of its blocks called silage (p >= 0.5).",
         "", "## Sampled counties", "", md_table(nat_tab), "",
         f"National silage share of corn in the sampled population (Census 2022): {cen_nat:.3f}.", ""]
    for k, lbl in models:
        L += [f"### County confusion matrix: {lbl}", "", md_table(class_matrix(ct, k)), ""]
    L += ["## Choice of the county-loss weight (out of fold by state)", "", md_table(choice), "",
          f"Chosen w = {w_best:g}: the lowest gap among the w whose BARC 2017-2020 held-out-year calls (each year "
          f"predicted by a model trained on the other three, with every sampled county) are at most {GUARD_FIELDS} "
          f"below w = 0's. w = 0 is the BARC-only model with BARC's latitude and longitude.", ""]
    if tc_tab is not None:
        L += ["## Test counties (never trained or chosen on)", "", md_table(tc_tab), "",
              md_table(tct.reset_index().round(3)), ""]
    if st_tab is not None:
        L += ["## States: sampled counties next to `state_silage_ratios.csv` (2022)", "",
              "Columns `sampled, ...`: corn-acre weighted silage share of the state's sampled counties (Census, and "
              "the share of blocks each model calls silage). The counties are drawn by share band, so they are not a "
              "proportional sample of the state; the state's own 2022 shares are context. The US row uses the "
              "stratified estimate.", "", md_table(st_tab), ""]
    L += ["## BARC 2021-2023 test field-years", "", md_table(field_tab), "",
          md_table(pd.DataFrame(confusion_rows("BARC-only", yb, p_real[te_barc]) +
                                confusion_rows(f"Proportion-trained, w {w_best:g}", yb, p_llp[te_barc]))), "",
          "## Québec field-years (no Québec label used)", "", md_table(q_tab), "",
          md_table(pd.DataFrame(confusion_rows("BARC-only", yq, p_real[q_rows]) +
                                confusion_rows(f"Proportion-trained, w {w_best:g}", yq, p_llp[q_rows]))), "",
          f"Checks: BARC-only refit vs the official BARC test predictions: largest difference "
          f"{checks['barc_max_abs_diff']:.4f}, same call {checks['barc_same_calls']}"
          + (f"; vs the official Québec run: {checks['quebec_max_abs_diff']:.4f}, {checks['quebec_same_calls']}."
             if "quebec_max_abs_diff" in checks else "."), ""]
    (out / "metrics.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)
    log("done")


if __name__ == "__main__":
    main()
