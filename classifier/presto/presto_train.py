"""Presto classifier for silage vs grain corn, one BARC field-year at a time (GPU if present, else CPU).

  python classifier/presto/presto_train.py --pixels /home/keeboh/work/presto/pixels \
      --presto-dir /home/keeboh/ext/presto --out classifier/presto/results/presto

Split: the same as classifier/prithvi/prithvi_train.py, so both models are scored on the same test parts. The
labeled field-years (--field-years) go to train, validation and test in a 70/15/15 ratio, with every year of one field
in the same part (StratifiedGroupKFold, 20 folds: 14 train, 3 validation, 3 test), once per --seeds value. Each
candidate is ranked on validation log loss only; the test part is scored once, for the chosen candidate. The 5-feature
logistic regression from research/grain_vs_silage is fitted on the same training part as the baseline. When the
Prithvi results folder holds a run on identical splits, its test predictions are shown alongside (read only).

Extra training rows (--extra-train, --extra-pixels, e.g. the Quebec field-years): added to the training part only,
never to validation or test, so the splits and the test rows are the same as without them. Each fine-tuning epoch
draws --extra-ratio x (training field-years) of them at random; the frozen head weighs them as that many. At most
EXTRA_MAX_PIX pixels are kept per extra field-year (a fixed random choice). The baseline is also refitted with the
extra rows that have hand features (p_baseline_extra; left out when none do); p_baseline stays the local-only baseline.

Model: Presto (Tseng et al. 2023, github.com/nasaharvest/presto, MIT), a small transformer pretrained on pixel time
series of Sentinel-1, Sentinel-2, ERA5 and SRTM. Input per pixel (presto_pixels.py): 12 monthly steps, January to
December; Sentinel-1 VV and VH as (dB + 25) / 25; Sentinel-2 B2-B8A, B11, B12 as reflectance; NDVI; the field
centroid's latitude and longitude. ERA5, SRTM and Dynamic World are masked. A month is used for a field-year when at
least half of its pixels have a clear reading; missing pixels in that month take the field-year's median, and
unused months are masked (Presto drops masked tokens). --last-month masks every later month, to test how much the
model relies on the end of the year.

Candidates:
  - frozen: pretrained encoder, 128-d pixel embeddings averaged per field-year, logistic regression head (C grid);
  - finetune: whole encoder plus a linear head trained on pixels (AdamW, weight decay 0.05, class-balanced loss, the
    same number of random pixels from every training field-year each epoch, early stopping on validation log loss).
    A field-year's probability is the mean over its pixels.
Control, never chosen: the same fine-tuning from random weights ("scratch"), to show what pretraining adds.
"""
import argparse
import copy
import hashlib
import json
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import binomtest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
SMALL = ["drop_doy", "ndvi_pre_drop", "drop_size", "ndvi_sep15", "ndti_post_drop"]
S2_ORDER = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
NORMED_BANDS = ["VV", "VH", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12",
                "temperature_2m", "total_precipitation", "elevation", "slope", "NDVI"]
T, D = 12, len(NORMED_BANDS)
MIN_FRAC = 0.5
DW_MISSING = 9
C_GRID = (0.001, 0.01, 0.1, 1.0)
EXTRA_MAX_PIX = 256
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- inputs

def field_year_input(s2, s1, last_month=12):
    """Presto x and mask (n_pix, 12, 17) for one field-year, and the months used for Sentinel-2 and Sentinel-1."""
    n = len(s2)
    x = np.zeros((n, T, D), np.float32)
    m = np.ones((n, T, D), np.float32)
    ok2, ok1 = np.isfinite(s2).all(2), np.isfinite(s1).all(2)
    in_range = np.arange(T) < last_month
    use2, use1 = (ok2.mean(0) >= MIN_FRAC) & in_range, (ok1.mean(0) >= MIN_FRAC) & in_range
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        s2 = np.where(ok2[..., None], s2, np.nanmedian(s2, 0)[None])
        s1 = np.where(ok1[..., None], s1, np.nanmedian(s1, 0)[None])
    x[:, use2, 2:12] = s2[:, use2]
    m[:, use2, 2:12] = 0
    x[:, use1, 0:2] = (s1[:, use1] + 25) / 25
    m[:, use1, 0:2] = 0
    nir, red = x[:, :, NORMED_BANDS.index("B8")], x[:, :, NORMED_BANDS.index("B4")]
    with np.errstate(all="ignore"):
        ndvi = np.where(nir + red > 0, (nir - red) / (nir + red), 0)
    x[:, use2, 16] = ndvi[:, use2]
    m[:, use2, 16] = 0
    return x, m, use2, use1


def masked_tokens(m, groups):
    """Tokens Presto drops for this mask (it needs the same count for every sample in a batch)."""
    return int(sum(m[:1 if g == "SRTM" else T][:, idx].max(1).sum() for g, idx in groups.items())) + T  # + DW


def field_inputs(fy, pixels_dir, groups, last_month=12, max_pix=None, skip=False):
    """Pixels of every field-year in fy order: X, M (N, 12, 17), LL (N, 2), offsets (n_fy + 1), key (n_fy),
    a coverage table, and the hashes of the pixel files. With skip, field-years without pixels or without a usable
    month are left out (the coverage table lists the ones kept) instead of stopping the run; with max_pix, at most
    that many pixels per field-year are kept, a fixed random choice."""
    by, hashes = {}, {}
    for f in sorted(Path(pixels_dir).glob("pixels_*.npz")):
        z = np.load(f)
        assert list(z["s2_bands"]) == S2_ORDER, z["s2_bands"]
        yr, off = int(f.stem.split("_")[1]), z["offsets"]
        for i, fid in enumerate(z["field_id"]):
            by[(int(fid), yr)] = (z["s2"][off[i]:off[i + 1]], z["s1"][off[i]:off[i + 1]])
        hashes[f.name] = sha256(f)
    missing = [(r.field_id, r.year) for r in fy.itertuples() if (r.field_id, r.year) not in by]
    if missing and not skip:
        sys.exit(f"{len(missing)} field-years have no pixels in {pixels_dir}, e.g. {missing[:5]}")
    rng = np.random.default_rng(0)
    X, M, LL, key, cov, offs = [], [], [], [], [], [0]
    for r in fy.itertuples():
        if (r.field_id, r.year) not in by:
            continue
        x, m, use2, use1 = field_year_input(*by[(r.field_id, r.year)], last_month)
        if not use2.any():
            if skip:
                continue
            sys.exit(f"field {r.field_id} {r.year}: no month with clear Sentinel-2 on half its pixels")
        if max_pix and len(x) > max_pix:
            sel = np.sort(rng.choice(len(x), max_pix, replace=False))
            x, m = x[sel], m[sel]
        X.append(x); M.append(m)
        LL.append(np.tile(np.float32([r.lat, r.lon]), (len(x), 1)))
        key.append(masked_tokens(m[0], groups))
        offs.append(offs[-1] + len(x))
        cov.append({"field_id": r.field_id, "year": r.year, "label": r.label, "n_pix": len(x),
                    "s2_months": int(use2.sum()), "s1_months": int(use1.sum()),
                    **{f"ndvi_{MONTHS[t]}": float(x[:, t, 16].mean()) if use2[t] else np.nan for t in range(T)}})
    return (np.concatenate(X), np.concatenate(M), np.concatenate(LL), np.array(offs), np.array(key),
            pd.DataFrame(cov), hashes)


def input_checks(cov):
    rows = []
    for t, mo in enumerate(MONTHS):
        r = {"month": mo}
        for lab in ("Grain", "Silage"):
            c = cov[cov.label == lab]
            r[f"{lab} median NDVI"] = round(float(c[f"ndvi_{mo}"].median()), 2)
            r[f"{lab} field-years with S2"] = f"{int(c[f'ndvi_{mo}'].notna().sum())}/{len(c)}"
        rows.append(r)
    return pd.DataFrame(rows)


def load_extra(path, pixels_dir, groups, last_month, fy):
    """Training-only field-years: their inputs (X, M, LL, offsets, key), the rows kept, the number left out, and the
    pixel-file hashes."""
    ex = pd.read_csv(path)
    clash = set(zip(ex.field_id, ex.year)) & set(zip(fy.field_id, fy.year))
    if clash:
        sys.exit(f"{len(clash)} extra field-years are also labeled field-years, e.g. {sorted(clash)[:3]}")
    X, M, LL, offs, key, cov, hashes = field_inputs(ex, pixels_dir, groups, last_month, EXTRA_MAX_PIX, skip=True)
    kept = ex.merge(cov[["field_id", "year"]], on=["field_id", "year"], how="inner")
    return (X, M, LL, offs, key), kept, len(ex) - len(kept), hashes


def stack_inputs(a, b):
    """Inputs b (X, M, LL, offsets, key) appended after inputs a."""
    return (np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1]]), np.concatenate([a[2], b[2]]),
            np.concatenate([a[3], a[3][-1] + b[3][1:]]), np.concatenate([a[4], b[4]]))


def train_weights(y, tr, extra, ratio):
    """Sample weights for the rows tr then extra: the extra rows together weigh ratio x len(tr), and the two classes
    weigh the same (as class_weight="balanced"). None without extra rows."""
    if not len(extra):
        return None
    rows = np.concatenate([tr, extra])
    w = np.concatenate([np.ones(len(tr)), np.full(len(extra), ratio * len(tr) / len(extra))])
    total = w.sum()
    for c in (0, 1):
        w[y[rows] == c] *= total / (2 * w[y[rows] == c].sum())
    return w


def fit_weighted(clf, feats, y, tr, extra=(), ratio=1.0):
    """clf (a pipeline ending in LogisticRegression) fitted on rows tr of feats, plus the extra rows when given."""
    w = train_weights(y, tr, np.asarray(extra, int), ratio)
    if w is None:
        return clf.fit(feats[tr], y[tr])
    rows = np.concatenate([tr, extra])
    clf.set_params(logisticregression__class_weight=None)
    return clf.fit(feats[rows], y[rows], logisticregression__sample_weight=w)


def epoch_rows(tr, extra, ratio, rng):
    """Field-years drawn for one training epoch: all of tr, plus ratio x len(tr) of the extra rows at random."""
    n = min(len(extra), int(round(ratio * len(tr))))
    return np.concatenate([tr, rng.choice(extra, n, replace=False)]) if n else tr


def pixel_weights(ypix, y, rows):
    """Class-balanced weight of every pixel for the field-years in rows."""
    n_pos = int(y[rows].sum())
    return torch.where(ypix == 1, len(rows) / (2 * n_pos), len(rows) / (2 * (len(rows) - n_pos)))


# ---------------------------------------------------------------- model

def load_presto(presto_dir, pretrained, seed=0):
    sys.path.insert(0, str(presto_dir))
    import single_file_presto as sfp  # MIT, nasaharvest/presto
    assert list(sfp.BANDS_GROUPS_IDX) == ["S1", "S2_RGB", "S2_Red_Edge", "S2_NIR_10m", "S2_NIR_20m", "S2_SWIR",
                                          "ERA5", "SRTM", "NDVI"] and sfp.NUM_DYNAMIC_WORLD_CLASSES == DW_MISSING
    torch.manual_seed(seed)
    model = sfp.Presto.construct()
    if pretrained:
        model.load_state_dict(torch.load(Path(presto_dir) / "data" / "default_model.pt", map_location="cpu"))
    return model.encoder, sfp.BANDS_GROUPS_IDX


class PixelClassifier(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc
        self.head = nn.Linear(enc.embedding_size, 1)

    def embed(self, x, m, ll):
        dw = torch.full(x.shape[:2], DW_MISSING, dtype=torch.long, device=x.device)
        return self.enc(x=x, dynamic_world=dw, latlons=ll, mask=m, month=0, eval_task=True)

    def forward(self, x, m, ll):
        return self.head(self.embed(x, m, ll)).squeeze(-1)


def batches(idx, key_pix, bs, rng=None):
    """Index arrays of at most bs pixels, each sharing one masked-token count; shuffled when rng is given."""
    out, k = [], key_pix[idx]
    for v in np.unique(k):
        sel = idx[k == v]
        if rng is not None:
            sel = rng.permutation(sel)
        out += [sel[i:i + bs] for i in range(0, len(sel), bs)]
    if rng is not None:
        out = [out[i] for i in rng.permutation(len(out))]
    return out


@torch.no_grad()
def field_outputs(model, data, rows, fn, bs=8192):
    """Per-field-year mean of fn(model, x, m, ll) over its pixels, for rows of fy."""
    X, M, LL, offs, key_pix = data
    model.eval()
    pix = np.concatenate([np.arange(offs[r], offs[r + 1]) for r in rows])
    owner = np.repeat(np.arange(len(rows)), [offs[r + 1] - offs[r] for r in rows])
    out = None
    pos = np.empty(offs[-1], np.int64)
    pos[pix] = np.arange(len(pix))
    for b in batches(pix, key_pix, bs):
        bt = torch.from_numpy(b).to(X.device)
        v = fn(model, X[bt], M[bt], LL[bt]).float().cpu().numpy()
        if out is None:
            out = np.zeros((len(pix),) + v.shape[1:], np.float32)
        out[pos[b]] = v
    s = np.zeros((len(rows),) + out.shape[1:])
    np.add.at(s, owner, out)
    return s / np.bincount(owner)[(slice(None),) + (None,) * (out.ndim - 1)]


def predict(model, data, rows):
    return field_outputs(model, data, rows, lambda mdl, x, m, ll: torch.sigmoid(mdl(x, m, ll)))


def finetune(enc, data, y, tr, va, device, seed, lr, epochs, patience, extra=(), ratio=1.0, ppf=64, bs=256,
             wd=0.05):
    X, M, LL, offs, key_pix = data
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = PixelClassifier(copy.deepcopy(enc)).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    owner = np.repeat(np.arange(len(offs) - 1), np.diff(offs))
    ypix = torch.from_numpy(y[owner].astype(np.float32)).to(device)
    best, best_state, best_ep, since = np.inf, None, 0, 0
    for ep in range(1, epochs + 1):
        model.train()
        rows = epoch_rows(tr, extra, ratio, rng)
        wpix = pixel_weights(ypix, y, rows)
        idx = np.concatenate([rng.integers(offs[r], offs[r + 1], ppf) for r in rows])
        for b in batches(idx, key_pix, bs, rng):
            bt = torch.from_numpy(b).to(device)
            loss = (F.binary_cross_entropy_with_logits(model(X[bt], M[bt], LL[bt]), ypix[bt], reduction="none")
                    * wpix[bt]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        vl = log_loss(y[va], np.clip(predict(model, data, va), 1e-6, 1 - 1e-6), labels=[0, 1])
        if vl < best - 1e-4:
            best, best_ep, since = vl, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_ep, ep


# ---------------------------------------------------------------- scoring (same as classifier/prithvi/prithvi_train.py)

def score(y, p):
    pred = p >= 0.5
    tp, fn = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum())
    fp, tn = int((pred & (y == 0)).sum()), int((~pred & (y == 0)).sum())
    return dict(n=len(y), correct=tp + tn, silage_caught=f"{tp}/{tp + fn}", grain_called_silage=f"{fp}/{fp + tn}",
                auc=round(roc_auc_score(y, p), 3) if 0 < y.sum() < len(y) else np.nan,
                log_loss=round(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]), 3))


def make_logreg():  # identical to research/grain_vs_silage/grain_silage.py
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000))


def split(fy, y, seed):
    folds = list(StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed).split(fy, y, groups=fy.field_id))
    part = np.empty(len(fy), dtype=object)
    for f, (_, idx) in enumerate(folds):
        part[idx] = "test" if f < 3 else "validation" if f < 6 else "train"
    return part


def md_table(df):
    cells = [[str(c) for c in df.columns]] + [["" if pd.isna(v) else str(v) for v in r] for r in df.to_numpy()]
    return "\n".join(["| " + " | ".join(cells[0]) + " |", "|" + "---|" * len(cells[0])] +
                     ["| " + " | ".join(r) + " |" for r in cells[1:]])


def prithvi_predictions(prithvi_dir, splits):
    """Prithvi's test predictions when its run used exactly these splits, else None and the reason."""
    sp, tp = Path(prithvi_dir) / "splits.csv", Path(prithvi_dir) / "test_predictions.csv"
    if not (sp.exists() and tp.exists()):
        return None, f"no Prithvi run in {prithvi_dir}"
    theirs = pd.read_csv(sp)
    cols = [c for c in splits.columns if c.startswith("seed")]
    if not set(cols) <= set(theirs.columns):
        return None, "Prithvi run used other seeds"
    j = splits.merge(theirs, on=["field_id", "year"], suffixes=("", "_p"), how="left")
    if len(theirs) != len(splits) or any((j[c] != j[c + "_p"]).any() for c in cols):
        return None, "Prithvi run used different splits"
    return pd.read_csv(tp)[["seed", "field_id", "year", "p_prithvi"]], {"splits.csv": sha256(sp),
                                                                        "test_predictions.csv": sha256(tp)}


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field-years", default=str(HERE / "field_years.csv"))
    ap.add_argument("--pixels", required=True, help="folder written by presto_pixels.py")
    ap.add_argument("--presto-dir", required=True, help="nasaharvest/presto files: single_file_presto.py, data/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prithvi-results", default=str(HERE.parent / "prithvi" / "results" / "prithvi"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-4, 3e-4])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--last-month", type=int, default=12, choices=range(1, 13), help="mask later months (10 = October)")
    ap.add_argument("--extra-train", default=None, help="field-years CSV used for training only (e.g. Quebec)")
    ap.add_argument("--extra-pixels", default=None, help="folder written by presto_pixels.py for --extra-train")
    ap.add_argument("--extra-ratio", type=float, default=1.0, help="extra field-years per training field-year")
    a = ap.parse_args(argv)
    if bool(a.extra_train) != bool(a.extra_pixels):
        ap.error("--extra-train and --extra-pixels go together")
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    fy = pd.read_csv(a.field_years)
    y = (fy.label == "Silage").to_numpy(int)
    enc_pre, groups = load_presto(a.presto_dir, pretrained=True)
    X, M, LL, offs, key, cov, pixel_hashes = field_inputs(fy, a.pixels, groups, a.last_month)
    y_all, extra, ex, extra_note = y, np.array([], int), None, None
    if a.extra_train:
        ein, ex, dropped, ehash = load_extra(a.extra_train, a.extra_pixels, groups, a.last_month, fy)
        X, M, LL, offs, key = stack_inputs((X, M, LL, offs, key), ein)
        extra = np.arange(len(fy), len(fy) + len(ex))
        y_all = np.concatenate([y, (ex.label == "Silage").to_numpy(int)])
        pixel_hashes.update({f"extra/{k}": v for k, v in ehash.items()})
        extra_note = (f"{len(ex)} extra field-years from {Path(a.extra_train).name} ({int(y_all[extra].sum())} Silage; "
                      f"{dropped} left out for missing pixels or no usable month), training only, "
                      f"{a.extra_ratio:g} per training field-year")
        print(extra_note, flush=True)
    key_pix = np.repeat(key, np.diff(offs))
    data = (torch.from_numpy(X).to(device), torch.from_numpy(M).to(device), torch.from_numpy(LL).to(device),
            offs, key_pix)
    checks = input_checks(cov)
    print(f"{len(fy)} field-years, {cov.n_pix.sum()} pixels (median {int(cov.n_pix.median())} per field-year), "
          f"{len(np.unique(key))} masked-token counts, device {device}", flush=True)
    print(checks.to_string(index=False), flush=True)
    cov.drop(columns=[c for c in cov if c.startswith("ndvi_")]).to_csv(out / "coverage.csv", index=False)

    parts = {s: split(fy, y, s) for s in a.seeds}
    splits = pd.DataFrame({"field_id": fy.field_id, "year": fy.year, "label": fy.label,
                           **{f"seed{s}": p for s, p in parts.items()}})
    splits.to_csv(out / "splits.csv", index=False)

    all_rows = np.arange(len(offs) - 1)
    Z = field_outputs(PixelClassifier(enc_pre).to(device), data, all_rows,
                      lambda mdl, x, m, ll: mdl.embed(x, m, ll))
    val_rows, cand = [], {}  # cand: (seed, name) -> (val probs, test probs); test read only for the chosen ones
    for s, part in parts.items():
        tr, va, te = (np.flatnonzero(part == k) for k in ("train", "validation", "test"))
        for C in C_GRID:
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))
            fit_weighted(clf, Z, y_all, tr, extra, a.extra_ratio)
            name = f"frozen C={C}"
            pv, pt = clf.predict_proba(Z[va])[:, 1], clf.predict_proba(Z[te])[:, 1]
            cand[(s, name)] = (pv, pt)
            val_rows.append({"seed": s, "candidate": name, "mode": "frozen", "C": C, "lr": np.nan, "epoch": np.nan,
                             "epochs_run": np.nan, **{f"val_{k}": v for k, v in score(y[va], pv).items()}})
        for mode in ("finetune", "scratch"):
            for lr in a.lrs:
                enc = enc_pre if mode == "finetune" else load_presto(a.presto_dir, pretrained=False, seed=s)[0]
                model, ep, ran = finetune(enc, data, y_all, tr, va, device, seed=s, lr=lr, epochs=a.epochs,
                                          patience=a.patience, extra=extra, ratio=a.extra_ratio)
                name = f"{mode} lr={lr:g}"
                pv, pt = predict(model, data, va), predict(model, data, te)
                cand[(s, name)] = (pv, pt)
                val_rows.append({"seed": s, "candidate": name, "mode": mode, "C": np.nan, "lr": lr, "epoch": ep,
                                 "epochs_run": ran, **{f"val_{k}": v for k, v in score(y[va], pv).items()}})
                print(f"seed {s} {name}: best epoch {ep} of {ran}, val log loss {val_rows[-1]['val_log_loss']}, "
                      f"{time.time() - t0:.0f} s", flush=True)
                del model
    val = pd.DataFrame(val_rows)
    val.to_csv(out / "validation_scores.csv", index=False)

    def pick(v):
        return v.sort_values(["val_log_loss", "val_auc"], ascending=[True, False]).candidate.iloc[0]

    test_rows, chosen = [], {}
    S = pd.concat([fy[SMALL], ex[SMALL]] if ex is not None else [fy[SMALL]]).to_numpy(float)
    extra_hand = extra[np.isfinite(S[extra]).any(1)]  # extra rows with hand features; others would be imputed
    for s, part in parts.items():
        tr, te = np.flatnonzero(part == "train"), np.flatnonzero(part == "test")
        v = val[val.seed == s]
        chosen[s] = (pick(v[v["mode"] != "scratch"]), pick(v[v["mode"] == "scratch"]))
        pb = make_logreg().fit(fy.loc[tr, SMALL], y[tr]).predict_proba(fy.loc[te, SMALL])[:, 1]
        rows_s = (fy.loc[te, ["field_id", "farm", "field", "year", "label", "label_source"]]
                  .assign(seed=s, chosen=chosen[s][0], p_presto=cand[(s, chosen[s][0])][1].round(4),
                          p_baseline=pb.round(4), scratch_chosen=chosen[s][1],
                          p_scratch=cand[(s, chosen[s][1])][1].round(4)))
        if len(extra_hand):
            pe = fit_weighted(make_logreg(), S, y_all, tr, extra_hand, a.extra_ratio).predict_proba(S[te])[:, 1]
            rows_s["p_baseline_extra"] = pe.round(4)
        test_rows.append(rows_s)
    pred = pd.concat(test_rows, ignore_index=True)
    pp, prithvi_note = prithvi_predictions(a.prithvi_results, splits)
    if pp is not None:
        pred = pred.merge(pp, on=["seed", "field_id", "year"], how="left", validate="1:1")
    pred.to_csv(out / "test_predictions.csv", index=False)

    rows = []
    for s in a.seeds:
        q = pred[pred.seed == s]
        yq = (q.label == "Silage").to_numpy(int)
        ca, cb = (q.p_baseline >= 0.5).to_numpy() == yq, (q.p_presto >= 0.5).to_numpy() == yq
        only_p, only_b = int((cb & ~ca).sum()), int((ca & ~cb).sum())
        sp, sb = score(yq, q.p_presto.to_numpy()), score(yq, q.p_baseline.to_numpy())
        rows.append({"seed": s, "test field-years": len(q), "Silage in test": int(yq.sum()), "chosen": chosen[s][0],
                     "Presto correct": sp["correct"], "Presto AUC": sp["auc"],
                     "Presto silage caught": sp["silage_caught"], "baseline correct": sb["correct"],
                     "baseline AUC": sb["auc"], "only Presto right": only_p, "only baseline right": only_b,
                     "McNemar p": round(binomtest(only_p, only_p + only_b).pvalue, 3) if only_p + only_b else 1.0})
    res = pd.DataFrame(rows)
    ytest = (pred.label == "Silage").to_numpy(int)
    models = [("Presto (chosen on validation)", "p_presto"), ("5-feature logistic baseline", "p_baseline"),
              ("Presto architecture from random weights (control)", "p_scratch")]
    if "p_baseline_extra" in pred:
        models.insert(2, ("5-feature logistic baseline with the extra rows", "p_baseline_extra"))
    if "p_prithvi" in pred:
        models.append(("Prithvi (chosen on validation)", "p_prithvi"))
    tot = []
    for label, col in models:
        sc = score(ytest, pred[col].to_numpy())
        tot.append({"model": label, "correct": f"{sc['correct']}/{sc['n']}", "silage caught": sc["silage_caught"],
                    "grain called silage": sc["grain_called_silage"],
                    "mean AUC over seeds": round(float(np.mean([roc_auc_score((g.label == "Silage").astype(int), g[col])
                                                               for _, g in pred.groupby("seed")])), 3)})
    tot = pd.DataFrame(tot)
    top = (val[val["mode"] != "scratch"].sort_values("val_log_loss").groupby("seed").head(3)
           [["seed", "candidate", "epoch", "epochs_run", "val_log_loss", "val_auc", "val_correct", "val_n"]])

    w = Path(a.presto_dir) / "data" / "default_model.pt"
    commit = Path(a.presto_dir) / "COMMIT"
    run = {"finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "runtime_s": round(time.time() - t0), "args": vars(a), "min_month_frac": MIN_FRAC, "c_grid": C_GRID,
           "presto": {"repo": "https://github.com/nasaharvest/presto", "license": "MIT",
                      "commit": commit.read_text().strip() if commit.exists() else None,
                      "weights_sha256": sha256(w), "single_file_presto_sha256": sha256(Path(a.presto_dir) /
                                                                                       "single_file_presto.py")},
           "inputs": {"field_years.csv": sha256(a.field_years), **pixel_hashes,
                      **({"extra_train.csv": sha256(a.extra_train)} if a.extra_train else {})},
           "extra_train": extra_note, "extra_max_pix": EXTRA_MAX_PIX,
           "prithvi_comparison": prithvi_note,
           "versions": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                        "sklearn": __import__("sklearn").__version__}}
    (out / "run.json").write_text(json.dumps(run, indent=1, default=str))

    n_tr, n_va, n_te = ([int((p == k).sum()) for p in parts.values()] for k in ("train", "validation", "test"))
    lines = ["# Presto classifier: results", "",
             f"Generated by `presto_train.py`. Split: {len(fy)} labeled field-years ({fy.year.min()}-{fy.year.max()}, "
             f"{int((y == 0).sum())} Grain, {int(y.sum())} Silage, {fy.field_id.nunique()} fields) into "
             f"train/validation/test by field (train {min(n_tr)}-{max(n_tr)}, validation {min(n_va)}-{max(n_va)}, "
             f"test {min(n_te)}-{max(n_te)} field-years per seed), "
             + ("the same splits as the Prithvi classifier" if pp is not None else "the split procedure of the "
                "Prithvi classifier") + f". Monthly inputs used: January to {MONTHS[a.last_month - 1]}. Each "
             "seed chooses its candidate on validation log loss; the test part is scored once for that candidate "
             "and for the hand-feature logistic baseline fitted on the same training part. Correct counts use a "
             "0.5 cut." + (f" Extra training rows: {extra_note}; never in validation or test." if len(extra) else "")
             + (" The extra rows have no hand features, so the baseline is fitted without them."
                if len(extra) and not len(extra_hand) else ""),
             "",
             "## All seeds together", "", md_table(tot), "",
             "Each seed is a different split; a field-year can be in more than one seed's test part. Prithvi: "
             + ("its own run on these splits, read from its results folder." if pp is not None else prithvi_note
                + "."), "",
             "## Test results by split seed", "", md_table(res), "",
             "## Best candidates on validation (top 3 per seed)", "", md_table(top), "",
             f"## Input check (median over field-years of the field's mean NDVI; {cov.n_pix.sum()} pixels in all)", "",
             md_table(checks), ""]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
