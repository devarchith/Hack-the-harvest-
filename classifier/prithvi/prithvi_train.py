"""Prithvi-EO-2.0 classifier for silage vs grain corn, one BARC field-year at a time (run on a GPU).

  python classifier/prithvi/prithvi_train.py --chips /home/keeboh/work/prithvi/chips \
      --out classifier/prithvi/results/prithvi

Split: the 127 labeled field-years (2017-2023) go to train, validation and test in a 70/15/15 ratio, with every year
of one field in the same part (StratifiedGroupKFold, 20 folds: 14 train, 3 validation, 3 test), so a field seen in
training never reappears in validation or test. Each candidate (model size, input scale, frozen encoder with a
logistic head or the last two encoder blocks fine-tuned) is ranked on validation log loss only; the test part is
scored once, for the chosen candidate. The hand-feature logistic regression from research/grain_vs_silage
(5 features, same settings) is fitted on the same training part and scored on the same test part. The whole
procedure is repeated for each --seeds value, because a 19-field-year test part moves a lot between splits.

Input per field-year: a 48 x 48 pixel (1.44 km) chip at 30 m centred on the field (prithvi_chips.py), four frames
that are per-pixel medians of the cloud-free readings in four windows (Jul 1-31, Aug 10-Sep 5, Sep 6-Oct 5,
Oct 6-Nov 15), normalised with Prithvi's own band means and standard deviations, plus the year and mid-window
day-of-year of each frame and the field's latitude and longitude (Prithvi's time and location inputs). Encoder patch
tokens are averaged per frame, weighted by the share of each 16 x 16 patch inside the field; the four frame vectors
are concatenated and go to the head.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy.stats import binomtest
from shapely import make_valid
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
SMALL = ["drop_doy", "ndvi_pre_drop", "drop_size", "ndvi_sep15", "ndti_post_drop"]
WINDOWS = [("07-01", "07-31"), ("08-10", "09-05"), ("09-06", "10-05"), ("10-06", "11-15")]
REPOS = {"tiny": "ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL", "100M": "ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL",
         "300M": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL", "600M": "ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL"}
CHIP, RES, NODATA = 48, 30.0, -32768
CRS = "EPSG:32618"
C_GRID = (0.001, 0.01, 0.1, 1.0)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- inputs

def field_inputs(fy, chips_dir):
    """Frames (N, 6, T, CHIP, CHIP) in reflectance x 10000 (NaN where no clear reading), field share per pixel
    (N, CHIP, CHIP), time inputs (N, T, 2) and location inputs (N, 2)."""
    geoms = gpd.GeoSeries.from_wkt(fy.wkt, crs=CRS).apply(make_valid)
    T, half = len(WINDOWS), CHIP // 2
    X = np.full((len(fy), 6, T, CHIP, CHIP), np.nan, np.float32)
    M = np.zeros((len(fy), CHIP, CHIP), np.float32)
    TC = np.zeros((len(fy), T, 2), np.float32)
    for yr in sorted(fy.year.unique()):
        z = np.load(Path(chips_dir) / f"{yr}.npz")
        x, tr = z["x"], Affine(*z["transform"])
        doy = pd.to_datetime(z["dates"]).day_of_year.values
        win = [(pd.Timestamp(f"{yr}-{a}").day_of_year, pd.Timestamp(f"{yr}-{b}").day_of_year) for a, b in WINDOWS]
        for i in np.flatnonzero(fy.year.values == yr):
            g = geoms.iloc[i]
            c = g.centroid
            col, row = int((c.x - tr.c) // RES), int((tr.f - c.y) // RES)
            crop = x[:, :, row - half:row + half, col - half:col + half].astype(np.float32)
            if crop.shape[-2:] != (CHIP, CHIP):
                raise SystemExit(f"field {fy.field_id.iat[i]} {yr}: chip runs off the grid; rebuild with a larger --pad-m")
            crop[crop == NODATA] = np.nan
            for t, (a, b) in enumerate(win):
                sel = (doy >= a) & (doy <= b)
                if sel.any():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        X[i, :, t] = np.nanmedian(crop[sel], axis=0)
                TC[i, t] = (yr, (a + b) / 2)
            # Share of each 30 m pixel inside the field, from a 10 m rasterisation.
            x0, y0 = tr.c + (col - half) * RES, tr.f - (row - half) * RES
            fine = rasterize([(g, 1)], out_shape=(CHIP * 3, CHIP * 3), transform=Affine(RES / 3, 0, x0, 0, -RES / 3, y0),
                             fill=0, dtype="uint8")
            M[i] = fine.reshape(CHIP, 3, CHIP, 3).mean((1, 3))
    LC = fy[["lat", "lon"]].to_numpy(np.float32)
    return X, M, TC, LC


def input_checks(fy, X, M):
    """Plain-number checks that the frames carry the expected signal, written to the summary."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ndvi = (X[:, 3] - X[:, 2]) / (X[:, 3] + X[:, 2])  # (N, T, H, W)
        inside = np.where((M > 0.5)[:, None], ndvi, np.nan)  # pixels outside the field become NaN
        field_ndvi = np.nanmean(inside, axis=(2, 3))
        empty = (np.isfinite(inside).reshape(len(fy), len(WINDOWS), -1).sum(-1) == 0).mean(0)
    rows = []
    for lab in ("Grain", "Silage"):
        v = field_ndvi[fy.label.values == lab]
        rows.append({"label": lab, **{f"frame {t + 1} NDVI": round(float(np.nanmedian(v[:, t])), 3)
                                       for t in range(len(WINDOWS))}})
    rows.append({"label": "share of field-years with no clear reading",
                 **{f"frame {t + 1} NDVI": round(float(e), 3) for t, e in enumerate(empty)}})
    return pd.DataFrame(rows)


def normalise(X, cfg):
    mean = np.array(cfg["mean"], np.float32)[None, :, None, None, None]
    std = np.array(cfg["std"], np.float32)[None, :, None, None, None]
    return np.nan_to_num((X - mean) / std, nan=0.0).astype(np.float32)


# ---------------------------------------------------------------- model

def load_encoder(size, n_frames, img_px, device):
    from huggingface_hub import hf_hub_download, list_repo_files
    repo = REPOS[size]
    ckpt = next(f for f in list_repo_files(repo) if f.endswith(".pt"))
    try:
        code = hf_hub_download(repo, "prithvi_mae.py")
    except Exception:  # some sizes ship weights only; the model code is shared
        code = hf_hub_download(REPOS["300M"], "prithvi_mae.py")
    cfg = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())["pretrained_cfg"]
    weights = hf_hub_download(repo, ckpt)
    spec = importlib.util.spec_from_file_location(f"prithvi_mae_{size}", code)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    enc = mod.PrithviViT(img_size=img_px, patch_size=cfg["patch_size"], num_frames=n_frames, in_chans=cfg["in_chans"],
                         embed_dim=cfg["embed_dim"], depth=cfg["depth"], num_heads=cfg["num_heads"],
                         mlp_ratio=cfg.get("mlp_ratio", 4.0), coords_encoding=cfg.get("coords_encoding"),
                         coords_scale_learn=cfg.get("coords_scale_learn", False))
    sd = torch.load(weights, map_location="cpu", weights_only=True)
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    sd = {k.removeprefix("encoder."): v for k, v in sd.items()
          if not k.startswith("decoder") and "pos_embed" not in k and not k.startswith("mask_token")}
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    if set(missing) - {"pos_embed"} or unexpected:
        raise SystemExit(f"{size}: weights do not match the encoder: missing {missing[:5]}, unexpected {unexpected[:5]}")
    info = {"repo": repo, "checkpoint": ckpt, "sha256": sha256(weights), "embed_dim": cfg["embed_dim"],
            "depth": cfg["depth"], "params_m": round(sum(p.numel() for p in enc.parameters()) / 1e6, 1)}
    return enc.to(device).eval(), cfg, info


class FieldClassifier(nn.Module):
    def __init__(self, enc, n_frames, scale, dropout=0.2):
        super().__init__()
        self.enc, self.T, self.scale = enc, n_frames, scale
        d = n_frames * enc.embed_dim
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, 1))

    def embed(self, x, m, tc, lc):
        if self.scale != 1:
            b, c, t, h, w = x.shape
            x = F.interpolate(x.reshape(b, c * t, h, w), scale_factor=self.scale, mode="bilinear",
                              align_corners=False).reshape(b, c, t, h * self.scale, w * self.scale)
            m = F.interpolate(m[:, None], scale_factor=self.scale, mode="bilinear", align_corners=False)[:, 0]
        tok = self.enc.forward_features(x, temporal_coords=tc, location_coords=lc)[-1][:, 1:]
        b, n = x.shape[0], (x.shape[-2] // 16) * (x.shape[-1] // 16)
        tok = tok.reshape(b, self.T, n, -1)
        wt = F.avg_pool2d(m[:, None], 16).reshape(b, 1, n, 1)
        return ((tok * wt).sum(2) / wt.sum(2).clamp_min(1e-6)).flatten(1)

    def forward(self, x, m, tc, lc):
        return self.head(self.embed(x, m, tc, lc)).squeeze(1)


def dihedral(x, m, k):
    """The k-th of the 8 rotations and flips, applied to frames (B, C, T, H, W) and field share (B, H, W)."""
    x, m = torch.rot90(x, k % 4, (-2, -1)), torch.rot90(m, k % 4, (-2, -1))
    return (x.flip(-1), m.flip(-1)) if k >= 4 else (x, m)


@torch.no_grad()
def embed_all(model, data, idx, device, bs=32):
    """Frozen-encoder features, averaged over the 8 rotations and flips."""
    X, M, TC, LC = data
    out = []
    for s in range(0, len(idx), bs):
        j = idx[s:s + bs]
        x, m = torch.from_numpy(X[j]).to(device), torch.from_numpy(M[j]).to(device)
        tc, lc = torch.from_numpy(TC[j]).to(device), torch.from_numpy(LC[j]).to(device)
        acc = 0
        for k in range(8):
            xk, mk = dihedral(x, m, k)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                acc = acc + model.embed(xk, mk, tc, lc).float()
        out.append((acc / 8).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def predict(model, data, idx, device, bs=32):
    X, M, TC, LC = data
    model.eval()
    out = []
    for s in range(0, len(idx), bs):
        j = idx[s:s + bs]
        x, m = torch.from_numpy(X[j]).to(device), torch.from_numpy(M[j]).to(device)
        tc, lc = torch.from_numpy(TC[j]).to(device), torch.from_numpy(LC[j]).to(device)
        acc = 0
        for k in range(8):
            xk, mk = dihedral(x, m, k)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                acc = acc + torch.sigmoid(model(xk, mk, tc, lc).float())
        out.append((acc / 8).cpu().numpy())
    return np.concatenate(out)


def finetune(model, data, y, tr, va, device, seed, blocks=2, epochs=40, bs=16, lr_enc=2e-5, lr_head=1e-3,
             patience=10):
    """Train the last `blocks` encoder blocks and the head; keep the epoch with the lowest validation log loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    enc = model.enc
    for p in enc.parameters():
        p.requires_grad = False
    train_enc = list(enc.blocks[-blocks:]) + [enc.norm] if blocks else []
    for mod in train_enc:
        for p in mod.parameters():
            p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True
    groups = [{"params": [p for mod in train_enc for p in mod.parameters()], "lr": lr_enc},
              {"params": list(model.head.parameters()), "lr": lr_head}]
    opt = torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=0.05)
    pos_weight = torch.tensor((y[tr] == 0).sum() / max(1, (y[tr] == 1).sum()), device=device)
    X, M, TC, LC = data
    best = (np.inf, 0, None)
    for ep in range(epochs):
        model.train()
        enc.eval()  # no dropout or drop-path in the encoder
        perm = np.random.permutation(tr)
        for s in range(0, len(perm), bs):
            j = perm[s:s + bs]
            x, m = torch.from_numpy(X[j]).to(device), torch.from_numpy(M[j]).to(device)
            x, m = dihedral(x, m, np.random.randint(8))
            tc, lc = torch.from_numpy(TC[j]).to(device), torch.from_numpy(LC[j]).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(x, m, tc, lc).float()
            loss = F.binary_cross_entropy_with_logits(logit, torch.from_numpy(y[j]).float().to(device),
                                                      pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        p = np.clip(predict(model, data, va, device), 1e-6, 1 - 1e-6)
        vl = log_loss(y[va], p, labels=[0, 1])
        if vl < best[0]:
            keep = {k: v.detach().clone() for k, v in model.state_dict().items()
                    if k.startswith("head") or any(k.startswith(f"enc.blocks.{len(enc.blocks) - i - 1}.")
                                                   for i in range(blocks)) or k.startswith("enc.norm")}
            best = (vl, ep + 1, keep)
        elif ep + 1 - best[1] >= patience:
            break
    model.load_state_dict(best[2], strict=False)
    return best[0], best[1]


# ---------------------------------------------------------------- scoring

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


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field-years", default=str(HERE / "field_years.csv"))
    ap.add_argument("--chips", required=True, help="folder written by prithvi_chips.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", nargs="+", default=["tiny", "100M", "300M", "600M"], choices=list(REPOS))
    ap.add_argument("--scales", type=int, nargs="+", default=[1, 2],
                    help="upsampling of the 30 m chip before the encoder; 2 makes one patch 240 m")
    ap.add_argument("--modes", nargs="+", default=["frozen", "finetune"], choices=["frozen", "finetune"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args(argv)
    t0 = time.time()
    device = "cuda"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    fy = pd.read_csv(a.field_years)
    y = (fy.label == "Silage").to_numpy(int)
    X, M, TC, LC = field_inputs(fy, a.chips)
    checks = input_checks(fy, X, M)
    print(checks.to_string(index=False), flush=True)
    parts = {s: split(fy, y, s) for s in a.seeds}
    pd.DataFrame({"field_id": fy.field_id, "year": fy.year, "label": fy.label,
                  **{f"seed{s}": p for s, p in parts.items()}}).to_csv(out / "splits.csv", index=False)

    val_rows, models_info, test_rows, chosen = [], {}, [], {}
    cand_preds = {}  # (seed, name) -> (val probs, test probs); test probs are read only for the chosen one
    for size in a.sizes:
        for scale in a.scales:
            enc, cfg, info = load_encoder(size, len(WINDOWS), CHIP * scale, device)
            models_info[size] = info
            base = {k: v.detach().clone() for k, v in enc.state_dict().items()}
            data = (normalise(X, cfg), M, TC, LC)
            model = FieldClassifier(enc, len(WINDOWS), scale).to(device)
            if "frozen" in a.modes:
                Z = embed_all(model, data, np.arange(len(fy)), device)
                for s, part in parts.items():
                    tr, va, te = (np.flatnonzero(part == k) for k in ("train", "validation", "test"))
                    for C in C_GRID:
                        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced",
                                                                                 max_iter=5000))
                        clf.fit(Z[tr], y[tr])
                        name = f"{size} x{scale} frozen C={C}"
                        pv, pt = clf.predict_proba(Z[va])[:, 1], clf.predict_proba(Z[te])[:, 1]
                        cand_preds[(s, name)] = (pv, pt)
                        val_rows.append({"seed": s, "candidate": name, "size": size, "scale": scale, "mode": "frozen",
                                         "C": C, "epoch": np.nan, **{f"val_{k}": v for k, v in score(y[va], pv).items()}})
            if "finetune" in a.modes:
                for s, part in parts.items():
                    tr, va, te = (np.flatnonzero(part == k) for k in ("train", "validation", "test"))
                    enc.load_state_dict(base)
                    model = FieldClassifier(enc, len(WINDOWS), scale).to(device)
                    vl, ep = finetune(model, data, y, tr, va, device, seed=s, epochs=a.epochs)
                    name = f"{size} x{scale} finetune last 2 blocks"
                    pv, pt = predict(model, data, va, device), predict(model, data, te, device)
                    cand_preds[(s, name)] = (pv, pt)
                    val_rows.append({"seed": s, "candidate": name, "size": size, "scale": scale, "mode": "finetune",
                                     "C": np.nan, "epoch": ep, **{f"val_{k}": v for k, v in score(y[va], pv).items()}})
                enc.load_state_dict(base)
            print(f"{size} x{scale} done, {time.time() - t0:.0f} s", flush=True)
            del enc, model
            torch.cuda.empty_cache()

    val = pd.DataFrame(val_rows)
    val.to_csv(out / "validation_scores.csv", index=False)
    for s, part in parts.items():
        tr, va, te = (np.flatnonzero(part == k) for k in ("train", "validation", "test"))
        v = val[val.seed == s].sort_values(["val_log_loss", "val_auc"], ascending=[True, False])
        name = v.candidate.iloc[0]
        chosen[s] = name
        pt = cand_preds[(s, name)][1]
        base_m = make_logreg().fit(fy.loc[tr, SMALL], y[tr])
        pb = base_m.predict_proba(fy.loc[te, SMALL])[:, 1]
        test_rows.append(fy.loc[te, ["field_id", "farm", "field", "year", "label", "label_source"]]
                         .assign(seed=s, chosen=name, p_prithvi=pt.round(4), p_baseline=pb.round(4)))
    pred = pd.concat(test_rows, ignore_index=True)
    pred.to_csv(out / "test_predictions.csv", index=False)

    rows = []
    for s in a.seeds:
        q = pred[pred.seed == s]
        yq = (q.label == "Silage").to_numpy(int)
        ca, cb = (q.p_baseline >= 0.5).to_numpy() == yq, (q.p_prithvi >= 0.5).to_numpy() == yq
        only_b, only_a = int((cb & ~ca).sum()), int((ca & ~cb).sum())
        sp, sb = score(yq, q.p_prithvi.to_numpy()), score(yq, q.p_baseline.to_numpy())
        rows.append({"seed": s, "test field-years": len(q), "Silage in test": int(yq.sum()), "chosen": chosen[s],
                     "Prithvi correct": sp["correct"], "Prithvi AUC": sp["auc"],
                     "Prithvi silage caught": sp["silage_caught"], "baseline correct": sb["correct"],
                     "baseline AUC": sb["auc"], "only Prithvi right": only_b, "only baseline right": only_a,
                     "McNemar p": round(binomtest(only_b, only_a + only_b).pvalue, 3) if only_a + only_b else 1.0})
    res = pd.DataFrame(rows)
    top = (val.sort_values("val_log_loss").groupby("seed").head(5)
           [["seed", "candidate", "epoch", "val_log_loss", "val_auc", "val_correct", "val_n"]])

    run = {"finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "runtime_s": round(time.time() - t0),
           "args": vars(a), "windows": WINDOWS, "chip_px": CHIP, "models": models_info,
           "inputs": {"field_years.csv": sha256(a.field_years),
                      **{p.name: sha256(p) for p in sorted(Path(a.chips).glob("*.npz"))}},
           "versions": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                        "sklearn": __import__("sklearn").__version__, "timm": __import__("timm").__version__}}
    (out / "run.json").write_text(json.dumps(run, indent=1, default=str))

    n_tr = [int((p == "train").sum()) for p in parts.values()]
    n_va = [int((p == "validation").sum()) for p in parts.values()]
    n_te = [int((p == "test").sum()) for p in parts.values()]
    lines = ["# Prithvi-EO-2.0 classifier: results", "",
             f"Generated by `prithvi_train.py`. Split: {len(fy)} labeled field-years ({fy.year.min()}-{fy.year.max()}, "
             f"{int((y == 0).sum())} Grain, {int(y.sum())} Silage, {fy.field_id.nunique()} fields) into "
             f"train/validation/test by field (train {min(n_tr)}-{max(n_tr)}, validation {min(n_va)}-{max(n_va)}, "
             f"test {min(n_te)}-{max(n_te)} field-years per seed). Each seed chooses its candidate on validation log "
             "loss; the test part is scored once for that candidate and for the hand-feature logistic baseline fitted "
             "on the same training part. Correct counts use a 0.5 cut.", "",
             "## Test results by split seed", "", md_table(res), "",
             f"Summed over seeds: Prithvi {int(res['Prithvi correct'].sum())} vs baseline "
             f"{int(res['baseline correct'].sum())} correct of {int(res['test field-years'].sum())} test field-years "
             "(each seed is a different split; a field-year can be in more than one seed's test part).", "",
             "## Best candidates on validation (top 5 per seed)", "", md_table(top), "",
             f"## Input check (median field NDVI per frame, all {len(fy)} field-years)", "",
             "Frames: " + "; ".join(f"{i + 1} = {a}..{b}" for i, (a, b) in enumerate(WINDOWS)), "",
             md_table(checks), ""]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
