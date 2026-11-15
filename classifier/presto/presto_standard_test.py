"""Presto on the standard test: train on the 2017-2020 farm records, score the 44 labeled 2021-2023 field-years once.

  python classifier/presto/presto_standard_test.py --pixels /home/keeboh/work/presto/pixels \
      --presto-dir /home/keeboh/ext/presto --out classifier/presto/results/standard_test

Same split and baseline as research/grain_vs_silage/ ("Train 2017-20, test 2021-23"). Every choice is made on
2017-2020 only, by holding out one year at a time (2017, 2018, 2019, 2020; 83 field-years, 19 Silage in the original
table); the 2021-2023 labels are read once, at the end. Inputs and model as presto_train.py. Extra training rows
(--extra-train, --extra-pixels, --extra-ratio) work as in presto_train.py: added to every training set, never held out
or tested.

Choice on 2017-2020:
  - frozen: logistic regression on the field-year mean embedding, C from the grid;
  - finetune: learning rate and number of epochs. Each held-out year is predicted after every epoch, the predictions
    are averaged over the training seeds (as in the final model), and the (learning rate, epochs) pair with the lowest
    pooled log loss is kept;
  - between the two, the lower pooled log loss.
Final model: the chosen setting fitted on all 2017-2020 field-years; for fine-tuning, one model per training seed,
trained for the chosen number of epochs, probabilities averaged. Both families are scored on the test; only the chosen
one was picked without the test labels. Before anything else runs, the 5-feature baseline must reproduce 68/83 on the
choice split and 39/44 on the test with the original table (REFERENCE; research/grain_vs_silage/results/
model_comparison.md); its counts with --field-years are then reported, not checked.
"""
import argparse
import copy
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from presto_train import (C_GRID, EXTRA_MAX_PIX, MONTHS, SMALL, PixelClassifier, batches,  # noqa: E402
                          epoch_rows, field_inputs, field_outputs, fit_weighted, load_extra, load_presto, make_logreg,
                          md_table, pixel_weights, predict, score, sha256, stack_inputs)

CHOICE_YEARS = (2017, 2018, 2019, 2020)
REFERENCE = HERE / "field_years.csv"  # the table research/grain_vs_silage scored
BASELINE = {"choice": 68, "test": 39}  # its published baseline counts on that table


def standard_split(fy):
    """Training rows (2017-2020), test rows (2021-2023), and the hold-one-year-out folds of the training rows."""
    tr = np.flatnonzero(fy.year.isin(CHOICE_YEARS))
    te = np.flatnonzero(fy.year >= 2021)
    return tr, te, [(tr[fy.year.iloc[tr] != yr], tr[fy.year.iloc[tr] == yr]) for yr in CHOICE_YEARS]


def baseline(fy):
    """The 5-feature baseline's held-out-year and test probabilities, and its correct counts."""
    y = (fy.label == "Silage").to_numpy(int)
    tr, te, folds = standard_split(fy)
    p_ho = np.concatenate([make_logreg().fit(fy.loc[f, SMALL], y[f]).predict_proba(fy.loc[h, SMALL])[:, 1]
                           for f, h in folds])
    p_te = make_logreg().fit(fy.loc[tr, SMALL], y[tr]).predict_proba(fy.loc[te, SMALL])[:, 1]
    ho = np.concatenate([h for _, h in folds])
    return p_ho, p_te, {"choice": score(y[ho], p_ho)["correct"], "test": score(y[te], p_te)["correct"]}


def train_curve(enc, data, y, tr, ho, device, seed, lr, epochs, extra=(), ratio=1.0, ppf=64, bs=256, wd=0.05):
    """Fine-tune as presto_train.finetune, for a fixed number of epochs; probabilities for ho after every epoch."""
    X, M, LL, offs, key_pix = data
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = PixelClassifier(copy.deepcopy(enc)).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    owner = np.repeat(np.arange(len(offs) - 1), np.diff(offs))
    ypix = torch.from_numpy(y[owner].astype(np.float32)).to(device)
    curve = np.zeros((epochs, len(ho)))
    for ep in range(epochs):
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
        curve[ep] = predict(model, data, ho)
    return curve


def frozen_head(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field-years", default=str(HERE / "field_years.csv"))
    ap.add_argument("--pixels", required=True, help="folder written by presto_pixels.py")
    ap.add_argument("--presto-dir", required=True, help="nasaharvest/presto files: single_file_presto.py, data/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="training seeds, averaged")
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-4, 3e-4])
    ap.add_argument("--epochs", type=int, default=60, help="longest fine-tuning considered")
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

    ref = baseline(pd.read_csv(REFERENCE))[2]
    if ref != BASELINE:
        sys.exit(f"baseline does not reproduce on {REFERENCE.name}: {ref} vs {BASELINE}")
    fy = pd.read_csv(a.field_years)
    y = (fy.label == "Silage").to_numpy(int)
    tr, te, folds = standard_split(fy)
    ho_all = np.concatenate([h for _, h in folds])
    y_ho, y_te = y[ho_all], y[te]

    def ll(yy, p):
        return log_loss(yy, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])

    pb_ho, pb_te, got = baseline(fy)
    print(f"baseline reproduced {ref} on {REFERENCE.name}; {len(tr)} training field-years, {len(te)} test "
          f"field-years in {Path(a.field_years).name}, baseline {got}", flush=True)

    enc, groups = load_presto(a.presto_dir, pretrained=True)
    X, M, LL, offs, key, cov, pixel_hashes = field_inputs(fy, a.pixels, groups, a.last_month)
    y_all, extra, extra_note, pbe_te = y, np.array([], int), None, None
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
        S = pd.concat([fy[SMALL], ex[SMALL]]).to_numpy(float)
        extra_hand = extra[np.isfinite(S[extra]).any(1)]  # extra rows with hand features; others would be imputed
        if len(extra_hand):
            pbe_ho = np.concatenate([fit_weighted(make_logreg(), S, y_all, f, extra_hand, a.extra_ratio)
                                     .predict_proba(S[h])[:, 1] for f, h in folds])
            pbe_te = fit_weighted(make_logreg(), S, y_all, tr, extra_hand, a.extra_ratio).predict_proba(S[te])[:, 1]
        else:
            extra_note += "; no hand features, so the baseline is fitted without them"
    data = (torch.from_numpy(X).to(device), torch.from_numpy(M).to(device), torch.from_numpy(LL).to(device),
            offs, np.repeat(key, np.diff(offs)))

    rows = []
    Z = field_outputs(PixelClassifier(enc).to(device), data, np.arange(len(offs) - 1),
                      lambda mdl, x, m, ll_: mdl.embed(x, m, ll_))
    for C in C_GRID:
        p = np.concatenate([fit_weighted(frozen_head(C), Z, y_all, f, extra, a.extra_ratio).predict_proba(Z[h])[:, 1]
                            for f, h in folds])
        rows.append({"family": "frozen", "setting": f"C={C}", "C": C, "lr": np.nan, "epochs": np.nan,
                     **score(y_ho, p), "log_loss": round(ll(y_ho, p), 4)})
    for lr in a.lrs:
        curves = [np.concatenate([train_curve(enc, data, y_all, f, h, device, s, lr, a.epochs, extra, a.extra_ratio)
                                  for f, h in folds], 1) for s in a.seeds]
        mean = np.mean(curves, 0)
        for ep in range(a.epochs):
            rows.append({"family": "finetune", "setting": f"lr={lr:g}, {ep + 1} epochs", "C": np.nan, "lr": lr,
                         "epochs": ep + 1, **score(y_ho, mean[ep]), "log_loss": round(ll(y_ho, mean[ep]), 4)})
        print(f"choice curves lr={lr:g} done, {time.time() - t0:.0f} s", flush=True)
    choice = pd.DataFrame(rows)
    choice.to_csv(out / "choice_scores.csv", index=False)
    best = {fam: g.sort_values(["log_loss", "auc"], ascending=[True, False]).iloc[0]
            for fam, g in choice.groupby("family")}
    chosen = min(best, key=lambda fam: (best[fam].log_loss, -best[fam].auc))

    p_frozen = (fit_weighted(frozen_head(best["frozen"].C), Z, y_all, tr, extra, a.extra_ratio)
                .predict_proba(Z[te])[:, 1])
    bf = best["finetune"]
    p_finetune = np.mean([train_curve(enc, data, y_all, tr, te, device, s, bf.lr, int(bf.epochs), extra,
                                      a.extra_ratio)[-1] for s in a.seeds], 0)
    p_presto = p_frozen if chosen == "frozen" else p_finetune
    pred = fy.loc[te, ["field_id", "farm", "field", "year", "label", "label_source"]].assign(
        p_presto=p_presto.round(4), p_frozen=p_frozen.round(4), p_finetune=p_finetune.round(4),
        p_baseline=pb_te.round(4))
    if pbe_te is not None:
        pred["p_baseline_extra"] = pbe_te.round(4)
    pred.to_csv(out / "test_predictions.csv", index=False)

    ca, cb = (pb_te >= 0.5) == y_te, (p_presto >= 0.5) == y_te
    only_p, only_b = int((cb & ~ca).sum()), int((ca & ~cb).sum())
    mcnemar = round(binomtest(only_p, only_p + only_b).pvalue, 3) if only_p + only_b else 1.0
    cols = ["n", "correct", "silage_caught", "grain_called_silage", "auc"]
    ch = pd.DataFrame([{"model": "5-feature logistic baseline", "setting": "C=0.5", **score(y_ho, pb_ho),
                        "log_loss": round(ll(y_ho, pb_ho), 4)}] +
                      [{"model": f"Presto {fam}" + (" (chosen)" if fam == chosen else ""),
                        "setting": best[fam].setting, **{c: best[fam][c] for c in cols},
                        "log_loss": best[fam].log_loss} for fam in ("frozen", "finetune")])
    names = {"frozen": f"Presto frozen, {best['frozen'].setting}", "finetune": f"Presto fine-tuned, {bf.setting}"}
    tt = pd.DataFrame([{"model": names[chosen] + " (chosen)", **score(y_te, p_presto)},
                       {"model": names["finetune" if chosen == "frozen" else "frozen"] + " (not chosen)",
                        **score(y_te, p_frozen if chosen == "finetune" else p_finetune)},
                       {"model": "5-feature logistic baseline", **score(y_te, pb_te)}])[["model"] + cols]
    if pbe_te is not None:
        name = "5-feature logistic baseline with the extra rows"
        ch.loc[len(ch)] = {"model": name, "setting": "C=0.5", **score(y_ho, pbe_ho),
                           "log_loss": round(ll(y_ho, pbe_ho), 4)}
        tt.loc[len(tt)] = {"model": name, **{c: score(y_te, pbe_te)[c] for c in cols}}
    top = choice[choice.family == "finetune"].sort_values("log_loss").head(5)[
        ["setting", "correct", "silage_caught", "grain_called_silage", "auc", "log_loss"]]

    w = Path(a.presto_dir) / "data" / "default_model.pt"
    commit = Path(a.presto_dir) / "COMMIT"
    run = {"finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "runtime_s": round(time.time() - t0), "args": vars(a), "chosen": chosen,
           "best": {fam: {k: (v.item() if hasattr(v, "item") else v) for k, v in b.items()} for fam, b in best.items()},
           "presto": {"commit": commit.read_text().strip() if commit.exists() else None, "weights_sha256": sha256(w)},
           "baseline_reference": {"table": REFERENCE.name, "expected": BASELINE, "got": ref},
           "baseline_this_table": got,
           "inputs": {"field_years.csv": sha256(a.field_years), **pixel_hashes,
                      **({"extra_train.csv": sha256(a.extra_train)} if a.extra_train else {})},
           "extra_train": extra_note, "extra_max_pix": EXTRA_MAX_PIX,
           "versions": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                        "sklearn": __import__("sklearn").__version__}}
    (out / "run.json").write_text(json.dumps(run, indent=1, default=str))

    lines = ["# Presto on the standard test: train 2017-2020, test 2021-2023", "",
             "Generated by `presto_standard_test.py`. Monthly inputs used: "
             f"{MONTHS[0]} to {MONTHS[a.last_month - 1]}. Everything is chosen on 2017-2020 only, holding out one "
             "year at a time; the 2021-2023 labels are read once. Fine-tuned predictions are the mean over training "
             f"seeds {', '.join(map(str, a.seeds))}. Correct counts use a 0.5 cut. Labels: "
             f"`{Path(a.field_years).name}`; before this run, the baseline reproduced {BASELINE['choice']} and "
             f"{BASELINE['test']} correct on `{REFERENCE.name}`."
             + (f" Extra training rows: {extra_note}; never held out or tested." if len(extra) else ""), "",
             f"## Choice on 2017-2020 (hold out one year; {len(tr)} field-years, {int(y[tr].sum())} Silage)", "",
             md_table(ch), "", f"Chosen: Presto {chosen}, {best[chosen].setting}.", "",
             f"## Test: the {len(te)} labeled 2021-2023 field-years ({int(y_te.sum())} Silage)", "",
             md_table(tt), "",
             f"Chosen Presto vs baseline on the field-years where they disagree: only Presto right {only_p}, only "
             f"baseline right {only_b}, exact McNemar p = {mcnemar}.", "",
             "## Best fine-tuning settings on the choice split", "", md_table(top), ""]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
