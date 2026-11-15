"""Presto tested outside Maryland: trained on BARC only, scored once on the Québec field-years.

  python classifier/presto/presto_quebec_test.py --pixels /home/keeboh/work/presto/pixels \
      --quebec-pixels /home/keeboh/work/quebec/pixels --presto-dir /home/keeboh/ext/presto \
      --setting-from classifier/presto/results/standard_test_fixed_labels --out classifier/presto/results/quebec_test

The test, fixed before any Québec prediction was made:
  - Model: the one chosen for the official test by presto_standard_test.py, with no new choice. Fine-tuned Presto with
    the learning rate, number of epochs, seeds and months stored in <--setting-from>/run.json, trained on the
    2017-2020 BARC field-years of --field-years (the same table, checked by its SHA-256), probabilities averaged over
    the seeds. The same run also predicts the 44 BARC 2021-2023 field-years; they must match that run's predictions.
  - Test: every Québec field-year with a usable month (../quebec/field_years_quebec.csv), scored once at a 0.5 cut.
    Québec labels are never used to train or choose anything. A field-year's probability is the mean over at most 256
    of its pixels (a fixed random choice, as for training-only rows).
  - Reported: AUC, balanced accuracy, silage and grain recall, which do not depend on the silage share, with 95%
    intervals from resampling whole parcels. Accuracy and silage precision are also reweighted to the silage share of
    the 30 km square the sample was drawn from (grain parcels in the square / grain parcels kept, per year; counts from
    quebec_selection.txt).
  - Reference: calling every field Grain. The 5-feature baseline cannot be scored here: Québec has no hand features.
"""
import argparse
import json
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "metrics"))
from metrics import cluster_bootstrap, point_metrics  # noqa: E402
from presto_standard_test import CHOICE_YEARS, train_curve  # noqa: E402
from presto_train import (EXTRA_MAX_PIX, MONTHS, field_inputs, load_extra, load_presto, md_table,  # noqa: E402
                          sha256, stack_inputs)

QUEBEC = HERE.parent / "quebec"


def square_weights(q, selection):
    """Per Québec row: parcels of its year and class in the square / parcels kept, so the classes count as in the
    square."""
    counts = {(int(yr), lab): (int(n), int(k)) for yr, lab, n, k in re.findall(
        r"^(\d{4}) (Grain|Silage): (\d+) parcels in the square, (\d+) kept$", Path(selection).read_text(), re.M)}
    return np.array([counts[(r.year, r.label)][0] / counts[(r.year, r.label)][1] for r in q.itertuples()])


def weighted(y, p, w):
    pred = p >= 0.5
    return {"accuracy": float((w * (pred == y)).sum() / w.sum()),
            "silage_precision": float((w * (pred & (y == 1))).sum() / (w * pred).sum()) if pred.any() else np.nan,
            "silage_share": float((w * y).sum() / w.sum()), "silage_share_called": float((w * pred).sum() / w.sum())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field-years", default=str(HERE.parent / "labels" / "results" / "field_years_fixed.csv"))
    ap.add_argument("--quebec", default=str(QUEBEC / "field_years_quebec.csv"))
    ap.add_argument("--selection", default=str(QUEBEC / "quebec_selection.txt"))
    ap.add_argument("--pixels", required=True, help="BARC folder written by presto_pixels.py")
    ap.add_argument("--quebec-pixels", required=True, help="Québec folder written by presto_pixels.py")
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
    X, M, LL, offs, key, _, hashes = field_inputs(fy, a.pixels, groups, last_month)
    qin, q, dropped, qhash = load_extra(a.quebec, a.quebec_pixels, groups, last_month, fy)
    X, M, LL, offs, key = stack_inputs((X, M, LL, offs, key), qin)
    te_q = np.arange(len(fy), len(fy) + len(q))
    y_fit = np.concatenate([y, np.zeros(len(q), int)])  # Québec labels never reach the model
    data = (torch.from_numpy(X).to(device), torch.from_numpy(M).to(device), torch.from_numpy(LL).to(device),
            offs, np.repeat(key, np.diff(offs)))
    print(f"{len(tr)} BARC training field-years; {len(q)} Québec field-years ({dropped} without a usable month); "
          f"lr {lr:g}, {epochs} epochs, seeds {seeds}, January to {MONTHS[last_month - 1]}; device {device}", flush=True)

    ho = np.concatenate([te_barc, te_q])
    p = np.mean([train_curve(enc, data, y_fit, tr, ho, device, s, lr, epochs)[-1] for s in seeds], 0)
    p_barc, p_q = p[:len(te_barc)], p[len(te_barc):]

    ref = pd.read_csv(src / "test_predictions.csv")
    chk = fy.loc[te_barc, ["field_id", "year", "label"]].assign(p_here=p_barc).merge(
        ref[["field_id", "year", "p_finetune"]], on=["field_id", "year"], validate="1:1")
    diff = float((chk.p_here - chk.p_finetune).abs().max())
    same_calls = int(((chk.p_here >= 0.5) == (chk.p_finetune >= 0.5)).sum())
    print(f"BARC 2021-2023 check: max |p - official run| = {diff:.4f}, same call on {same_calls}/{len(chk)}", flush=True)

    yq = (q.label == "Silage").to_numpy(int)
    w = square_weights(q, a.selection)
    pred = q[["field_id", "field", "year", "label", "area_ha"]].assign(p_presto=p_q.round(4), weight=w.round(4))
    pred.to_csv(out / "test_predictions.csv", index=False)

    def row(name, yy, pp, ww, groups, with_ci=True):
        pm = point_metrics(yy, pp)
        ci = cluster_bootstrap(yy, {"m": pp}, groups)["m"] if with_ci else {}
        wt = weighted(yy, pp, ww)
        f = lambda k, d=2: (f"{pm[k]:.{d}f}" + (f" ({ci[k][0]:.{d}f} to {ci[k][1]:.{d}f})" if k in ci else "")
                            if np.isfinite(pm[k]) else "n/a")
        return {"": name, "field-years (silage)": f"{pm['n']} ({pm['silage']})", "AUC": f("auc", 3),
                "balanced accuracy": f("balanced_accuracy"), "silage caught": pm["silage_caught"],
                "grain called silage": pm["grain_called_silage"],
                "accuracy at the square's share": f"{wt['accuracy']:.2f}",
                "silage precision at the square's share": "n/a" if np.isnan(wt["silage_precision"])
                else f"{wt['silage_precision']:.2f}",
                "silage share in the square / called": f"{wt['silage_share']:.2f} / {wt['silage_share_called']:.2f}"}

    rows = [row("All years", yq, p_q, w, q.field_id.to_numpy())]
    rows += [row(str(yr), yq[m], p_q[m], w[m], q.field_id.to_numpy()[m], with_ci=False)
             for yr in sorted(q.year.unique()) for m in [(q.year == yr).to_numpy()]]
    allgrain = weighted(yq, np.zeros(len(yq)), w)["accuracy"]

    runj = {"finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "runtime_s": round(time.time() - t0), "args": vars(a),
            "setting": {"from": str(src), "lr": lr, "epochs": epochs, "seeds": seeds, "last_month": last_month},
            "barc_check": {"max_abs_diff": diff, "same_calls": same_calls, "n": len(chk)},
            "quebec_dropped": dropped, "quebec_max_pix": EXTRA_MAX_PIX,
            "inputs": {"field_years.csv": sha256(a.field_years), "field_years_quebec.csv": sha256(a.quebec),
                       "quebec_selection.txt": sha256(a.selection), **hashes,
                       **{f"quebec/{k}": v for k, v in qhash.items()}},
            "versions": {"python": platform.python_version(), "torch": torch.__version__,
                         "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None}}
    (out / "run.json").write_text(json.dumps(runj, indent=1, default=str))

    lines = [f"# Presto outside Maryland: trained on BARC, tested on Québec ({MONTHS[0]} to {MONTHS[last_month - 1]})",
             "", "Generated by `presto_quebec_test.py`. The model is the one chosen for the official test "
             f"(`{src.name}`: fine-tuned, lr {lr:g}, {epochs} epochs, seeds {', '.join(map(str, seeds))}), trained on "
             f"the {len(tr)} BARC 2017-2020 field-years; no Québec label was used to train or choose anything. "
             f"Québec: {len(q)} field-years from FADQ declarations ({dropped} left out for no usable month). "
             "Brackets: 95% interval from resampling whole parcels. The 5-feature baseline cannot be scored: Québec "
             "has no hand features.", "",
             md_table(pd.DataFrame(rows)), "",
             "AUC, balanced accuracy and the two recalls do not depend on the silage share. The sample keeps every "
             "silage parcel and 400 grain parcels per year, so accuracy and precision are reweighted to the silage "
             f"share of the 30 km square. Calling every field Grain scores {allgrain:.2f} accuracy at that share.", "",
             f"Check: the same run predicts the 44 BARC 2021-2023 field-years; the largest difference from "
             f"`{src.name}` is {diff:.4f}, and the call is the same on {same_calls} of {len(chk)}.", ""]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
