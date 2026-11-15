"""Test-set metrics for silage (1) vs grain (0) field-years, with field-level confidence intervals.

Every function takes y (1 = Silage, 0 = Grain) and p (predicted probability of Silage). A field-year is called
Silage when p >= THRESHOLD; the threshold is fixed before testing, never tuned on test labels.
"""
import numpy as np
from scipy.stats import binomtest, rankdata

THRESHOLD = 0.5
CLIP = 1e-6  # probabilities are clipped to [CLIP, 1 - CLIP] for log loss, as in the training scripts


def _ratio(a, b):
    return a / b if b else np.nan


def confusion(y, p, threshold=THRESHOLD):
    """(tp, fn, fp, tn) with Silage as the positive class."""
    y, pred = np.asarray(y).astype(bool), np.asarray(p) >= threshold
    return int((pred & y).sum()), int((~pred & y).sum()), int((pred & ~y).sum()), int((~pred & ~y).sum())


def auc(y, p):
    """Area under the ROC curve (Mann-Whitney, ties count half); NaN when y has one class."""
    y, p = np.asarray(y).astype(bool), np.asarray(p, float)
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return np.nan
    r = rankdata(p)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def point_metrics(y, p, threshold=THRESHOLD):
    """Every metric for one set of predictions. Counts are exact; rates are fractions (0-1)."""
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    tp, fn, fp, tn = confusion(y, p, threshold)
    n = len(y)
    rec1, rec0 = _ratio(tp, tp + fn), _ratio(tn, tn + fp)
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    q = np.clip(p, CLIP, 1 - CLIP)
    return {
        "n": n, "silage": int(y.sum()), "correct": tp + tn,
        "silage_caught": f"{tp}/{tp + fn}", "grain_called_silage": f"{fp}/{fp + tn}",
        "accuracy": _ratio(tp + tn, n),
        "balanced_accuracy": np.nanmean([rec1, rec0]) if np.isfinite([rec1, rec0]).any() else np.nan,
        "silage_recall": rec1, "grain_recall": rec0,
        "silage_precision": _ratio(tp, tp + fp),
        "silage_f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "grain_precision": _ratio(tn, tn + fn),
        "grain_f1": _ratio(2 * tn, 2 * tn + fn + fp),
        "macro_f1": np.nanmean([_ratio(2 * tp, 2 * tp + fp + fn), _ratio(2 * tn, 2 * tn + fn + fp)]) if n else np.nan,
        "mcc": (tp * tn - fp * fn) / den if den else 0.0,
        "auc": auc(y, p),
        "log_loss": float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))) if n else np.nan,
        "brier": float(np.mean((p - y) ** 2)) if n else np.nan,
        "silage_share_true": _ratio(int(y.sum()), n), "silage_share_pred": _ratio(tp + fp, n),
    }


RATES = ["accuracy", "balanced_accuracy", "silage_recall", "grain_recall", "silage_precision", "silage_f1",
         "grain_precision", "grain_f1", "macro_f1", "mcc", "auc", "log_loss", "brier", "silage_share_pred"]


def mcnemar(y, p_model, p_base, threshold=THRESHOLD):
    """Exact McNemar test on the field-years where the two models disagree about being right.
    Returns (only model right, only baseline right, two-sided p)."""
    y = np.asarray(y).astype(bool)
    a = (np.asarray(p_model) >= threshold) == y
    b = (np.asarray(p_base) >= threshold) == y
    only_a, only_b = int((a & ~b).sum()), int((b & ~a).sum())
    pval = binomtest(only_a, only_a + only_b).pvalue if only_a + only_b else 1.0
    return only_a, only_b, float(pval)


def cluster_bootstrap(y, probs, groups, base=None, n_boot=2000, seed=0, level=0.95, threshold=THRESHOLD):
    """Percentile confidence intervals from resampling whole groups (fields) with replacement.

    probs: {model name: probabilities}. Rows of one field (other years, other split seeds) are drawn together, so the
    interval reflects the number of distinct fields, not rows. With base set, also intervals for each model's
    difference from that model, computed on the same resamples (paired).
    Returns {model: {metric: (low, high)}} and, with base, {model: {"diff_" + metric: (low, high)}} merged in.
    """
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    rows_of = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    draws = {m: {k: [] for k in RATES} for m in probs}
    diffs = {m: {k: [] for k in RATES} for m in probs if base is not None and m != base}
    for _ in range(n_boot):
        idx = np.concatenate([rows_of[g] for g in rng.integers(0, len(uniq), len(uniq))])
        pm = {m: point_metrics(y[idx], np.asarray(p)[idx], threshold) for m, p in probs.items()}
        for m in probs:
            for k in RATES:
                draws[m][k].append(pm[m][k])
            if m in diffs:
                for k in RATES:
                    diffs[m][k].append(pm[m][k] - pm[base][k])
    lo, hi = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    out = {}
    for m in probs:
        out[m] = {k: tuple(np.nanpercentile(v, [lo, hi])) if np.isfinite(v).any() else (np.nan, np.nan)
                  for k, v in draws[m].items()}
        if m in diffs:
            out[m].update({f"diff_{k}": tuple(np.nanpercentile(v, [lo, hi])) if np.isfinite(v).any()
                           else (np.nan, np.nan) for k, v in diffs[m].items()})
    return out
