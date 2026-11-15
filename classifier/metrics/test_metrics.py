"""Tests for metrics.py and report.py. Run either way (no pytest needed):

  python classifier/metrics/test_metrics.py
  pytest classifier/metrics/test_metrics.py
"""
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics as skm

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import metrics as M  # noqa: E402
import report  # noqa: E402

PRESTO = HERE.parent / "presto" / "results"


def _random_case(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(6, 80))
    y = rng.integers(0, 2, n)
    y[:2] = [0, 1]  # both classes present
    p = rng.random(n)
    if seed % 3 == 0:
        p = np.round(p, 1)  # many ties
    return y, p


def test_point_metrics_match_sklearn():
    for seed in range(40):
        y, p = _random_case(seed)
        pred = (p >= 0.5).astype(int)
        m = M.point_metrics(y, p)
        q = np.clip(p, M.CLIP, 1 - M.CLIP)
        expect = {"accuracy": skm.accuracy_score(y, pred), "balanced_accuracy": skm.balanced_accuracy_score(y, pred),
                  "silage_recall": skm.recall_score(y, pred), "grain_recall": skm.recall_score(y, pred, pos_label=0),
                  "silage_f1": skm.f1_score(y, pred), "grain_f1": skm.f1_score(y, pred, pos_label=0),
                  "macro_f1": skm.f1_score(y, pred, average="macro"), "mcc": skm.matthews_corrcoef(y, pred),
                  "auc": skm.roc_auc_score(y, p), "log_loss": skm.log_loss(y, q), "brier": skm.brier_score_loss(y, p)}
        if pred.any():
            expect["silage_precision"] = skm.precision_score(y, pred)
        if (pred == 0).any():
            expect["grain_precision"] = skm.precision_score(y, pred, pos_label=0)
        for k, v in expect.items():
            assert abs(m[k] - v) < 1e-9, (seed, k, m[k], v)
        assert m["correct"] == int((pred == y).sum()) and m["n"] == len(y)
        tn, fp, fn, tp = skm.confusion_matrix(y, pred, labels=[0, 1]).ravel()
        assert (m["silage_caught"], m["grain_called_silage"]) == (f"{tp}/{tp + fn}", f"{fp}/{fp + tn}")


def test_threshold_is_inclusive():
    m = M.point_metrics([1, 0], [0.5, 0.4999])
    assert m["correct"] == 2 and m["silage_recall"] == 1.0


def test_auc_all_ties_is_half():
    assert M.auc([0, 1, 0, 1], [0.3, 0.3, 0.3, 0.3]) == 0.5


def test_one_class_and_no_silage_calls():
    m = M.point_metrics([0, 0, 0], [0.1, 0.7, 0.2])
    assert np.isnan(m["auc"]) and np.isnan(m["silage_recall"])
    assert m["balanced_accuracy"] == m["grain_recall"] == 2 / 3
    m = M.point_metrics([1, 0, 0], [0.1, 0.2, 0.3])
    assert np.isnan(m["silage_precision"]) and m["silage_f1"] == 0 and m["mcc"] == 0
    assert m["silage_share_pred"] == 0 and abs(m["silage_share_true"] - 1 / 3) < 1e-12


def test_mcnemar_known_values():
    y = np.ones(9, int)
    pa = np.array([0.9] * 7 + [0.1] * 2)  # right on the first 7
    pb = np.array([0.1] * 7 + [0.9] * 2)  # right on the last 2
    a, b, pv = M.mcnemar(y, pa, pb)
    assert (a, b) == (7, 2) and abs(pv - 92 / 512) < 1e-12  # exact two-sided binomial, n = 9
    assert M.mcnemar(y, pa, pa) == (0, 0, 1.0)


def test_bootstrap_perfect_model_and_identical_models():
    y = np.array([0, 1] * 10)
    p = y * 0.8 + 0.1
    ci = M.cluster_bootstrap(y, {"a": p, "b": p}, np.arange(20), base="b", n_boot=200)
    assert ci["a"]["accuracy"] == (1.0, 1.0) and ci["a"]["auc"] == (1.0, 1.0)
    assert ci["a"]["diff_accuracy"] == (0.0, 0.0) and ci["a"]["diff_brier"] == (0.0, 0.0)


def test_bootstrap_is_deterministic_and_covers_the_estimate():
    y, p = _random_case(7)
    a = M.cluster_bootstrap(y, {"m": p}, np.arange(len(y)), n_boot=300, seed=3)
    b = M.cluster_bootstrap(y, {"m": p}, np.arange(len(y)), n_boot=300, seed=3)
    assert a == b
    acc = M.point_metrics(y, p)["accuracy"]
    assert a["m"]["accuracy"][0] <= acc <= a["m"]["accuracy"][1]


def test_bootstrap_resamples_fields_not_rows():
    # Every field appears twice (as when split seeds share test field-years). Drawing whole fields must give the same
    # intervals as one row per field; drawing rows would make them narrower.
    y, p = _random_case(11)
    g = np.arange(len(y))
    one = M.cluster_bootstrap(y, {"m": p}, g, n_boot=300, seed=1)
    two = M.cluster_bootstrap(np.repeat(y, 2), {"m": np.repeat(p, 2)}, np.repeat(g, 2), n_boot=300, seed=1)
    for k in ("accuracy", "auc", "brier", "silage_f1"):
        assert np.allclose(one["m"][k], two["m"][k]), k


def test_report_end_to_end():
    rng = np.random.default_rng(0)
    n = 40
    y = np.array([1] * 10 + [0] * 30)
    df = pd.DataFrame({"field_id": np.arange(n) // 2, "year": 2021 + np.arange(n) % 2, "seed": np.arange(n) % 2,
                       "label": np.where(y == 1, "Silage", "Grain"),
                       "p_a": np.clip(y * 0.6 + rng.random(n) * 0.5, 0, 1), "p_b": rng.random(n)})
    with tempfile.TemporaryDirectory() as d:
        src, out = Path(d) / "pred.csv", Path(d) / "m.md"
        df.to_csv(src, index=False)
        report.main(["--predictions", str(src), "--models", "p_a=Model A", "p_b=Model B", "--baseline", "p_b",
                     "--out", str(out), "--n-boot", "100"])
        text, tab = out.read_text(encoding="utf-8"), pd.read_csv(out.with_suffix(".csv"))
    assert "## Against Model B" in text and "## By split seed" in text
    tp, fn, fp, tn = M.confusion(y, df.p_a)
    assert f"| Model A | Silage | {tp} | {fn} | {tp + fn} |" in text
    assert f"|  | Grain | {fp} | {tn} | {fp + tn} |" in text
    assert list(tab.model) == ["Model A", "Model B"]
    assert tab.loc[0, "correct"] == M.point_metrics(y, df.p_a)["correct"]
    for k in ("silage_precision", "silage_recall", "silage_f1", "accuracy", "macro_f1", "auc"):
        assert {k, f"{k}_ci_low", f"{k}_ci_high", f"diff_{k}"} <= set(tab.columns), k


def test_join_adds_a_run_on_the_same_rows():
    df = pd.DataFrame({"seed": [0, 0, 1], "field_id": [1, 2, 1], "year": [2021, 2021, 2022],
                       "label": ["Silage", "Grain", "Grain"], "p_a": [0.9, 0.2, 0.4]})
    other = df.sample(frac=1, random_state=0).assign(p_a=lambda d: 1 - d.p_a)  # same rows, another order
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "other.csv"
        other.to_csv(f, index=False)
        j = report.join(df, f"{f}:p_a=p_other")
        assert list(j.columns) == list(df.columns) + ["p_other"] and np.allclose(j.p_other, 1 - df.p_a)
        other.iloc[:2].to_csv(f, index=False)
        try:
            report.join(df, f"{f}:p_a=p_other")
            raise AssertionError("missing rows were accepted")
        except ValueError:
            pass


def test_published_presto_numbers():
    """The metric code reproduces the counts already published in the Presto results, when they are present."""
    f = PRESTO / "presto" / "test_predictions.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    y = (d.label == "Silage").to_numpy(int)
    assert M.point_metrics(y, d.p_presto)["correct"] == 95
    assert M.point_metrics(y, d.p_baseline)["correct"] == 90
    if "p_prithvi" in d:
        assert M.point_metrics(y, d.p_prithvi)["correct"] == 87
    f = PRESTO / "standard_test" / "test_predictions.csv"
    if f.exists():
        d = pd.read_csv(f)
        assert M.point_metrics((d.label == "Silage").to_numpy(int), d.p_baseline)["correct"] == 39


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"pass  {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}\n{traceback.format_exc()}")
    print(f"{len(tests) - failed} of {len(tests)} passed")
    sys.exit(1 if failed else 0)
