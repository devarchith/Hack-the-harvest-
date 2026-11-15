# Test metrics

Shared scoring for the silage vs grain classifiers. Silage is the positive class. Any model that saves test
predictions (one row per test field-year: `field_id`, `year`, `label`, one probability column per model, and `seed`
when several splits are pooled) can be scored the same way.

| File | What it does |
|---|---|
| `metrics.py` | The metric functions: point metrics, field-resampled confidence intervals, McNemar test. |
| `report.py` | Writes `metrics.md` and `metrics.csv` for a predictions file. |
| `test_metrics.py` | Checks the metric code. Runs with plain `python`; pytest is optional. |

## Run

```
python classifier/metrics/test_metrics.py

python classifier/metrics/report.py \
    --predictions classifier/presto/results/standard_test/test_predictions.csv \
    --models p_presto=Presto p_baseline="5-feature baseline" --baseline p_baseline \
    --out classifier/presto/results/standard_test/metrics.md
```

`--join OTHER.csv:COLUMN=NEW` adds a column from another run on the same test rows (it stops if the rows or labels
differ), so two runs can be compared pair by pair, for example with and without extra training data.

## Metrics

A field-year is called Silage when its probability is at least 0.5. TP = silage called silage, FN = silage called
grain, FP = grain called silage, TN = grain called grain.

| Metric | Definition | Why it is reported |
|---|---|---|
| correct, accuracy | (TP + TN) / n | Easy to read, but about 25% of field-years are Silage, so calling everything Grain already scores about 0.75. |
| balanced accuracy | mean of silage recall and grain recall | Calling everything one class scores 0.5. |
| silage recall ("silage caught") | TP / (TP + FN) | Share of silage field-years found. |
| grain recall | TN / (TN + FP) | "Grain called silage" is FP out of FP + TN. |
| silage precision | TP / (TP + FP) | Of the field-years called silage, the share that are silage. n/a when nothing is called silage. |
| silage F1 | 2TP / (2TP + FP + FN) | One number for the silage class that needs both precision and recall. |
| grain precision, grain F1 | the same with Grain as the positive class | The majority class, for completeness. |
| macro F1 | mean of silage F1 and grain F1 | Both classes count equally. |
| MCC | (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN)) | Uses all four cells; 0 is chance level, 1 is perfect, and it is not inflated by the class imbalance. |
| AUC | chance that a random silage field-year gets a higher probability than a random grain one (ties count half) | Ranking quality without the 0.5 cut. |
| log loss | mean of −log(probability of the true class), probabilities clipped to [1e-6, 1 − 1e-6] | Penalises confident mistakes. The training scripts choose models on this, on validation data only. |
| Brier | mean of (probability − label)² | Probability accuracy; a constant 0.5 scores 0.25. |
| predicted silage share | share of field-years called silage, next to the true share | Shows whether a model would over- or under-count silage area. |

## Decisions

- **Fixed cut.** The cut is 0.5, set before testing and never tuned on test labels. The models are trained with
  class-balanced losses, so 0.5 is their natural cut.
- **Choice is not made on these numbers.** Every model is chosen on validation or held-out training years. Test
  metrics are computed once, for reporting.
- **Intervals resample fields.** The years of one field are not independent, and in the mixed-year splits the same
  field-year can be tested under several split seeds. The 95% intervals (percentile, 2000 resamples) therefore draw
  whole fields with replacement, carrying every row of a field. Resampling rows instead would make the intervals
  too narrow. `test_bootstrap_resamples_fields_not_rows` checks this.
- **Paired comparisons.** Differences against the baseline are computed on the same resamples, so they account for
  the two models being scored on the same field-years. An interval that excludes 0 means the difference holds
  across the field resamples. The exact McNemar test uses only the field-years where exactly one of the two models
  is right. When split seeds are pooled, rows repeat and its p is optimistic; use the interval instead.

## Reading the numbers

- The test sets are small. On the 44-field-year standard test, one field-year is 2.3 points of accuracy, and
  intervals are wide. Differences of one or two field-years are not evidence of a better model.
- Precision and F1 can be n/a in a resample where nothing is called silage. Such resamples are left out of that
  metric's interval.
- The labels themselves have errors; `classifier/labels/` documents the fixes.

## Tests

`test_metrics.py` checks the following:

- Every point metric matches scikit-learn on 40 random cases, including many tied probabilities.
- The 0.5 cut is inclusive.
- AUC with all ties is 0.5.
- One-class and no-silage-call cases return n/a or 0 where defined.
- McNemar returns a known exact value (7 against 2 gives p = 92/512).
- The bootstrap returns (1, 1) for a perfect model and (0, 0) for differences between identical models.
- The bootstrap is deterministic for a seed, and its interval covers the point estimate.
- The bootstrap resamples fields, not rows.
- The report runs end to end.
- `--join` matches rows by key and refuses mismatched rows.
- When results are present, the code reproduces the published Presto counts.
