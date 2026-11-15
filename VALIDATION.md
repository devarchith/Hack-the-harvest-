# Validation

How every model was tested, and the results on one page. Each number comes from the result file named in its row.

## The test that counts

- **Test set:** the 44 labeled 2021-2023 field-years in `04_Evaluation/BARC_Corn.2021-2023.gpkg` (32 Grain, 12 Silage,
  34 fields). Each model scores them once.
- **Training:** BARC farm records from 2017-2020 only (83 field-years; 81 after the label fixes in
  `classifier/labels/`). No 2021-2023 label is used for training.
- **Choosing settings:** hold out one year at a time inside 2017-2020. No 2021-2023 label is read before the test.
- **Not used for results:** splits that keep each field in one part but mix years (`presto_train.py`,
  `prithvi_train.py`). Fields of the same year share weather, harvest timing and image dates, so those scores are not
  out-of-time, and the ranking flipped when two label rows were dropped.

## Results: train 2017-2020, test 2021-2023

| Model | Correct /44 | Silage caught /12 | Grain called silage /32 | AUC | vs baseline (exact McNemar) | Result file |
|---|---|---|---|---|---|---|
| 5 features, logistic regression (baseline) | 39 | 11 | 4 | 0.982 | | `research/grain_vs_silage/results/report.md` |
| Presto, fine-tuned, January-December | 41 | 11 | 2 | 0.969 | 4 vs 2, p = 0.69 | `classifier/presto/results/standard_test_fixed_labels/` |
| Presto, fine-tuned, January-October | 40 | 8 | 0 | 0.958 | 4 vs 3, p = 1.0 | `classifier/presto/results/standard_test_fixed_labels_jan_oct/` |
| Prithvi-EO-2.0 100M, fine-tuned | 40 | 11 | 3 | 0.906 | 3 vs 2, p = 1.0 | `classifier/prithvi/results/year_split/` |
| All 38 hand features, XGBoost (chosen on 2017-2020) | 40 | 11 | 3 | 0.961 | 3 vs 2, p = 1.0 | `research/grain_vs_silage/results/model_comparison.md` |

- **No model beats the baseline significantly.** The baseline is wrong on 5 of 44, so even a perfect model would only
  reach p = 0.0625 on this test. Presto's accuracy difference is +0.05 (95% interval −0.04 to +0.14).
- **Presto's lead depends on November and December.** BARC's silage fields green up again after the cut, probably from a
  cover crop. The baseline leaves that regrowth out on purpose, since it may be a BARC habit. Without those months
  Presto catches 8 of 12 silage fields against the baseline's 11 of 12.
- **Every model in the table misses Central Farm 4-7 in 2021.** It is labeled Silage, but the farm record shows it was combined on
  November 4, which is grain timing.
- The 5 features plus AlphaEarth embeddings scored 43 of 44, but that was noticed by looking at the test results, so it
  is a lead, not a result (`research/grain_vs_silage/README.md`).

## Other year-based checks

| Check | Baseline | Prithvi | Result file |
|---|---|---|---|
| Train 2017-2020 on fields never seen in 2021-2023, test 2021-2023 | 40/44 | 38/44 | `classifier/prithvi/results/year_split/summary.md` |
| Train 2017-2020 on other farms, test Central Farm 2021-2023 | 24/25 | 21/25 | same |
| Hold out one year at a time, all seven years | 113/127 | 119/127 (p = 0.15) | same |

## Outside Maryland: Québec

Presto, trained on BARC 2017-2020 only, scored once on 2,245 Québec field-years from 2021-2023 (FADQ crop-insurance
declarations, 1,045 silage). No Québec label was used to train or choose anything.

| Model | AUC | Balanced accuracy | Silage caught | Grain called silage | Result file |
|---|---|---|---|---|---|
| Presto, January-December | 0.78 (0.76-0.80) | 0.57 | 149/1,045 | 6/1,200 | `classifier/presto/results/quebec_test/` |
| Presto, January-October | 0.71 (0.69-0.73) | 0.64 | 422/1,045 | 158/1,200 | `classifier/presto/results/quebec_test_jan_oct/` |

Presto ranks silage above grain better than chance, but at the cut learned at BARC it calls almost every Québec field
Grain. At the 19% silage share of the area sampled, its accuracy (0.83) is barely above calling every field Grain
(0.81). The baseline cannot be scored on Québec yet, because the Québec rows have no hand features.

## Limits

- All training labels and the official test come from one farm, BARC.
- With 12 silage test cases, one field-year moves silage recall by 8 points.
- Most 2017-2020 training labels rest on a planting record only (`classifier/labels/README.md`).
- The Québec labels are declarations made before harvest, not harvest records.
