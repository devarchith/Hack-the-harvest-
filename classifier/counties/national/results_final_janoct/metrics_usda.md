# Metrics: results_final_janoct

Threshold 0.5 on the predicted silage probability. Silage is the positive class. Precision is USDA's user's accuracy, recall its producer's accuracy.

### Maryland (BARC) 2021-2023, held-out years: 44 field-years

Fields by year and class: 2021 silage 4, grain 14; 2022 silage 4, grain 3; 2023 silage 4, grain 15

BARC-only: rows are the label, columns the call.

| | called Silage | called Grain |
|---|---|---|
| Silage | 8 | 4 |
| Grain | 0 | 32 |

Per year, correct: 2021 16/18; 2022 6/7; 2023 18/19

County-share: rows are the label, columns the call.

| | called Silage | called Grain |
|---|---|---|
| Silage | 7 | 5 |
| Grain | 1 | 31 |

Per year, correct: 2021 16/18; 2022 6/7; 2023 16/19

| model | overall accuracy (95%) | balanced accuracy | Silage precision / recall / F1 | Grain precision / recall / F1 | Cohen's kappa (95%) | ROC AUC (95%) |
|---|---|---|---|---|---|---|
| BARC-only | 40/44 = 0.909 (0.788-0.964) | 0.833 | 1.000 / 0.667 / 0.800 | 0.889 / 1.000 / 0.941 | 0.744 (0.463-0.941) | 0.958 (0.875-1.000) |
| County-share | 38/44 = 0.864 (0.733-0.936) | 0.776 | 0.875 / 0.583 / 0.700 | 0.861 / 0.969 / 0.912 | 0.616 (0.320-0.861) | 0.870 (0.713-0.994) |

### Québec 2021-2023, no Québec label used in training: 2245 field-years

Fields by year and class: 2021 silage 360, grain 400; 2022 silage 336, grain 400; 2023 silage 349, grain 400

BARC-only: rows are the label, columns the call.

| | called Silage | called Grain |
|---|---|---|
| Silage | 422 | 623 |
| Grain | 158 | 1042 |

Per year, correct: 2021 517/760; 2022 439/736; 2023 508/749

County-share: rows are the label, columns the call.

| | called Silage | called Grain |
|---|---|---|
| Silage | 744 | 301 |
| Grain | 133 | 1067 |

Per year, correct: 2021 610/760; 2022 597/736; 2023 604/749

| model | overall accuracy (95%) | balanced accuracy | Silage precision / recall / F1 | Grain precision / recall / F1 | Cohen's kappa (95%) | ROC AUC (95%) |
|---|---|---|---|---|---|---|
| BARC-only | 1464/2245 = 0.652 (0.632-0.672) | 0.636 | 0.728 / 0.404 / 0.519 | 0.626 / 0.868 / 0.727 | 0.280 (0.243-0.318) | 0.711 (0.690-0.731) |
| County-share | 1811/2245 = 0.807 (0.790-0.822) | 0.801 | 0.848 / 0.712 / 0.774 | 0.780 / 0.889 / 0.831 | 0.607 (0.575-0.639) | 0.864 (0.848-0.879) |

### Counties: 163 sampled counties in 36 states, against the 2022 Census-only share

| estimate | MAE (95%) | RMSE | bias | R^2 | Spearman | 3-class right | weighted kappa (95%) |
|---|---|---|---|---|---|---|---|
| BARC-only | 0.246 (0.184-0.313) | 0.336 | -0.104 | -0.54 | 0.30 | 62/163 | 0.27 (0.125-0.414) |
| County-share (out of fold) | 0.148 (0.112-0.189) | 0.221 | -0.050 | 0.33 | 0.69 | 88/163 | 0.57 (0.460-0.664) |
| Last Census reused | 0.082 (0.061-0.105) | 0.129 | -0.019 | 0.77 | 0.90 | 125/163 | 0.76 (0.642-0.852) |
| Last Census + satellite (other folds) | 0.086 (0.066-0.109) | 0.125 | -0.004 | 0.79 | 0.90 | 133/163 | 0.81 (0.714-0.883) |

Test counties never trained on (7): mean absolute error BARC-only 0.256, county-share 0.077.

National silage share of corn (band-stratified estimate): sampled, Census 0.071, sampled, BARC-only 0.118, sampled, proportion-trained (out of fold) 0.070, state 2022, survey 0.080, state 2022, Census 0.069.
