# Metrics: results_final_barconly

Threshold 0.5 on the predicted silage probability. Silage is the positive class. Precision is USDA's user's accuracy, recall its producer's accuracy.

### Maryland (BARC) 2021-2023, held-out years: 44 field-years

Fields by year and class: 2021 silage 4, grain 14; 2022 silage 4, grain 3; 2023 silage 4, grain 15

BARC-only: rows are the label, columns the call.

| | called Silage | called Grain |
|---|---|---|
| Silage | 11 | 1 |
| Grain | 2 | 30 |

Per year, correct: 2021 15/18; 2022 7/7; 2023 19/19

County-share: rows are the label, columns the call.

| | called Silage | called Grain |
|---|---|---|
| Silage | 11 | 1 |
| Grain | 1 | 31 |

Per year, correct: 2021 16/18; 2022 7/7; 2023 19/19

| model | overall accuracy (95%) | balanced accuracy | Silage precision / recall / F1 | Grain precision / recall / F1 | Cohen's kappa (95%) | ROC AUC (95%) |
|---|---|---|---|---|---|---|
| BARC-only | 41/44 = 0.932 (0.818-0.977) | 0.927 | 0.846 / 0.917 / 0.880 | 0.968 / 0.938 / 0.952 | 0.832 (0.616-1.000) | 0.977 (0.924-1.000) |
| County-share | 42/44 = 0.955 (0.849-0.987) | 0.943 | 0.917 / 0.917 / 0.917 | 0.969 / 0.969 / 0.969 | 0.885 (0.692-1.000) | 0.982 (0.939-1.000) |

### Québec 2021-2023, no Québec label used in training: 2245 field-years

Fields by year and class: 2021 silage 360, grain 400; 2022 silage 336, grain 400; 2023 silage 349, grain 400

BARC-only: rows are the label, columns the call.

| | called Silage | called Grain |
|---|---|---|
| Silage | 185 | 860 |
| Grain | 8 | 1192 |

Per year, correct: 2021 492/760; 2022 411/736; 2023 474/749

County-share: rows are the label, columns the call.

| | called Silage | called Grain |
|---|---|---|
| Silage | 739 | 306 |
| Grain | 104 | 1096 |

Per year, correct: 2021 618/760; 2022 598/736; 2023 619/749

| model | overall accuracy (95%) | balanced accuracy | Silage precision / recall / F1 | Grain precision / recall / F1 | Cohen's kappa (95%) | ROC AUC (95%) |
|---|---|---|---|---|---|---|
| BARC-only | 1377/2245 = 0.613 (0.593-0.633) | 0.585 | 0.959 / 0.177 / 0.299 | 0.581 / 0.993 / 0.733 | 0.180 (0.155-0.207) | 0.807 (0.789-0.824) |
| County-share | 1835/2245 = 0.817 (0.801-0.833) | 0.810 | 0.877 / 0.707 / 0.783 | 0.782 / 0.913 / 0.842 | 0.628 (0.595-0.659) | 0.876 (0.861-0.891) |

### Counties: 163 sampled counties in 36 states, against the 2022 Census-only share

| estimate | MAE (95%) | RMSE | bias | R^2 | Spearman | 3-class right | weighted kappa (95%) |
|---|---|---|---|---|---|---|---|
| BARC-only | 0.252 (0.184-0.331) | 0.348 | -0.158 | -0.65 | 0.24 | 55/163 | 0.10 (-0.033-0.271) |
| County-share (out of fold) | 0.141 (0.114-0.171) | 0.205 | +0.002 | 0.42 | 0.69 | 91/163 | 0.57 (0.431-0.685) |
| Last Census reused | 0.082 (0.061-0.105) | 0.129 | -0.019 | 0.77 | 0.90 | 125/163 | 0.76 (0.642-0.852) |
| Last Census + satellite (other folds) | 0.084 (0.064-0.106) | 0.123 | -0.003 | 0.79 | 0.90 | 127/163 | 0.77 (0.674-0.849) |

Test counties never trained on (7): mean absolute error BARC-only 0.324, county-share 0.119.

National silage share of corn (band-stratified estimate): sampled, Census 0.071, sampled, BARC-only 0.085, sampled, proportion-trained (out of fold) 0.106, state 2022, survey 0.080, state 2022, Census 0.069.
