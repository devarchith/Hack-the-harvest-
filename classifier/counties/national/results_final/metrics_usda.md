# Metrics: results_final

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
| Grain | 2 | 30 |

Per year, correct: 2021 15/18; 2022 7/7; 2023 19/19

| model | overall accuracy (95%) | balanced accuracy | Silage precision / recall / F1 | Grain precision / recall / F1 | Cohen's kappa (95%) | ROC AUC (95%) |
|---|---|---|---|---|---|---|
| BARC-only | 41/44 = 0.932 (0.818-0.977) | 0.927 | 0.846 / 0.917 / 0.880 | 0.968 / 0.938 / 0.952 | 0.832 (0.616-1.000) | 0.969 (0.903-1.000) |
| County-share | 41/44 = 0.932 (0.818-0.977) | 0.927 | 0.846 / 0.917 / 0.880 | 0.968 / 0.938 / 0.952 | 0.832 (0.616-1.000) | 0.982 (0.938-1.000) |

### Québec 2021-2023, no Québec label used in training: 2245 field-years

Fields by year and class: 2021 silage 360, grain 400; 2022 silage 336, grain 400; 2023 silage 349, grain 400

BARC-only: rows are the label, columns the call.

| | called Silage | called Grain |
|---|---|---|
| Silage | 149 | 896 |
| Grain | 6 | 1194 |

Per year, correct: 2021 475/760; 2022 412/736; 2023 456/749

County-share: rows are the label, columns the call.

| | called Silage | called Grain |
|---|---|---|
| Silage | 841 | 204 |
| Grain | 263 | 937 |

Per year, correct: 2021 582/760; 2022 606/736; 2023 590/749

| model | overall accuracy (95%) | balanced accuracy | Silage precision / recall / F1 | Grain precision / recall / F1 | Cohen's kappa (95%) | ROC AUC (95%) |
|---|---|---|---|---|---|---|
| BARC-only | 1343/2245 = 0.598 (0.578-0.618) | 0.569 | 0.961 / 0.143 / 0.248 | 0.571 / 0.995 / 0.726 | 0.146 (0.123-0.171) | 0.780 (0.761-0.799) |
| County-share | 1778/2245 = 0.792 (0.775-0.808) | 0.793 | 0.762 / 0.805 / 0.783 | 0.821 / 0.781 / 0.801 | 0.583 (0.550-0.616) | 0.865 (0.849-0.880) |

### Counties: 163 sampled counties in 36 states, against the 2022 Census-only share

| estimate | MAE (95%) | RMSE | bias | R^2 | Spearman | 3-class right | weighted kappa (95%) |
|---|---|---|---|---|---|---|---|
| BARC-only | 0.248 (0.185-0.323) | 0.340 | -0.171 | -0.58 | 0.23 | 53/163 | 0.09 (-0.040-0.265) |
| County-share (out of fold) | 0.139 (0.107-0.175) | 0.205 | +0.004 | 0.43 | 0.66 | 97/163 | 0.58 (0.436-0.700) |
| Last Census reused | 0.082 (0.061-0.105) | 0.129 | -0.019 | 0.77 | 0.90 | 125/163 | 0.76 (0.642-0.852) |
| Last Census + satellite (other folds) | 0.080 (0.062-0.100) | 0.116 | -0.004 | 0.81 | 0.91 | 126/163 | 0.76 (0.670-0.841) |

Test counties never trained on (7): mean absolute error BARC-only 0.339, county-share 0.127.

National silage share of corn (band-stratified estimate): sampled, Census 0.071, sampled, BARC-only 0.072, sampled, proportion-trained (out of fold) 0.133, state 2022, survey 0.080, state 2022, Census 0.069.
