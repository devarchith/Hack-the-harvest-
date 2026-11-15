# County silage share: satellite vs the last Census

## /home/keeboh/work/national/results_final_barconly

163 of 163 counties scored (Census-only 2022 share available; folds as in the run). The run's own census_share differs from the Census-only share by more than 0.01 in 0. Last-Census source: {'2017': 156, '2012': 4, 'state 5-yr': 3}.

| estimate | counties | mean abs gap | median abs gap | Spearman | class right |
|---|---|---|---|---|---|
| Last Census (2017) alone | 163 | 0.082 | 0.0479 | 0.90 (p 1e-60) | 125/163 |
| State 2022 survey share | 163 | 0.1442 | 0.0922 | 0.69 (p 3e-24) | 109/163 |
| Satellite, Maryland-only model | 163 | 0.2302 | 0.1534 | 0.33 (p 2e-05) | 55/163 |
| Satellite, county-share model | 163 | 0.142 | 0.1053 | 0.71 (p 3e-26) | 109/163 |
| Satellite, recalibrated on other folds | 163 | 0.1409 | 0.1007 | 0.70 (p 9e-26) | 108/163 |
| Last Census + satellite (fitted on other folds) | 163 | 0.0836 | 0.0517 | 0.90 (p 1e-60) | 127/163 |

Last Census + satellite vs last Census alone, per-county absolute gap: better in 80, worse in 83; mean 0.0836 vs 0.0820; Wilcoxon signed-rank p 0.98.

3-class matrix, last Census + satellite (rows: Census 2022 class):

estimate      40% and over  5-40%  under 5%
Census                                     
40% and over            41      6         0
5-40%                   12     65         4
under 5%                 0     14        21
