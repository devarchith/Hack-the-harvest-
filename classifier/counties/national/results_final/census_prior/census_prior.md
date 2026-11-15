# County silage share: satellite vs the last Census

## /home/keeboh/work/national/results_final

163 of 163 counties scored (Census-only 2022 share available; folds as in the run). The run's own census_share differs from the Census-only share by more than 0.01 in 0. Last-Census source: {'2017': 156, '2012': 4, 'state 5-yr': 3}.

| estimate | counties | mean abs gap | median abs gap | Spearman | class right |
|---|---|---|---|---|---|
| Last Census (2017) alone | 163 | 0.082 | 0.0479 | 0.90 (p 1e-60) | 125/163 |
| State 2022 survey share | 163 | 0.1442 | 0.0922 | 0.69 (p 3e-24) | 109/163 |
| Satellite, Maryland-only model | 163 | 0.2314 | 0.1558 | 0.31 (p 4e-05) | 60/163 |
| Satellite, county-share model | 163 | 0.1385 | 0.0819 | 0.68 (p 1e-23) | 108/163 |
| Satellite, recalibrated on other folds | 163 | 0.1446 | 0.0936 | 0.67 (p 6e-23) | 104/163 |
| Last Census + satellite (fitted on other folds) | 163 | 0.0801 | 0.0502 | 0.91 (p 3e-62) | 126/163 |

Last Census + satellite vs last Census alone, per-county absolute gap: better in 83, worse in 80; mean 0.0801 vs 0.0820; Wilcoxon signed-rank p 0.67.

3-class matrix, last Census + satellite (rows: Census 2022 class):

estimate      40% and over  5-40%  under 5%
Census                                     
40% and over            39      8         0
5-40%                    7     65         9
under 5%                 0     13        22
