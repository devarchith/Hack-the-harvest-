# County silage share: satellite vs the last Census

## /home/keeboh/work/national/results_final_janoct

163 of 163 counties scored (Census-only 2022 share available; folds as in the run). The run's own census_share differs from the Census-only share by more than 0.01 in 0. Last-Census source: {'2017': 156, '2012': 4, 'state 5-yr': 3}.

| estimate | counties | mean abs gap | median abs gap | Spearman | class right |
|---|---|---|---|---|---|
| Last Census (2017) alone | 163 | 0.082 | 0.0479 | 0.90 (p 1e-60) | 125/163 |
| State 2022 survey share | 163 | 0.1442 | 0.0922 | 0.69 (p 3e-24) | 109/163 |
| Satellite, Maryland-only model | 163 | 0.2291 | 0.173 | 0.31 (p 5e-05) | 70/163 |
| Satellite, county-share model | 163 | 0.1354 | 0.0885 | 0.72 (p 3e-27) | 103/163 |
| Satellite, recalibrated on other folds | 163 | 0.1419 | 0.1008 | 0.71 (p 2e-26) | 102/163 |
| Last Census + satellite (fitted on other folds) | 163 | 0.0856 | 0.0529 | 0.90 (p 4e-59) | 133/163 |

Last Census + satellite vs last Census alone, per-county absolute gap: better in 79, worse in 84; mean 0.0856 vs 0.0820; Wilcoxon signed-rank p 0.61.

3-class matrix, last Census + satellite (rows: Census 2022 class):

estimate      40% and over  5-40%  under 5%
Census                                     
40% and over            42      5         0
5-40%                    8     69         4
under 5%                 0     13        22
