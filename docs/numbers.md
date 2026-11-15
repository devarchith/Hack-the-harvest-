# Numbers sheet: silage corn vs grain corn from public satellite data

Copy numbers from here. Every row names its source. `RF` = `classifier/counties/national/results_final` (the headline run: January-December imagery, loss weight w = 1). `usda` = `RF/metrics_usda.md`, `prior` = `RF/census_prior/census_prior.md`. Counts were recomputed from the CSVs next to those files and agree.

**Two models.** *Maryland-only model*: trained on field labels from BARC, the Beltsville, Maryland research farm (columns `p_barc`, `barc_*`). *County-share model*: the same training plus the 2022 Census of Agriculture silage share of 163 counties as a target, with no field label outside Maryland (columns `p_llp`, `llp_*`). A field is called silage when its probability is 0.5 or more.

## 1. Headline

1. **Maryland, held-out years:** the county-share model gets 41 of 44 field-years right, the same calls as the Maryland-only model.
2. **Québec, no Québec label used:** 1,778 of 2,245 right (79%) against 1,343 (60%); silage found rises from 149 to 841 of 1,045. Cost: grain called silage rises from 6 to 263 of 1,200.
3. **163 US counties, each scored by a model that never saw its state:** mean gap from the Census share 0.139 against 0.248 (county estimate: share of blocks called silage). Reusing the last Census (2017) is still closer: 0.082 against 0.138 for the model on the like-for-like comparison of section 6 (county estimate: mean probability). And the model's national total is too high (0.133 against 0.071).

## 2. Maryland held-out years (44 field-years, 2021-2023; trained on 2017-2020)

| Year | Silage | Grain | Source |
|---|---|---|---|
| 2021 / 2022 / 2023 | 4 / 4 / 4 (12) | 14 / 3 / 15 (32) | `usda` line 7; `RF/field_predictions.csv` kind=barc |

| Measure | Maryland-only | County-share | Source |
|---|---|---|---|
| Silage called silage / called grain | 11 / 1 | 11 / 1 | `usda` lines 11-23 |
| Grain called silage / called grain | 2 / 30 | 2 / 30 | same |
| Accuracy (95% interval) | 41/44 = 0.932 (0.818-0.977) | same | `usda` lines 29-30 |
| Silage precision / recall | 0.846 / 0.917 | same | same |
| Grain precision / recall | 0.968 / 0.938 | same | same |
| Kappa, agreement beyond chance (95%) | 0.832 (0.616-1.000) | same | same |
| AUC, ranking quality (95%) | 0.969 (0.903-1.000) | 0.982 (0.938-1.000) | same |
| Right per year 2021 / 2022 / 2023 | 15/18, 7/7, 19/19 | same | `usda` lines 16, 25 |

All 44 calls are identical between the two models (recomputed from `p_barc`, `p_llp`).

## 3. Québec (2,245 field-years, 2021-2023; no Québec label used in training)

Fields: 2021 silage 360, grain 400; 2022 336, 400; 2023 349, 400 (1,045 silage, 1,200 grain). `usda` line 34.

| Measure | Maryland-only | County-share | Source |
|---|---|---|---|
| Silage called silage / called grain | 149 / 896 | 841 / 204 | `usda` lines 38-50 |
| Grain called silage / called grain | 6 / 1,194 | 263 / 937 | same |
| Accuracy (95%) | 1,343/2,245 = 0.598 (0.578-0.618) | 1,778/2,245 = 0.792 (0.775-0.808) | `usda` lines 56-57 |
| Silage precision / recall | 0.961 / 0.143 | 0.762 / 0.805 | same |
| Grain precision / recall | 0.571 / 0.995 | 0.821 / 0.781 | same |
| Balanced accuracy | 0.569 | 0.793 | same |
| Kappa (95%) | 0.146 (0.123-0.171) | 0.583 (0.550-0.616) | same |
| AUC (95%) | 0.780 (0.761-0.799) | 0.865 (0.849-0.880) | same |
| **Cost: false silage** (grain called silage) | 6/1,200 = 0.5% | 263/1,200 = 21.9% | cells above |
| Accuracy at the area's real silage share (19%) | 0.83 | 0.79 | `RF/metrics.md` lines 128-131 |

The sample is 47% silage by design. At the 19% share of the area it was drawn from, calling every field grain scores 0.81 (`classifier/presto/README.md` line 121), so the gain in Québec is in finding silage, not in raw accuracy.

## 4. 163 US counties in 36 states, scored out of fold by state

Each county is predicted by a model trained without any county of its state group. Truth: 2022 Census silage share of harvested corn acres. County estimate: share of its 100 corn blocks called silage. Classes: under 5%, 5-40%, 40% and over (35, 81 and 47 counties).

| Measure | Maryland-only | County-share | Source |
|---|---|---|---|
| Mean absolute gap (95%, resampling states) | 0.248 (0.185-0.323) | 0.139 (0.107-0.175) | `usda` lines 63-64 |
| Root mean square error | 0.340 | 0.205 | same |
| Bias (estimate minus Census) | -0.171 | +0.004 | same |
| R² | -0.58 | 0.43 | same |
| Spearman rank correlation | 0.23 | 0.66 | same |
| Counties in the right class | 53/163 | 97/163 | same |
| Weighted kappa (95%) | 0.09 (-0.040-0.265) | 0.58 (0.436-0.700) | same |
| Of 47 counties at 40%+ silage, called under 5% | 23 | 1 | `RF/metrics.md` lines 15-37 |
| Of 35 counties under 5%, called higher | 11 | 15 | same |

Spearman on the mean probability instead of the share called is 0.31 and 0.68 (`RF/metrics.md` lines 9, 11): same data, different estimate. The out-of-fold numbers also chose w, so they are slightly optimistic.

## 5. The 7 counties never trained on (200 blocks each; MD, PA, NY, VT)

| Measure | Maryland-only | County-share | Source |
|---|---|---|---|
| Mean absolute gap | 0.339 | 0.127 | `usda` line 68; `RF/test_county_predictions.csv` |
| Counties in the right class | 2/7 | 5/7 | `RF/metrics.md` lines 53-56 |
| Spearman (mean probability) | -0.04 | 0.93 | same |
| Washington, NY (Census 0.759) share called silage | 0.115 | 0.570 | `RF/test_county_predictions.csv` |
| Addison, VT (Census 0.921) | 0.210 | 0.780 | same |
| Queen Anne's, MD (Census 0.013) | 0.220 | 0.120 | same |

Seven counties is a small check. Jan-Oct variant: 0.077 (exploratory: these counties were not blind for that comparison; `classifier/counties/national/results_final_janoct/metrics_usda.md` line 68).

## 6. Honest baseline: reuse the last Census (2017) vs the model (like-for-like, `prior` only)

Same 163 counties, same truth, county estimate = mean probability, out of fold. Last Census = 2017 share (156 counties), 2012 (4), state 5-year share (3).

| Estimate | Mean gap | Median gap | Spearman | Right class | Source |
|---|---|---|---|---|---|
| Last Census (2017) reused | 0.082 | 0.048 | 0.90 | 125/163 | `prior` line 9 |
| County-share model | 0.138 | 0.082 | 0.68 | 108/163 | `prior` line 12 (prints 0.1385; unrounded 0.13846 from the CSV) |
| Maryland-only model | 0.231 | 0.156 | 0.31 | 60/163 | `prior` line 11 |
| State 2022 survey share for every county | 0.144 | 0.092 | 0.69 | 109/163 | `prior` line 10 |
| Last Census + county-share model | 0.080 | 0.050 | 0.91 | 126/163 | `prior` line 14 |

- The last Census is closer than the model in 116 of 163 counties, the model in 47 (exploratory; recomputed from `RF/census_prior/census_prior_counties.csv`).
- Adding the model to the last Census: better in 83 counties, worse in 80, Wilcoxon p = 0.67. No measurable gain (`prior` line 16).
- 108/163 and 0.138 here, 97/163 and 0.139 in section 4, are the same model with two county estimates (mean probability here; share of blocks called there). Truth and counties are identical in both. Do not pair 125 with 97, or 0.082 with 0.139: pair the last Census with 108 and 0.138.
- The charts `docs/figures/last_census_comparison.png` and `docs/figures/county_class_matrices.png` use this section's figures (0.231 / 0.138 / 0.082 / 0.080 and 60 / 108 / 125); the scatter and three-runs charts use section 4's share of blocks called.

## 7. National total: silage share of US corn acres, 2022

| Estimate | Share | Source |
|---|---|---|
| Census, sampled population (963 counties) | 0.071 | `RF/state_check.csv` last row; `usda` line 70 |
| County-share model (out of fold) | 0.133, about 1.9 times the Census | same |
| Maryland-only model | 0.072, close only because errors cancel (section 4) | same; `classifier/counties/README.md` lines 165-167 |
| USDA state file, US 2022: survey / Census | 0.080 / 0.069 | same |

## 8. Training setup

| Fact | Value | Source |
|---|---|---|
| Labeled training field-years | 81 (19 silage, 62 grain), BARC 2017-2020 | `RF/guard_field_predictions.csv`; `classifier/counties/presto_national.py` line 13 |
| Counties with a Census share as target | 163 in 36 states, 2022 imagery | `RF/county_predictions.csv` |
| Corn blocks | 100 blocks of 90 m per county = 16,300 | `classifier/counties/national/county_squares.csv` |
| State groups (folds) | 5, with 32 / 36 / 32 / 32 / 31 counties; no state split | `RF/county_predictions.csv` column `fold` |
| Location inputs | latitude and longitude of every non-Maryland pixel set to BARC's | `classifier/counties/presto_national.py` lines 302-304 |
| Loss-weight grid | w = 0, 0.3, 1, 3, 10 (the value 10 was added after an early run: exploratory) | `RF/run.json` `setting.w_grid` |
| Out-of-fold county gap per w | 0.255 / 0.166 / **0.139** / 0.144 / 0.151 | `RF/run.json` `choice` |
| Maryland guard: right of 81 per w | 72 / 74 / 73 / 77 / 76 | same; `RF/guard_field_predictions.csv` |
| Guard rule | each 2017-2020 year predicted by a model fit on the other three; a w may lose at most 1 of the 81 calls against w = 0 (needs 71 or more). No w lost any; all five pass | `classifier/counties/presto_national.py` lines 26-27, 68, 450 |
| Chosen w | 1: lowest county gap among the w that pass | `RF/run.json` `setting.w_chosen` |
| Headline rule | lower out-of-fold county gap among passing w, Jan-Dec against Jan-Oct: 0.139 against 0.148, so Jan-Dec | `classifier/counties/README.md` lines 185-188 |
| How close | on the mean-probability gap the two are about tied: 0.138 Jan-Dec, 0.135 Jan-Oct (recomputed from the two `census_prior_counties.csv` files: 0.13846 and 0.13544; `classifier/counties/README.md` line 187 prints the first as 0.139) | `prior` line 12; `classifier/counties/national/results_final_janoct/census_prior/census_prior.md` line 12 |
| When declared | before either final result (`classifier/counties/README.md` line 186). Earlier runs on Census shares that wrongly included survey rows had already been seen (team decision log, not in the repo; the fix is at README lines 86-87) | as named |

## 9. Say / do not say

- **Say:** "Training on county shares keeps Maryland intact (41 of 44) and finds 841 of 1,045 Québec silage fields instead of 149, with no Québec label."
- **Say:** "It gives an answer per field, every year, where the Census gives one number per county every five years."
- **Say:** "County gap 0.139 against 0.248 for the Maryland-only model, on states the model never saw." (share of blocks called silage, section 4)
- **Do not say** "beats the Census" or "beats the baseline": the 2017 Census reused is closer (0.082 against 0.138, section 6).
- **Do not say** the national total is right: 0.133 against the Census 0.071.
- **Do not say** Québec is solved: 263 of 1,200 grain fields are called silage, and labels there are insurance declarations.
