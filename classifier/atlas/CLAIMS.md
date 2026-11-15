# Claims sheet: silage corn vs grain corn from public satellite data

Every number here is recomputed by `classifier/atlas/claims.py` and stored in `classifier/atlas/claims.json`.
Paths below are under `classifier/counties/national/` unless they start elsewhere.
Two models. The **Maryland-only model** learns from Maryland field labels only.
The **county-share model** also learns from published 2022 Census county silage shares. It uses no field labels outside Maryland.
A field is called silage when its probability is 0.5 or more.

## Headline numbers

- **Maryland, official test: 41 of 44 field-years correct (93%)**, both models. The years 2021-2023 were held out of training. `results_final/field_predictions.csv`
- On that test both models catch 11 of 12 silage fields and call 2 of 32 grain fields silage. Kappa 0.83. AUC 0.97 (Maryland-only) and 0.98 (county-share). `results_final/metrics_usda.md`
- **Québec, 2,245 field-years, no Québec label used in training: 1,778 correct (79%)** with the county-share model, against 1,343 (60%) with the Maryland-only model. AUC 0.86 against 0.78. `results_final/field_predictions.csv`
- **Silage caught in Québec: 841 of 1,045 (80%)** with the county-share model, against 149 of 1,045 (14%). That is 5.6 times as many. Grain fields called silage rise from 6 to 263 of 1,200. `results_final/field_predictions.csv`
- **163 counties in 36 states.** The county-share model scores each county without having trained on any county share from that county's state: mean gap to the 2022 Census silage share is 0.139 for the county-share model, against 0.248 for the Maryland-only model. Rank correlation 0.66 against 0.23. Average bias +0.004 against -0.171. `results_final/county_predictions.csv`
- **7 counties never used in training** (other counties in their states were): mean gap 0.127 against 0.339. Rank correlation with the Census 0.93 (p = 0.003, 7 counties). `results_final/test_county_predictions.csv`

## What USDA gets

- A grain-or-silage call with a probability for every corn field, every year. The Census gives one number per county every five years.
- The step from Maryland to the rest of the country cost no new field labels. The extra training signal is the county silage shares USDA already publishes.
- County totals that land near the Census: 89 of 163 counties (55%) within 0.10 of the Census share and 121 (74%) within 0.20. The Maryland-only model gets 60 (37%) and 92 (56%). `results_final/county_predictions.csv`
- The right county class (under 5%, 5-40%, 40% and over) in 97 of 163 counties, against 53. `results_final/metrics.md`
- Scale already run: 16,300 corn blocks in the 163 counties, plus 1,400 blocks in the 7 test counties. `results_final/block_predictions.csv`

## Supporting numbers

**Confident calls** (probability under 0.2 or over 0.8; computed after the main tests). `results_final/field_predictions.csv`, `results_final/block_predictions.csv`

- Maryland, county-share model: 37 of 44 fields are confident, and 36 of those 37 are correct (97%). Maryland-only model: 35 confident, 34 correct (97%).
- Québec, county-share model: 1,098 of 2,245 fields are confident (49%), and 995 of those are correct (91%). The other 1,147 are 68% correct.
- Québec, Maryland-only model: 1,956 confident (87%) but only 63% of those correct. It is sure and often wrong outside Maryland; the county-share model is not.
- Across the country, 12,679 of 16,300 blocks (78%) get a confident call from the county-share model.

**Québec calibration, county-share model** (10 equal bins; computed after the main tests). `results_final/field_predictions.csv`

- When the model says 0.8-0.9, 91% are silage (234 fields). When it says 0.9-1.0, 96% are silage (389 fields).
- When it says 0.1-0.2, 13% are silage (282 fields). When it says under 0.1, 16% are silage (193 fields).
- The middle is too high: fields scored 0.4-0.6 are silage 24% to 34% of the time. Expected calibration error is 0.083.

**Counties whose silage share moved** (computed after the main tests; this grouping was defined after the results were seen). `results_final/census_prior/census_prior_counties.csv`

- In the 46 counties where the share moved by more than 0.10 since the previous Census, the satellite estimate has a mean gap of 0.153. Reusing the previous Census gives 0.205. The satellite is closer in 30 of the 46.
- Read this with care: these counties are picked by the previous Census being off by more than 0.10, so it must do badly there, and which counties moved is only known once the next Census is out.
- In the other 117 counties the previous Census is closer: 0.034 against 0.133.
- Overall, across all 163, reusing the previous Census is closer: 0.082 against 0.138 (mean block probability, as in the two lines above; 0.139 with the share of blocks called silage).
- Combining the two gives 0.080 overall, not a significant gain over the previous Census alone (p = 0.67). `results_final/census_prior/census_prior.md`

**In-season, January to October only.** `results_final_janoct/metrics_usda.md`

- Maryland-only model: 40 of 44 correct in Maryland, with no grain field called silage (0 of 32) and 8 of 12 silage fields caught.
- County-share model: 38 of 44 in Maryland (7 of 12 silage caught). In Québec, 1,811 of 2,245 correct (81%), AUC 0.86.
- So a call before the year ends is possible. It costs silage recall in Maryland.

## Say this carefully

- **Maryland is not significantly better than the simple baseline.** A 5-feature logistic regression gets 39 of 44. The Maryland-only model gets 41 of 44 (p = 0.69). The gain is outside Maryland. `VALIDATION.md`
- **Both models score the same 41 of 44 in Maryland.** The county-share model adds reach, not Maryland accuracy.
- **On county totals, the previous Census reused is still closer overall** (0.082 against 0.139). Say the satellite adds field-level and yearly detail, and helps where a county is changing. Do not say it beats the Census.
- **All field labels for training and for the official test come from one farm.** 12 silage test cases means one field moves silage recall by 8 points. `VALIDATION.md`
- **Québec labels are crop-insurance declarations made before harvest**, not harvest records. The Québec sample is 47% silage by design, so its precision does not carry over to regions with little silage. `VALIDATION.md`
- **The county comparison uses 100 sampled blocks per county** (200 in the 7 test counties) against whole-county Census totals. `results_final/metrics.md`
- **The county-share model's 163-county numbers are out of fold by state.** Use those, not the in-sample columns. The Maryland-only model never trains on county shares, so its numbers need no fold.
- **Confident calls, calibration and the moved-county split were computed after the main tests.** Present them as supporting, not as pre-planned tests. Confident is not certain: 103 of the 1,098 confident Québec calls are wrong.
- County numbers above use the share of blocks called silage. Using the mean probability instead gives rank correlations 0.68 and 0.31, and a mean gap of 0.138 for the county-share model. `results_final/metrics.md`
