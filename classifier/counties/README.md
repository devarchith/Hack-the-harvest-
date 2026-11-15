# County check: predicted silage share against the 2022 Census

A test of the classifier on county totals instead of labeled fields. No field labels exist outside BARC and Québec,
but the 2022 Census of Agriculture reports silage and grain corn acres for each county
(`../../02_QuickStats/corn_grain_silage_acres_2012_2025.csv`). If the model works away from Maryland, the share of corn
it calls silage should rise and fall with the Census share across counties.

The test and its outputs are in `../presto/presto_county_test.py` and `../presto/README.md`.

## Counties

Seven counties in UTM zone 18N (the grid of the BARC pixels), chosen for a wide range of Census silage shares:

| FIPS | County | Census 2022 silage share of harvested corn acres |
|---|---|---|
| 24035 | Queen Anne's, MD | 0.013 |
| 24013 | Carroll, MD | 0.089 |
| 42133 | York, PA | 0.078 |
| 42071 | Lancaster, PA | 0.274 |
| 36011 | Cayuga, NY | 0.525 |
| 36115 | Washington, NY | 0.759 |
| 50001 | Addison, VT | 0.921 |

Share = silage / (silage + grain), harvested acres, rows with `source` CENSUS and `domain` TOTAL. The file also holds
2022 county SURVEY estimates for some counties; they are left out. Withheld values are left out.

## Sample

`county_squares.py` uses no field outlines. The 2022 Cropland Data Layer (CDL, 30 m, USDA NASS, from CropScape) marks
corn pixels. Per county it keeps 90 m blocks (3 x 3 CDL pixels) that are all corn (code 1, or a double crop with corn:
225, 226, 237, 241). It then picks the 20 km square (5 km steps) with the most such blocks and draws 200 blocks there
at random (seed 0). Each block is one row of `county_squares.csv`, in the format of `../quebec/field_years_quebec.csv`
with an empty label. Blocks are 4,900 m² after the 10 m inward buffer of `presto_pixels.py`, so about 49 pixels each.

Equal-size blocks drawn at random from corn area make the mean prediction an area share, which matches the Census acre
share. `county_selection.txt` lists each county's Census acres, the square, and the square's share of the county's
all-corn blocks (22% to 51%).

## Limits

- The Census share covers the whole county. The sample covers one 20 km square of it.
- Blocks sit inside fields, so fields narrower than about 90 m are under-sampled.
- CDL corn includes CDL errors. The Census counts harvested acres; CDL counts planted corn.
- A county total cannot show which squares are wrong. Over- and under-calls can cancel.

## Run

```bash
PY=/home/keeboh/venvs/prithvi/bin/python
W=/home/keeboh/work/counties
$PY classifier/counties/county_squares.py --cdl-dir $W/cdl --out classifier/counties

# Pixels: the unchanged presto_pixels.py, one run per county (it reads one box around all rows of a year), from a copy
# outside /workspace as for Québec.
$PY -c "import pandas as pd; d = pd.read_csv('classifier/counties/county_squares.csv', dtype={'fips': str}); \
[g.to_csv(f'$W/fy/{f}.csv', index=False) for f, g in d.groupby('fips')]"
R=$W/run
mkdir -p $R/classifier/presto && ln -sfn "$PWD/preprocessing" $R/preprocessing
cp classifier/presto/presto_pixels.py $R/classifier/presto/
for f in 24035 24013 42133 42071 36011 36115 50001; do
  (cd $R && $PY classifier/presto/presto_pixels.py --field-years $W/fy/$f.csv --out $W/pixels/$f --years 2022 \
     --workers 6) &
done; wait
```

## Files

| File | What it holds |
|---|---|
| `county_squares.py` | Block selection |
| `county_squares.csv` | The 1,400 blocks (200 per county) |
| `county_selection.txt` | Census acres and share, block counts and the square per county |

## Whole US: sampled counties and a model trained on county shares

The seven-county check above, widened to the whole country, plus a Presto variant that also learns from the Census
county shares (learning from label proportions). Files: `national_counties.py` (sample), `national_pixels.py` and
`launch_pulls.sh` (pixels), `fix_shares.py` (Census-only shares of the drawn sample), `presto_national.py` (models;
`launch_runs.sh` holds the commands of the three runs), `guard_fields.py` (BARC held-out-year predictions),
`census_prior.py` (comparison with the last Census), `national_page.py` (results page), `national_metrics.py`
(challenge and USDA-style metrics with 95% intervals, `national/<run>/metrics_usda.md`); results in `national/`.

### Sample

`national_counties.py` draws counties with both 2022 Census numbers and at least 10,000 acres of corn. County shares
use the Census rows only: the QuickStats file also has 2022 survey rows for some counties, which `census_shares` used
to add in (fixed; `fix_shares.py` recomputed the drawn sample's shares, bands and population without new pulls). The
population is 963 counties after leaving out the seven test counties above and Prince George's, MD (BARC's county).
The draw is in five silage-share bands (under 5%, 5-15%, 15-40%, 40-70%, 70% and over), in random order (seed 0), at
most 3 counties per state per band: 40 counties in each of the three lower bands, 26 and 17 in the two upper bands,
which run out after the state cap. The draw used the old shares; with Census-only shares the 163 counties (36 states)
fall 35, 32, 49, 28 and 19 into the five bands. Per county it uses the same all-corn 90 m blocks as
`county_squares.py`, in the county's own UTM zone, from the 10 km square (2.5 km steps) with the most blocks, and
draws 100 blocks there (seed 0). A county with fewer than 150 blocks in that square is skipped and the next one in its
band is used. `national/selection.csv` lists every county tried, `national/population.csv` the population and
`national/county_squares.csv` the 16,300 blocks.

### Pixels

`national_pixels.py` runs `../presto/presto_pixels.py` unchanged, once per county, 2022 only. It sets the module's
grid projection to the county's UTM zone (the script fixes UTM 18N). `launch_pulls.sh` keeps a capped number of pulls
running. A county is used only once its pull wrote `pixels_2022.provenance.json`; one that did not finish by the
start of the run is listed in `run.json` and left out.

### Models

- **BARC-only**: the model of the official test (fine-tuned, lr 0.0003, 12 epochs, seeds 0-4, BARC 2017-2020). The
  refit reproduces the official BARC 2021-2023 predictions and the official Québec predictions (largest difference
  below 0.0001, every call the same).
- **Proportion-trained**: the same fine-tuning plus a county loss at every step. For 8 counties, 32 blocks x 4 pixels
  each, the binary cross-entropy between the county's mean predicted probability and its Census share, times a weight
  w. Latitude and longitude of every non-BARC pixel are set to BARC's, so location cannot stand in for the share.
  w is chosen from 0, 0.3, 1, 3 and 10 by the out-of-fold county error: counties in 5 groups by state, each group
  predicted by models that never saw it. Only a w that keeps the BARC fields is eligible: each BARC training year
  (2017-2020) is predicted by a model trained on the other three, and a w may lose at most 1 of those 81 calls
  against w 0. The final model is scored once on the BARC 2021-2023 test, Québec and the seven test counties.
- **Adapted starting weights** (comparison): the same run with Presto weights that continued Presto's own
  masked-reconstruction pretraining on BARC 2017-2020 pixels (no labels or Census numbers read), w from 0, 3 and 10.

A county's predicted share is the share of its blocks called silage (p >= 0.5). County classes by silage share: under
5%, 5-40%, 40% and over. The national share weights each band's sampled counties by corn acres and each band by its
corn acres in the population.

### Results

`national/results_final/` (all 163 sampled counties). `national_page.py` builds a self-contained HTML page per run
(not in git).

Choice of w, out of fold by state (county gap: mean absolute difference between the share of blocks called silage
and the Census share; guard: BARC 2017-2020 calls when each year is held out, of 81):

| w | county gap | Spearman | class right | BARC guard (of 81) |
|---|---|---|---|---|
| 0 | 0.255 | 0.28 | 53/163 | 72 |
| 0.3 | 0.166 | 0.62 | 90/163 | 74 |
| **1** | **0.139** | **0.68** | **97/163** | **73** |
| 3 | 0.144 | 0.70 | 91/163 | 77 |
| 10 | 0.151 | 0.69 | 96/163 | 76 |

Every w passes the guard; w 1 has the lowest county gap.

Final model (w 1) next to BARC-only:

| | BARC-only | Proportion-trained |
|---|---|---|
| BARC 2021-2023 (44 field-years): right | 41/44 | 41/44 (same calls) |
| silage called silage / grain called silage | 11/12, 2/32 | 11/12, 2/32 |
| AUC | 0.969 | 0.982 |
| Québec (2,245, no Québec labels): right | 1,343 | 1,778 |
| silage called silage / grain called silage | 149/1,045, 6/1,200 | 841/1,045, 263/1,200 |
| AUC | 0.780 | 0.865 |
| 163 counties, class right (proportion-trained out of fold) | 53/163 | 97/163 |
| county gap / Spearman | 0.248 / 0.31 | 0.139 / 0.68 |
| seven test counties, gap | 0.339 | 0.127 |
| national silage share (Census 0.071) | 0.072 | 0.133 |

County classes (rows: Census; columns: predicted):

| BARC-only | under 5% | 5-40% | 40%+ | | Proportion-trained, out of fold | under 5% | 5-40% | 40%+ |
|---|---|---|---|---|---|---|---|---|
| under 5% | 24 | 9 | 2 | | under 5% | 20 | 12 | 3 |
| 5-40% | 46 | 24 | 11 | | 5-40% | 26 | 39 | 16 |
| 40%+ | 23 | 19 | 5 | | 40%+ | 1 | 8 | 38 |

BARC-only's national share is close only because its errors cancel (it calls 46 of the 81 middle counties under 5%
and 23 of the 47 top counties under 5%). The proportion-trained model over-calls silage in low-silage counties, which
hold most corn acres, so its national share is too high.

Against the last Census (`census_prior.py`, `results_final/census_prior/`): the 2017 Census share alone is closer to
the 2022 share than either satellite model (gap 0.082, Spearman 0.90, 125/163 counties in the right class). The 2017
share combined with the proportion-trained estimate (fitted on the other folds) gives 0.080 and 126/163: better in 83
counties, worse in 80 (Wilcoxon p 0.67). So county totals alone do not justify the satellite model. Its use is where
the Census gives no answer: which fields are silage, the years between censuses, and places it does not cover.

Same run with the adapted starting weights (`results_final_barconly/`): w 10 chosen (w 0 / 3 / 10: gap 0.256 /
0.144 / 0.141; guard 72 / 76 / 76). Out of fold 91/163 counties in the right class (gap 0.141, Spearman 0.71); BARC
2021-2023 42/44; Québec 1,835 right (AUC 0.876); seven test counties gap 0.119; national share 0.106. Not a candidate
for the headline (rule fixed before the results).

January-October inputs (`results_final_janoct/`, the setting of the official Jan-Oct test: 34 epochs; w from 0, 1
and 3; same counties and groups): w 3 chosen (w 0 / 1 / 3: gap 0.238 / 0.159 / 0.148; guard 69 / 69 / 71). Out of
fold 88/163 counties in the right class (gap 0.148, Spearman 0.72); BARC 2021-2023 38/44 (its BARC-only refit 40/44,
the official Jan-Oct result); Québec 1,811 right (AUC 0.864); seven test counties gap 0.077; national share 0.070.

The headline model is the one with the lower out-of-fold county gap among the w that pass the BARC guard, Jan-Dec or
Jan-Oct (rule fixed before either result): `results_final` (Jan-Dec, w 1), 0.139 against 0.148. On the
mean-probability gap the two are about tied (0.139 and 0.135). The seven test counties had already been scored with a
Maryland-only Jan-Oct model, so they are not blind for this comparison.

### Limits

- County totals cannot show which blocks are wrong; over- and under-calls cancel.
- One 10 km square stands in for a whole county. The corn map has its own errors, and all-corn blocks favor large
  fields.
- 2022 was a drought year in the Plains, where some corn planted for grain was chopped for silage.
- The out-of-fold county numbers also chose w (five values), so they are slightly optimistic for the chosen model.
