# Presto classifier: silage vs grain corn

A pretrained pixel time-series model (Presto, NASA Harvest, MIT) fine-tuned on the labeled BARC field-years to tell
silage corn from grain corn. It is scored against the 5-feature logistic regression in `research/grain_vs_silage/`
and against the Prithvi classifier in `../prithvi/`.

## At a glance

- **Official test** (train on 2017-2020, score the 44 labeled 2021-2023 field-years once): fine-tuned Presto gets 41
  of 44 right, the 5-feature model 39 of 44. The difference is not significant (accuracy +0.05, 95% interval −0.04 to
  +0.14). Presto matches the 5-feature model; it is not shown to beat it.
- **Presto's edge comes from November and December.** Silage fields at BARC green up again after the cut, probably a
  cover crop. The 5-feature model leaves that regrowth out on purpose, because it may be a BARC habit
  (`research/grain_vs_silage/README.md`). With November and December hidden, Presto gets 40 of 44 but catches 8 of 12
  silage fields against the 5-feature model's 11 of 12 (balanced accuracy 0.83 against 0.90).
- **Mixed-year splits are the weaker test:** fields of the same year sit in training and test, and the order flipped
  when two label rows were dropped. Presto 92 of 101, the 5-feature model 96 of 101.
- **Outside Maryland it mostly says Grain.** Trained on BARC only and scored once on 2,245 Québec field-years, Presto
  ranks silage above grain better than chance (AUC 0.78) but catches 149 of 1,045 silage fields at the 0.5 cut. At the
  silage share of the sampled area, its accuracy (0.83) is barely above calling every field Grain (0.81).
- **Where to read the numbers:** `results/standard_test_fixed_labels/summary.md` (official test),
  `results/standard_test_fixed_labels_jan_oct/summary.md` (January-October), `results/quebec_test*/summary.md`
  (Québec). Each BARC run folder's `metrics.md` has every metric with its interval.
- **To rerun:** the commands under Pipeline. Each run takes a few minutes on one GPU once the pixels are pulled.

## Hypothesis

The monthly Sentinel-1 and Sentinel-2 series of a field carries the signals behind the hand features of
`research/grain_vs_silage/`. Those signals are when the canopy disappears (silage is cut in late August or September,
grain is harvested later), whether it is still green when it does, and what is left afterwards (bare soil after
silage, stover after grain). A model pretrained on pixel time series can learn them from a few labeled field-years.
Monthly steps blur the harvest date to within a month, so the timing signal is coarser here than in the hand
features.

## Results

Labels: `../labels/results/field_years_fixed.csv` (125 field-years; see Inputs). "Correct" counts field-years at the
0.5 cut. Brackets are 95% intervals from resampling whole fields. Every metric, the confusion matrices and the paired
comparisons are in each run's `metrics.md`; definitions are in `../metrics/README.md`.

**Standard test:** train on 2017-2020 (81 field-years, 19 Silage) and score the 44 labeled 2021-2023 field-years
(12 Silage) once (`results/standard_test_fixed_labels/`):

| Model | Correct | Silage caught | Grain called silage | Silage precision | Silage F1 | Balanced accuracy | MCC | AUC |
|---|---|---|---|---|---|---|---|---|
| Presto fine-tuned, chosen on 2017-2020 (lr 3e-4, 12 epochs) | 41/44 | 11/12 | 2/32 | 0.85 | 0.88 (0.71-1.00) | 0.93 (0.82-1.00) | 0.83 | 0.969 |
| Presto frozen (not chosen) | 36/44 | 11/12 | 7/32 | 0.61 | 0.73 (0.52-0.88) | 0.85 (0.73-0.94) | 0.63 | 0.885 |
| 5-feature logistic baseline | 39/44 | 11/12 | 4/32 | 0.73 | 0.81 (0.61-0.95) | 0.90 (0.78-0.97) | 0.74 | 0.982 |
| Presto fine-tuned, January-October (lr 3e-4, 34 epochs) | 40/44 | 8/12 | 0/32 | 1.00 | 0.80 (0.56-0.96) | 0.83 (0.69-0.96) | 0.77 | 0.958 |

**Mixed-year splits:** 5 field-level splits (seeds), each test part scored once, 101 test rows
(`results/presto_fixed_labels/`, `results/presto_fixed_labels_jan_oct/`). AUC is over the pooled rows:

| Model | Correct | Silage caught | Grain called silage | Silage precision | Silage F1 | Balanced accuracy | MCC | AUC |
|---|---|---|---|---|---|---|---|---|
| Presto, chosen on validation | 92/101 | 23/27 | 5/74 | 0.82 | 0.84 (0.72-0.94) | 0.89 (0.81-0.96) | 0.78 | 0.967 |
| Presto, January-October | 90/101 | 17/27 | 1/74 | 0.94 | 0.76 (0.56-0.88) | 0.81 (0.69-0.90) | 0.71 | 0.918 |
| 5-feature logistic baseline | 96/101 | 24/27 | 2/74 | 0.92 | 0.91 (0.78-0.98) | 0.93 (0.86-0.99) | 0.87 | 0.968 |
| Presto architecture from random weights (control) | 83/101 | 9/27 | 0/74 | 1.00 | 0.50 (0.25-0.67) | 0.67 (0.57-0.75) | 0.52 | 0.784 |

What the results show:

- **Presto and the baseline cannot be separated on these data.** On the standard test, fine-tuned Presto is right on
  4 field-years where the baseline is wrong, and wrong on 2 where it is right: accuracy +0.05 (−0.04 to +0.14),
  McNemar p = 0.69. On the mixed-year splits, the baseline is ahead: −0.04 (−0.12 to +0.04). Both intervals include
  0.
- **The mixed-year ranking depends on the split.** With the original 127-row table, Presto led 95 to 90 on the same
  procedure (below). Dropping two rows reassigned the folds, and the order reversed.
- **Pretraining is what makes Presto work.** The same network trained from random weights is worse than the baseline
  on every mixed-year run (fixed labels: accuracy −0.13, −0.23 to −0.04).
- **Fine-tuning beats the frozen encoder.** Frozen embeddings trail the baseline on the standard test and on 5 of the
  6 shared-evaluator tests.
- **November and December help Presto find silage.** Silage fields at BARC green up again late in the year (median
  field NDVI 0.64-0.69 against 0.31-0.33 for grain), probably a cover crop after the cut. Masking those months drops
  silage caught from 11 to 8 of 12 on the standard test and from 23 to 17 of 27 on the mixed-year splits, while false
  silage calls fall from 2 to 0 and from 5 to 1. A model that has to call a field before November gives up silage
  recall for precision.

**Where it fails (standard test).** Both models miss Central Farm 4-7 in 2021. It is labeled Silage, but the farm
record dates its harvest to November 4, very late for silage (`research/grain_vs_silage/README.md`, "Where it fails").
Presto calls two APU grain fields silage in 2021 (APU 1 and APU 30, probabilities 0.67 and 0.66); the baseline gets
both right. The baseline calls four grain field-years silage that Presto gets right: Central Farm 2-32B 2022, APU 30
2023, East Farm 5-21 2023 and East Farm 5-40 2023.

**Before the label fixes** (original table, `field_years.csv`, 127 field-years; `results/standard_test/`,
`results/standard_test_jan_oct/`, `results/presto/`, `results/presto_jan_oct/`):

| Test | Presto fine-tuned | Presto, January-October | Presto frozen | 5-feature baseline | Prithvi | Control |
|---|---|---|---|---|---|---|
| Standard test (/44) | 41 | 38 | 35 | 39 | | |
| Mixed-year splits (/101) | 95 | 95 | | 90 | 87 (its run on the same splits) | 72 |

**Through the team's shared evaluator** (`../prithvi/evaluate.py` on frozen field embeddings from `presto_embed.py`,
original labels; the feature set, PCA components and C are chosen on 2017-2020 only; output copied to
`results/shared_evaluator_frozen/`):

| Test | 5-feature baseline | Presto frozen (chosen) | Presto frozen + 5 features (not chosen) |
|---|---|---|---|
| Train 2017-2020, test 2021-2023 (/44) | 39 | 34 | 39 |
| Hold out one year, 2021-2023 (/44) | 42 | 36 | 38 |
| Hold out one year, 2017-2023 (/127) | 113 | 109 | 116 |
| Train other farms 2017-2020, test Central Farm 2021-2023 (/25) | 24 | 18 | 19 |
| Train 2017-2020 fields never seen in 2021-2023, test 2021-2023 (/44) | 40 | 31 | 38 |
| Hold out one year, 2017-2020 (/83) | 68 | 70 | 70 |

**Outside Maryland:** the setting chosen for the standard test, trained on the 81 BARC 2017-2020 field-years only,
scores the 2,245 Québec field-years once (`results/quebec_test/`, `results/quebec_test_jan_oct/`). No Québec label is
used to train or choose anything. Brackets resample whole parcels. The sample is 47% silage; accuracy is reweighted to
the 19% of the 30 km area it was drawn from (`../quebec/README.md`). The 5-feature baseline cannot be scored: Québec has
no hand features.

| Model | Field-years (silage) | AUC | Balanced accuracy | Silage caught | Grain called silage | Accuracy at 19% silage |
|---|---|---|---|---|---|---|
| Presto fine-tuned (lr 3e-4, 12 epochs) | 2,245 (1,045) | 0.780 (0.760-0.799) | 0.57 (0.56-0.58) | 149/1,045 | 6/1,200 | 0.83 |
| Presto fine-tuned, January-October (lr 3e-4, 34 epochs) | 2,245 (1,045) | 0.711 (0.689-0.732) | 0.64 (0.62-0.65) | 422/1,045 | 158/1,200 | 0.78 |
| Every field called Grain | | | 0.50 | 0/1,045 | 0/1,200 | 0.81 |

What BARC teaches does not carry over at the BARC cut: the full-year model calls 3% of the sampled area's fields silage
where 19% are. Hiding November and December lowers the ranking (AUC 0.71) and does not fix the calls. The Québec labels
are crop-insurance declarations, not harvest records.

## Pipeline

| Step | Script | Runs on | Output |
|---|---|---|---|
| 1. Monthly 10 m pixel series, January-December, 2017-2023 | `presto_pixels.py` | CPU, network | `<pixels>/pixels_<year>.npz`, `results/pixels/<year>.provenance.json` |
| 2a. Mixed-year splits: split, train, choose on validation, test once | `presto_train.py` | NVIDIA GPU (CPU works, slowly) | `results/presto*/` |
| 2b. Standard test: choose on 2017-2020, test 2021-2023 once | `presto_standard_test.py` | NVIDIA GPU | `results/standard_test*/` |
| 2c. Québec test: the model chosen in 2b, trained on BARC only, scores Québec once | `presto_quebec_test.py` | NVIDIA GPU | `results/quebec_test*/` |
| 3. Metrics with confidence intervals | `../metrics/report.py` | CPU | `results/*/metrics.md`, `metrics.csv` |
| Optional: frozen field embeddings for `../prithvi/evaluate.py` | `presto_embed.py` | NVIDIA GPU | one CSV |

```bash
PY=/home/keeboh/venvs/prithvi/bin/python
mkdir -p /home/keeboh/ext/presto/data && cd /home/keeboh/ext/presto && C=11e207a668a34336ced1d8e492a1bd5849b96c4a
for f in single_file_presto.py LICENSE data/default_model.pt; do
  curl -sfL -o $f https://raw.githubusercontent.com/nasaharvest/presto/$C/$f
done; echo $C > COMMIT; cd -
for y in 2017 2018 2019 2020 2021 2022 2023; do
  $PY classifier/presto/presto_pixels.py --out /home/keeboh/work/presto/pixels --years $y --workers 6 &
done; wait
FY=classifier/labels/results/field_years_fixed.csv
A="--field-years $FY --pixels /home/keeboh/work/presto/pixels --presto-dir /home/keeboh/ext/presto"
Q="--extra-train classifier/quebec/field_years_quebec.csv --extra-pixels /home/keeboh/work/quebec/pixels"
R=classifier/presto/results
$PY classifier/presto/presto_train.py $A --out $R/presto_fixed_labels
$PY classifier/presto/presto_train.py $A --last-month 10 --out $R/presto_fixed_labels_jan_oct
$PY classifier/presto/presto_train.py $A $Q --out $R/presto_fixed_labels_quebec
$PY classifier/presto/presto_standard_test.py $A --out $R/standard_test_fixed_labels
$PY classifier/presto/presto_standard_test.py $A --last-month 10 --out $R/standard_test_fixed_labels_jan_oct
$PY classifier/presto/presto_standard_test.py $A $Q --out $R/standard_test_fixed_labels_quebec
QT="--pixels /home/keeboh/work/presto/pixels --quebec-pixels /home/keeboh/work/quebec/pixels --presto-dir /home/keeboh/ext/presto"
$PY classifier/presto/presto_quebec_test.py $QT --setting-from $R/standard_test_fixed_labels --out $R/quebec_test
$PY classifier/presto/presto_quebec_test.py $QT --setting-from $R/standard_test_fixed_labels_jan_oct \
  --out $R/quebec_test_jan_oct
$PY classifier/metrics/report.py --predictions $R/standard_test_fixed_labels/test_predictions.csv \
  --models p_finetune="Presto fine-tuned" p_frozen="Presto frozen" p_baseline="5-feature baseline" \
  --baseline p_baseline --out $R/standard_test_fixed_labels/metrics.md
```

The Québec pixels come from the same `presto_pixels.py`, run with `--field-years
classifier/quebec/field_years_quebec.csv --out /home/keeboh/work/quebec/pixels`. Without `--field-years`, the scripts
use `field_years.csv`, the original 127-row table (the `results/presto`, `presto_jan_oct`, `standard_test` and
`standard_test_jan_oct` runs). Step 1 took 6-7 minutes per BARC year with the years in parallel (1,342 scene-days,
none failed). On one NVIDIA B200, a mixed-year run took 2-5 minutes and a standard-test run 7-20 minutes, depending on
how many runs shared the GPU. The scripts need `torch`, `einops` and the packages in
`research/grain_vs_silage/requirements.txt`.

## Inputs

- **Labels:** `../labels/results/field_years_fixed.csv`: 125 labeled field-years 2017-2023 (94 Grain, 31 Silage) with
  polygons and the 5 hand features for the baseline. It is the original table (`field_years.csv`, a copy of
  `../prithvi/field_years.csv`, 127 field-years) without 235/2020 (soybeans) and 278/2017 (sorghum-sudangrass), both
  Grain in the original. How far the labels can be trusted: `../labels/README.md`.
- **Extra training rows (optional):** `../quebec/field_years_quebec.csv`: 2,245 field-years from Québec, 2021-2023
  (1,045 Silage, 1,200 Grain; FADQ declarations). Used for training only: never in validation, the choice split or a
  test.
- **Imagery:** Sentinel-2 L2A (10 bands: B2-B8, B8A, B11, B12) and Sentinel-1 RTC (VV, VH) from Microsoft Planetary
  Computer on a 10 m grid, with the cloud rules and reflectance scaling of `preprocessing/loader.py`. Fields are
  shrunk 10 m inward; a scene-day counts for a field when at least 60% of its pixels are clear. Per pixel and month:
  median of clear Sentinel-2 readings, mean Sentinel-1 power in dB. BARC: 56,596 pixels in the 125 field-years,
  median 450 per field-year.
- **Presto:** `nasaharvest/presto` at commit 11e207a, `single_file_presto.py` and `data/default_model.pt`
  (SHA-256 559c9d76...; full hashes in each `run.json`).

## Method

- **Model input per pixel:** 12 monthly steps; Sentinel-1 as (dB + 25) / 25, Sentinel-2 as reflectance, NDVI, and the
  field centroid's latitude and longitude. ERA5, SRTM and Dynamic World are masked. A month is used for a
  field-year when at least half of its pixels have a clear reading; missing pixels in that month take the
  field-year's median, and unused months are masked. `--last-month 10` also masks November and December.
- **Candidates:**
  - frozen encoder: 128-d pixel embeddings averaged per field-year, logistic regression head, C in 0.001-1;
  - fine-tuned: whole encoder plus a linear head trained on pixels; AdamW, learning rate 1e-4 or 3e-4, weight decay
    0.05, class-balanced loss, 64 random pixels per training field-year per epoch. A field-year's probability is the
    mean over its pixels.
- **Mixed-year splits (`presto_train.py`):** 70/15/15 train/validation/test by field, every year of a field in the
  same part (StratifiedGroupKFold, 20 folds: 14 train, 3 validation, 3 test), seeds 0-4; the same procedure as the
  Prithvi splits. Fine-tuning runs up to 100 epochs and keeps the epoch with the lowest validation log loss (stop
  after 10 without improvement). Each seed picks the candidate with the lowest validation log loss; its test part is
  scored once for that candidate and for the baseline fitted on the same training part. Control: the same
  fine-tuning from random weights, never a candidate.
- **Standard test (`presto_standard_test.py`):** the split of `research/grain_vs_silage/`. Everything is chosen on
  2017-2020 by holding out one year at a time: C for the frozen head; learning rate and number of epochs (1-60) for
  fine-tuning, on held-out predictions averaged over training seeds 0-4; then the family with the lower pooled log
  loss. The chosen settings are refitted on all 2017-2020 field-years (fine-tuning: one model per seed, probabilities
  averaged), and the 44 labeled 2021-2023 field-years are scored once. Before anything else runs, the baseline must
  reproduce its published 68/83 (choice split) and 39/44 (test) on the original table.
- **Extra training rows:** added to every training set only. Each fine-tuning epoch draws as many Québec field-years
  as there are BARC training field-years (`--extra-ratio 1`), at most 256 pixels kept per Québec field-year; the class
  weights balance the field-years drawn. The frozen head and the baseline weigh the Québec rows to the same total.
  The splits, the choice split and the test rows are the same as without them, so the runs compare pair by pair.
- **Metrics:** `../metrics/README.md`. Silage is the positive class, the cut is 0.5, and 95% intervals resample
  whole fields.

## Outputs (each folder in `results/`)

- `summary.md`: the run's results; mixed-year runs also list each seed and the top validation candidates.
- `metrics.md`, `metrics.csv`: every metric with its interval and the paired comparison with the baseline.
- `test_predictions.csv`: test probabilities for Presto, the baseline and the other models of the run.
- `validation_scores.csv` or `choice_scores.csv`: every candidate on validation or on the choice split.
- `run.json`: arguments, versions, GPU, and hashes of the labels, pixel files and Presto weights. Mixed-year runs also
  write `splits.csv` and `coverage.csv` (pixels and usable months per field-year).

## Caveats

- One farm (BARC, Maryland). The tests are small: one field-year is 2.3 points on the 44-field-year standard test and
  about 5 points on a 20-field-year seed. In the mixed-year runs a field-year can be in more than one seed's test
  part, so the 101 test rows are not 101 different field-years.
- The mixed-year results depend on the split: the label fixes removed two rows, which reassigned the folds, and
  Presto went from 5 more correct than the baseline to 4 fewer. Read the standard test and the intervals, not one
  split.
- Validation picks both the epoch and the candidate, so validation scores are optimistic. The test numbers are the
  honest ones.
- The Québec rows come from 2021-2023, the years of the standard test (other fields, other region). With them, the
  standard test no longer measures training only on earlier years.
- Presto was pretrained on Sentinel-2 Level-1C top-of-atmosphere reflectance (`COPERNICUS/S2` through `openmapflow`
  0.2.2, the version its repository pins). The inputs here are Level-2A surface reflectance, in the same scaling
  (DN / 10000). The Presto repository's own embedding script also uses Level-2A (`S2_SR_HARMONIZED`). The effect of
  the difference was not measured.
- Time steps are calendar months: a cut on August 2 and one on August 30 look the same.
- The shared-evaluator table uses the original 127-row labels; the other headline numbers use the fixed 125.
- The control is not a tuned model without pretraining; it got the same short training budget as the fine-tuned
  Presto.
- 56 of the 83 farm-record labels in the original table rest on a planting record only (see `../labels/README.md`).
