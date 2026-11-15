# Presto classifier: silage vs grain corn

A pretrained pixel time-series model (Presto, NASA Harvest, MIT) fine-tuned on the labeled BARC field-years to tell
silage corn from grain corn. It is scored against the 5-feature logistic regression in `research/grain_vs_silage/`
and against the Prithvi classifier in `../prithvi/`, on the same splits.

## Results

Same 5 field-level splits (seeds) as Prithvi; each seed's test part scored once, 101 test field-years in all
(`results/presto/summary.md`, `results/presto_jan_oct/summary.md`):

| Model | Correct | Silage caught | Grain called silage | Mean AUC |
|---|---|---|---|---|
| Presto, fine-tuned, January-December | 95/101 | 21/25 | 2/76 | 0.948 |
| Presto, fine-tuned, January-October | 95/101 | 20/25 | 1/76 | 0.922 |
| 5-feature logistic baseline | 90/101 | 21/25 | 7/76 | 0.940 |
| Prithvi (its run on the same splits) | 87/101 | 14/25 | 3/76 | 0.891 |
| Presto architecture from random weights (control) | 72/101 | 0/25 | 4/76 | 0.519 |

| Seed | Test field-years (silage) | Presto Jan-Dec | Presto Jan-Oct | 5-feature | McNemar p, Jan-Dec vs 5-feature |
|---|---|---|---|---|---|
| 0 | 19 (4) | 18 | 19 | 18 | 1.0 |
| 1 | 21 (6) | 20 | 19 | 20 | 1.0 |
| 2 | 21 (5) | 19 | 19 | 18 | 1.0 |
| 3 | 20 (5) | 19 | 19 | 17 | 0.5 |
| 4 | 20 (5) | 19 | 19 | 17 | 0.5 |

- Validation chose the fine-tuned encoder over the frozen one on every seed, in both runs.
- Presto makes fewer false silage calls than the baseline (2 vs 7 of 76) and catches as much silage (21 of 25).
  No single seed is significant (McNemar p 0.5-1.0).
- Pretraining is what works here: the same network trained from random weights calls almost everything grain.
- Silage fields at BARC green up again in November and December (median field NDVI 0.64-0.69 vs 0.31-0.33 for
  grain), probably a cover crop after the cut. Hiding those months (`--last-month 10`) leaves the correct count
  unchanged (AUC 0.948 to 0.922), so the result does not rest on them.

## Pipeline

| Step | Script | Runs on | Output |
|---|---|---|---|
| 1. Monthly 10 m pixel series, January-December, 2017-2023 | `presto_pixels.py` | CPU, network | `<pixels>/pixels_<year>.npz`, `results/pixels/<year>.provenance.json` |
| 2. Split, train, choose on validation, test once | `presto_train.py` | NVIDIA GPU (CPU works, slowly) | `results/presto/` |

```bash
PY=/home/keeboh/venvs/prithvi/bin/python
mkdir -p /home/keeboh/ext/presto/data && cd /home/keeboh/ext/presto && C=11e207a668a34336ced1d8e492a1bd5849b96c4a
for f in single_file_presto.py LICENSE data/default_model.pt; do
  curl -sfL -o $f https://raw.githubusercontent.com/nasaharvest/presto/$C/$f
done; echo $C > COMMIT; cd -
for y in 2017 2018 2019 2020 2021 2022 2023; do
  $PY classifier/presto/presto_pixels.py --out /home/keeboh/work/presto/pixels --years $y --workers 6 &
done; wait
P="$PY classifier/presto/presto_train.py --pixels /home/keeboh/work/presto/pixels --presto-dir /home/keeboh/ext/presto"
$P --out classifier/presto/results/presto
$P --last-month 10 --out classifier/presto/results/presto_jan_oct
```

Step 1 took 6-7 minutes per year with the years in parallel (1,342 scene-days, none failed). Step 2 took about
2 minutes per run on one NVIDIA B200 and needs `torch`, `einops` and the packages in
`research/grain_vs_silage/requirements.txt`.

## Inputs

- **Labels:** `field_years.csv`, a copy of `../prithvi/field_years.csv`: 127 labeled field-years 2017-2023 (96 Grain,
  31 Silage, 64 fields) with polygons and the 5 hand features for the baseline. How far the farm-record labels can be
  trusted: `../labels/README.md`.
- **Imagery:** Sentinel-2 L2A (10 bands: B2-B8, B8A, B11, B12) and Sentinel-1 RTC (VV, VH) from Microsoft Planetary
  Computer on a 10 m grid, with the cloud rules and reflectance scaling of `preprocessing/loader.py`. Fields are
  shrunk 10 m inward; a scene-day counts for a field when at least 60% of its pixels are clear. Per pixel and month:
  median of clear Sentinel-2 readings, mean Sentinel-1 power in dB. 57,440 pixels, median 450 per field-year.
- **Presto:** `nasaharvest/presto` at commit 11e207a, `single_file_presto.py` and `data/default_model.pt`
  (SHA-256 559c9d76...; full hashes in `results/presto/run.json`).

## Method

- **Model input per pixel:** 12 monthly steps; Sentinel-1 as (dB + 25) / 25, Sentinel-2 as reflectance, NDVI, and the
  field centroid's latitude and longitude. ERA5, SRTM and Dynamic World are masked. A month is used for a
  field-year when at least half of its pixels have a clear reading; missing pixels in that month take the
  field-year's median, and unused months are masked.
- **Split:** 70/15/15 train/validation/test by field, every year of a field in the same part (StratifiedGroupKFold,
  20 folds: 14 train, 3 validation, 3 test), seeds 0-4. Identical to the Prithvi splits (checked in the script).
- **Candidates:**
  - frozen encoder: 128-d pixel embeddings averaged per field-year, logistic regression head, C in 0.001-1;
  - fine-tuned: whole encoder plus a linear head trained on pixels; AdamW, learning rate 1e-4 or 3e-4, weight decay
    0.05, class-balanced loss, 64 random pixels per training field-year per epoch, up to 100 epochs, stop after 10
    without a lower validation log loss, keep the best epoch. A field-year's probability is the mean over its pixels.
- **Choice and test:** each seed picks the candidate with the lowest validation log loss. The test part is scored once
  for that candidate and for the baseline (5 hand features, logistic regression, fitted on the same training part).
- **Control:** the fine-tuning above from random weights, chosen on validation the same way, never a candidate.

## Outputs (`results/presto/`, `results/presto_jan_oct/`)

- `summary.md`: results for all seeds together and per seed, top validation candidates, input check.
- `validation_scores.csv`: every candidate on every seed's validation part.
- `test_predictions.csv`: test-part probabilities for Presto, the baseline, the control and Prithvi.
- `splits.csv`, `coverage.csv` (pixels and usable months per field-year), `run.json` (versions, GPU, hashes).

## Caveats

- One farm (BARC, Maryland). About 20 test field-years per seed, so one field-year is about 5 percentage points; a
  field-year can be in more than one seed's test part, so the 101 are not 101 different field-years.
- Validation picks both the epoch and the candidate, so validation scores are optimistic (the chosen candidates got
  all but 0 or 1 validation field-years right on every seed). The test numbers are the honest ones.
- The splits mix years, so a field's neighbours in time can be in training. The numbers are not comparable with the
  "train 2017-2020, test 2021-2023" results in `research/grain_vs_silage/`.
- Sentinel-2 values are L2A surface reflectance in Presto's scaling (DN / 10000); whether this matches the product
  Presto was pretrained on was not checked.
- The control is not a tuned model without pretraining; it got the same short training budget as the fine-tuned
  Presto.
- 56 of the 83 farm-record labels rest on a planting record only; 5 Grain labels look like soybeans in the USDA
  Cropland Data Layer (see `../labels/README.md`).
