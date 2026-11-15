# Methodology: grain vs silage corn from public satellite data (BARC, Maryland)

This document explains what the method does, why, and how to reproduce every number. It is written so a
teammate, a judge or an AI assistant can rebuild the results from nothing but public data and this folder.

- **Fastest way to reproduce:** open [`reproduce.ipynb`](reproduce.ipynb) and run all cells (section 9).
- **Results:** [`README.md`](README.md) (summary), [`results/report.md`](results/report.md) (every confusion
  matrix), [`results/model_comparison.md`](results/model_comparison.md) (XGBoost and AlphaEarth).

## Contents

1. [The question and the claim](#1-the-question-and-the-claim)
2. [Data sources](#2-data-sources)
3. [Labels](#3-labels)
4. [Processing pipeline](#4-processing-pipeline)
5. [Features](#5-features)
6. [Models](#6-models)
7. [Validation design](#7-validation-design)
8. [Results and uncertainty](#8-results-and-uncertainty)
9. [How to reproduce](#9-how-to-reproduce)
10. [Limitations and known sources of error](#10-limitations-and-known-sources-of-error)
11. [Files and licenses](#11-files-and-licenses)

---

## 1. The question and the claim

**Question (UC Davis challenge):** from public geospatial data alone, was a corn field harvested for silage or
for grain?

**Hypotheses, stated before modelling:**

| # | Hypothesis | What it predicts in the satellite record |
|---|---|---|
| H1 | Timing: silage is cut earlier (late Aug to early Sep) than grain is harvested (late Sep to Nov) | The big fall in greenness (NDVI) comes earlier for silage |
| H2 | Cut green: silage is cut while the canopy is still green | NDVI falls from a high value in one step; grain fades gradually as it dries |
| H3 | Residue: silage removes the whole plant | Bare soil afterwards, so a low crop-residue index (NDTI); grain leaves stalks (higher NDTI) |

**Claim:** five features built from these hypotheses separate grain from silage on held-out years at BARC,
including when the model is trained only on older years (2017-2020) from a different label source.

**Not claimed:** that the method works outside BARC. Transfer to other regions is untested (section 10).

## 2. Data sources

All sources are public. Imagery is read from Microsoft Planetary Computer (no account needed).

| Source | Resolution | What we use | Access (exact) | When it was pulled |
|---|---|---|---|---|
| Sentinel-2 L2A | 10 m (20 m bands resampled to 10 m, nearest) | Blue B02, Green B03, Red B04, Red-edge B05, NIR B08, NIR-narrow B8A, SWIR B11, B12; scene classification `SCL` | STAC `https://planetarycomputer.microsoft.com/api/stac/v1`, collection `sentinel-2-l2a`, bbox of the fields, 1 Mar to 31 Dec each year, `eo:cloud_cover < 95` | 2021-23: 2026-11-14 21:42 UTC (loader 0.1.1). 2017-20: 2026-11-15 00:55 UTC (loader 0.1.3) |
| Landsat 8 and 9 Collection 2 Level-2 | 30 m | Blue, Green, Red, NIR `nir08`, SWIR `swir16`, `swir22`; `qa_pixel` | Same STAC, collection `landsat-c2-l2`, `platform in [landsat-8, landsat-9]`, same dates and cloud filter | 2021-23: 2026-11-14 22:20 UTC (loader 0.1.2). 2017-20: with Sentinel-2 above |
| Daymet daily weather | 1 km | Daily max/min temperature, precipitation | `https://daymet.ornl.gov/single-pixel/api/data?lat=..&lon=..&vars=prcp,tmax,tmin&start=YYYY-01-01&end=YYYY-12-31`, one request per field centroid (rounded to 0.01 deg) | With each imagery pull |
| BARC evaluation file | field polygons | 83 polygons; `Corn_2021..2023` labels | `04_Evaluation/BARC_Corn.2021-2023.gpkg`, layer `barc_silage_grain_corn20212023` (challenge starter kit) | n/a |
| Kraatz (2023) BARC farm records and outlines, CC-BY 4.0 | per field | Planting tables 2017-2021, harvest tables 2019-2021, field outlines | Zenodo record [8087301](https://doi.org/10.5281/zenodo.8087301); md5 `65590c4f15b2179599bf676172ab36ba` (farm database) and `35fae5dc89622d483b11e4a370f26d9c` (outlines) | Downloaded by `reproduce.py` |
| AlphaEarth Foundations Satellite Embedding V1, annual, CC-BY 4.0 | 10 m, 64 numbers per pixel | One embedding per pixel per calendar year | Source Cooperative [tge-labs/aef](https://source.coop/tge-labs/aef): `aef_index.parquet`, then two UTM 18N tiles per year read over HTTPS | 2026-11-15, about 02:30 UTC |

Each imagery pull writes a provenance JSON beside its parquet (`data/series_<tag>.provenance.json`) with the
loader version, every STAC query, the cloud rules, the scaling, and any failed scene-days.

## 3. Labels

**Blank labels are unlabeled, not negatives.** A blank `Corn_YYYY` means "no information", never "not silage".

| Set | Field-years | Grain | Silage | Source | Used for |
|---|---|---|---|---|---|
| Test | 44 (2021: 18, 2022: 7, 2023: 19) | 32 | 12 | Challenge evaluation file | Testing only, never for choosing features or models |
| Training | 83 on 54 polygons (2017: 24, 2018: 17, 2019: 24, 2020: 18) | 64 | 19 | Kraatz farm records | Training and model choice |
| Harvest-date check | 42 records from 2021 (38 with a harvest date) | | | Kraatz farm records | Checking whether the NDVI drop dates the harvest; never trained on |

How the 2017-2020 labels are built (`kraatz_fields.py`):

1. Keep whole-field rows only. "Corn Field (FLDCRN)" = Grain, "Corn Silage (CSIL)" = Silage.
2. If the harvest table has the field-year, use its class (what was harvested). Otherwise use the planting
   table (what was planned). By year: 2017 and 2018 are planting records only; 2019 has 18 harvest and 6
   planting records; 2020 has 9 of each.
3. Drop field-years listed as both Grain and Silage (ambiguous): none in 2017-2020. Planting and harvest
   class disagree for 1 field-year (the harvest class is kept).
4. Polygon: the evaluation-file polygon when the field name matches (so a field keeps one `field_id` across all
   years), otherwise the Kraatz outline (`field_id` = 1000 + row). Field-years with no polygon are dropped:
   52 of 135, because field names change between years (for example 1-7 split into 1-7A to 1-7F). The 83 kept
   field-years sit on 54 polygons: 46 from the evaluation file, 8 Kraatz outlines.

## 4. Processing pipeline

```
Zenodo farm records ──> kraatz_fields.py ──> 2017-2020 labels + polygons
                                                    │
Evaluation polygons ──┐                             │
                      ├──> preprocessing/loader.py ─┴─> per-field, per-date series (parquet) + Daymet weather
Planetary Computer ───┘                                          │
                                                                 v
                                           grain_silage.py: harmonize, clean, features, validation
                                                                 │
                       aef_embeddings.py (AlphaEarth) ──> model_comparison.py
```

### 4.1 Field pixels

- Fields are projected to UTM zone 18N (EPSG:32618) and shrunk 10 m inward so edge pixels mixing a road or a
  neighbouring field are left out. (If a shrunk field had fewer than 2 pixels the loader would fall back to the
  whole polygon; that fallback was needed for 0 fields.)
- Sentinel-2 is read on a 10 m grid aligned with its native pixels; Landsat on its native 30 m grid (pixel
  edges 15 m past multiples of 30 m).

### 4.2 One reading per field, date and sensor

For each scene-day (scenes grouped by local solar day):

1. **Duplicates:** Planetary Computer lists some Sentinel-2 scenes twice (original and reprocessed). Keep the
   newest copy so one day never mixes two scalings.
2. **Reflectance:** Sentinel-2 `(DN - 1000) / 10000` for processing baseline 04.00 or later (Jan 2022 on), else
   `DN / 10000`. Landsat `DN * 2.75e-5 - 0.2`. DN 0 = no data.
3. **Clear pixels:** Sentinel-2 `SCL` in {4 vegetation, 5 bare, 6 water, 7 unclassified}; Landsat `qa_pixel`
   bits 0-5 all off (fill, dilated cloud, cirrus, cloud, shadow, snow); and red > 0 and NIR > 0.
4. **Indices per pixel:**
   - NDVI = (NIR - Red) / (NIR + Red): greenness
   - NDTI = (SWIR1 - SWIR2) / (SWIR1 + SWIR2): crop residue; low = bare soil
   - NDMI = (NIR - SWIR1) / (NIR + SWIR1): canopy moisture
5. **Field value:** the median over the field's clear pixels.
6. **Usable reading:** at least 60% of the field's pixels are clear. Only usable readings go further.

### 4.3 Weather

Daily Daymet at each field centroid. Corn growing degree days (GDD) per day:
`max(0, (min(tmax, 30) + max(tmin, 10)) / 2 - 10)` in deg C, summed over a range of days.

### 4.4 Putting Landsat on the Sentinel-2 scale

On the same field and day, Sentinel-2 NDVI runs slightly below Landsat. For each year, the median
Sentinel-2 minus Landsat difference over all fields (no labels used) is added to every Landsat reading.

| Year | Same-day pairs | NDVI | NDTI | NDMI |
|---|---|---|---|---|
| 2017 | 106 | -0.014 | -0.024 | -0.053 |
| 2018 | 48 | -0.043 | -0.026 | -0.042 |
| 2019 | 140 | -0.009 | -0.016 | -0.036 |
| 2020 | 220 | -0.024 | -0.008 | -0.041 |
| 2021 | 267 | -0.021 | -0.012 | -0.026 |
| 2022 | 345 | -0.027 | -0.011 | -0.030 |
| 2023 | 448 | -0.027 | -0.020 | -0.056 |

### 4.5 Merge and clean

1. Merge the two sensors and average readings on the same field and day.
2. **Drop isolated dips (missed clouds):** a reading at least 0.15 NDVI below both neighbours, with both
   neighbours within 10 days, is removed; repeat until none is left. This removes 114 of 13,684 merged
   readings (0.8%) for 2021-2023 and 138 of 8,949 (1.5%) for 2017-2020. A real silage cut stays low
   afterwards, so it is not an isolated dip.

Result: at least 13 usable readings from August to November for every labeled 2021-2023 field-year, and at
least 11 for every 2017-2020 field-year.

## 5. Features

One row per field-year. DOY = day of year (Aug 1 = 213 in a non-leap year).

**The five hypothesis features (used by the main model):**

| Feature | Exact definition | Hypothesis |
|---|---|---|
| `drop_doy` | Find the summer peak (below). From the peak on, take every pair of consecutive readings at most 20 days apart, ending by DOY 335. The pair with the largest NDVI fall is "the drop"; `drop_doy` is the midpoint of the pair | H1 |
| `drop_size` | NDVI fall across that pair | H2 |
| `ndvi_pre_drop` | NDVI of the first reading of that pair | H2 |
| `ndvi_sep15` | NDVI linearly interpolated to DOY 258 (Sep 15) | H1 |
| `ndti_post_drop` | Median NDTI of readings 5 to 40 days after `drop_doy` | H3 |

**All 38 hand-built features (used by the random forest and XGBoost):** the five above, plus:

| Feature | Exact definition |
|---|---|
| `peak_ndvi`, `peak_doy` | Highest NDVI between DOY 150 and 260, on a 3-reading running median |
| `greenup_doy` | Find the lowest NDVI between DOY 90 and 180 (skips winter cover crops); the first rise through NDVI 0.4 after it, interpolated |
| `drop_rate` | `drop_size` divided by the days between the pair |
| `ndvi_post_drop` | NDVI of the second reading of the drop pair |
| `regrowth` | Highest NDVI from 20 days after the drop onward, minus `ndvi_post_drop` (fall green-up; probably a cover crop) |
| `t_below_0.6`, `0.5`, `0.4`, `0.3` | Interpolated date NDVI first falls below 0.6 (0.5, 0.4, 0.3) after the peak, with the next reading staying below the threshold + 0.05 |
| `ndvi_`, `ndti_`, `ndmi_` + `aug01`, `aug15`, `sep01`, `sep15`, `oct01`, `oct15`, `nov01` | Each index linearly interpolated to DOY 213, 227, 244, 258, 274, 288, 305 (21 features). Fixed DOYs, so in leap-year 2020 each lands one calendar day earlier |
| `season_days` | `t_below_0.4` minus `greenup_doy` |
| `gdd_greenup_to_drop`, `gdd_greenup_to_0.4` | Corn GDD from green-up to `drop_doy`, and to `t_below_0.4` |

`n_obs`, `n_obs_aug_nov` and `acres` are kept for quality checks and are not model inputs. Missing values
(for example no crossing below 0.3) are filled with the training-fold median for logistic regression and the
random forest; XGBoost handles them natively.

**AlphaEarth embeddings (model comparison only):** `aef_embeddings.py` reads the two UTM 18N tiles covering
BARC for each year, flips them (the files are stored bottom-up), de-quantizes each value
(`(v / 127.5)^2 * sign(v)`, -128 = no data), averages the 64 numbers over each field's pixels after the same
10 m inward buffer, and rescales the average to length 1. Fields have 43 to 1,387 pixels (median 450).
Check that the flip and position are right: pixels inside a field are more alike (cosine 0.975 to the field
mean) than with the rows wrongly flipped (0.889) or shifted 200 m (0.90).

## 6. Models

All models predict a probability of silage; a field-year is called Silage when it is at least 0.5.

| Model | Inputs | Settings (none tuned on the test labels) |
|---|---|---|
| Logistic regression (main model) | The 5 hypothesis features | Median imputation, standardize, `LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000)` |
| Random forest | All 38 | Median imputation, `RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced", random_state=0)` |
| XGBoost | Any feature set | Teammate's `baseline_xgboost.ipynb` settings: `n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=1, reg_lambda=1.0, scale_pos_weight=grain/silage ratio in training, random_state=0` |
| Single-threshold rules (baselines) | One feature | Cut and direction maximizing balanced accuracy on the training data only |
| Farm only (confound check) | Farm name | Silage if Central Farm |

`C=0.5` keeps the weights small (fewer than 100 training rows), and `class_weight="balanced"` counts each
silage mistake more, since silage is the rarer class.

## 7. Validation design

Rules followed throughout:

- The 44 labeled 2021-2023 field-years are the test set. Features, the five-feature list, and model settings
  were fixed before looking at 2021-2023 performance. Where we noticed something by looking at test results, the
  write-up says so.
- No random field splits (the challenge warns against them): every split holds out whole years, whole fields
  or a whole farm.
- When models are compared, the winner is chosen on a split inside 2017-2020 only, then its 2021-2023 scores
  are read off.

| Split | Train | Test | What it tests |
|---|---|---|---|
| Hold out one year, 2021-23 | The other two of 2021-2023 | The held-out year (44 in total) | Year-to-year transfer |
| Same, strict | As above, minus every field that appears in the test year | Same 44 | Field memory |
| Out of time | 2017-2020 farm records only (83) | 2021-2023 (44) | Different years and a different label source; the headline test |
| Out of time, unseen fields | The 53 2017-2020 field-years whose fields never appear in 2021-2023 | Same 44 | Field memory, out of time |
| Hold out one year, 2017-23 | The other six years | Each year in turn (127) | Stability across seven seasons |
| Central Farm holdout | 2017-2020 from APU, East, Linkage and North farms (47: 39 Grain, 8 Silage) | Central Farm 2021-2023 (25: 14 Grain, 11 Silage) | Transfer to another farm (and later years) |
| Model choice | Hold out one year within 2017-2020 (83) | Each 2017-2020 year | Used only to pick a model in `model_comparison.py` |

A full spatial holdout is not possible: 11 of the 12 test silage field-years are on Central Farm.

**Metrics:** confusion matrix, precision and recall for both classes, accuracy, ROC-AUC, and counts per
held-out year (`results/report.md`).

**Uncertainty:** 95% Wilson intervals for recall; exact McNemar tests (on the field-years where two models
disagree) when comparing models on the same test set.

## 8. Results and uncertainty

| Split | Main model (logistic, 5 features) | Random forest (38) |
|---|---|---|
| Hold out one year, 2021-23 | 42/44 (silage 11/12, grain 31/32), AUC 0.96 | 43/44, AUC 0.95 |
| Same, strict | 42/44 | |
| **Out of time (train 2017-20)** | **39/44** (silage 11/12, grain 28/32), AUC 0.98 | **41/44**, AUC 0.95 |
| Out of time, unseen fields | 40/44, AUC 0.98 | |
| Hold out one year, 2017-23 | 113/127 (silage 25/31, grain 88/96), AUC 0.94 | |
| Central Farm holdout | 24/25 | 24/25 |

- **Baselines on the 44:** farm only 29/44; one cut on the drop date 29/44; best single rule chosen on 2017-2020
  (`ndti_sep15 < 0.225`) 35/44.
- **Uncertainty:** silage recall 11/12 has a 95% Wilson interval of 0.65-0.99; grain recall 31/32 has
  0.84-0.99. With 12 silage test cases, one error moves silage recall by 8 points.
- **Model comparison** (`results/model_comparison.md`): XGBoost does not beat the main model on any 2021-2023
  test. AlphaEarth alone is no better than the five features. The five features plus AlphaEarth score best on
  every 2021-2023 test (43/44 out of time), but only tie the main model on the 2017-2020 model-choice split
  (69 vs 68 of 83), and the gain is not significant (McNemar p = 0.22). It is a lead, not a result.
- **Harvest dates:** against 2021 farm records, the NDVI drop lands within 2 days of the cut for 2 of 3 silage
  fields; for grain it lands a median 34 days before harvest (it captures dry-down). `drop_doy` is not a grain
  harvest date.

## 9. How to reproduce

### 9.1 Requirements

- Python 3.12 (tested 3.12.10). Packages: `requirements.txt` in this folder (it includes the shared loader's
  `../../requirements.txt`). Tested versions: scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.4, geopandas 1.2.0,
  rasterio 1.5.2, xgboost 3.4.1.
- Internet access to Zenodo, Planetary Computer, Daymet and Source Cooperative. No accounts or keys.
- About 1 GB free disk for `data/` (imagery series, AlphaEarth index and cached tiles).

### 9.2 Option A: the notebook

Open [`reproduce.ipynb`](reproduce.ipynb) in Jupyter and run all cells. It runs the same scripts as option B,
step by step, skips any step whose outputs already exist, and ends with a check cell that compares the
headline numbers to the expected ones. Settings at the top: `INSTALL` (install requirements into the current
kernel), `WORKERS`, `FORCE_PULL`.

### 9.3 Option B: one command

From the repository root:

```bash
uv venv .venv
uv pip install --python .venv -r research/grain_vs_silage/requirements.txt
.venv/Scripts/python research/grain_vs_silage/reproduce.py      # macOS/Linux: .venv/bin/python
```

`--analysis-only` re-runs only the analysis, figures and model comparison (about a minute once `data/` exists);
`--workers 4` if many scene-days fail; `--force-pull` to re-pull imagery.

### 9.4 Option C: step by step (what `reproduce.py` runs)

| Step | Command (from the repo root) | Writes | Time |
|---|---|---|---|
| 1. Farm records | download the two Zenodo files into `data/kraatz_zenodo_8087301/`, check md5 | the zips | seconds |
| 2. Training labels | `python research/grain_vs_silage/kraatz_fields.py --kraatz-dir data/kraatz_zenodo_8087301 --gpkg 04_Evaluation/BARC_Corn.2021-2023.gpkg --out data` | `data/fields_kraatz_2017_2020.gpkg`, `labels_kraatz_2017_2020.csv`, `records_kraatz_2021.csv` | seconds |
| 3. Imagery 2021-23 | `python preprocessing/loader.py --gpkg 04_Evaluation/BARC_Corn.2021-2023.gpkg --years 2021 2022 2023 --tag barc --cell-m 50000 --workers 8 --weather` | `data/series_barc.parquet`, `weather_barc.parquet`, provenance JSON | 10-30 min |
| 4. Imagery 2017-20 | `python preprocessing/loader.py --fields data/fields_kraatz_2017_2020.gpkg --years 2017 2018 2019 2020 --tag kraatz_2017_2020 --cell-m 50000 --workers 8 --weather` | `data/series_kraatz_2017_2020.parquet`, weather, provenance | 10-30 min |
| 4b. Only if failures | the same loader command with `--retry-failed` instead of `--weather` | merges recovered scene-days | minutes |
| 5. Features + validation | `python research/grain_vs_silage/grain_silage.py` | `results/report.md`, feature tables, predictions | under a minute |
| 6. Figures | `python research/grain_vs_silage/make_figures.py` | `results/fig_*.png` | seconds |
| 7. AlphaEarth | `python research/grain_vs_silage/aef_embeddings.py` | `results/aef_embeddings.csv`, `data/aef/` cache | about 6 min the first time |
| 8. Model comparison | `python research/grain_vs_silage/model_comparison.py` | `results/model_comparison.md` and CSVs | under a minute |

### 9.5 What a correct run looks like

| Check | Where | Expected |
|---|---|---|
| No failed scene-days left | `"failures"` in each `data/series_*.provenance.json` | empty list |
| Enough fall readings | "Min usable readings Aug-Nov" in `results/report.md` | 13 (2021-23), 11 (2017-20) on these inputs |
| Hold out one year, 2021-23, logistic / forest | `results/report.md` | 42/44, 43/44 |
| Out of time, logistic / forest | `results/report.md` | 39/44, 41/44 |
| Out of time, unseen fields, logistic | `results/report.md` | 40/44 |
| Hold out one year, 2017-23, logistic | `results/report.md` | 113/127 |
| Baseline check | `model_comparison.py` | exits with "baseline mismatch" if any of the six numbers above differ |

**If your numbers differ by a field-year or two:** first compare the package versions above. Second, the
2021-2023 series in this repo's results was assembled from two loader runs (Sentinel-2 from loader 0.1.1,
Landsat from 0.1.2, see `data/series_barc.provenance.json`), in which 4 of 1,136 and 1 of 82 scene-day reads
failed and were not retried. A fresh pull retries failures and may add a reading or two. To reproduce these exact
numbers, use the same input files (`data/pod_inputs_grain_vs_silage.zip` on the team machine, with SHA256 sums).

### 9.6 Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Many scene-days in `failures` | Planetary Computer throttles bursts | Re-run the loader command with `--retry-failed`; use `--workers 4` |
| AlphaEarth `.vrt` will not open (`hSrcDS is not set`) | The `.vrt` files point at `/vsis3/` paths | Already handled: `aef_embeddings.py` reads the `.tiff` files directly |
| `ModuleNotFoundError: xgboost` | Older environment | `pip install -r research/grain_vs_silage/requirements.txt` |
| Shared Jupyter server (team pod) | Installing into the shared kernel affects teammates | Use your own virtual environment or leave `INSTALL = False` |

## 10. Limitations and known sources of error

- **Small test set:** 12 silage field-years. Every percentage has wide intervals (section 8).
- **One site, one management team.** BARC cuts silage green in late August to early September and usually
  plants a fall cover crop after. A late silage cut (Central Farm 4-7, 2021, harvested Nov 4) or an early grain
  harvest (2019: grain harvested Sep 10-30) fools the timing features.
- **Fixed calendar dates** (`ndvi_sep15`, the fixed-date features) assume BARC's planting window; other
  latitudes need dates relative to green-up or growing degree days.
- **Label noise:** 2017-2018 labels come from planting records (what was planned, not what was harvested).
  An earlier check outside this folder also flagged five labeled 2022-2023 Grain field-years that look like
  soybeans in the USDA Cropland Data Layer; they were not removed.
- **Clouds:** autumn gaps can hide the cut; Sentinel-1 radar would help and is not used yet.
- **Not tested elsewhere:** silage is a small share of corn in most US counties (median about 3.5% of
  harvested corn acres across counties with at least 20,000 acres, 2022 Census in `02_QuickStats`), so silage
  precision would fall sharply where silage is rare even if the error rates held.

## 11. Files and licenses

| Path | What it is |
|---|---|
| `reproduce.ipynb` | Step-by-step reproduction notebook |
| `reproduce.py` | The same pipeline as one command |
| `kraatz_fields.py` | 2017-2020 training labels and polygons from the Zenodo farm records |
| `../../preprocessing/loader.py` | Shared imagery and weather loader |
| `grain_silage.py` | Harmonization, cleaning, features, validation; writes `results/report.md` |
| `make_figures.py` | The two figures |
| `aef_embeddings.py`, `model_comparison.py` | AlphaEarth features; model comparison |
| `results/` | Every output that is small enough to commit |
| `data/` (repo root, git-ignored) | Downloads and imagery series; rebuilt by the pipeline |

Licenses and attribution: Sentinel-2 (Copernicus) and Landsat (USGS) data are free and open; Daymet is
distributed by ORNL DAAC; the Kraatz farm records (Zenodo 8087301) are CC-BY 4.0; AlphaEarth embeddings are
CC-BY 4.0: "The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind."
