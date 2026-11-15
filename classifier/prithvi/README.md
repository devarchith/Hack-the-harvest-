# Prithvi-EO-2.0 classifier: silage vs grain corn

A pretrained Earth-observation model (Prithvi-EO-2.0, IBM and NASA, Apache 2.0) fine-tuned on the labeled BARC
field-years to tell silage corn from grain corn, scored against the 5-feature logistic regression in
`research/grain_vs_silage/` on the same split.

## Pipeline

| Step | Script | Runs on | Output |
|---|---|---|---|
| 1. Labeled field-years with polygons and label checks | `prithvi_field_years.py` | any machine with the repo `data/` folder | `field_years.csv` (127 rows) |
| 2. 30 m six-band image stacks, June-November, 2017-2023 | `prithvi_chips.py` | CPU, network | `<chips>/<year>.npz`, `<chips>/<year>.provenance.json` |
| 3. Split, train, choose on validation, test once | `prithvi_train.py` | NVIDIA GPU | `results/prithvi/` |
| 4. Frozen embeddings scored with the team's shared evaluator | `prithvi_evaluate.sh` (runs `prithvi_embed.py`, `evaluate.py`, `prithvi_select.py`) | NVIDIA GPU, then CPU | `results/evaluate/` |
| 5. One test on 2014-2016 and 2024 field-years no model has seen | `prithvi_fresh.sh` (runs `prithvi_fresh.py`, the team loader, `prithvi_chips.py`, `prithvi_embed.py`) | CPU, network, NVIDIA GPU | `results/fresh/` |

```bash
PY=/home/keeboh/venvs/prithvi/bin/python
$PY classifier/prithvi/prithvi_field_years.py
for y in 2017 2018 2019 2020 2021 2022 2023; do
  $PY classifier/prithvi/prithvi_chips.py --out /home/keeboh/work/prithvi/chips --years $y --workers 6 &
done; wait
HF_HOME=/home/keeboh/hf $PY classifier/prithvi/prithvi_train.py --chips /home/keeboh/work/prithvi/chips \
  --out classifier/prithvi/results/prithvi
```

Step 3 needs `torch`, `timm`, `einops`, `huggingface_hub` and the packages in `research/grain_vs_silage/requirements.txt`.
It downloads the Prithvi checkpoints from Hugging Face (`ibm-nasa-geospatial/Prithvi-EO-2.0-{tiny,100M,300M,600M}-TL`).

## Inputs

- **Labels:** 83 field-years from the BARC farm records 2017-2020 (Kraatz 2023, Zenodo 8087301, CC BY 4.0) and
  the 44 labeled field-years of the 2021-2023 evaluation file. 96 Grain, 31 Silage, 64 distinct fields.
  Unlabeled field-years are not used. How far the farm-record labels can be trusted: `../labels/README.md`.
  `field_years.csv` carries two columns from those checks (`../labels/outline_check.py`): `label_basis`
  (`harvest record`, `planting record only`, or `official`) and `outline_fit` (`ok`, `misfit`, or `official`).
- **Imagery:** Sentinel-2 L2A and Landsat 8/9 Collection 2 Level-2 from Microsoft Planetary Computer, the same
  collections, cloud rules and reflectance scaling as `preprocessing/loader.py`. Both are put on one 30 m grid
  (EPSG:32618, Landsat pixel edges) in Prithvi's band order: blue, green, red, narrow NIR, SWIR 1, SWIR 2.
  Prithvi was pretrained on NASA HLS; Planetary Computer holds HLS only from 2020, so the six bands are built from
  the source collections. HLS's BRDF and bandpass adjustments are not applied.
- **Model input per field-year:** a 48 x 48 pixel chip (1.44 km) centred on the field; four frames, each the
  per-pixel median of cloud-free readings in Jul 1-31, Aug 10-Sep 5, Sep 6-Oct 5 and Oct 6-Nov 15; the year and
  day-of-year of each frame and the field's latitude and longitude as Prithvi's time and location inputs.

## Method (the run in `results/prithvi/`)

- **Split:** 70/15/15 train/validation/test by field: every year of a field is in the same part
  (StratifiedGroupKFold, 20 folds: 14 train, 3 validation, 3 test). Repeated for 5 split seeds (0-4).
- **Candidates:** model size (tiny 5M, 100M, 300M, 600M parameters) x input scale (30 m, or upsampled to 15 m so one
  16 x 16 patch is 240 m, about one field) x training mode:
  - frozen encoder, features averaged over the 8 rotations and flips, logistic regression head (C chosen with the rest);
  - last two encoder blocks and a linear head fine-tuned (AdamW, encoder 2e-5, head 1e-3, class-weighted loss,
    random rotations and flips, batch 16, up to 40 epochs, stop after 10 epochs without a lower validation log loss,
    keep the best epoch).
- **Pooling:** encoder patch tokens are averaged per frame, weighted by the share of each patch inside the field
  polygon, then the four frame vectors are concatenated.
- **Choice and test:** each seed picks the candidate with the lowest validation log loss. The test part is scored once
  for that candidate and for the baseline (5 hand features, logistic regression, fitted on the same training part).

## Shared evaluator (step 4)

`prithvi_evaluate.sh` scores frozen Prithvi embeddings the same way the Presto, AlphaEarth and TESSERA embeddings were
scored: the 5-feature baseline must first reproduce its six published numbers; the head (embedding alone or with the
5 hand features, PCA size, C) is chosen on the 2017-2020 hold-out-one-year split only; the chosen head is then run once
on every split, including train 2017-2020, test on the 44 official 2021-2023 labels.

- Four embedding tables: the 300M model, chip at 30 m or upsampled to 15 m, and either the 4 frames above or 8
  two-week frames from Aug 1 to Nov 20 (a pixel with no clear reading in a frame takes the nearest frame's value).
- `prithvi_select.py` picks one table, also on the 2017-2020 split only; the others are listed for reference.
- `evaluate.py` is the shared evaluator with one change: the PCA size is capped at what the training fold allows.
  Plain PCA fails when the chosen size (32) exceeds the 26 field-years of the other-farms split; nothing changes
  where it ran before.
- It needs `research/grain_vs_silage/model_comparison.py` and `results/aef_embeddings.csv` from `main`, so the script
  runs it inside a fresh copy of `main`.

## Test on unseen years (step 5)

`prithvi_fresh.py` fixes the test before any input for these years exists (its docstring is the declaration):

- **Test set:** the 45 field-years in `../labels/results/extra_labels_outside_2021_2023.csv` that have an outline in
  the 2021-2023 evaluation file and are not among the 127 used so far: 2014, 2015, 2016 and 2024, 35 Grain, 10 Silage.
- **Models:** the 8-frame and 4-frame 300M tables and the 5-feature baseline, each fitted once on all 127 labeled
  field-years with the head `evaluate.py` chose on the 2017-2020 split.
- **Inputs:** the hand features come from the team loader and `grain_silage.py`'s feature functions, run over every
  field in the evaluation file. 2021 is fetched as a check, and the rebuilt features match the team's 2021 values
  exactly. 2014 and 2015 have no same-day Sentinel-2 and Landsat readings, so their Landsat offset is the median of
  the other years'.
- **Result:** correct of 45 at p >= 0.5: baseline 29, 8 frames 31, 4 frames 35, all Grain 35. Per year, per label
  basis, McNemar tests and the checks are in `results/fresh/summary.md`.

## Runs without the least certain labels (not run yet)

`--exclude` drops farm-record field-years before the split. The run in `results/prithvi/` used all 127 field-years.

```bash
$PY classifier/prithvi/prithvi_train.py --chips /home/keeboh/work/prithvi/chips \
  --out classifier/prithvi/results/prithvi_no_misfit --exclude outline-misfit
$PY classifier/prithvi/prithvi_train.py --chips /home/keeboh/work/prithvi/chips \
  --out classifier/prithvi/results/prithvi_no_misfit_no_planting_silage --exclude outline-misfit planting-only-silage
```

| `--exclude` | Field-years left out | Field-years left |
|---|---|---|
| (none) | 0 | 127 (96 Grain, 31 Silage) |
| `outline-misfit` | 10 Grain whose outline does not fit the field in that year | 117 (86 Grain, 31 Silage) |
| `outline-misfit planting-only-silage` | also 12 Silage that rest on a planting record only | 105 (86 Grain, 19 Silage) |

The chips do not need rebuilding. The splits differ from the full run, so compare each run with its own baseline
column, not test rows one by one. `summary.md` states what was left out.

## Outputs (`results/prithvi/`)

- `summary.md`: test results per seed against the baseline, McNemar test, top validation candidates, input check.
- `validation_scores.csv`: every candidate on every seed's validation part; `epoch` is the kept fine-tuning epoch.
- `test_predictions.csv`: test-part probabilities for the chosen candidate and the baseline.
- `splits.csv`: which part each field-year was in, per seed. `run.json`: versions, GPU, checkpoint and input hashes.
  The run was made when this folder was `research/pretrained_classifier/`, so `run.json` records those paths.
  It also predates the `label_basis` and `outline_fit` columns, so its `field_years.csv` hash is of the 14-column file.

## Caveats

- One farm (BARC, Maryland). About 20 test field-years per seed, so one field-year is about 5 percentage points.
- 56 of the 83 farm-record labels rest on a planting record only (what was planned), 12 of them Silage. A model
  fitted on the official labels agrees with 6 of those 12 (`../labels/README.md`).
- 10 farm-record Grain field-years use an outline that does not fit the field in that year: the recorded acres differ
  from the outline by more than 20-25%, and in four 2017 cases the record covers 2-7 acres of a 27-41 acre outline
  (`../labels/README.md`). They are in the run in `results/prithvi/`.
- 5 official Grain field-years look like soybeans in the USDA Cropland Data Layer (field_id 251 and 286 in 2022;
  230, 231 and 397 in 2023).
- 14 of the 40 fine-tuning runs kept epoch 39 or 40, so the 40-epoch limit may have stopped them early.
- This split mixes years, unlike the "train 2017-2020, test 2021-2023" result in `research/grain_vs_silage/`, so the
  two sets of numbers are not directly comparable.
