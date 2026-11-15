# Classifier: silage vs grain corn at BARC

This folder holds four pieces of work on telling silage cornfields from grain cornfields at the USDA Beltsville
Agricultural Research Center (BARC), using only public satellite data:

1. **`prithvi/`**: a pretrained Earth-observation model, Prithvi-EO-2.0 (IBM and NASA), adapted to this task and
   trained on the GPU here. It is compared with the team's simple 5-feature model from `research/grain_vs_silage/`.
2. **`labels/`**: the public BARC farm records for years outside 2021-2023, turned into silage/grain labels, and
   checks of how far those labels can be trusted.
3. **`presto/`**: Presto (NASA Harvest), a pretrained pixel time-series model, fine-tuned on the labeled
   field-years. Results and method: `presto/README.md`.
4. **`metrics/`**: shared test metrics for any of these models (accuracy, balanced accuracy, precision, recall and
   F1 per class, MCC, AUC, log loss, Brier, with field-resampled confidence intervals), and their tests.

## Status: Prithvi stops here

Prithvi-EO-2.0 does not beat the 5-feature model on any test, and its best-looking version (8 frames) did not hold
up on years no model had seen (below). Fine-tuned Presto does better than Prithvi on the same seeded splits; frozen
Presto embeddings do worse than Prithvi's through the shared evaluator on two of the three tests below (original
127-row labels):

| Test (correct field-years) | 5-feature model | Prithvi | Presto |
|---|---|---|---|
| Same 5 seeded splits, 101 test field-years, version chosen on validation | 90 | 87 | 95 (fine-tuned) |
| Shared evaluator: train 2017-2020, test the 44 official labels | 39 | 38 | 34 (frozen) |
| Shared evaluator: hold out one year, 2017-2023 (127) | 113 | 115 | 109 (frozen) |
| Shared evaluator: train on other farms, test Central Farm (25) | 24 | 16 | 18 (frozen) |

Neither pretrained model beats the 5-feature model significantly. Fine-tuned Presto, chosen on 2017-2020 only, gets 41
of 44 on the official test (right where the 5-feature model is wrong 4 times, the reverse twice, McNemar p = 0.69).
Both do worse than the 5-feature model on Central Farm when trained on the other farms. The Presto numbers are in
`classifier/presto/results/`: `presto/` and `standard_test/` for the fine-tuned model,
`shared_evaluator_frozen/` for its frozen embeddings through `evaluate.py`.

## Results so far

**Prithvi does not beat the simple model.** Same field-level splits, 5 different splits (seeds), each test part
scored once:

| | Prithvi | 5-feature model |
|---|---|---|
| Correct | 87 of 101 | 90 of 101 |
| Silage fields caught | 14 of 25 | 21 of 25 |
| Grain fields called silage | 3 of 76 | 7 of 76 |

| Seed | Test field-years (silage) | Prithvi version chosen on validation | Prithvi correct | 5-feature correct | McNemar p |
|---|---|---|---|---|---|
| 0 | 19 (4) | 600M, 30 m, frozen | 16 | 18 | 0.63 |
| 1 | 21 (6) | tiny, 15 m, frozen | 17 | 20 | 0.25 |
| 2 | 21 (5) | 100M, 15 m, fine-tuned | 18 | 18 | 1.0 |
| 3 | 20 (5) | 100M, 15 m, fine-tuned | 18 | 17 | 1.0 |
| 4 | 20 (5) | 100M, 30 m, frozen | 18 | 17 | 1.0 |

- No seed shows a significant difference. With about 20 test field-years per seed, one field-year is 5 percentage
  points. A field-year can be in more than one seed's test part, so the 101 are not 101 different field-years.
- Prithvi misses more silage. It calls fewer fields silage overall, so it also makes fewer false silage calls.
- Validation parts are small (19-21 field-years) and many versions score perfectly on them (seed 0: the top 5 all
  21 of 21), so the choice between versions is weakly informed.

**Through the team's shared evaluator** (`prithvi/prithvi_evaluate.sh`; same protocol as Presto, AlphaEarth and
TESSERA: the head and the Prithvi version are chosen on 2017-2020 only, then every test is run once):

| Test | 5-feature baseline | Prithvi, chosen (300M, 30 m, 4 frames) |
|---|---|---|
| Train 2017-2020, test the 44 official 2021-2023 labels | 39 of 44 | 38 of 44 |
| Hold out one year, 2017-2023 | 113 of 127 | 115 of 127 |
| Hold out one year, 2021-2023 | 42 of 44 | 40 of 44 |
| Train on other farms, test Central Farm 2021-2023 | 24 of 25 | 16 of 25 |
| Train on fields never in 2021-2023, test 2021-2023 | 40 of 44 | 36 of 44 |
| Choice split: hold out one year, 2017-2020 | 68 of 83 | 76 of 83 |

- The chosen version adds the 5 hand features to 32 principal components of the embedding.
- The two versions with 8 two-week frames (Aug 1 to Nov 20) lost the choice by 2 field-years (74 vs 76 of 83) but
  matched or beat the chosen one on every test: 42 and 41 of 44 on the official test, 122 and 119 of 127 holding out
  one year, 24 and 23 of 25 on Central Farm. Because they were not chosen on 2017-2020, these are not results to claim;
  they were tested once on new years (below).
- For comparison, frozen Presto embeddings plus the 5 hand features scored 39 of 44 on the official test (also not
  the evaluator's choice for Presto, which kept the embedding alone: 34 of 44).

**On years no model has seen, the 8-frame version does not hold up** (`prithvi/prithvi_fresh.sh`). 45 field-years
from the farm reports in 2014, 2015, 2016 and 2024 (35 Grain, 10 Silage), never used for training or any choice.
Each model was fitted once on all 127 labeled field-years, with its head already chosen; the test was written down
and committed before these years' inputs were built, and run once.

| Model | Correct of 45 | Silage caught | Grain called silage | AUC |
|---|---|---|---|---|
| 5-feature baseline | 29 | 3 of 10 | 9 of 35 | 0.66 |
| Prithvi, 8 frames (the version under test) | 31 | 4 of 10 | 8 of 35 | 0.71 |
| Prithvi, 4 frames (the evaluator's choice) | 35 | 4 of 10 | 4 of 35 | 0.76 |
| Calling every field Grain | 35 | 0 of 10 | 0 of 35 | |

- 8 frames against the baseline: right where the baseline is wrong 3 times, the reverse once (McNemar p = 0.63).
  4 frames: 7 and 1 (p = 0.07). Neither difference is significant.
- Every model does poorly here; none beats calling every field Grain. In 2014 (Landsat 8 only, 8 usable days from
  June to November against 50 in 2021) no model catches any of the 5 silage fields. In 2024 the baseline and the
  8-frame version call 7 of 8 grain fields silage.
- Every silage label here rests on a planting record only, the least certain kind (see below). The hand features
  rebuild the team's 2021 values exactly (83 of 83), so the baseline's inputs are built the same way as before.

**The extra labels hold up where they are backed by a harvest record.** The farm's public records never contradict
the official labels where both exist. A model fitted only on the official labels agrees with 27 of 27 harvest-backed
training labels, but with only 6 of 12 silage labels that rest on a planting record alone (what the farm planned to
grow). 12 of the 19 silage labels in the 2017-2020 training data are planting-only. Details: `labels/README.md`.

**Presto matches the 5-feature model but is not shown to beat it.** With the fixed labels (125 field-years, without
235/2020 and 278/2017), the fine-tuned Presto gets 41 of 44 on the standard test (train on 2017-2020, score the 44
official 2021-2023 labels) and the 5-feature model 39 of 44. On the mixed-year splits it is 92 of 101 against 96 of
101. In both cases the 95% interval of the accuracy difference includes 0 (`presto/README.md`, `metrics/README.md`).
Presto's small lead on the standard test comes from November and December, when BARC's silage fields green up again
(probably a cover crop, left out of the 5-feature model on purpose). Without those months Presto gets 40 of 44 and
catches 8 of 12 silage fields, against 11 of 12 for the 5-feature model.

## Folder layout

| Path | What it holds |
|---|---|
| `prithvi/README.md` | How the Prithvi classifier is built, trained and chosen |
| `prithvi/field_years.csv` | The 127 labeled field-years used (field outline, label, label source, 5 hand features) |
| `prithvi/prithvi_field_years.py`, `prithvi_chips.py`, `prithvi_train.py` | Label list, image stacks, training and scoring |
| `prithvi/results/` | Written by the scripts (full result tables are kept on the GPU workspace; the key numbers are above) |
| `labels/README.md` | Label sources, licenses, checks and results |
| `labels/*.py`, `labels/run_all.sh` | Download, parse and check the farm records |
| `labels/results/` | `extra_labels_outside_2021_2023.csv`, the combined label table; `run_all.sh` rebuilds it and every check |
| `labels/apply_yield_fixes.py`, `labels/results/field_years_fixed.csv` | Label fixes from the combine yield records; the fixed 125-row table Presto trains on |
| `quebec/` | 2,245 Québec field-years (FADQ declarations, 2021-2023), optional training-only rows for Presto |
| `presto/README.md` | Presto results, pipeline and method |
| `presto/presto_pixels.py`, `presto_train.py`, `presto_standard_test.py`, `presto_embed.py` | Pixel series, mixed-year splits, standard test, embeddings for `prithvi/evaluate.py` |
| `presto/results/` | Summaries, test predictions and `metrics.md` per run |
| `metrics/README.md` | Metric definitions, decisions, how to run and read them |
| `metrics/metrics.py`, `report.py`, `test_metrics.py` | Metric code, report writer (`metrics.md`, `metrics.csv`), tests |

## Run

Both parts use the Python environment `/home/keeboh/venvs/prithvi` and run from the repository root.

```bash
PY=/home/keeboh/venvs/prithvi/bin/python

# Labels: download the sources (checksums checked), rebuild every table and check. About a minute.
PY=$PY bash classifier/labels/run_all.sh /home/keeboh/work/labels

# Prithvi: label list, image stacks (one process per year), then training on the GPU.
$PY classifier/prithvi/prithvi_field_years.py
for y in 2017 2018 2019 2020 2021 2022 2023; do
  $PY classifier/prithvi/prithvi_chips.py --out /home/keeboh/work/prithvi/chips --years $y --workers 6 &
done; wait
HF_HOME=/home/keeboh/hf $PY classifier/prithvi/prithvi_train.py --chips /home/keeboh/work/prithvi/chips \
  --out classifier/prithvi/results/prithvi
```

Building the image stacks reads Sentinel-2 and Landsat from Microsoft Planetary Computer: 5.5-7.5 minutes per year,
with the years run in parallel. Training all 5 seeds and 40 candidates per seed took 137 seconds on one NVIDIA B200.

## Large files (outside `/workspace`)

| Path | Contents |
|---|---|
| `/home/keeboh/work/prithvi/chips/` | 30 m six-band image stacks per year, 2017-2023 |
| `/home/keeboh/work/prithvi/logs/` | Run logs (`train.log` is the full training run) |
| `/home/keeboh/hf/` | Prithvi-EO-2.0 checkpoints (Hugging Face cache) |
| `/home/keeboh/venvs/prithvi/` | Python environment (PyTorch, timm, geopandas, scikit-learn, pypdf) |
| `/home/keeboh/work/labels/` | Downloaded label sources and the text of the farm-report PDFs |
| `/home/keeboh/work/fresh/` | Inputs for the 2014-2016 and 2024 test: test list, loader series, image stacks, embeddings |

## Limitations

- One farm. All labels, official and extra, come from BARC.
- The Prithvi splits mix years, so a field's neighbours in time can be in training. The results are not directly
  comparable with the train-on-2017-2020, test-on-2021-2023 numbers.
- Prithvi was pretrained on NASA HLS imagery. HLS is on Planetary Computer only from 2020, so the same six bands are
  built from Sentinel-2 and Landsat without HLS's BRDF and bandpass adjustments.
- 14 of the 40 fine-tuning runs kept their last or second-to-last epoch (39 or 40 of 40), so they may have stopped
  before converging.
- 56 of the 83 farm-record training labels rest on a planting record only. 5 official Grain labels look like soybeans
  in the USDA Cropland Data Layer (field_id 251 and 286 in 2022; 230, 231 and 397 in 2023).
