# Grain vs silage research folder: notes for Claude

This folder is a self-contained, reproducible line of work for the UC Davis "Silage or Grain?" challenge
(`01_Challenge/challenge_statement.md`). Read `README.md` here first: it has the hypothesis, the features,
the validation design, the current results and the open questions.

## Ground rules (from the challenge; do not break them)

- **Blank labels are unlabeled, not negatives.** Only the 44 labeled 2021-2023 field-years (32 Grain,
  12 Silage) are evaluation data. Never treat a blank `Corn_YYYY` as "not corn" or "not silage".
- **Public data only.** Every source must be documented (URL, dates, query, processing). The loader writes a
  provenance JSON beside each parquet; keep that habit for any new source.
- **The 2021-2023 labels are the test set.** Do not tune thresholds, pick features or choose models by looking
  at 2021-2023 performance and then report that same performance as a result. Choose on 2017-2020 (or inside
  the training folds) and test on 2021-2023. If you explore on 2021-2023, say so in the report.
- **A proxy is not a label.** "Early NDVI drop" is a hypothesis about harvest timing, not ground truth.
- Report, at minimum: confusion matrix, precision and recall for both classes, counts by year and class, the
  split design, and uncertainty. `grain_silage.py` already prints all of these to `results/report.md`.

## Commands (run from the repo root)

```bash
uv venv .venv && uv pip install --python .venv -r research/grain_vs_silage/requirements.txt   # or pip install -r
python research/grain_vs_silage/reproduce.py                 # everything; skips pulls whose outputs exist
python research/grain_vs_silage/reproduce.py --analysis-only # just features + validation + figures (seconds)
```

`data/` is git-ignored at every level (the root `.gitignore` has `data/`), so the imagery pulls never reach
GitHub. Never name a new folder `data` inside this folder expecting it to be committed. Results that should be
shared go in `results/` (small CSVs, `report.md`, PNGs).

## Code map

| File | What it does |
|---|---|
| `reproduce.py` | Pipeline driver: Zenodo download (md5-checked) -> `kraatz_fields.py` -> loader pulls -> analysis -> figures |
| `kraatz_fields.py` | Turns the Kraatz BARC farm records (Zenodo 8087301) into 2017-2020 Grain/Silage field-years with polygons; also writes the 2021 records used only for the harvest-date check |
| `../../preprocessing/loader.py` | Shared team loader. `--gpkg` for the evaluation fields, `--fields` for any polygon file (added for this work) |
| `grain_silage.py` | `merged_series()` harmonizes Landsat to Sentinel-2 and drops isolated cloud dips; `features_one()` computes every feature for one field-year; `main()` / `extra_section()` run every validation and write `results/` |
| `make_figures.py` | NDVI trajectories by class and the key-feature dot plots |
| `aef_embeddings.py` | AlphaEarth annual embeddings (source.coop, CC-BY 4.0) averaged per labeled field-year -> `results/aef_embeddings.csv`; caches tile windows in `data/aef/` |
| `model_comparison.py` | 3 models x 4 feature sets x 6 splits; asserts the baseline numbers first; picks a model on 2017-2020 only -> `results/model_comparison.md` |

To add a feature: compute it in `features_one()` and it automatically appears in
`results/univariate_separation.csv` and in the random forest. Add it to the `small` list (the pre-stated
hypothesis features used by the logistic regression) only with a written hypothesis, decided before you look at
its 2021-2023 separation.

## Gotchas already found

- **Sentinel-2 vs Landsat offset:** S2 NDVI runs about 0.02-0.03 below Landsat on the same field and day.
  `sensor_offsets()` measures it per year from all fields (no labels) and shifts Landsat.
- **Missed clouds:** isolated NDVI dips (>= 0.15 below both neighbours within 10 days) look like a silage cut.
  `merged_series()` removes them; about 1% of readings.
- **Fall regrowth:** silage fields at BARC green up again after the cut (probably a cover crop). It is the
  single strongest separator but is a management practice and may not transfer; it is deliberately not in the
  `small` feature list.
- **The NDVI drop is the silage cut, but for grain it is dry-down, not harvest.** Against farm-record harvest
  dates, the drop lands within 2 days of the cut for silage and weeks before harvest for grain.
- **Planetary Computer throttles bursts.** At 32 workers, 50 of 384 scene-days of the 2017-2020 pull failed
  (mostly 2017 Jul-Nov, which left 2017 with 2-7 usable fall dates). `loader.py --retry-failed` re-reads only
  the failed scene-days listed in the provenance JSON and merges them; it recovered all 50. `reproduce.py`
  defaults to 8 workers and runs one retry automatically. Always check `failures` in the provenance JSON and the
  "Min usable readings Aug-Nov" line in `results/report.md` before trusting a run.
- **This machine's 2021-2023 input** (`data/series_barc.parquet`) was assembled from two earlier loader runs
  (Sentinel-2 from 0.1.1 after the duplicate-scene fix, Landsat from 0.1.2 after the grid fix); see
  `data/series_barc.provenance.json`. A fresh `reproduce.py` pull produces the same content in one run.
- **CDL 2022-2023:** CropScape refused connections during the hackathon; Planetary Computer's `usda-cdl` ends at
  2021; the NASS national zips are 2 GB each with deflate-compressed GeoTIFFs inside (no partial reads).
  Grain vs silage does not need CDL (all labeled field-years are corn); labeling unlabeled field-years does.
- **Field names change between years** (1-7 split into 1-7A to 1-7F, etc.). `kraatz_fields.py` drops
  field-years with no matching polygon (52 of 135) rather than guess.
- **AlphaEarth files are stored bottom-up**, and their `.vrt` wrappers point at `/vsis3/` paths that need S3
  settings. `aef_embeddings.py` reads the raw `.tiff` windows over HTTPS and flips them. Checked: field masks on
  the flipped mosaic give within-field cosine 0.975 vs 0.889 on a wrongly flipped one.
- **Choose models on 2017-2020 only.** `model_comparison.py` ranks every model on a hold-out-one-year split
  inside 2017-2020 and only then reads its 2021-2023 scores. "Hand 5 + AlphaEarth" looks best on 2021-2023 but
  ties the baseline on that split; do not promote it on 2021-2023 numbers alone.
- Windows: run scripts with the venv's `python`; paths in this folder are all relative to the repo root.

## Open tasks (pick up here)

See "Next steps" in `README.md`. Short version: Sentinel-1 for cloudy fall periods, a residue/harvest-date
detector that works for grain, a spatial holdout, label-noise checks on the 2017-2018 planting-record labels,
and a corn mask (CDL) so the method can label the unlabeled field-years.
