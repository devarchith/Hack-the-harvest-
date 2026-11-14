# Hack the Harvest 2026: silage or grain?

Team repository for Hack the Harvest (USDA NASS, NVIDIA, UC Davis CATS and HMCI), Aggie Square, Sacramento,
November 14 to 15, 2026.

**Question:** using only public geospatial data, can we tell whether a corn field was harvested for silage
(chopped green, the whole plant) or for grain (combined after it dries)?

## Working hypotheses

- **H1, timing.** Silage fields lose their greenness abruptly (a "cliff"), weeks before grain fields finish
  drying down.
- **H2, surface.** After harvest, silage fields show bare soil and grain fields show stalk residue. We measure
  residue with the tillage index NDTI, built from Sentinel-2 and Landsat shortwave-infrared bands.
- **H3, exploratory.** Fields chopped early may green up again in the fall from a cover crop. Treat this as
  untested until we test it.

## Evaluation

The organizers' evaluation labels come from the USDA Beltsville Agricultural Research Center (BARC) in
Maryland: 44 labeled field-years for 2021 to 2023 (32 Grain, 12 Silage). They are in the organizers' starter kit,
which is copied into this repository (`04_Evaluation/`); cite the sources below when using them. We report:

- a confusion matrix
- precision and recall for both classes
- the false positive rate
- counts by year and class
- a leave-one-year-out split
- the same metrics for simple baselines, on the same rows

## Layout

| Path | What |
|---|---|
| `01_Challenge/` to `04_Evaluation/`, `Prior_Work/` | The organizers' starter kit (its README is below) |
| `preprocessing/` | Scene search, cloud masking, per-field time series |
| `notebooks/` | Analysis notebooks, one per person; combined at the end |
| `docs/PLAN.md` | Team plan: facts, assumptions and proposed decisions to review |
| `docs/PREPROCESSING.md` | What "data done" means, and the known pitfalls |
| `data/` | Local only (git-ignored): downloads and derived tables |

## Using the data loader

`preprocessing/loader.py` builds one table of per-field satellite readings that everyone shares. Its columns
are listed at the top of the file; they are a proposal (see `docs/PREPROCESSING.md`).

```bash
pip install -r requirements.txt
python preprocessing/smoke_test.py      # about 1-2 min: two small squares near BARC, no evaluation data needed
```

From the repository root (on the Keeboh pod the same file is at `/workspace/04_Evaluation/`):

```bash
python preprocessing/loader.py --gpkg 04_Evaluation/BARC_Corn.2021-2023.gpkg     --years 2021 2022 2023 --months 3 12 --weather --out data
```

This writes `data/series_barc.parquet`, `data/series_barc.provenance.json` and `data/weather_barc.parquet`.
In a notebook:

```python
import sys; sys.path.insert(0, "..")          # repository root, if the notebook is in notebooks/
from preprocessing.loader import load_series, add_rain_flag
import pandas as pd
series = load_series("../data/series_barc.parquet")          # usable readings only by default
series = add_rain_flag(series, pd.read_parquet("../data/weather_barc.parquet"))
```

Labels come only from `barc_labels()`, and only in the validation notebook.

## Rules for this repository

- API keys go in a `.env` file only, never in a notebook or source file.
- No imagery, rasters or large files in git; `.gitignore` covers the usual ones.
- The official submission is the team's Keeboh workspace, so keep this repository in sync with it.
- All code here is written during the build window (Sat 10:30 AM to Sun 9:00 AM PT).

## Public data sources

| Source | Use |
|---|---|
| [Sentinel-2 L2A](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a) (Microsoft Planetary Computer); [Earth Search](https://www.element84.com/earth-search/) | 10 to 20 m reflectance, NDVI, NDTI, red edge |
| [Landsat Collection 2 Level-2](https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2) | 30 m reflectance to fill cloud gaps |
| [gridMET](https://www.climatologylab.org/gridmet.html) or [Daymet](https://daymet.ornl.gov/) | Rain before each reading (wet-soil flag) |
| [USDA Cropland Data Layer](https://www.nass.usda.gov/Research_and_Science/Cropland/SARS1a.php) | Corn mask, if we scale beyond BARC |
| [USDA NASS Quick Stats](https://quickstats.nass.usda.gov/) | County and state totals for context, not field labels |

## Citations for the evaluation data

- Dulaney, W., et al. (2024). Development of a gridded yield data archive for farm management and research at
  the USDA Beltsville Agricultural Research Center. *Agrosystems, Geosciences & Environment*, 7(1), e20474.
  https://doi.org/10.1002/agg2.20474
- Kraatz, S., et al. (2023). Comparing NISAR (Using Sentinel-1), USDA/NASS CDL, and Ground Truth Crop/Non-Crop
  Areas in an Urban Agricultural Region. *Sensors*, 23(20), 8595. https://doi.org/10.3390/s23208595

---

*Below: the starter kit's README, as the organizers wrote it.*

# UC Davis Hackathon: Silage or Grain?

## Detecting Corn Harvest Type with Open Geospatial Data

Can satellite imagery and other public geospatial data tell whether a corn field
was harvested for **silage** or **grain**?

This repository contains the challenge statement, open-data references, prior
work, Quick Stats context, and labeled BARC evaluation data for the UC Davis
Hackathon. Teams may choose their own sensors, data sources, features, and
models. The included notebooks are starting points, not a required solution.

## Repository Guide

### `01_Challenge`

- [challenge_statement.md](01_Challenge/challenge_statement.md): the complete
  problem statement, data rules, evaluation expectations, and suggested lines
  of inquiry.

### `02_QuickStats`

- [download_corn_acres.py](02_QuickStats/download_corn_acres.py): downloads
  USDA NASS Quick Stats corn acreage records.
- [corn_grain_silage_acres_2012_2025.csv](02_QuickStats/corn_grain_silage_acres_2012_2025.csv):
  committed context data covering planted and harvested acreage.
- [methodology_and_caveats.md](02_QuickStats/methodology_and_caveats.md): API
  setup, schema, source details, and limitations.

The downloader uses county aggregation for the 2012, 2017, and 2022 Census
years, and state aggregation for other years. NASS provides all-corn planted
acreage and separate grain/silage harvested acreage series.

To download a fresh copy, request a NASS Quick Stats API key and set it in the
current PowerShell session:

```powershell
$env:NASS_API_KEY = "your-key-here"
python .\02_QuickStats\download_corn_acres.py
```

### `03_Geospatial_References`

- [data_sources.md](03_Geospatial_References/data_sources.md): multiple open
  geospatial data pathways, source-validation guidance, and feature ideas.
- [starter_notebook.ipynb](03_Geospatial_References/starter_notebook.ipynb):
  one optional Sentinel-2 workflow using a field GeoJSON AOI, cloud masking,
  NDVI, and harvest-timing features.

The Sentinel-2 workflow is only an example. Teams are encouraged to explore
other public sensors, catalogs, weather sources, processing methods, and
modeling strategies, provided they validate and document their sources.

### `04_Evaluation`

- `BARC_Corn.2021-2023.gpkg`: BARC field multipolygons with yearly Grain or
  Silage reference labels.
- [barc_fields_explorer.ipynb](04_Evaluation/barc_fields_explorer.ipynb): loads,
  summarizes, reshapes, and maps the labeled fields.

The GeoPackage contains 83 BARC fields in Maryland with labels for 2021-2023.
The `Corn_2021`, `Corn_2022`, and `Corn_2023` columns mark silage or grain corn
that was planted and harvested in that year: 44 labeled field-years (32 Grain,
12 Silage) on 34 unique fields. Blank yearly labels are unlabeled records, not
negative examples. The explorer map colors Grain fields gold and Silage fields
green.

The BARC field polygons were developed and published in prior studies. Please
cite them as described in [Data Citation](#data-citation).

## Prior Work

- [reading_list.md](Prior_Work/reading_list.md): annotated DOI-linked
  papers on harvest timing, phenology, radar, and crop residue.
- Three included PDF papers provide additional methods for phenology and
  management-event detection.

## Suggested Workflow

1. Read the [challenge statement](01_Challenge/challenge_statement.md).
2. Inspect the BARC data and its label coverage with the evaluation notebook.
3. Choose and validate public geospatial sources.
4. Build a transparent timing or residue baseline before adding complexity.
5. Hold out a year or spatial group when possible.
6. Report precision, recall, a confusion matrix, class counts, uncertainty,
   and known geographic or temporal limits.

## Data and Reproducibility Rules

- Use public data only.
- Do not treat a proxy such as an early NDVI decline as ground truth.
- Record source URLs, acquisition dates, spatial and temporal resolution,
  processing level, cloud filters, feature definitions, and assumptions.
- Keep API keys and other credentials out of source control.
- Validate results on held-out fields, years, or regions rather than relying
  only on neighboring random splits.

## Data Citation

The BARC field polygons in `04_Evaluation` were originally developed and
published as part of the studies below. If you use or redistribute these data,
please cite the relevant publications and datasets.

**Publications**

- Dulaney, W., Anderson, M., Gao, F., Stern, A., Moglen, G., Meyers, G.,
  Daughtry, C., White, W., Akumaga, U., & Showalter, J. (2024). Development of
  a gridded yield data archive for farm management and research at the USDA
  Beltsville Agricultural Research Center. *Agrosystems, Geosciences &
  Environment*, 7(1), e20474. https://doi.org/10.1002/agg2.20474
- Kraatz, S., Lamb, B., Hively, W., Jennewein, J., Gao, F., Cosh, M., &
  Siqueira, P. (2023). Comparing NISAR (Using Sentinel-1), USDA/NASS CDL, and
  Ground Truth Crop/Non-Crop Areas in an Urban Agricultural Region. *Sensors*,
  23(20), 8595. https://doi.org/10.3390/s23208595

**Datasets**

- USDA Beltsville Agricultural Research Center (BARC) Gridded Yield Data
  Archive, version 1. USDA National Agricultural Library, Ag Data Commons.
  https://agdatacommons.nal.usda.gov/articles/dataset/USDA_Beltsville_Agricultural_Research_Center_BARC_Gridded_Yield_Data_Archive_version_1/28950746
- Kraatz, S. (2023). Validating NISAR's cropland mapping approach and the
  USDA/NASS Cropland Data Layer against ground truth data in a fragmented
  urban agricultural region (Version 1.0.0) [Data set]. Zenodo.
  https://doi.org/10.5281/zenodo.8087301

## Environment Notes

The Quick Stats downloader uses only the Python standard library. The notebooks
install their own geospatial dependencies in their first code cells. Python
3.10 or later is recommended for the notebook examples.
