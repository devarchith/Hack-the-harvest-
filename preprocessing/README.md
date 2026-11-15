# Preprocessing

`loader.py` builds one table with one row per field, date and satellite (Sentinel-2 or Landsat 8/9), using
imagery from Microsoft Planetary Computer.

## Steps

1. Shrink each field 10 m inward to avoid edge pixels.
2. Search Sentinel-2 L2A and Landsat Collection 2 Level-2, skipping scenes that are 95% or more cloud. When a
   Sentinel-2 scene is listed twice (original and reprocessed), keep the newest copy.
3. Read each scene on the sensor's native pixel grid in UTM 18N. Landsat pixel edges sit 15 m past multiples of
   30 m (`grid_anchor`); reads time out after 60 s and are retried.
4. Mask clouds per pixel:
   - Sentinel-2: keep scene classification (SCL) classes 4 to 7, as the starter notebook does.
   - Landsat: drop QA_PIXEL bits 0 to 5 (fill, dilated cloud, cirrus, cloud, shadow, snow).
   - Both: drop pixels with red or NIR at or below 0.
5. Convert to surface reflectance:
   - Sentinel-2: (DN - 1000) / 10000 for processing baseline 04.00 or later, otherwise DN / 10000.
   - Landsat: DN x 2.75e-5 - 0.2.
6. Compute NDVI, NDTI, NDMI and NDRE (Sentinel-2 only) per pixel. Each field gets the median of every band
   and index over its clear pixels, plus the NDVI 25th and 75th percentiles.
7. Mark a reading `usable` when at least 60% of the field's pixels are clear.
8. Optional (`--weather`): daily Daymet rain and temperature at each field's centre. `add_rain_flag()` adds the
   rain from the 3 days before each reading.
9. Write the searches, image IDs, settings and failures to `series_<tag>.provenance.json`.

## Output: `data/series_<tag>.parquet`

| Column | Meaning |
|---|---|
| `field_id` | BARC `OBJECTID` |
| `year`, `date` | Acquisition date (local solar day) |
| `sensor`, `platform` | `S2` or `LANDSAT`; the specific satellite |
| `items`, `baseline` | Image IDs used; Sentinel-2 processing baseline |
| `n_pix`, `n_valid`, `valid_frac`, `usable` | Field pixels, clear pixels, clear share, clear share at least 60% |
| `blue`, `green`, `red`, `nir`, `swir1`, `swir2`, `re1`, `nir_narrow` | Median surface reflectance (`re1`, `nir_narrow`: Sentinel-2 only) |
| `ndvi`, `ndvi_p25`, `ndvi_p75`, `ndti`, `ndmi`, `ndre` | Medians and percentiles over clear pixels (`ndre`: Sentinel-2 only) |

Readings with no clear pixels are kept with `n_valid = 0`.

`data/weather_<tag>.parquet` columns: `field_id`, `date`, `prcp_mm`, `tmax_c`, `tmin_c`.

## Run

```bash
pip install -r requirements.txt
python preprocessing/smoke_test.py
python preprocessing/loader.py --gpkg 04_Evaluation/BARC_Corn.2021-2023.gpkg --years 2021 2022 2023 --months 3 12 --weather
```

```python
import pandas as pd
from preprocessing.loader import load_series, add_rain_flag
series = load_series("data/series_barc.parquet")  # usable readings only by default
series = add_rain_flag(series, pd.read_parquet("data/weather_barc.parquet"))
```

### Extra training years (2017 to 2020)

The Kraatz farm records (Zenodo 8087301: `farmdb.zip`, `outlines.zip`) give Grain or Silage for BARC fields
before 2021. `research/grain_vs_silage/kraatz_fields.py` turns them into field-years with polygons, then
`--fields` points the loader at them (`research/grain_vs_silage/reproduce.py` runs both):

```bash
python research/grain_vs_silage/kraatz_fields.py --kraatz-dir data/kraatz_zenodo_8087301
python preprocessing/loader.py --fields data/fields_kraatz_2017_2020.gpkg --years 2017 2018 2019 2020 --tag kraatz_2017_2020 --cell-m 50000 --weather
```

- Label: the harvest-record class when there is one (2019 to 2020), otherwise the planting-record class (all of
  2017 and 2018). Whole-field rows only.
- Polygon: the evaluation polygon when the field name matches, keeping its `OBJECTID` as `field_id`; otherwise
  the Kraatz outline (`field_id` 1000 and up). Field-years whose sub-field (for example 1-7A to 1-7F) has no
  polygon are dropped: 52 of 135.
- Result: 83 field-years (64 Grain, 19 Silage) on 54 polygons.

Run cost, all 83 fields, March to December 2021 to 2023, 32 threads on a 12-core laptop:

- Default `cell_m=20000`: the fields straddle a cell corner, so each scene is read 4 times. 1,136 scene-days,
  13 min. With `fetch_series(..., cell_m=50000)` the fields fit one cell: 351 scene-days.
- Daymet: 16 s.
- Output: 25,700 rows, 1.4 MB; weather 0.5 MB.
- The run is limited by network speed. Without a read timeout, one stalled read hung a run indefinitely.

## Data

| Source | Use | Access | Over BARC, 2021 to 2023 |
|---|---|---|---|
| BARC fields | Field outlines; validation labels | `04_Evaluation/` | 83 polygons, 6 to 43 acres. After the buffer: at least 106 Sentinel-2 pixels (median 405) and 11 Landsat pixels (median 45) |
| Sentinel-2 L2A | Optical, 10 to 20 m | Planetary Computer `sentinel-2-l2a` | Tile 18SUJ only. 269 dates under 95% scene cloud, March to December. Baselines 02.12 to 05.10 |
| Landsat 8/9 C2 L2 | Optical, 30 m | `landsat-c2-l2` | 81 dates |
| Daymet | Daily rain and temperature | ORNL single-pixel API | 26 distinct 1 km cells |
| NASS Quick Stats | County and state acres | `02_QuickStats/` | National, 39,906 rows. County rows only for 2012, 2017 and 2022 |
| Cropland Data Layer | Crop type per field-year | `usda-cdl` (2008 to 2021 only); CropScape from 2022 | Not in the loader |
| Sentinel-1 RTC (optional) | Radar through cloud | `sentinel-1-rtc` | 43 to 47 dates a year. Sentinel-1A only, ascending, relative orbits 4 and 106 |
| HLS v2 (optional) | Sentinel-2 and Landsat on one 30 m grid and scale | `hls2-s30`, `hls2-l30` | 93 to 104 and 20 to 35 dates a year |
| Crop Sequence Boundaries | Field outlines beyond BARC, with CDL codes | NASS national zip, 3.5 GB. GeoParquet copy on Source Cooperative, 4.3 GB, 2023 crop only | A Maryland read touches 43 of 648 row groups, about 0.5 GB |

Planetary Computer's `daymet-daily-na` and `gridmet` collections end in 2020, so the loader uses the ORNL API.

## Checks on the full pull

These checks ran on the 0.1.1 pull, before the Landsat grid fix. Landsat rows shift slightly after the fix.
Labels were used only to count labeled field-years, never their class.

- 15 Sentinel-2 field-days, covering all five baselines, were recomputed from the raw files with rasterio. The
  offset came from each product's `MTD_MSIL2A.xml`. Red, NIR, NDVI and clear-pixel counts match exactly.
- The same check found Landsat read 15 m off its native grid, with NDVI off by up to 0.06 on 26-pixel fields.
  Fixed in 0.1.2. After the fix, 3 of 3 field-days match a direct read exactly.
- In June to August, the darkest field on each scene has red reflectance of 0.02 to 0.03 under every
  baseline. No +1000 offset is left in.
- There are no duplicate field-date-sensor rows, and no day mixes two copies of one scene. Pixel counts are
  constant per field.
- August to November: every field-year has at least 12 usable dates, with medians of 15 to 30 by year. The longest gap
  is 26 days (2022). 1 of the 44 labeled field-years has a gap over 20 days.
- Weather has no missing days or NaNs, tmax ≥ tmin, and annual rain is 1,037 to 1,239 mm.
- Contact sheets of all 249 field-years were checked by eye.

## Known issues

- **Missed clouds.** 107 of 14,749 usable readings are isolated NDVI dips: at least 0.15 below both
  neighbours, with both within 10 days. 106 of them are Sentinel-2, and they cluster by scene: 2021-07-09 (15
  fields), 2021-08-13 (15) and 2021-08-28 (11). Another 907 usable readings have blue above 0.10. A late-August
  dip looks like a silage cut. Filter isolated dips, or require a drop to hold at the next reading.
- **Sentinel-2 vs Landsat.** Median Sentinel-2 minus Landsat on the same field and day, measured before the
  Landsat grid fix:

  | Year | Baseline | Pairs | NDVI | NDTI |
  |---|---|---|---|---|
  | 2021 | 02.12 | 108 | -0.064 | -0.023 |
  | 2021 | 03.00 | 125 | 0.000 | -0.010 |
  | 2021 | 04.00 | 34 | -0.024 | -0.014 |
  | 2022 | 04.00 | 218 | -0.038 | -0.014 |
  | 2022 | 05.10 | 128 | -0.027 | -0.013 |
  | 2023 | 05.09 | 152 | -0.041 | -0.041 |
  | 2023 | 05.10 | 296 | -0.027 | -0.014 |

  The difference varies by year and baseline. Part of it may be season, since the 02.12 scenes are in
  spring. Build features per sensor or adjust per sensor; do not mix raw values.
- **Most unlabeled polygons are not corn.** By the 2021 CDL majority class, all 18 labeled field-years are
  corn. Of the 65 unlabeled polygons, 16 are corn, 26 soybeans, 10 grass or pasture, 5 double-crop wheat and
  soybeans, and 8 other. Restrict summaries and thresholds to corn field-years.
- **Fall green-up.** Some labeled field-years green up again after the late-summer drop (2022: 252, 278, 429;
  2023: 246, 292, 323), probably from a cover crop. Late-season NDVI is not only standing corn.
- **Quick Stats CSV.** County-years have several records:
  - County numbers come from both CENSUS and SURVEY. For Washington County 2022 grain, they are 17,619 and
    7,500 acres.
  - State rows include monthly forecasts.
  - Irrigated and organic are separate series.

  Keep `domain == "TOTAL"`, `reference_period == "YEAR"`, the three plain `short_description` series and one
  `source`, and drop `OTHER (COMBINED) COUNTIES`. That leaves no duplicates. Values carry commas, and `(D)`
  means withheld (6,035 rows). Maryland county silage is CENSUS only.
- Landsat scene LC08_L2SP_015033_20220804 has an unreadable QA file on Planetary Computer. It is skipped and
  listed under `failures` in the provenance file.
- Daymet has no Dec 31 in leap years (none in 2021 to 2023).

## Not built yet

- Crop-type flag per field-year. CDL 2022 and 2023 are not on Planetary Computer, and CropScape did not
  respond in testing.
- Missed-cloud filter.
- Sentinel-2 to Landsat adjustment.
- Quick Stats cleaning function.
- Sentinel-1 features (optional).

## Sources

- Sentinel-2 L2A: https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a
- Landsat Collection 2 Level-2: https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2
- Daymet: https://daymet.ornl.gov/
- NASS Quick Stats: https://quickstats.nass.usda.gov/
- Cropland Data Layer: https://planetarycomputer.microsoft.com/dataset/usda-cdl; https://nassgeodata.gmu.edu/CropScape/
- Sentinel-1 RTC: https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc
- HLS: https://planetarycomputer.microsoft.com/dataset/hls2-s30
- Crop Sequence Boundaries: https://www.nass.usda.gov/Research_and_Science/Crop-Sequence-Boundaries/;
  https://source.coop/fiboa/us-usda-cropland
- BARC fields: Dulaney et al. (2024), https://doi.org/10.1002/agg2.20474; Kraatz et al. (2023),
  https://doi.org/10.3390/s23208595
