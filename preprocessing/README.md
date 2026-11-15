# Preprocessing

`loader.py` turns raw satellite images into one clean table: **one row per field, per date, per satellite**.
Everyone works from that table, so the cleaning happens once, the same way for everybody.

## What it does, step by step

1. **Field shapes.** It reads the 83 BARC fields from `04_Evaluation/BARC_Corn.2021-2023.gpkg` with the
   Grain/Silage answer columns removed. Each field is shrunk 10 m inward so that edge pixels (roads, trees,
   neighbouring fields) don't mix in.
2. **Find images.** It searches Microsoft Planetary Computer for every image over the fields in the chosen years
   and months, from two satellites:
   - Sentinel-2: 10 to 20 m pixels, a pass every few days
   - Landsat 8/9: 30 m pixels

   Scenes that are 95% or more cloud are skipped.
3. **Remove clouds pixel by pixel.**
   - Sentinel-2: uses its scene classification layer and keeps vegetation, bare soil, water and unclassified
     pixels, as the organizers' starter notebook does.
   - Landsat: uses its quality band and drops missing pixels, cloud, cloud edge, cirrus, shadow and snow.
4. **Put brightness on one scale.**
   - Sentinel-2 images processed from January 2022 onward are stored with a +1000 offset, which is removed.
     Without this, 2021 would not be comparable with 2022 and 2023.
   - Landsat values get their standard scale factor.
5. **Compute measures for every pixel.**
   - NDVI: greenness
   - NDTI: crop residue against bare soil
   - NDMI: moisture
   - NDRE: red-edge greenness, Sentinel-2 only
6. **Summarize each field.** For each field it keeps:
   - the median of every band and measure over the clear pixels
   - the 25th and 75th percentile of NDVI (a patchy, half-harvested field shows a wide spread)
   - pixel counts

   A reading is marked `usable` when at least 60% of the field is clear.
7. **Weather.** Daily rain and temperature at the centre of each field comes from Daymet (1 km grid). `add_rain_flag()` adds the rain in
   the 3 days before each reading. Wet soil reads darker and can make stalk residue look like bare soil.
8. **Record where everything came from.** Every search, image ID, setting and failure is saved to a
   `.provenance.json` file next to the table. The challenge requires documenting each data source.

## Output: `data/series_<tag>.parquet`

| Column | Meaning |
|---|---|
| `field_id` | BARC `OBJECTID` |
| `year`, `date` | Acquisition date (local solar day) |
| `sensor`, `platform` | `S2` or `LANDSAT`; the specific satellite |
| `items`, `baseline` | Image IDs used; Sentinel-2 processing version |
| `n_pix`, `n_valid`, `valid_frac`, `usable` | Pixels in the field, clear pixels, clear share, whether at least 60% is clear |
| `blue` to `swir2`, `re1`, `nir_narrow` | Median surface reflectance (0 to 1). `re1` and `nir_narrow` are Sentinel-2 only |
| `ndvi`, `ndvi_p25`, `ndvi_p75`, `ndti`, `ndmi`, `ndre` | Median measures over the clear pixels |

Readings with no clear pixel are kept (with `n_valid = 0`), so cloud gaps stay visible. Because the raw band
medians are saved, any new measure can be computed later without downloading again.

Weather table: `data/weather_<tag>.parquet`, with columns `field_id`, `date`, `prcp_mm`, `tmax_c`, `tmin_c`.

## Radar: `data/s1_<tag>.parquet`

`fetch_s1()` builds the same kind of table from Sentinel-1 radar (RTC product: gamma0, terrain corrected, 10 m),
also from Microsoft Planetary Computer. Radar sees through cloud. One row per field, date and pass.

| Column | Meaning |
|---|---|
| `field_id`, `year`, `date`, `sensor` (`S1`), `platform`, `items` | As in the optical table |
| `orbit_state`, `relative_orbit` | Pass direction and track. Over BARC: ascending, tracks 4 and 106, which view the fields at different angles |
| `n_pix`, `n_valid`, `valid_frac`, `usable` | Pixels in the field, pixels with data, their share, whether at least 60% has data |
| `vv_db`, `vh_db` | Field mean backscatter, averaged in linear power and then converted to dB |
| `vh_vv_db`, `rvi` | VH/VV ratio in dB; radar vegetation index 4·VH / (VV + VH) |

Each scene is read as a window on its own 10 m grid (no resampling). Scenes that only clip the edge of the
area give rows with `n_valid = 0`. Searches, settings and failures are saved to `s1_<tag>.provenance.json`.

```bash
python preprocessing/loader.py --gpkg 04_Evaluation/BARC_Corn.2021-2023.gpkg --years 2021 2022 2023 --months 6 11 --s1-only --workers 8
```

Run on Sun Nov 15 2026: 274 passes, 6,263 rows, 64% usable, no failures, 18 minutes on a laptop.
A median of 13 usable passes per field and year (June to November).

## How to run

```bash
pip install -r requirements.txt
python preprocessing/smoke_test.py      # about 1.5 min check on two small squares near BARC
python preprocessing/loader.py --gpkg 04_Evaluation/BARC_Corn.2021-2023.gpkg --years 2021 2022 2023 --months 3 12 --weather
```

On the Keeboh pod, the same evaluation file is also at `/workspace/04_Evaluation/`. Then, in a notebook:

```python
import sys; sys.path.insert(0, "..")      # repository root, if the notebook sits one folder down
import pandas as pd
from preprocessing.loader import load_series, add_rain_flag
series = load_series("../data/series_barc.parquet")          # usable readings only, by default
series = add_rain_flag(series, pd.read_parquet("../data/weather_barc.parquet"))
```

The answers (Grain/Silage) are never in these tables. `barc_labels()` reads them, for testing only.

## Checks done

- **Smoke test** (two squares near BARC, Aug to Sep of 2021 and 2022): passes. Sentinel-2 versions were 03.00
  in 2021 and 04.00 or later in 2022, so the offset fix applies only where it should.
- **All 83 BARC fields, Aug to Nov 2022, on a laptop:**
  - 3,451 rows, 62% of readings usable, 164 s
  - 16 to 32 usable dates per field (median 22)
  - the smallest field keeps 105 Sentinel-2 pixels; the median field keeps 412
  - median NDVI over all fields (answers not used): Aug 0.84, Sep 0.64, Oct 0.38, Nov 0.38

## Things to know when using the data

- **The two satellites disagree slightly.** On 153 field-days seen by both, Sentinel-2 NDVI is about 0.05 lower
  than Landsat's. Compare each satellite with itself, or correct for the gap; don't mix raw values.
- **Clouds cover part of the silage harvest window.** Counting whole scenes under 20% cloud:
  - 2022: no clear scene from Aug 18 to Sep 15
  - 2023: no clear scene in August before Aug 31

  Per-field masking recovers some partly cloudy days.
- **One broken image is skipped.** Landsat scene LC08_L2SP_015033_20220804 has an unreadable quality file on
  Planetary Computer. It is listed under `failures` in the provenance file.
- **The weather calendar has 365 days**, so Dec 31 is missing in leap years. This doesn't matter for harvest
  season.

## Data sources

| Source | Link |
|---|---|
| Sentinel-2 Level-2A | https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a |
| Landsat Collection 2 Level-2 | https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2 |
| Sentinel-1 RTC | https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc |
| Daymet daily weather | https://daymet.ornl.gov/ |
| BARC fields | Dulaney et al. (2024), https://doi.org/10.1002/agg2.20474, and Kraatz et al. (2023), https://doi.org/10.3390/s23208595 |
