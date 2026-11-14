# Preprocessing

`loader.py` builds one table with one row per field, date and satellite (Sentinel-2 or Landsat 8/9), using
imagery from Microsoft Planetary Computer.

## Steps

1. Shrink each field 10 m inward to avoid edge pixels.
2. Search Sentinel-2 L2A and Landsat Collection 2 Level-2, skipping scenes that are 95% or more cloud. When a
   Sentinel-2 scene is listed twice (original and reprocessed), keep the newest copy.
3. Mask clouds per pixel:
   - Sentinel-2: keep scene classification (SCL) classes 4 to 7, as the starter notebook does.
   - Landsat: drop QA_PIXEL bits 0 to 5 (fill, dilated cloud, cirrus, cloud, shadow, snow).
   - Both: drop pixels with red or NIR at or below 0.
4. Convert to surface reflectance:
   - Sentinel-2: (DN - 1000) / 10000 for processing baseline 04.00 or later, otherwise DN / 10000.
   - Landsat: DN x 2.75e-5 - 0.2.
5. Compute NDVI, NDTI, NDMI and NDRE (Sentinel-2 only) per pixel. Each field gets the median of every band
   and index over its clear pixels, plus the NDVI 25th and 75th percentiles.
6. Mark a reading `usable` when at least 60% of the field's pixels are clear.
7. Optional (`--weather`): daily Daymet rain and temperature at each field's centre. `add_rain_flag()` adds the
   rain from the 3 days before each reading.
8. Write the searches, image IDs, settings and failures to `series_<tag>.provenance.json`.

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

## Known issues

- Sentinel-2 NDVI is about 0.05 lower than Landsat on the same field and day (median over 153 pairs).
- Landsat scene LC08_L2SP_015033_20220804 has an unreadable QA file on Planetary Computer. It is skipped and
  listed under `failures` in the provenance file.
- Daymet has no Dec 31 in leap years.

## Sources

- Sentinel-2 L2A: https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a
- Landsat Collection 2 Level-2: https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2
- Daymet: https://daymet.ornl.gov/
- BARC fields: Dulaney et al. (2024), https://doi.org/10.1002/agg2.20474; Kraatz et al. (2023),
  https://doi.org/10.3390/s23208595
