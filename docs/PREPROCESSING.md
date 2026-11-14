# Preprocessing: what "data done" means

Target: finished by about **4 PM Saturday**. Anything that is not done by then is cut or simplified, not
allowed to push the rest of the timeline back.

Code: `preprocessing/loader.py` (fetch and load), tested with `preprocessing/smoke_test.py`.
The output columns in the loader's docstring are a proposal made while the team was away: add columns
freely, but do not rename existing ones without telling everyone.

## Definition of done

Each item is a checkbox.

- [ ] All 83 BARC polygons are loaded from the starter kit (layer `barc_silage_grain_corn20212023`) and
  reprojected:
  - EPSG:4326 for catalog searches
  - EPSG:32618 (UTM 18N) for pixels; **not** the starter notebook's EPSG:32610
  - buffered 10 m inward to drop edge pixels
- [ ] Every Sentinel-2 L2A and Landsat 8/9 Collection 2 Level-2 scene from Jun 1 to Dec 31 of 2021, 2022 and
  2023 is searched. All BARC scenes fall on Sentinel-2 tile 18SUJ.
- [ ] Clouds are masked per pixel:
  - Sentinel-2 SCL: keep classes 4, 5, 6 and 7, as the starter does
  - Landsat QA_PIXEL: drop cloud, dilated cloud, shadow and snow bits
- [ ] Reflectance is on one scale across years:
  - **Sentinel-2:** 2021 scenes are processing baseline 03.00, while 2022 and 2023 scenes are 04.00 or later,
    which add a +1000 offset to the stored values. Remove it before dividing by 10000, or use a catalog that
    is already harmonized. Verify on one field by comparing 2021 against 2022.
  - **Landsat Collection 2 L2:** reflectance = DN × 0.0000275 − 0.2.
- [ ] There is one row per field, date and sensor, with these columns:
  - `field_id` (OBJECTID), `year`, `date`, `sensor`, `scene_id`
  - `n_pix`, `n_valid`, `valid_frac`
  - median `ndvi`, `ndti`, `ndmi`, and `ndre` (Sentinel-2 only)

  It is saved as a small table in `data/` (git-ignored).
- [ ] Each reading is flagged as usable when `valid_frac` is at least 0.6. Keep the unusable rows, but mark
  them.
- [ ] Daily precipitation at each field (gridMET or Daymet) is joined, with a `rain_3d_mm` column per reading
  for the wet-soil flag.
- [ ] Quality-check outputs, with **labels hidden**:
  - a contact sheet of all 249 field-year curves
  - usable readings per field-year
  - the longest gap around late August to September
- [ ] A data card: every source URL, collection name, query dates, bands, cloud rule, CRS and processing step
  (the challenge's Data Rules require this).

## Indices

| Index | Sentinel-2 | Landsat 8/9 | Meaning |
|---|---|---|---|
| NDVI | (B08 − B04) / (B08 + B04) | (B5 − B4) / (B5 + B4) | Greenness |
| NDTI | (B11 − B12) / (B11 + B12) | (B6 − B7) / (B6 + B7) | Residue vs bare soil (moisture lowers it) |
| NDMI | (B08 − B11) / (B08 + B11) | (B5 − B6) / (B5 + B6) | Canopy and surface moisture |
| NDRE | (B8A − B05) / (B8A + B05) | not available | Red-edge greenness |

## Known pitfalls

From a metadata-only scene search over the BARC area, Aug 1 to Nov 30, counting scenes under 20% cloud
(Sentinel-2 or Landsat 8/9):

| Year | Clear dates | Longest gap |
|---|---|---|
| 2021 | 18 | 20 days (Sep 30 to Oct 20) |
| 2022 | 23 | 28 days (Aug 18 to Sep 15), inside the silage window |
| 2023 | 18 | no clear date in August before Aug 31 |

Masking clouds per field, rather than per scene, should recover more dates. Script:
`preprocessing/00_scene_probe.py`.

Other pitfalls:

- Wet soil lowers NDTI and can make residue look like bare soil. Use the rain flag.
- A cover crop sown after an early silage harvest turns green in the fall and can mask the bare-soil signal.

## Label hygiene

Preprocessing outputs never contain the labels. Only the validation notebook joins them. That keeps the rule
honest when only 44 labels exist.
