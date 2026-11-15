# Extra BARC labels: where they come from and how far to trust them

The challenge provides 44 labeled field-years at BARC, Maryland, 2021-2023. The farm's own records for other years are
public. This folder downloads them, turns them into silage and grain labels, and checks them two ways: against the
official labels where they overlap, and against the satellite signal through a model fitted on the official labels only.

## Sources

| Dataset | License | What it holds | md5 |
|---|---|---|---|
| Kraatz, S. (2023). Validating NISAR's cropland mapping approach and the USDA/NASS Cropland Data Layer against ground truth data in a fragmented urban agricultural region. Zenodo, [10.5281/zenodo.8087301](https://doi.org/10.5281/zenodo.8087301) | CC BY 4.0 | BARC farm-database records: planting 2017-2021, harvest 2019-2021, each corn field marked `Corn Silage (CSIL)` or `Corn Field (FLDCRN)`; the study's field outlines | `65590c4f…` (records), `35fae5dc…` (outlines) |
| USDA BARC Gridded Yield Data Archive, version 1. Ag Data Commons, [10.15482/USDA.ADC/28950746.v2](https://doi.org/10.15482/USDA.ADC/28950746.v2), file `BARCFarmMgmt.zip` | U.S. Public Domain | FarmLogic field reports (49 PDFs): planting and harvest records 2014-2024, crop `Corn, Silage` or `Corn, Field`, and the harvest machine | `1d1d01cf…` |

`get_sources.py` holds the full checksums and stops if a download does not match.

## Run

```bash
/home/keeboh/venvs/prithvi/bin/pip install pypdf   # once
PY=/home/keeboh/venvs/prithvi/bin/python bash classifier/labels/run_all.sh /home/keeboh/work/labels
```

Needs pandas, geopandas, scikit-learn and pypdf. Downloads go to the source folder (about 12 MB); everything written to
`results/` is rebuilt on each run.

| Step | Script | Output in `results/` |
|---|---|---|
| 0. Download and check the sources | `get_sources.py` | `0_sources.txt` |
| 1. PDF text | `pdf2txt.py` | `1_pdf_text.txt` (text files go to the source folder) |
| 2. Corn planting and harvest events from the field reports | `parse_farmmgmt.py` | `farm_report_events.csv`, `2_farm_report_events.txt` |
| 3. Zenodo labels per field-year; agreement with the official 2021 labels | `kraatz_vs_eval.py` | `zenodo_labels.csv`, `3_zenodo_vs_official.txt` |
| 4. Field-report labels per field-year; agreement with the official labels and with Zenodo | `mgmt_vs_eval.py` | `farm_report_field_years.csv`, `4_reports_vs_official.txt` |
| 5. One table outside 2021-2023; label basis, changes of use, outline fit | `combine_check.py` | `extra_labels_outside_2021_2023.csv`, `5_combined_checks.txt` |
| 6. Satellite check of the 2017-2020 training labels | `satellite_check.py` | `satellite_check.csv`, `6_satellite_check.txt` |
| 7. Fixes from the combine yield records | `apply_yield_fixes.py` | `label_fixes.csv`, `field_years_fixed.csv`, `field_years_combine_additions.csv`, `extra_labels_outside_2021_2023_fixed.csv`, `7_yield_fixes.txt` |

A field-year's label is its whole-field harvest record when there is one, otherwise its whole-field planting record.
Fields are matched to outlines by farm and field name only (for example `CentralFarm_1-14B`).

## Results

**Agreement with the official labels, where both exist.** No class conflicts.

| Comparison | Planting records | Harvest records |
|---|---|---|
| Zenodo vs official, 2021 | 12 of 12 | 13 of 13 |
| Farm reports vs official, 2021-2023 | 20 of 20 | 20 of 21 (`CentralFarm_1-32` 2023 was partly chopped, partly combined) |
| Farm reports vs Zenodo, 2017-2020 | 47 of 47 (planting or harvest label) | |

**Labels outside 2021-2023:** 203 field-years (161 Grain, 42 Silage); 114 have an outline in the evaluation file
(86 Grain, 28 Silage). Of those 114, 40 rest on a harvest record and 74 on a planting record only. 2014, 2016 and 2024
have no harvest records.

**Planting records only state the plan.** Where a field-year has both records, planned silage was harvested wholly or
partly as grain 1 of 17 times in Zenodo and 3 of 25 times in the farm reports. Planned grain never changed (0 of 49,
0 of 55).

**Satellite check.** A 5-feature logistic regression fitted on the 44 official labels agrees with official labels it
did not see 42 of 44 times (each year predicted from the other two). On the 83 Zenodo field-years used for training:

| Zenodo label based on | Model agrees |
|---|---|
| Harvest record | 27 of 27 (silage 7 of 7) |
| Planting record only, Grain | 44 of 44 |
| Planting record only, Silage | 6 of 12 |

All 6 disagreements are silage labels from a planting record only. This does not prove those labels wrong, but harvest-
backed silage labels agree 7 of 7.

**Outline fit.** 74 of the 114 field-years have recorded acres. 11 differ from the 2021-2023 outline by more than
20-25%; 10 of them are Grain field-years in the 2017-2020 training set. In 4 (2017) the record covers 2-7 acres of a
27-41 acre outline, so most of the outline describes other ground that year.

**Combine yield records.** The same Ag Data Commons record has yearly zips (`2014.zip`, `2016.zip` to `2021.zip`) with
the raw combine yield-monitor files of each field, year and crop; a combine corn file means grain harvest (there are no
chopper files). Two separate checks of those files against the outlines (share of the outline the combine covered;
harvest dates, header width and grain moisture from the raw shapefiles) found the same conflicts, and the farm reports
agree where they have a harvest record. `apply_yield_fixes.py` holds the fixes with their evidence and writes fixed
copies of the tables; the original tables are not changed.

| Field-year | Label | Combine record | Fix |
|---|---|---|---|
| `CentralFarm_1-14C` 2020 (training) | Grain | soybeans on this outline (the farm redrew the 1-14 sub-fields in 2020) | dropped |
| `CentralFarm_2-32A` 2017 (training) | Grain | redrilled to sorghum-sudangrass in June, combined as sorghum | dropped |
| `CentralFarm_4-7` 2014, `EastFarm_5-12` 2014, `CentralFarm_1-37` 2016 | Silage (planting record only) | combined for grain over 83-100% in Oct-Nov | relabeled Grain |
| `CentralFarm_2-32B` 2014 | Silage (planting record only) | 30% combined for grain, rest unknown | dropped |
| `CentralFarm_4-7` 2021 (official) | Silage | combined Nov 4 over 90% of the field, 19.6% moisture, median 80 bu/ac | kept; score with and without it |

Of the 16 official 2021 Grain labels, 15 have combine corn on the field (one over only 17%); the other has none
either way. `field_years_fixed.csv` is `../prithvi/field_years.csv` without the 2 dropped rows (125 field-years).
`field_years_combine_additions.csv` lists 13 more Grain field-years (2020 and 2021, outlines with no label) where the
combine covered at least 70% and no other crop 10%; they are not merged into any table and have no image data yet.

## Caveats

- One farm. The official labels likely come from the same farm database, so agreement shows the sources are
  consistent, not that either is independently correct.
- 5 official Grain field-years look like soybeans in the USDA Cropland Data Layer (field_id 251 and 286 in 2022;
  230, 231 and 397 in 2023). No farm record exists for any of them.
- Planting-only silage labels are the least certain. Results that depend on them should also be checked with them
  left out.
