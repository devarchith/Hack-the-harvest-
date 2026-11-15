# Silage Atlas

An interactive map of the silage/grain predictions, for looking at the results field by field.

- `atlas.html`: the page. Open it in a browser. In JupyterLab, open it and click "Trust HTML".
- `build_atlas.py`: builds `atlas.html` from the committed result files and `template.html`.
- `claims.py`: recomputes the numbers in `CLAIMS.md` from the same files and writes `claims.json`.
- `CLAIMS.md`: the results in short sentences, each with its source file.

## What the page shows

- **United States:** the 163 sampled counties and the 7 test counties never used in training, colored by the share of
  sampled corn blocks called silage, by the 2022 Census share, or by the difference. Click a county to see its
  blocks on satellite imagery, the Census comparison and the spread of block probabilities.
- **Maryland test fields:** the 44 evaluation field-years of 2021-2023 as field outlines, with the farm record, the
  call and the misses.
- **Québec test fields:** the 2,245 Québec field-years. No Québec label was used in training.
- The model switch compares the Maryland-only model with the county-share model in every view.
- The county table and each county's blocks download as CSV.

For sampled counties the county-share numbers are out of fold by state: each county is predicted by a model that
was not trained on that county's state.

## Inputs

| Data | File |
|---|---|
| County predictions and Census shares | `classifier/counties/national/results_final/county_predictions.csv`, `test_county_predictions.csv` |
| Block predictions | `classifier/counties/national/results_final/block_predictions.csv` |
| Block outlines | `classifier/counties/national/county_squares.csv`, `classifier/counties/county_squares.csv` |
| Field predictions | `classifier/counties/national/results_final/field_predictions.csv` |
| Field outlines and positions | `classifier/presto/field_years.csv`, `classifier/quebec/field_years_quebec.csv` |
| Previous Census shares | `classifier/counties/national/results_final/census_prior/census_prior_counties.csv` |
| County and state outlines | US Census Bureau cartographic boundary files, 2022, 1:20,000,000 (downloaded on first run) |

The page loads Leaflet from cdnjs and imagery tiles from Esri World Imagery, so the map background needs an
internet connection. The predictions and outlines are inside the file.

## Rebuild

```
python classifier/atlas/claims.py
python classifier/atlas/build_atlas.py
```

The build stops if a block or field outline does not sit on the longitude and latitude stored for it.

## Demo images

`demo/` holds screenshots of each view and `atlas_demo.gif`, a short walk through them: the US map, one county's
blocks, block outlines on the imagery, the Maryland test fields, and Québec with each model.
