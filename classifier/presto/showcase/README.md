# Presto, pixel by pixel

Pictures of fine-tuned Presto's calls on the 44 labeled test field-years (2021-2023). Presto scores every 10 m pixel
from its own 12-month series, and a field-year's score is the mean over its pixels (`../README.md`, Method). These
pictures show the pixel scores and the monthly true color Presto was given.

- `presto_pixels.html`: interactive map in one self-contained file; open it in a browser. Pick a year, show satellite
  color by month or Presto's pixel scores, and click a field for its 12 monthly images, pixel vote and NDVI curve.
- `season_strip.png`: a silage and a grain field-year (Central Farm 2-32C and 2-72A, 2021), June to December, with
  Presto's pixel scores and their NDVI against the median silage and grain field-year.
- `pixel_map.png`: every test pixel colored by its silage score, one panel per year, misses outlined and named.

## What they show

- 41 of 44 field-years right (2021: 15 of 18, 2022: 7 of 7, 2023: 19 of 19), the standard-test result of
  `../results/standard_test_fixed_labels/`.
- Pixels mostly agree. In the 41 right calls, the median share of pixels on the side of the field's call is 99.8%;
  6 of them are below 90% (lowest 56%).
- The two grain field-years called silage, APU 1 and APU 30 in 2021, are split: 72% of their pixels score silage.
  The third miss, Central Farm 4-7 in 2021 (silage, combined November 4 per the farm record), is called grain by
  every pixel. Pixel agreement is therefore a hint of confidence, not a guarantee.
- The silage field-years turn bare in September and green again in November and December; the grain field-years
  stand into October.

## How the data are made

`presto_pixel_map.py` refits the setting chosen in the standard test (fine-tuning, lr 3e-4, 12 epochs, seeds 0-4) on
2017-2020 with `presto_standard_test.train_curve`, unchanged, and keeps each test pixel's probability. Each
field-year's mean over pixels and seeds equals `p_finetune` in the standard test's `test_predictions.csv` (largest
difference 0.0000, same call on 44 of 44; `results/field_check.csv`). Pixel positions are recomputed from the polygons in
`../field_years.csv` on the grid of `presto_pixels.py` (10 m, shrunk 10 m inward); the script stops if a field-year's
count differs from the pulled pixels. `build_showcase.py` draws the pictures from `results/pixel_map.npz` (3.5 MB, not
committed: step 1 rebuilds it in under a minute on one NVIDIA B200).

```bash
PY=/home/keeboh/venvs/prithvi/bin/python
FY=classifier/labels/results/field_years_fixed.csv
$PY classifier/presto/showcase/presto_pixel_map.py --field-years $FY --pixels /home/keeboh/work/presto/pixels \
  --presto-dir /home/keeboh/ext/presto \
  --predictions classifier/presto/results/standard_test_fixed_labels/test_predictions.csv \
  --out classifier/presto/showcase/results
$PY classifier/presto/showcase/build_showcase.py --data classifier/presto/showcase/results --field-years $FY \
  --out classifier/presto/showcase --pair 280:2021 301:2021
```

## Notes

- Only the test years are shown. The model was trained on the 2017-2020 labels, so its scores there say little.
- True color is Sentinel-2 B4, B3, B2 reflectance, brightened for display (x 3.2, gamma 0.8). A month without a clear
  view on half of a field's pixels was masked for Presto and is gray here; within a used month, pixels without a
  clear reading carry the field-year's median, as Presto received them.
- `--pair` picks the season-strip field-years by `field_id:year`; without it, the script takes the same-farm pair
  called right with the most clear months from June to December.
- Results, the comparison with the 5-feature baseline and the caveats: `../README.md`.
