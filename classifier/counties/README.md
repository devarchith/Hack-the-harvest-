# County check: predicted silage share against the 2022 Census

A test of the classifier on county totals instead of labeled fields. No field labels exist outside BARC and Québec,
but the 2022 Census of Agriculture reports silage and grain corn acres for each county
(`../../02_QuickStats/corn_grain_silage_acres_2012_2025.csv`). If the model works away from Maryland, the share of corn
it calls silage should rise and fall with the Census share across counties.

The test and its outputs are in `../presto/presto_county_test.py` and `../presto/README.md`.

## Counties

Seven counties in UTM zone 18N (the grid of the BARC pixels), chosen for a wide range of Census silage shares:

| FIPS | County | Census 2022 silage share of harvested corn acres |
|---|---|---|
| 24035 | Queen Anne's, MD | 0.013 |
| 24013 | Carroll, MD | 0.089 |
| 42133 | York, PA | 0.078 |
| 42071 | Lancaster, PA | 0.274 |
| 36011 | Cayuga, NY | 0.525 |
| 36115 | Washington, NY | 0.759 |
| 50001 | Addison, VT | 0.921 |

Share = silage / (silage + grain), harvested acres, rows with `source` CENSUS and `domain` TOTAL. The file also holds
2022 county SURVEY estimates for some counties; they are left out. Withheld values are left out.

## Sample

`county_squares.py` uses no field outlines. The 2022 Cropland Data Layer (CDL, 30 m, USDA NASS, from CropScape) marks
corn pixels. Per county it keeps 90 m blocks (3 x 3 CDL pixels) that are all corn (code 1, or a double crop with corn:
225, 226, 237, 241). It then picks the 20 km square (5 km steps) with the most such blocks and draws 200 blocks there
at random (seed 0). Each block is one row of `county_squares.csv`, in the format of `../quebec/field_years_quebec.csv`
with an empty label. Blocks are 4,900 m² after the 10 m inward buffer of `presto_pixels.py`, so about 49 pixels each.

Equal-size blocks drawn at random from corn area make the mean prediction an area share, which matches the Census acre
share. `county_selection.txt` lists each county's Census acres, the square, and the square's share of the county's
all-corn blocks (22% to 51%).

## Limits

- The Census share covers the whole county. The sample covers one 20 km square of it.
- Blocks sit inside fields, so fields narrower than about 90 m are under-sampled.
- CDL corn includes CDL errors. The Census counts harvested acres; CDL counts planted corn.
- A county total cannot show which squares are wrong. Over- and under-calls can cancel.

## Run

```bash
PY=/home/keeboh/venvs/prithvi/bin/python
W=/home/keeboh/work/counties
$PY classifier/counties/county_squares.py --cdl-dir $W/cdl --out classifier/counties

# Pixels: the unchanged presto_pixels.py, one run per county (it reads one box around all rows of a year), from a copy
# outside /workspace as for Québec.
$PY -c "import pandas as pd; d = pd.read_csv('classifier/counties/county_squares.csv', dtype={'fips': str}); \
[g.to_csv(f'$W/fy/{f}.csv', index=False) for f, g in d.groupby('fips')]"
R=$W/run
mkdir -p $R/classifier/presto && ln -sfn "$PWD/preprocessing" $R/preprocessing
cp classifier/presto/presto_pixels.py $R/classifier/presto/
for f in 24035 24013 42133 42071 36011 36115 50001; do
  (cd $R && $PY classifier/presto/presto_pixels.py --field-years $W/fy/$f.csv --out $W/pixels/$f --years 2022 \
     --workers 6) &
done; wait
```

## Files

| File | What it holds |
|---|---|
| `county_squares.py` | Block selection |
| `county_squares.csv` | The 1,400 blocks (200 per county) |
| `county_selection.txt` | Census acres and share, block counts and the square per county |
