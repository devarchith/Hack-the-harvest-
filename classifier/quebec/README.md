# Québec silage and grain corn field-years

Extra labeled field-years from outside Maryland, in the same format as `../prithvi/field_years.csv`, with Presto pixel
inputs. They are used in two separate ways, never both in one run: as training-only rows for Presto
(`--extra-train`), and as a test outside Maryland (`../presto/presto_quebec_test.py`: the model chosen for the BARC
official test, trained on BARC only, scores Québec once).

The sample is about 47% silage. In the 30 km square it was drawn from, silage is 18-21% of the parcels each year; among
all parcels of at least 3 ha in UTM zone 18N it is about 10%. Accuracy and precision on Québec must be reweighted to
the share of interest; AUC, balanced accuracy and the per-class recalls need no reweighting.

## Source

| Dataset | License | What it holds |
|---|---|---|
| La Financière agricole du Québec (FADQ), Base de données des parcelles et productions agricoles déclarées (BDPPAD), version 3, [fadq.qc.ca](https://www.fadq.qc.ca/documents/donnees/base-de-donnees-des-parcelles-et-productions-agricoles-declarees) | CC BY 4.0 | Insured parcels per year with the declared crop. Crop group `MAF` = maïs fourrager (silage corn, codes `MFO`, `MFE`), `MGR` = maïs-grain (grain corn, codes `MGR`, `MGN`). |

Downloads (one zip per year, about 170 MB each): `BDPPAD_V03_2021.zip`, `BDPPAD_v03_2022.zip`, `BDPPAD_V03_2023.zip`
from `https://www.fadq.qc.ca/fileadmin/geo4tb/BDPPAD/`. SHA-256 sums of the files used are in
`/home/keeboh/work/quebec/src/SHA256SUMS`.

## What the labels mean

The labels are crop insurance declarations, not harvest records. FADQ's user guide says farmers can switch corn between
grain and silage until mid-September, and that farms insured for silage corn under the feed-needs option are not in the
data, so silage is under-represented and an unlabeled parcel can still be silage. Treat them like the BARC
planting-record labels: useful in bulk, not field by field.

## Selection

`quebec_field_years.py` keeps parcels whose crop group is `MAF` or `MGR` (every such parcel has one crop) and that are at
least 3 ha. It keeps those in UTM zone 18N, so the pixels sit on the same EPSG:32618 grid as BARC. It then picks the
30 km square (10 km steps) where silage is most common in its worst year:
x 680-710 km, y 5090-5120 km (Centre-du-Québec). It keeps every silage parcel there, plus a random 400 grain parcels per year (seed 0).

| Year | Silage | Grain kept (of all in the square) |
|---|---|---|
| 2021 | 360 | 400 (1,352) |
| 2022 | 336 | 400 (1,524) |
| 2023 | 349 | 400 (1,439) |

That gives 2,245 field-years on 1,883 parcels. `field_id` is 1,000,000 plus a rank of FADQ's parcel ID `IDPAR`, so a
parcel keeps one `field_id` across years. Split by `field_id` and the same ground never lands in both training and
test. 46 parcels change class between years. The five hand-feature columns are empty. Extra columns: `area_ha`, and
`idanpar` (FADQ's parcel-year ID).

## Run

```bash
PY=/home/keeboh/venvs/prithvi/bin/python
$PY classifier/quebec/quebec_field_years.py --src /home/keeboh/work/quebec/src --out classifier/quebec

# Pixels: the unchanged classifier/presto/presto_pixels.py, run from a copy outside /workspace so that its provenance
# copy does not overwrite classifier/presto/results/pixels/ (the copy's folder links preprocessing/ to the repository).
R=/home/keeboh/work/quebec/run
mkdir -p $R/classifier/presto && ln -sfn "$PWD/preprocessing" $R/preprocessing
cp classifier/presto/presto_pixels.py $R/classifier/presto/
for y in 2021 2022 2023; do
  (cd $R && $PY classifier/presto/presto_pixels.py --field-years "$OLDPWD/classifier/quebec/field_years_quebec.csv" \
     --out /home/keeboh/work/quebec/pixels --years $y --workers 8) &
done; wait
```

With 12 workers per year, the 2023 pull stopped once without an error after 200 of 228 scene-days and was rerun with
8. Failed scene-days are listed in each `.provenance.json`.

Output: `/home/keeboh/work/quebec/pixels/pixels_<year>.npz` and `.provenance.json` (same format as the BARC pixels).

## Files

| File | What it holds |
|---|---|
| `quebec_field_years.py` | Parcel selection |
| `field_years_quebec.csv` | The 2,245 field-years |
| `quebec_selection.txt` | Counts printed by the selection |
