#!/usr/bin/env bash
# One test of the Prithvi classifier on field-years no model has seen (2014, 2015, 2016, 2024); see prithvi_fresh.py.
#   bash classifier/prithvi/prithvi_fresh.sh
# Uses the copy of main and the evaluate.py result folders that prithvi_evaluate.sh made under $WORK.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-/home/keeboh/venvs/prithvi/bin/python}"
WORK="${WORK:-/home/keeboh/work/eval}"
FRESH="${FRESH:-/home/keeboh/work/fresh}"
export HF_HOME="${HF_HOME:-/home/keeboh/hf}"
TREE="$WORK/Hack-the-harvest--main"
"$PY" "$HERE/prithvi_fresh.py" prepare --tree "$TREE" --fresh "$FRESH"
# Hand-feature inputs: the team loader over every field in the evaluation file, as for 2021-2023. 2021 is fetched
# only to check that the features rebuild the team's 2021 values.
(cd "$TREE" && "$PY" preprocessing/loader.py --gpkg 04_Evaluation/BARC_Corn.2021-2023.gpkg \
  --years 2014 2015 2016 2021 2024 --out "$FRESH" --tag fresh)
"$PY" "$HERE/prithvi_chips.py" --field-years "$FRESH/fresh_field_years.csv" --out "$FRESH/chips"
for f in 8 4; do
  "$PY" "$HERE/prithvi_embed.py" --field-years "$FRESH/fresh_field_years.csv" --chips "$FRESH/chips" \
    --out "$FRESH/emb" --size 300M --scale 1 --frames "$f"
done
"$PY" "$HERE/prithvi_fresh.py" score --tree "$TREE" --fresh "$FRESH" --out "$HERE/results/fresh"
