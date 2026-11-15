#!/usr/bin/env bash
# Rebuild the extra BARC labels and every check from the published sources.
#   PY=/home/keeboh/venvs/prithvi/bin/python bash classifier/labels/run_all.sh /home/keeboh/work/labels
# The source folder holds the downloads and the PDF text; tables and printed outputs go to classifier/labels/results/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SRC="${1:?usage: run_all.sh <source folder>}"
PY="${PY:-python}"
OUT="$HERE/results"
GPKG="$ROOT/04_Evaluation/BARC_Corn.2021-2023.gpkg"
mkdir -p "$OUT" "$SRC/farm_reports/txt"

"$PY" "$HERE/get_sources.py" "$SRC" | tee "$OUT/0_sources.txt"
"$PY" "$HERE/pdf2txt.py" "$SRC/farm_reports/pdf" "$SRC/farm_reports/txt" > "$OUT/1_pdf_text.txt"
"$PY" "$HERE/parse_farmmgmt.py" "$SRC/farm_reports/txt" "$OUT/farm_report_events.csv" | tee "$OUT/2_farm_report_events.txt"
"$PY" "$HERE/kraatz_vs_eval.py" "$SRC/zenodo/farmdb" "$GPKG" "$(ls "$SRC"/zenodo/outlines/*.shp)" \
  "$OUT/zenodo_labels.csv" | tee "$OUT/3_zenodo_vs_official.txt"
"$PY" "$HERE/mgmt_vs_eval.py" "$OUT/farm_report_events.csv" "$GPKG" "$OUT/zenodo_labels.csv" \
  "$OUT/farm_report_field_years.csv" | tee "$OUT/4_reports_vs_official.txt"
"$PY" "$HERE/combine_check.py" --zenodo-dir "$SRC/zenodo/farmdb" | tee "$OUT/5_combined_checks.txt"
"$PY" "$HERE/satellite_check.py" | tee "$OUT/6_satellite_check.txt"
