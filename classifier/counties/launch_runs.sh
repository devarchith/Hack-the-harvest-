#!/bin/bash
# The three final runs on Census-only county shares (national_fix: fix_shares.py; county_squares.census_shares now
# keeps source == CENSUS), started together on the same counties:
#   results_final           Jan-Dec, original Presto weights, w 0, 0.3, 1, 3, 10
#   results_final_barconly  Jan-Dec, presto_barc_only weights, w 0, 3, 10
#   results_final_janoct    Jan-Oct (setting and Québec check of the official Jan-Oct runs), original weights, w 0, 1, 3
set -e
W=/home/keeboh/work/national
H=/workspace/Hack-the-harvest-
PY=/home/keeboh/venvs/prithvi/bin/python
R=$H/classifier/presto/results
common=(--field-years $H/classifier/labels/results/field_years_fixed.csv --pixels /home/keeboh/work/presto/pixels
        --national $W/national_fix --test-counties /home/keeboh/work/counties
        --census $H/02_QuickStats/corn_grain_silage_acres_2012_2025.csv
        --quebec $H/classifier/quebec/field_years_quebec.csv --quebec-selection $H/classifier/quebec/quebec_selection.txt
        --quebec-pixels /home/keeboh/work/quebec/pixels
        --state-ratios /home/keeboh/work/eval/Hack-the-harvest--main/02_QuickStats/state_silage_ratios.csv)
cd $W/tree/classifier/counties
nohup $PY presto_national.py "${common[@]}" --presto-dir /home/keeboh/ext/presto \
      --setting-from $R/standard_test_fixed_labels --quebec-official $R/quebec_test \
      --out $W/results_final --parallel 2 > $W/logs/national_final.log 2>&1 &
nohup $PY presto_national.py "${common[@]}" --presto-dir /home/keeboh/work/unsup/presto_barc_only --w-grid 0,3,10 \
      --setting-from $R/standard_test_fixed_labels --quebec-official $R/quebec_test \
      --out $W/results_final_barconly --parallel 1 > $W/logs/national_final_barconly.log 2>&1 &
nohup $PY presto_national.py "${common[@]}" --presto-dir /home/keeboh/ext/presto --w-grid 0,1,3 \
      --setting-from $R/standard_test_fixed_labels_jan_oct --quebec-official $R/quebec_test_jan_oct \
      --out $W/results_final_janoct --parallel 1 > $W/logs/national_final_janoct.log 2>&1 &
echo started
