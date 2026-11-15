#!/bin/bash
# Keeps pixel pulls (national_pixels.py, one per county) running for the counties in selection.csv with status ok
# that have no pixels_2022.npz yet: at most $(cat $W/max_pulls) pulls at a time (edit that file to change it), each
# with <workers> threads. Checks every 60 s; stops when every county has its npz or a pull that ended in a traceback.
# Usage: launch_pulls.sh <workers per county>
W=/home/keeboh/work/national
P=/home/keeboh/venvs/prithvi/bin/python
cd $W/tree
while true; do
  todo=0
  running=$(pgrep -fc "$W/code/national_pixels.py")  # counted once; a pull just started may not be listed yet
  for fc in $($P -I -c "
import pandas as pd
s = pd.read_csv('$W/selection.csv', dtype={'fips': str})
print(' '.join(f'{r.fips}:{r.crs}' for r in s[s.status == 'ok'].itertuples()))
"); do
    fips=${fc%%:*}; crs=${fc#*:}
    [ -f $W/pixels/$fips/pixels_2022.npz ] && continue
    grep -q Traceback $W/logs/pixels_$fips.log 2>/dev/null && continue
    todo=$((todo + 1))
    pgrep -f "$W/fy/$fips.csv" > /dev/null && continue
    [ $running -ge $(cat $W/max_pulls) ] && continue
    setsid nohup $P $W/code/national_pixels.py --tree $W/tree --crs $crs -- --field-years $W/fy/$fips.csv \
        --out $W/pixels/$fips --years 2022 --workers ${1:-8} > $W/logs/pixels_$fips.log 2>&1 < /dev/null &
    running=$((running + 1))
    echo "$(date -u +%H:%M:%S) started $fips $crs"
  done
  [ $todo -eq 0 ] && break
  sleep 60
done
echo "$(date -u +%H:%M:%S) every county has pixels or a traceback"
