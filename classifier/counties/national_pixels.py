"""Runs classifier/presto/presto_pixels.py unchanged for one county: in that county's UTM zone, with the band files of
each scene-day read in parallel.

  python national_pixels.py --tree /home/keeboh/work/national/tree --crs EPSG:32615 --pool 12 -- \
      --field-years fy/19027.csv --out pixels/19027 --years 2022 --workers 6

presto_pixels.py fixes the grid to EPSG:32618 (UTM 18N) and calls odc.stac.load once per scene-day without a thread
pool, so the band files of a scene-day are read one after another. Both are set from here, without editing the file:
presto_pixels.CRS, and presto_pixels.load = odc.stac.load with pool=N (the same reads, run at the same time).
--tree is a copy of the repository folders classifier/presto and preprocessing, outside /workspace, so the
provenance copy that presto_pixels.py writes next to itself stays out of the repository.
"""
import argparse
import functools
import sys
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--tree", required=True)
ap.add_argument("--crs", required=True)
ap.add_argument("--pool", type=int, default=0, help="threads per scene-day load (0: as in presto_pixels.py)")
a, rest = ap.parse_known_args()
sys.path.insert(0, str(Path(a.tree) / "classifier" / "presto"))
import presto_pixels as pp  # noqa: E402
from odc.stac import load  # noqa: E402

pp.CRS = a.crs
if a.pool > 1:
    pp.load = functools.partial(load, pool=a.pool)
sys.argv = [pp.__file__] + [r for r in rest if r != "--"]
pp.main()
