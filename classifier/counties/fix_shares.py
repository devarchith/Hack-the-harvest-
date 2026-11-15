"""Census-only county shares for the drawn national sample.

county_squares.census_shares added the 2022 SURVEY county rows to the CENSUS rows (fixed: source == "CENSUS"). This
keeps the drawn counties, their blocks and pixels, and recomputes grain and silage acres, the share and the band from
the fixed function; population.csv (for the stratified national share) is recomputed the same way.

  python fix_shares.py --national <national folder> --census 02_QuickStats/corn_grain_silage_acres_2012_2025.csv \
      --out <new folder>

<new folder> gets selection.csv (old share kept as census_share_with_survey), population.csv, and links to fy/ and
pixels/ of <national folder>.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import national_counties as nc  # noqa: E402  (imports the fixed county_squares next to it)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--national", required=True)
ap.add_argument("--census", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
nat, out = Path(a.national).resolve(), Path(a.out)
out.mkdir(parents=True, exist_ok=True)

cen = nc.cs.census_shares(a.census)
pop, _ = nc.ordered(cen)
pop.assign(band_name=[nc.BAND_NAMES[b] for b in pop.band]).to_csv(out / "population.csv")

sel = pd.read_csv(nat / "selection.csv", dtype={"fips": str})
ok = (sel.status == "ok").to_numpy()
assert set(sel.band[ok]) <= set(nc.BAND_NAMES), set(sel.band[ok])
c = cen.reindex(sel.fips)
sel["census_share_with_survey"] = sel.census_share
for col, src in (("grain_acres", "grain"), ("silage_acres", "silage"), ("census_share", "share")):
    sel.loc[ok, col] = c[src].to_numpy()[ok]
band = pd.cut(c.share, nc.BANDS, right=False, labels=nc.BAND_NAMES).astype(str).to_numpy()
moved = ok & (band != sel.band.to_numpy())
sel.loc[ok, "band"] = band[ok]
gone = ok & c.share.isna().to_numpy()
sel.loc[gone, "status"] = "withheld in the Census-only numbers"
sel.to_csv(out / "selection.csv", index=False)
for name in ("fy", "pixels"):
    if not (out / name).exists():
        (out / name).symlink_to(nat / name)

d = (sel.census_share - sel.census_share_with_survey)[ok & ~gone]
print(f"{ok.sum()} sampled counties: {gone.sum()} withheld, {moved.sum()} change band, |change| > 0.01: "
      f"{int((d.abs() > 0.01).sum())}, > 0.05: {int((d.abs() > 0.05).sum())}, median change {d.median():+.3f}")
print(f"population: {len(pop)} counties, Census-only silage share {pop.silage.sum() / (pop.grain + pop.silage).sum():.4f}")
print("bands of the sample:", sel[ok & ~gone].band.value_counts().to_dict())
