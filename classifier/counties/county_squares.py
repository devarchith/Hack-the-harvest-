"""Corn sample squares in seven US counties for 2022, for a check of predicted silage share against the 2022 Census.

  python classifier/counties/county_squares.py --cdl-dir /path/to/cdl_cache --out classifier/counties

No field outlines are needed. The 2022 Cropland Data Layer (CDL, 30 m, USDA NASS) marks corn pixels. Per county:
  1. Download the county's CDL GeoTIFF from CropScape (cached in --cdl-dir).
  2. Cut the CDL grid into 3 x 3-pixel blocks (90 m) and keep blocks whose 9 pixels are all corn (codes in CORN).
  3. Keep blocks in the 20 km square (EPSG:32618, 5 km steps) that holds the most kept blocks, so one pixel pull per
     county stays small.
  4. Draw SAMPLE blocks at random (seed 0) and write each as a polygon in EPSG:32618 (coordinates to the cm).
Equal-size blocks drawn at random from corn area make the mean predicted silage share an area share, comparable with
the Census acre share. Blocks sit inside fields, so very small fields are under-sampled.

Writes county_squares.csv (field-year table in the format of ../quebec/field_years_quebec.csv, no labels) and
county_selection.txt (per county: Census acres and share, blocks, the square, its share of the county's CDL corn).
"""
import argparse, hashlib, re, urllib.request
from pathlib import Path

import numpy as np, pandas as pd, geopandas as gpd, rasterio
from shapely import wkt
from shapely.geometry import box

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CRS = "EPSG:32618"
YEAR = 2022
SQUARE, STEP = 20_000, 5_000
SAMPLE = 200
# Corn (1) and double crops with corn (Dbl Crop WinWht/Corn, Oats/Corn, Barley/Corn, Corn/Soybeans). Sweet and
# popcorn (12, 13) are left out: the Census silage and grain series do not count them.
CORN = (1, 225, 226, 237, 241)
COUNTIES = ["24035", "24013", "42133", "42071", "36011", "36115", "50001"]  # state + county FIPS
CENSUS = ROOT / "02_QuickStats" / "corn_grain_silage_acres_2012_2025.csv"
CROPSCAPE = "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile?year={y}&fips={f}"


def census_shares(path):
    """2022 Census harvested acres per county: grain, silage, silage share. Withheld values are left out."""
    c = pd.read_csv(path, dtype=str)
    c = c[(c.year == str(YEAR)) & (c.aggregation_level == "COUNTY") & (c.source == "CENSUS") & (c.domain == "TOTAL")
          & c.short_description.isin(["CORN, GRAIN - ACRES HARVESTED", "CORN, SILAGE - ACRES HARVESTED"])]
    c = c.assign(fips=c.state_ansi + c.county_ansi,
                 acres=pd.to_numeric(c.value.str.replace(",", "").str.strip(), errors="coerce"))
    w = c.pivot_table(index=["fips", "state_name", "county_name"], columns="crop_use", values="acres",
                      aggfunc=lambda x: x.sum(min_count=1)).reset_index()
    w["share"] = w.silage / (w.silage + w.grain)
    return w.set_index("fips")


def cdl(fips, cache):
    """Path of the county's 2022 CDL GeoTIFF, downloaded once."""
    p = Path(cache) / f"CDL_{YEAR}_{fips}.tif"
    if not p.exists():
        xml = urllib.request.urlopen(CROPSCAPE.format(y=YEAR, f=fips), timeout=120).read().decode()
        url = re.search(r"<returnURL>(.*?)</returnURL>", xml).group(1)
        urllib.request.urlretrieve(url, p)
    return p


def corn_blocks(path):
    """GeoDataFrame (EPSG:32618) of 90 m blocks whose 9 CDL pixels are all corn, with the main CDL code."""
    with rasterio.open(path) as src:
        a, t, crs = src.read(1), src.transform, src.crs
    h, w = (a.shape[0] // 3) * 3, (a.shape[1] // 3) * 3
    b = a[:h, :w].reshape(h // 3, 3, w // 3, 3).transpose(0, 2, 1, 3).reshape(h // 3, w // 3, 9)
    ok = np.isin(b, CORN).all(-1)
    r, c = np.nonzero(ok)
    code = np.array([np.bincount(v).argmax() for v in b[r, c]])
    x0, y0 = t * (c * 3, r * 3)
    x1, y1 = t * (c * 3 + 3, r * 3 + 3)
    g = gpd.GeoDataFrame({"cdl_code": code}, geometry=[box(*v) for v in zip(x0, y1, x1, y0)], crs=crs)
    return g.to_crs(CRS)


def best_square(cent):
    """(x0, y0) of the SQUARE-wide square on STEP steps holding the most block centers."""
    xs, ys = cent.x.to_numpy(), cent.y.to_numpy()
    gx = np.arange(np.floor(xs.min() / STEP) * STEP - SQUARE + STEP, xs.max() + STEP, STEP)
    gy = np.arange(np.floor(ys.min() / STEP) * STEP - SQUARE + STEP, ys.max() + STEP, STEP)
    best = max(((((xs >= x) & (xs < x + SQUARE) & (ys >= y) & (ys < y + SQUARE)).sum(), x, y)
                for x in gx for y in gy))
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cdl-dir", required=True)
    ap.add_argument("--census", default=str(CENSUS))
    ap.add_argument("--out", default=str(HERE))
    a = ap.parse_args()
    Path(a.cdl_dir).mkdir(parents=True, exist_ok=True)
    cs = census_shares(a.census)
    rows, log = [], []
    for k, fips in enumerate(COUNTIES):
        cen = cs.loc[fips]
        p = cdl(fips, a.cdl_dir)
        g = corn_blocks(p)
        cent = g.geometry.centroid
        x0, y0 = best_square(cent)
        inside = (cent.x >= x0) & (cent.x < x0 + SQUARE) & (cent.y >= y0) & (cent.y < y0 + SQUARE)
        pick = g[inside.to_numpy()].sample(SAMPLE, random_state=0).sort_index()
        ll = pick.geometry.centroid.to_crs(4326)
        for i, (geom, code, pt) in enumerate(zip(pick.geometry, pick.cdl_code, ll)):
            rows.append({"field_id": 2_000_000 + 1_000 * k + i, "farm": f"{cen.county_name.title()} {fips}",
                         "field": f"{fips}-{i:03d}", "year": YEAR, "label": "", "label_source": "",
                         "lon": round(pt.x, 6), "lat": round(pt.y, 6), "wkt": wkt.dumps(geom, rounding_precision=2),
                         "fips": fips, "cdl_code": int(code)})
        log.append(f"{fips} {cen.county_name.title()}, {cen.state_name.title()}: Census {YEAR} harvested acres "
                   f"grain {cen.grain:.0f}, silage {cen.silage:.0f}, silage share {cen.share:.4f}; CDL "
                   f"{hashlib.sha256(p.read_bytes()).hexdigest()[:16]}; all-corn blocks {len(g)}, in the square "
                   f"{int(inside.sum())} ({inside.mean():.1%}); square x {x0:.0f}-{x0 + SQUARE:.0f}, "
                   f"y {y0:.0f}-{y0 + SQUARE:.0f}; sampled {len(pick)}, CDL codes "
                   f"{dict(sorted(pick.cdl_code.value_counts().items()))}")
        print(log[-1], flush=True)
    out = pd.DataFrame(rows)
    hand = ["drop_doy", "ndvi_pre_drop", "drop_size", "ndvi_sep15", "ndti_post_drop"]
    out = out.reindex(columns=["field_id", "farm", "field", "year", "label", *hand, "label_source", "lon", "lat", "wkt",
                               "fips", "cdl_code"])
    out.to_csv(Path(a.out) / "county_squares.csv", index=False)
    (Path(a.out) / "county_selection.txt").write_text("\n".join(log) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
