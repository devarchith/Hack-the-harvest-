"""Corn sample squares in US counties drawn across five silage-share bands, for a national check of Presto against the
2022 Census and for training on county shares.

  python national_counties.py --cdl-dir /path/to/cdl_cache --out /path/to/national [--select-only]

Population: counties with both 2022 Census numbers (grain and silage harvested acres) and at least MIN_ACRES of corn.
Left out: the seven counties of county_squares.py (kept as a test set) and Prince George's, MD (BARC's county).
Selection (seed 0): counties in random order within each silage-share band, at most CAP per state per band; the
first PER_BAND of each band are wave 1, the next PER_BAND2 wave 2, the next PER_BAND3 wave 3. A county with too few all-corn blocks in its best
square is skipped and the next one in its band takes its place.
Per county (blocks from county_squares.corn_blocks, the same corn codes): the UTM zone of the county's corn, the
SQUARE-wide square on STEP steps holding the most all-corn 90 m blocks, SAMPLE blocks drawn at random (seed 0).

Writes <out>/selection.csv (one row per county tried), <out>/fy/<fips>.csv (field-year table per county, polygons
in that county's UTM zone; the format of county_squares.csv) and <out>/national_blocks.csv (all of them, with crs).
"""
import argparse, hashlib, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import county_squares as cs  # noqa: E402  copy of classifier/counties/county_squares.py

MIN_ACRES = 10_000
BANDS = [0, 0.05, 0.15, 0.40, 0.70, 1.0001]
BAND_NAMES = ["under 5%", "5-15%", "15-40%", "40-70%", "70% and over"]
PER_BAND, PER_BAND2, PER_BAND3, CAP = 12, 8, 20, 3
SQUARE, STEP, SAMPLE, MIN_BLOCKS = 10_000, 2_500, 100, 150
TEST = set(cs.COUNTIES)
BARC_COUNTY = "24033"


def ordered(cen):
    """Population in draw order: band, then wave rank within band (rank < PER_BAND = wave 1)."""
    pop = cen[(cen.grain + cen.silage >= MIN_ACRES) & cen.grain.notna() & cen.silage.notna()].copy()
    pop = pop[~pop.index.isin(TEST | {BARC_COUNTY})]
    pop["band"] = pd.cut(pop.share, BANDS, right=False, labels=range(len(BAND_NAMES))).astype(int)
    rng = np.random.default_rng(0)
    out = []
    for b in range(len(BAND_NAMES)):
        d = pop[pop.band == b]
        d = d.iloc[rng.permutation(len(d))]
        d = d.assign(state_rank=d.groupby("state_name").cumcount())
        d = d[d.state_rank < CAP]
        out.append(d.assign(rank=np.arange(len(d))))
    return pop, pd.concat(out)


def utm_epsg(lon, lat):
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def best_square(cent):
    xs, ys = cent.x.to_numpy(), cent.y.to_numpy()
    gx = np.arange(np.floor(xs.min() / STEP) * STEP - SQUARE + STEP, xs.max() + STEP, STEP)
    gy = np.arange(np.floor(ys.min() / STEP) * STEP - SQUARE + STEP, ys.max() + STEP, STEP)
    return max(((((xs >= x) & (xs < x + SQUARE) & (ys >= y) & (ys < y + SQUARE)).sum(), x, y) for x in gx for y in gy))


def get_cdl(fips, cdl_dir, tries=5):
    """cs.cdl with retries (CropScape answers 502 at times); a partial or unreadable file is deleted and fetched again."""
    for t in range(tries):
        try:
            p = cs.cdl(fips, cdl_dir)
            with cs.rasterio.open(p):
                return p
        except Exception:
            (Path(cdl_dir) / f"CDL_{cs.YEAR}_{fips}.tif").unlink(missing_ok=True)
            if t == tries - 1:
                raise
            time.sleep(10 * (t + 1))


def county(fips, cdl_dir):
    """(selection row, blocks DataFrame or None) for one county."""
    try:
        p = get_cdl(fips, cdl_dir)
    except Exception as e:
        return {"fips": fips, "status": f"CDL download failed: {repr(e)[:120]}"}, None
    cs.CRS = "EPSG:4326"
    ll = cs.corn_blocks(p)
    row = {"fips": fips, "cdl_sha256_16": hashlib.sha256(p.read_bytes()).hexdigest()[:16], "corn_blocks": len(ll)}
    if len(ll) < MIN_BLOCKS:
        return {**row, "status": "too few corn blocks in county"}, None
    c = ll.geometry.centroid
    epsg = utm_epsg(float(np.median(c.x)), float(np.median(c.y)))
    g = ll.to_crs(epsg)
    cent = g.geometry.centroid
    n, x0, y0 = best_square(cent)
    row.update(crs=f"EPSG:{epsg}", square_blocks=int(n), square_x0=x0, square_y0=y0)
    if n < MIN_BLOCKS:
        return {**row, "status": "too few corn blocks in best square"}, None
    inside = ((cent.x >= x0) & (cent.x < x0 + SQUARE) & (cent.y >= y0) & (cent.y < y0 + SQUARE)).to_numpy()
    pick = g[inside].sample(SAMPLE, random_state=0).sort_index()
    pt = pick.geometry.centroid.to_crs(4326)
    b = pd.DataFrame({"wkt": pick.geometry.to_wkt().to_numpy(), "cdl_code": pick.cdl_code.to_numpy(),
                      "lon": pt.x.round(6).to_numpy(), "lat": pt.y.round(6).to_numpy()})
    return {**row, "status": "ok"}, b


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cdl-dir", required=True)
    ap.add_argument("--census", default=str(cs.CENSUS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--waves", type=int, nargs="+", default=[1])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--select-only", action="store_true", help="print the draw order and stop (no downloads)")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "fy").mkdir(parents=True, exist_ok=True)
    Path(a.cdl_dir).mkdir(parents=True, exist_ok=True)
    cen = cs.census_shares(a.census)
    pop, order = ordered(cen)
    pop.assign(band_name=[BAND_NAMES[b] for b in pop.band]).to_csv(out / "population.csv")
    want = {1: PER_BAND, 2: PER_BAND + PER_BAND2, 3: PER_BAND + PER_BAND2 + PER_BAND3}[max(a.waves)]
    if a.select_only:
        for b in range(len(BAND_NAMES)):
            d = order[order.band == b]
            print(BAND_NAMES[b], len(pop[pop.band == b]), "in population,", len(d), "after state cap;",
                  ", ".join(f"{r.county_name.title()} {r.state_name[:2]} {r.share:.2f}" for r in d.head(want).itertuples()))
        return

    sel = pd.read_csv(out / "selection.csv", dtype={"fips": str}).set_index("fips") if (out / "selection.csv").exists() \
        else pd.DataFrame()
    rows, blocks = [], {}
    for b in range(len(BAND_NAMES)):
        d = order[order.band == b]
        ok, i = 0, 0
        while ok < want and i < len(d):
            batch = d.iloc[i:i + max(want - ok, 1)]
            i += len(batch)
            with ThreadPoolExecutor(a.workers) as pool:
                res = list(pool.map(lambda f: county(f, a.cdl_dir), batch.index))
            for (fips, r), (row, blk) in zip(batch.iterrows(), res):
                wave = 1 if ok < PER_BAND else 2 if ok < PER_BAND + PER_BAND2 else 3
                row.update(state=r.state_name.title(), county=r.county_name.title(), band=BAND_NAMES[b], rank=int(r["rank"]),
                           grain_acres=r.grain, silage_acres=r.silage, census_share=round(r.share, 4),
                           wave=wave if blk is not None else None)
                rows.append(row)
                print(row, flush=True)
                if blk is not None:
                    ok += 1
                    blocks[fips] = (blk, row)
    k_of = {f: k for k, f in enumerate(sorted(pop.index))}  # stable across waves
    allb = []
    for fips, (blk, row) in blocks.items():
        k = k_of[fips]
        t = pd.DataFrame({"field_id": 3_000_000 + 1_000 * k + np.arange(len(blk)),
                          "farm": f"{row['county']} {fips}", "field": [f"{fips}-{i:03d}" for i in range(len(blk))],
                          "year": cs.YEAR, "label": "", **{h: np.nan for h in
                          ["drop_doy", "ndvi_pre_drop", "drop_size", "ndvi_sep15", "ndti_post_drop"]},
                          "label_source": "", "lon": blk.lon, "lat": blk.lat, "wkt": blk.wkt, "fips": fips,
                          "cdl_code": blk.cdl_code})
        f = out / "fy" / f"{fips}.csv"
        if not f.exists():  # an earlier wave wrote the same table; a pull may be reading it
            t.to_csv(f, index=False)
        allb.append(t.assign(crs=row["crs"], census_share=row["census_share"], band=row["band"], wave=row["wave"]))
    pd.DataFrame(rows).to_csv(out / "selection.csv.tmp", index=False)
    os.replace(out / "selection.csv.tmp", out / "selection.csv")  # launch_pulls.sh reads it every minute
    pd.concat(allb).to_csv(out / "national_blocks.csv", index=False)
    print(f"{len(blocks)} counties with blocks; {sum(len(b) for b, _ in blocks.values())} blocks", flush=True)


if __name__ == "__main__":
    main()
