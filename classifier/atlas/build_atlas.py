"""Build atlas.html: an interactive map of the silage/grain predictions.

Reads only committed result files (and claims.json, written by claims.py) and the public
Census boundary files, and writes one
self-contained HTML page (data embedded; Leaflet and the imagery tiles load from the web).

    python classifier/atlas/build_atlas.py

Views: US counties (163 sampled + 7 test counties) with their sampled 90 m corn blocks,
the 44 Maryland evaluation field-years, and the 2,245 Quebec field-years.
"""
from __future__ import annotations

import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import wkt as shapely_wkt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NATIONAL = REPO / "classifier" / "counties" / "national"
RUN = NATIONAL / "results_final"
STATES_URL = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_state_20m.zip"
COUNTIES_URL = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_county_20m.zip"
SKIP_STATES = {"02", "15", "60", "66", "69", "72", "78"}  # outside the lower 48


def fix_fips(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)


def fetch_boundary(url, cache):
    """Download a Census cartographic boundary zip once and read it."""
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / url.rsplit("/", 1)[1]
    if not path.exists():
        with urllib.request.urlopen(url, timeout=120) as r:
            data = r.read()
        zipfile.ZipFile(io.BytesIO(data)).testzip()
        path.write_bytes(data)
    return gpd.read_file(f"zip://{path}").to_crs(4326)


def rings(geom, nd=3):
    """Polygon or MultiPolygon -> list of outer rings as [lat, lon] pairs."""
    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    return [[[round(y, nd), round(x, nd)] for x, y in p.exterior.coords] for p in polys]


def utm_epsg(lon):
    return 32600 + int((lon + 180) // 6) + 1


def to_latlon(wkt_series, epsg, nd=5):
    """UTM outline text -> list of ring lists in [lat, lon]."""
    tr = Transformer.from_crs(epsg, 4326, always_xy=True)
    out = []
    for text in wkt_series:
        g = shapely_wkt.loads(text)
        polys = list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
        rr = []
        for p in polys:
            xs, ys = p.exterior.coords.xy
            lon, lat = tr.transform(np.asarray(xs), np.asarray(ys))
            rr.append([[round(float(a), nd), round(float(b), nd)] for a, b in zip(lat, lon)])
        out.append(rr)
    return out


def check_position(ring_lists, lon, lat, what, tol=0.01):
    """The reprojected outline must sit on the lon/lat stored in the table."""
    for rr, x, y in zip(ring_lists, lon, lat):
        a = np.asarray(rr[0])
        if abs(a[:, 0].mean() - y) > tol or abs(a[:, 1].mean() - x) > tol:
            raise SystemExit(f"{what}: outline is not at its stored lon/lat ({x}, {y})")


def counts(label, p):
    s = label == "Silage"
    c = p >= 0.5
    return {"ss": int((s & c).sum()), "sg": int((s & ~c).sum()),
            "gs": int((~s & c).sum()), "gg": int((~s & ~c).sum())}


def build(a):
    cp = pd.read_csv(RUN / "county_predictions.csv", dtype={"fips": str})
    cp["fips"] = fix_fips(cp["fips"])
    tc = pd.read_csv(RUN / "test_county_predictions.csv", dtype={"fips": str})
    tc["fips"] = fix_fips(tc["fips"])
    prior = pd.read_csv(RUN / "census_prior" / "census_prior_counties.csv", dtype={"fips": str})
    prior["fips"] = fix_fips(prior["fips"])
    bp = pd.read_csv(RUN / "block_predictions.csv", dtype={"fips": str})
    bp["fips"] = fix_fips(bp["fips"])
    fp = pd.read_csv(RUN / "field_predictions.csv")

    sq = pd.concat([
        pd.read_csv(NATIONAL / "county_squares.csv", dtype={"fips": str}),
        pd.read_csv(REPO / "classifier" / "counties" / "county_squares.csv", dtype={"fips": str}),
    ], ignore_index=True)
    sq["fips"] = fix_fips(sq["fips"])
    blocks = bp.merge(sq[["field_id", "lon", "lat", "wkt"]], on="field_id", how="left", validate="one_to_one")
    if blocks["wkt"].isna().any():
        raise SystemExit("block predictions without an outline")
    # sampled counties: out-of-fold prediction; test counties: the final model (never trained on them)
    blocks["p"] = np.where(blocks["kind"] == "national", blocks["p_llp_oof"], blocks["p_llp"])
    if blocks["p"].isna().any():
        raise SystemExit("block without a county-share prediction")

    # ---- counties
    last = prior.set_index("fips")["last_census"]
    last_year = prior.set_index("fips")["prior_source"]
    rows = []
    for r in cp.itertuples():
        rows.append({"fips": r.fips, "name": r.county, "state": r.state, "band": r.band, "test": 0,
                     "census": r.census_share, "pred": r.llp_oof_mean_p, "called": r.llp_oof_called,
                     "md": r.barc_mean_p, "md_called": r.barc_called,
                     "grain_acres": r.grain_acres, "silage_acres": r.silage_acres,
                     "last": None if pd.isna(last.get(r.fips, np.nan)) else float(last[r.fips]),
                     "last_year": None if pd.isna(last_year.get(r.fips, np.nan)) else str(last_year[r.fips])})
    for r in tc.itertuples():
        rows.append({"fips": r.fips, "name": r.county_name.title(), "state": r.state_name.title(), "band": "",
                     "test": 1, "census": r.census_share, "pred": r.llp_mean_p, "called": r.llp_called,
                     "md": r.barc_mean_p, "md_called": r.barc_called,
                     "grain_acres": None, "silage_acres": None, "last": None, "last_year": None})
    counties = pd.DataFrame(rows)
    if counties["fips"].duplicated().any():
        raise SystemExit("a county is both sampled and a test county")

    cache = Path(a.boundaries_cache)
    cgeo = fetch_boundary(COUNTIES_URL, cache)
    cgeo["fips"] = cgeo["STATEFP"] + cgeo["COUNTYFP"]
    cgeo = cgeo.set_index("fips")
    sgeo = fetch_boundary(STATES_URL, cache)
    sgeo = sgeo[~sgeo["STATEFP"].isin(SKIP_STATES)]

    crs = cp.set_index("fips")["crs"].to_dict()
    out_counties, out_blocks = [], {}
    for c in counties.to_dict("records"):
        b = blocks[blocks["fips"] == c["fips"]]
        if c["fips"] not in cgeo.index or b.empty:
            raise SystemExit(f"county {c['fips']}: no outline or no blocks")
        epsg = int(crs[c["fips"]].split(":")[1]) if c["fips"] in crs else utm_epsg(b["lon"].mean())
        rr = to_latlon(b["wkt"], epsg)
        check_position(rr, b["lon"], b["lat"], f"blocks of {c['fips']}")
        out_blocks[c["fips"]] = [[r[0][:4], round(float(p), 3), round(float(m), 3)]
                                 for r, p, m in zip(rr, b["p"], b["p_barc"])]
        c["n"] = int(len(b))
        c["lat"] = round(float(b["lat"].mean()), 4)
        c["lon"] = round(float(b["lon"].mean()), 4)
        c["geom"] = rings(cgeo.loc[c["fips"], "geometry"])
        out_counties.append({k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in c.items()})

    # ---- Maryland evaluation fields
    fy = pd.read_csv(REPO / "classifier" / "presto" / "field_years.csv")
    barc = fp[fp["kind"] == "barc"].merge(fy[["field_id", "year", "farm", "field", "lon", "lat", "wkt"]],
                                         on=["field_id", "year"], how="left", validate="one_to_one")
    if barc["wkt"].isna().any():
        raise SystemExit("Maryland field-year without an outline")
    rr = to_latlon(barc["wkt"], 32618)
    check_position(rr, barc["lon"], barc["lat"], "Maryland fields")
    out_barc = [{"farm": r.farm, "field": r.field, "year": int(r.year), "label": r.label,
                 "md": round(float(r.p_barc), 3), "cs": round(float(r.p_llp), 3), "geom": g}
                for r, g in zip(barc.itertuples(), rr)]

    # ---- Quebec field-years (points at the field's stored lon/lat)
    qy = pd.read_csv(REPO / "classifier" / "quebec" / "field_years_quebec.csv")
    que = fp[fp["kind"] == "quebec"].merge(qy[["field_id", "year", "lon", "lat", "area_ha"]],
                                          on=["field_id", "year"], how="left", validate="one_to_one")
    if que["lon"].isna().any():
        raise SystemExit("Quebec field-year without a position")
    out_que = [[round(float(r.lat), 5), round(float(r.lon), 5), int(r.year), 1 if r.label == "Silage" else 0,
                round(float(r.p_barc), 3), round(float(r.p_llp), 3)] for r in que.itertuples()]

    # ---- scorecard numbers, all computed here from the same files
    samp = counties[counties["test"] == 0]
    test = counties[counties["test"] == 1]
    b_md, b_cs = fp[fp["kind"] == "barc"], fp[fp["kind"] == "quebec"]
    stats = {
        "n_counties": int(len(samp)), "n_states": int(samp["state"].nunique()),
        "n_test": int(len(test)), "n_blocks": int((blocks["kind"] == "national").sum()),
        "n_test_blocks": int((blocks["kind"] == "test_county").sum()),
        "md": {"md": counts(b_md["label"], b_md["p_barc"]), "cs": counts(b_md["label"], b_md["p_llp"])},
        "qc": {"md": counts(b_cs["label"], b_cs["p_barc"]), "cs": counts(b_cs["label"], b_cs["p_llp"])},
        "county_gap": {"md": float((samp["md_called"] - samp["census"]).abs().mean()),
                       "cs": float((samp["called"] - samp["census"]).abs().mean())},
        "county_gap_mean_p": {"md": float((samp["md"] - samp["census"]).abs().mean()),
                              "cs": float((samp["pred"] - samp["census"]).abs().mean())},
        "test_gap": {"md": float((test["md_called"] - test["census"]).abs().mean()),
                     "cs": float((test["called"] - test["census"]).abs().mean())},
        "last_census_gap": float((prior["last_census"] - prior["truth"]).abs().mean()),
        "n_last": int(prior["last_census"].notna().sum()),
    }
    # supporting numbers from claims.py (confident calls, counties whose share moved)
    claims = json.loads((HERE / "claims.json").read_text(encoding="utf-8"))
    stats["confident"] = claims["E_confident_calls"]
    stats["moved"] = claims["G_fast_changing_counties"]
    data = {"stats": stats, "counties": out_counties, "blocks": out_blocks, "barc": out_barc, "quebec": out_que,
            "states": [r for g in sgeo.geometry for r in rings(g, 2)]}

    page = (HERE / "template.html").read_text(encoding="utf-8")
    if "/*DATA*/" not in page:
        raise SystemExit("template.html has no /*DATA*/ marker")
    page = page.replace("/*DATA*/", json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    out = Path(a.out)
    out.write_text(page, encoding="utf-8", newline="\n")
    print(json.dumps(stats, indent=1))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "atlas.html"), help="output HTML path")
    ap.add_argument("--boundaries-cache", default=str(HERE / "data"),
                    help="where to keep the Census boundary zips (not committed)")
    build(ap.parse_args(argv))


if __name__ == "__main__":
    main()
