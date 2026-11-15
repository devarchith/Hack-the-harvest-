#!/usr/bin/env python3
"""Turn a presto_national.py results folder into one self-contained HTML page (figures embedded as PNG).

    python national_page.py --results R --national N --test-counties T --out page.html [--boundaries-cache DIR]

Needs pandas, numpy, matplotlib. geopandas + shapely draw the state basemap (Albers, EPSG:5070) and the block
polygons; without them, or without the Census boundary download, the page falls back to plain lon/lat axes.
Every number on the page is read or computed from the files. A missing optional file skips its section with a note.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
import sys
import traceback
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

try:
    import geopandas as gpd
except Exception:  # pragma: no cover - the pod has it; the page still renders without it
    gpd = None
try:
    from shapely import wkt as shp_wkt
except Exception:  # pragma: no cover
    shp_wkt = None

STATES_URL = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_state_20m.zip"
COUNTIES_URL = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_county_20m.zip"
NOT_LOWER48 = {"02", "15", "60", "66", "69", "72", "78"}
CLASS_CUTS = [0.05, 0.40]                      # same classes as presto_national.py (left-closed)
CLASS_NAMES = ["under 5%", "5-40%", "40% and over"]
TEST_CRS = "EPSG:32618"                       # county_squares.py writes its blocks in UTM 18N
DPI = 110
C_BARC, C_LLP = "#2f6db5", "#d9701e"          # BARC-only blue, proportion-trained orange
INK, MUTED, GRID = "#222222", "#666666", "#e6e6e6"
SHARE_CMAP = LinearSegmentedColormap.from_list("share", plt.get_cmap("YlGn")(np.linspace(0.12, 1.0, 256)))
P_CMAP = plt.get_cmap("BrBG")                 # brown = grain side, teal = silage side, light at p = 0.5
SHARE_NORM = Normalize(0, 1)

plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5, "axes.edgecolor": "#888888",
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"})


def log(*a):
    print("[national_page]", *a, file=sys.stderr, flush=True)


def esc(x):
    return html.escape(str(x))


def note(msg):
    return f'<p class="note">{esc(msg)}</p>'


def fig_img(fig, alt):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f'<img alt="{esc(alt)}" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def pct(v, nd=0):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{100 * v:.{nd}f}%"


def pts(v):
    k = round(100 * float(v))
    return f"{k} percentage point" + ("" if k == 1 else "s")


def num(v, nd=3):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "n/a" if v is None else str(v)
    return "n/a" if math.isnan(v) else f"{v:.{nd}f}"


def nice_name(s):
    """'QUEEN ANNE'S' -> "Queen Anne's" (str.title would give "Anne'S")."""
    return " ".join(w[:1].upper() + w[1:].lower() for w in str(s).split())


def spearman(a, b):
    d = pd.DataFrame({"a": np.asarray(a, float), "b": np.asarray(b, float)}).dropna()
    if len(d) < 3 or d.a.nunique() < 2 or d.b.nunique() < 2:
        return float("nan")
    return float(d.a.rank().corr(d.b.rank()))


def share_class(v):
    v = np.asarray(v, float)
    return np.digitize(v, CLASS_CUTS)          # 0: < 0.05, 1: [0.05, 0.40), 2: >= 0.40


def class_matrix(true, pred):
    t, p = np.asarray(true, float), np.asarray(pred, float)
    ok = ~(np.isnan(t) | np.isnan(p))
    m = np.zeros((3, 3), int)
    for i, j in zip(share_class(t[ok]), share_class(p[ok])):
        m[i, j] += 1
    return m


def table_html(df, hl_rows=(), bold_last=False):
    cols = list(df.columns)
    out = ['<div class="tablewrap"><table><thead><tr>' + "".join(f"<th>{esc(c)}</th>" for c in cols) + "</tr></thead><tbody>"]
    for k, (_, r) in enumerate(df.iterrows()):
        cls = ' class="hl"' if (k in hl_rows or (bold_last and k == len(df) - 1)) else ""
        out.append(f"<tr{cls}>" + "".join(f"<td>{esc(r[c])}</td>" for c in cols) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


# ---------------------------------------------------------------- inputs

def read_csv(path, **kw):
    path = Path(path)
    if not path.exists():
        log("missing", path)
        return None
    try:
        return pd.read_csv(path, **kw)
    except Exception as e:
        log("could not read", path, e)
        return None


def fix_fips(s):
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(5)


class Data:
    def __init__(self, a):
        R, N = Path(a.results), Path(a.national)
        T = Path(a.test_counties) if a.test_counties else None
        self.R, self.N, self.T = R, N, T
        self.run = {}
        if (R / "run.json").exists():
            try:
                self.run = json.loads((R / "run.json").read_text(encoding="utf-8"))
            except Exception as e:
                log("run.json unreadable", e)
        self.setting = self.run.get("setting", {}) or {}
        self.w = self.setting.get("w_chosen")
        self.w_txt = f"{float(self.w):g}" if self.w is not None else "?"

        ct = read_csv(R / "county_predictions.csv", dtype={"fips": str})
        if ct is not None:
            if "fips" not in ct.columns:
                ct = ct.rename(columns={ct.columns[0]: "fips"})
            ct["fips"] = fix_fips(ct.fips)
            ct = ct.set_index("fips")
        self.ct = ct
        tct = read_csv(R / "test_county_predictions.csv", dtype={"fips": str})
        if tct is not None:
            if "fips" not in tct.columns:
                tct = tct.rename(columns={tct.columns[0]: "fips"})
            tct["fips"] = fix_fips(tct.fips)
            tct = tct.set_index("fips")
            tct["name"] = [f"{nice_name(c)}, {nice_name(s)}" for c, s in
                           zip(tct.get("county_name", tct.index), tct.get("state_name", ""))]
        self.tct = tct
        bp = read_csv(R / "block_predictions.csv", dtype={"fips": str, "field_id": str})
        if bp is not None:
            bp["fips"] = bp.fips.where(bp.fips.isna(), fix_fips(bp.fips.fillna("")))
        self.blocks = bp
        self.fields = read_csv(R / "field_predictions.csv", dtype={"fips": str, "field_id": str})
        self.states_check = read_csv(R / "state_check.csv")
        mp = R / "metrics.md"
        self.metrics_md = mp.read_text(encoding="utf-8", errors="replace") if mp.exists() else None

        sel = read_csv(N / "selection.csv", dtype={"fips": str})
        if sel is not None and "status" in sel.columns:
            sel = sel[sel.status == "ok"]
        if sel is not None:
            sel["fips"] = fix_fips(sel.fips)
            sel = sel.drop_duplicates("fips").set_index("fips")
        self.sel = sel
        pop = read_csv(N / "population.csv", dtype={"fips": str})
        if pop is not None:
            pop["fips"] = fix_fips(pop.fips)
        self.pop = pop
        sq = read_csv(T / "county_squares.csv", dtype={"fips": str, "field_id": str}) if T else None
        if sq is not None:
            sq["fips"] = fix_fips(sq.fips)
        self.squares = sq
        self._fy = {}

    # ---- block geometry for one county: DataFrame field_id, lon, lat, wkt (+ crs string)
    def county_blocks(self, fips, test):
        if test:
            if self.squares is None:
                return None, None
            d = self.squares[self.squares.fips == fips]
            crs = str(d.crs.iloc[0]) if "crs" in d.columns and len(d) else TEST_CRS
        else:
            if fips not in self._fy:
                self._fy[fips] = read_csv(self.N / "fy" / f"{fips}.csv", dtype={"field_id": str, "fips": str})
            d = self._fy[fips]
            crs = None
            for src in (self.ct, self.sel):
                if src is not None and "crs" in src.columns and fips in src.index:
                    crs = str(src.loc[fips, "crs"])
                    break
        if d is None or not len(d):
            return None, crs
        return d.drop_duplicates("field_id"), crs

    def county_location(self, fips, test):
        d, _ = self.county_blocks(fips, test)
        if d is None or "lon" not in d.columns:
            return None
        lon, lat = pd.to_numeric(d.lon, errors="coerce").mean(), pd.to_numeric(d.lat, errors="coerce").mean()
        return None if (math.isnan(lon) or math.isnan(lat)) else (lon, lat)

    def choice_row(self, w):
        for r in self.run.get("choice", []) or []:
            try:
                if abs(float(r.get("w")) - float(w)) < 1e-9:
                    return r
            except (TypeError, ValueError):
                continue
        return None

    def field_counts(self, kind, col):
        f = self.fields
        if f is None or col not in f.columns:
            return None
        d = f[f.kind == kind].dropna(subset=[col])
        if not len(d):
            return None
        y = d.label.astype(str).str.lower().eq("silage").to_numpy()
        p = d[col].to_numpy(float) >= 0.5
        return {"n": len(d), "correct": int((y == p).sum()), "tp": int((y & p).sum()), "fn": int((y & ~p).sum()),
                "fp": int((~y & p).sum()), "tn": int((~y & ~p).sum()), "silage": int(y.sum()),
                "years": sorted(pd.to_numeric(d.year, errors="coerce").dropna().astype(int).unique().tolist())}


# ---------------------------------------------------------------- boundaries

def fetch_boundary(url, cache):
    name = url.rstrip("/").rsplit("/", 1)[-1]
    stem = name[:-4] if name.lower().endswith(".zip") else name
    shp_dir = cache / stem
    shp = shp_dir / f"{stem}.shp"
    if shp.exists():
        return shp
    try:
        cache.mkdir(parents=True, exist_ok=True)
        zp = cache / f"{stem}.zip"
        if not zp.exists() or zp.stat().st_size == 0:
            log("downloading", url)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (national_page.py)"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            tmp = zp.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(zp)
        with zipfile.ZipFile(zp) as z:
            z.extractall(shp_dir)
        if shp.exists():
            return shp
        found = sorted(shp_dir.rglob("*.shp"))
        return found[0] if found else None
    except Exception as e:
        log(f"boundary download failed ({url}): {e}; falling back")
        return None


def load_boundaries(a, out):
    if gpd is None:
        return None, None, "geopandas is not available, so the map uses plain longitude/latitude axes."
    cache = Path(a.boundaries_cache) if a.boundaries_cache else out.parent / "boundaries"
    states = counties = None
    shp = fetch_boundary(a.states_url, cache)
    if shp is not None:
        try:
            s = gpd.read_file(shp)
            s = s[~s["STATEFP"].astype(str).isin(NOT_LOWER48)]
            states = s.to_crs(5070)
        except Exception as e:
            log("states unreadable", e)
    if states is None:
        return None, None, ("The Census state boundary file could not be downloaded or read, so the map shows dots "
                            "on plain longitude/latitude axes.")
    if not a.no_county_polygons:
        shp = fetch_boundary(a.counties_url, cache)
        if shp is not None:
            try:
                c = gpd.read_file(shp)
                c = c[~c["STATEFP"].astype(str).isin(NOT_LOWER48)]
                counties = c.to_crs(5070)
            except Exception as e:
                log("counties unreadable", e)
    return states, counties, None


# ---------------------------------------------------------------- sections

def section_headline(D):
    run, ct, tct = D.run, D.ct, D.tct
    w0, ww = D.choice_row(0), D.choice_row(D.w) if D.w is not None else None
    tiles, sent, fsent, tsent = [], [], [], ""
    n_used = run.get("counties_used", len(ct) if ct is not None else None)
    n_states = ct.state.nunique() if ct is not None and "state" in ct.columns else None
    missing = run.get("counties_missing_pixels") or []
    n_test = len(tct) if tct is not None else 0
    sub = (f"in {n_states} states" if n_states else "") + (f"; {len(missing)} left out for missing pixels" if missing else "")
    tiles.append(("Sampled counties used", f"{n_used}" if n_used is not None else "n/a", sub))
    if n_test:
        tiles.append(("Test counties (never trained on)", f"{n_test}", ", ".join(sorted(tct.get("state_name", pd.Series(dtype=str))
                                                                              .map(nice_name).unique()))))
    wl = f"w {D.w_txt}"
    if w0 and ww:
        tiles.append(("County share gap, out of fold (mean abs)", f"{num(w0.get('gap called (mean abs)'))} → "
                      f"{num(ww.get('gap called (mean abs)'))}", f"BARC-only (w 0) → proportion-trained ({wl})"))
        sp0 = str(w0.get("Spearman (mean p vs Census)", "n/a")).split(" (")[0]
        sp1 = str(ww.get("Spearman (mean p vs Census)", "n/a")).split(" (")[0]
        tiles.append(("Spearman with Census share, out of fold", f"{sp0} → {sp1}", "rank correlation, mean probability"))
        tiles.append(("County class right, out of fold", f"{w0.get('class right', 'n/a')} → {ww.get('class right', 'n/a')}",
                      "; ".join(CLASS_NAMES)))
        sent.append(f"Across {n_used} sampled counties, out of fold, the mean absolute gap between predicted and Census "
                    f"county silage share is {num(w0.get('gap called (mean abs)'))} for BARC-only (w 0) and "
                    f"{num(ww.get('gap called (mean abs)'))} for the proportion-trained model ({wl}); Spearman "
                    f"{sp0} vs {sp1}; county class right {w0.get('class right')} vs {ww.get('class right')}.")
    md0, md1 = D.field_counts("barc", "p_barc"), D.field_counts("barc", "p_llp")
    if md0 and md1:
        yrs = f"{min(md0['years'])}-{max(md0['years'])}" if md0["years"] else ""
        tiles.append((f"Maryland test fields {yrs}, correct", f"{md0['correct']} → {md1['correct']} of {md0['n']}",
                      "BARC-only → proportion-trained"))
        fsent.append(f"Maryland {yrs} {md0['correct']} vs {md1['correct']} of {md0['n']} right")
    q0, q1 = D.field_counts("quebec", "p_barc"), D.field_counts("quebec", "p_llp")
    if q0 and q1:
        tiles.append(("Québec fields, correct", f"{q0['correct']:,} → {q1['correct']:,} of {q0['n']:,}",
                      "no Québec label used in training"))
        fsent.append(f"Québec {q0['correct']:,} vs {q1['correct']:,} of {q0['n']:,} (no Québec label used)")
    if tct is not None and {"census_share", "barc_called", "llp_called"} <= set(tct.columns):
        g0 = float((tct.barc_called - tct.census_share).abs().mean())
        g1 = float((tct.llp_called - tct.census_share).abs().mean())
        tiles.append(("Test-county share gap (mean abs)", f"{num(g0)} → {num(g1)}", f"{n_test} counties, final model"))
        tsent = (f"On the {n_test} test counties, never trained on, the mean absolute share gap is {num(g0)} vs "
                 f"{num(g1)}.")
    if D.states_check is not None and len(D.states_check):
        us = D.states_check.iloc[-1]
        cols = list(D.states_check.columns)
        try:
            ic = [c for c in cols if c.startswith("sampled, Census")][0]
            ib = [c for c in cols if c.startswith("sampled, BARC")][0]
            il = [c for c in cols if c.startswith("sampled, proportion")][0]
            npop = len(D.pop) if D.pop is not None else None
            tiles.append(("US silage share of corn (band-stratified estimate)", f"{num(us[ib])} → {num(us[il])}",
                          f"Census 2022: {num(us[ic])}" + (f" over {npop:,} eligible counties" if npop else "")))
        except (IndexError, KeyError):
            pass
    lead = ("Presto (NASA Harvest's pretrained pixel time-series transformer) is fine-tuned on field labels from BARC "
            "(Beltsville, Maryland) to tell silage from grain corn. The proportion-trained model adds one loss: the mean "
            "predicted silage probability of each sampled US county's corn blocks should match that county's 2022 "
            "Census of Agriculture silage share. This page compares it with the BARC-only model.")
    body = f"<p>{esc(lead)}</p>"
    if fsent:
        sent.append("On held-out fields (BARC-only vs proportion-trained): " + "; ".join(fsent) + ".")
    if tsent:
        sent.append(tsent)
    if sent:
        body += "<p>" + esc(" ".join(sent)) + "</p>"
    body += '<div class="tiles">' + "".join(
        f'<div class="tile"><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div><div class="s">{esc(s)}</div></div>'
        for k, v, s in tiles) + "</div>"
    body += note("Arrows read BARC-only → proportion-trained. w 0 in the weight choice is BARC-only with BARC's "
                 "latitude/longitude on every pixel (the same input as the proportion-trained model). Out-of-fold "
                 "numbers come from models that never saw the county's state group.")
    return body


def section_map(D, states, counties, map_note):
    ct, tct = D.ct, D.tct
    if ct is None:
        return note("county_predictions.csv is missing, so there is no map.")
    pts = []
    for f in ct.index:
        loc = D.county_location(f, test=False)
        pts.append((f, False, loc))
    if tct is not None:
        for f in tct.index:
            pts.append((f, True, D.county_location(f, test=True)))
    # counties without block coordinates: use the county polygon centroid if we have it
    cent = {}
    if counties is not None:
        cc = counties.set_index(counties["GEOID"].astype(str))
        cent = {g: (p.x, p.y) for g, p in cc.geometry.representative_point().items()}
    proj = states is not None
    rows, lost = [], []
    for f, test, loc in pts:
        if loc is not None:
            rows.append((f, test, loc[0], loc[1], True))
        elif proj and f in cent:
            rows.append((f, test, cent[f][0], cent[f][1], False))
        else:
            lost.append(f)
    if not rows:
        return note("No county coordinates were found (fy/<fips>.csv, county_squares.csv), so there is no map.")
    P = pd.DataFrame(rows, columns=["fips", "test", "x", "y", "lonlat"])
    if proj:
        ll = P[P.lonlat]
        if len(ll):
            g = gpd.GeoSeries(gpd.points_from_xy(ll.x, ll.y), crs=4326).to_crs(5070)
            P.loc[ll.index, "x"] = g.x.to_numpy()
            P.loc[ll.index, "y"] = g.y.to_numpy()
    nat, tst = P[~P.test].set_index("fips"), P[P.test].set_index("fips")

    def col(df, name):
        return df[name] if df is not None and name in df.columns else pd.Series(dtype=float)

    panels = [("Census 2022 silage share", col(ct, "census_share"), col(tct, "census_share")),
              ("BARC-only: share of blocks called silage", col(ct, "barc_called"), col(tct, "barc_called")),
              (f"Proportion-trained (w {D.w_txt}), out of fold", col(ct, "llp_oof_called"), col(tct, "llp_called"))]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    for ax, (title, vn, vt) in zip(axes, panels):
        if proj:
            states.plot(ax=ax, facecolor="#f3f3f0", edgecolor="#a0a0a0", linewidth=0.5, zorder=1)
            if counties is not None and D.pop is not None:
                el = counties[counties["GEOID"].astype(str).isin(set(D.pop.fips))]
                el.plot(ax=ax, facecolor="#dcdcd4", edgecolor="none", zorder=1.5)
            x0, y0, x1, y1 = states.total_bounds
            ax.set_xlim(x0 - 5e4, x1 + 5e4)
            ax.set_ylim(y0 - 5e4, y1 + 5e4)
            ax.set_aspect("equal")
            ax.set_axis_off()
        else:
            ax.set_xlim(-125, -66)
            ax.set_ylim(24, 50)
            ax.set_aspect(1 / math.cos(math.radians(37)))
            ax.grid(color=GRID, lw=0.6)
            ax.set_xlabel("longitude")
            ax.set_ylabel("latitude")
        a = nat.join(vn.rename("v"), how="inner").dropna(subset=["v"])
        a = a.sort_values("v")
        ax.scatter(a.x, a.y, c=a.v, cmap=SHARE_CMAP, norm=SHARE_NORM, s=30, edgecolor="#333333", linewidth=0.45,
                   zorder=3)
        b = tst.join(vt.rename("v"), how="inner").dropna(subset=["v"])
        ax.scatter(b.x, b.y, c=b.v, cmap=SHARE_CMAP, norm=SHARE_NORM, s=58, marker="D", edgecolor="#c2185b",
                   linewidth=1.6, zorder=4)
        ax.set_title(title, color=INK)
    sm = plt.cm.ScalarMappable(cmap=SHARE_CMAP, norm=SHARE_NORM)
    cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.045, pad=0.03 if proj else 0.16, shrink=0.35,
                      aspect=40)
    cb.set_label("silage share of harvested corn (Census) or share of a county's blocks called silage (models)")
    cb.ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    hand = [Line2D([], [], marker="o", ls="", mfc="#9ccf8a", mec="#333333", ms=7, label="sampled county (one 10 km square)"),
            Line2D([], [], marker="D", ls="", mfc="#9ccf8a", mec="#c2185b", mew=1.6, ms=7,
                   label="test county (never trained on; final model)")]
    if proj and counties is not None and D.pop is not None:
        hand.append(Patch(facecolor="#dcdcd4", label=f"eligible counties ({len(D.pop):,})"))
    axes[1].legend(handles=hand, loc="lower center", ncol=len(hand), frameon=False, bbox_to_anchor=(0.5, 1.08),
                   fontsize=9.5)
    body = fig_img(fig, "US maps: Census silage share and the two models' predicted county shares")
    body += note("Same color scale in all three maps. Each dot is one sampled county, placed at its 10 km square. "
                 "Sampled counties show the out-of-fold prediction (the county's state group was held out). Test "
                 "counties (pink-edged diamonds) are never trained on, so they show the final proportion-trained model.")
    if map_note:
        body += note(map_note)
    if lost:
        body += note(f"{len(lost)} county(ies) had no coordinates and are not drawn: {', '.join(lost[:20])}.")
    return body


def section_scatter(D):
    ct, tct = D.ct, D.tct
    if ct is None:
        return note("county_predictions.csv is missing.")
    specs = [("BARC-only", C_BARC, "barc_called", "barc_mean_p", "barc_called", "barc_mean_p"),
             (f"Proportion-trained (w {D.w_txt}), out of fold", C_LLP, "llp_oof_called", "llp_oof_mean_p",
              "llp_called", "llp_mean_p")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True)
    for ax, (title, color, cn, mn, tc, tm) in zip(axes, specs):
        ax.plot([0, 1], [0, 1], ls="--", color="#888888", lw=1, zorder=1, label="y = x")
        for c in CLASS_CUTS:
            ax.axhline(c, color=GRID, lw=0.8, zorder=0)
            ax.axvline(c, color=GRID, lw=0.8, zorder=0)
        if cn not in ct.columns:
            ax.set_title(f"{title}: no column {cn}")
            continue
        d = ct[["census_share", cn]].dropna()
        rho = spearman(d.census_share, d[cn])
        gap = float((d[cn] - d.census_share).abs().mean()) if len(d) else float("nan")
        ax.scatter(d.census_share, d[cn], s=26, color=color, alpha=0.8, edgecolor="white", linewidth=0.5, zorder=3,
                   label=f"sampled counties (n {len(d)}), Spearman {num(rho, 2)}")
        if tct is not None and tc in tct.columns:
            t = tct[["census_share", tc]].dropna()
            rt = spearman(t.census_share, t[tc])
            ax.scatter(t.census_share, t[tc], s=60, marker="D", color=color, edgecolor="#c2185b", linewidth=1.6,
                       zorder=4, label=f"test counties (n {len(t)}), Spearman {num(rt, 2)}")
        ax.set_title(f"{title}\nmean |predicted - Census| {num(gap)}", color=INK)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_xlabel("Census 2022 silage share of corn")
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    axes[0].set_ylabel("predicted share (blocks with p ≥ 0.5)")
    body = fig_img(fig, "Predicted vs Census county silage share for both models")
    body += note("Spearman here is between the plotted share of blocks called silage and the Census share. The weight "
                 "table below uses the mean probability instead, so its Spearman can differ slightly. Light lines mark "
                 "the class edges at 5% and 40%. Test counties use the final proportion-trained model.")
    if tct is not None and len(tct):
        cols = {"name": "test county", "census_share": "Census", "barc_called": "BARC-only",
                "llp_called": f"proportion-trained (w {D.w_txt})"}
        t = tct[[c for c in cols if c in tct.columns]].rename(columns=cols).copy()
        for c in t.columns[1:]:
            t[c] = t[c].map(lambda v: pct(v))
        body += "<h3>Test counties: share of corn that is silage</h3>" + table_html(t.reset_index(drop=True))
    return body


def heatmap(ax, m, rows, cols, title, cmap="Purples"):
    """Counts annotated; shading is the share of the row (so rows of different size compare)."""
    m = np.asarray(m)
    rs = m.sum(1)
    frac = np.divide(m, rs[:, None], out=np.zeros(m.shape, float), where=rs[:, None] > 0)
    ax.imshow(frac, cmap=cmap, vmin=0, vmax=1.15, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m[i, j]
            share = f"\n{100 * frac[i, j]:.0f}% of row" if rs[i] else ""
            ax.text(j, i, f"{v:,}{share}", ha="center", va="center", fontsize=10,
                    color="white" if frac[i, j] > 0.6 else INK, fontweight="bold" if i == j else "normal")
    for k in range(min(m.shape)):
        ax.add_patch(Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False, edgecolor="#222222", lw=1.4))
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, color=INK)


def section_county_cm(D):
    ct = D.ct
    if ct is None:
        return note("county_predictions.csv is missing.")
    specs = [("BARC-only", "barc_called"), (f"Proportion-trained (w {D.w_txt}), out of fold", "llp_oof_called")]
    specs = [s for s in specs if s[1] in ct.columns]
    if not specs:
        return note("No prediction columns found.")
    fig, axes = plt.subplots(1, len(specs), figsize=(5.6 * len(specs), 4.4))
    axes = np.atleast_1d(axes)
    for k, (ax, (title, c)) in enumerate(zip(axes, specs)):
        m = class_matrix(ct.census_share, ct[c])
        heatmap(ax, m, [f"Census\n{n}" for n in CLASS_NAMES], [f"called\n{n}" for n in CLASS_NAMES],
                f"{title}\nclass right {int(np.trace(m))}/{int(m.sum())}")
        if k:
            ax.tick_params(labelleft=False)
    fig.subplots_adjust(wspace=0.08)
    axes[0].set_ylabel("Census class (rows)")
    for ax in axes:
        ax.set_xlabel("predicted class (columns)")
    body = fig_img(fig, "County confusion matrices, Census class vs predicted class")
    body += note("Classes by silage share of harvested corn: under 5%, 5-40%, 40% and over. The predicted class uses the "
                 "share of the county's blocks with p ≥ 0.5. The diagonal (outlined) is right.")
    return body


def section_field_cm(D):
    if D.fields is None:
        return note("field_predictions.csv is missing, so the field-level tests are skipped.")
    sets = []
    for kind, label in (("barc", "Maryland (BARC)"), ("quebec", "Québec")):
        for col, mname in (("p_barc", "BARC-only"), ("p_llp", f"proportion-trained (w {D.w_txt})")):
            c = D.field_counts(kind, col)
            if c:
                yrs = "" if not c["years"] else (f" {c['years'][0]}" if len(c["years"]) == 1
                                                 else f" {min(c['years'])}-{max(c['years'])}")
                sets.append((f"{label}{yrs}, {mname}\n{c['correct']:,} of {c['n']:,} correct", c))
    if not sets:
        return note("No BARC or Québec rows in field_predictions.csv.")
    ncol = 2
    nrow = math.ceil(len(sets) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.5, 3.9 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, c) in zip(axes, sets):
        m = np.array([[c["tp"], c["fn"]], [c["fp"], c["tn"]]])
        heatmap(ax, m, ["true silage", "true grain"], ["called silage", "called grain"], title)
    for ax in axes[len(sets):]:
        ax.set_axis_off()
    fig.tight_layout()
    body = fig_img(fig, "Field-level confusion matrices for Maryland and Québec, both models")
    body += note("A field-year is called silage when its probability is ≥ 0.5. Maryland: BARC field-years held out "
                 "from training. Québec: labeled fields from the Québec crop insurance data; no Québec label is used "
                 "in training, by either model.")
    return body


def section_choice(D):
    ch = D.run.get("choice") or []
    if not ch:
        return note("run.json has no weight-choice table.")
    df = pd.DataFrame(ch)
    if "w" in df.columns:
        df["w"] = df.w.map(lambda v: f"{float(v):g}")
    for c in df.columns:
        if c != "w" and pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: num(v))
    hl = [k for k, r in enumerate(ch) if D.w is not None and abs(float(r.get("w", -1)) - float(D.w)) < 1e-9]
    folds = D.setting.get("folds", "several")
    grid = ", ".join(f"{float(w):g}" for w in D.setting.get("w_grid", [])) or "a grid"
    guard = D.setting.get("guard_fields")
    rule = ("gives the smallest mean absolute gap between the share of blocks called silage and the Census share"
            if guard is None else
            f"gives the smallest mean absolute gap between the share of blocks called silage and the Census share "
            f"among the w that lose at most {guard} of the BARC 2017-2020 field calls against w 0, when each training "
            f"year is predicted by a model trained on the other three (the scored evaluation data are BARC fields)")
    body = (f"<p>{esc(f'The weight w of the county-share loss was chosen out of fold. Counties were split into {folds} groups by state; for each w in {{{grid}}}, every group was predicted by models trained without its counties. The chosen w (w {D.w_txt}, highlighted) {rule}. w 0 is BARC-only with BARC latitude/longitude on every pixel.')}</p>")
    body += table_html(df, hl_rows=hl)
    body += note("“gap called” uses the share of blocks with p ≥ 0.5; “gap mean p” uses the mean probability. "
                 "Spearman is between mean probability and Census share, with its p-value.")
    return body


GUARD_COL = "BARC 2017-2020 held-out year, correct"
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def variant_summary(d):
    """One run's headline numbers, from its run.json, field_predictions.csv and test_county_predictions.csv."""
    d = Path(d)
    run = json.loads((d / "run.json").read_text(encoding="utf-8"))
    s = run.get("setting", {})
    ch = pd.DataFrame(run.get("choice") or [])
    w = float(s["w_chosen"])
    row = lambda v: ch[(ch.w.astype(float) - v).abs() < 1e-9].iloc[0]  # noqa: E731
    r0, rw = row(0.0), row(w)
    fp = pd.read_csv(d / "field_predictions.csv")

    def right(kind, col):
        f = fp[fp.kind == kind]
        return f"{int(((f[col] >= 0.5) == (f.label == 'Silage')).sum()):,}/{len(f):,}"
    tc = d / "test_county_predictions.csv"
    tgap = "–"
    if tc.exists():
        t = pd.read_csv(tc)
        tgap = " → ".join(f"{(t[c + '_called'] - t.census_share).abs().mean():.3f}" for c in ("barc", "llp"))
    return {"run": d.name, "gap": float(rw["gap called (mean abs)"]), "cols": {
        "inputs": f"Jan-{MONTHS[int(s.get('last_month', 12)) - 1]}, {int(s.get('epochs', 0))} epochs",
        "starting weights": Path(run.get("args", {}).get("presto_dir", "?")).name,
        "w tried / chosen": f"{', '.join(f'{float(v):g}' for v in s.get('w_grid', []))} / {w:g}",
        "county gap, out of fold (w 0 → chosen)": f"{float(r0['gap called (mean abs)']):.3f} → {float(rw['gap called (mean abs)']):.3f}",
        "county class right, out of fold (w 0 → chosen)": f"{r0['class right']} → {rw['class right']}",
        "BARC 2017-2020 held-out-year calls, of 81 (w 0 → chosen)":
            f"{int(r0[GUARD_COL])} → {int(rw[GUARD_COL])}" if GUARD_COL in ch.columns else "–",
        "Maryland 2021-2023 correct (BARC-only → county-share)": f"{right('barc', 'p_barc')} → {right('barc', 'p_llp')}",
        "Québec correct (BARC-only → county-share)": f"{right('quebec', 'p_barc')} → {right('quebec', 'p_llp')}",
        "7 test counties, mean gap (BARC-only → county-share)": tgap}}


def section_variants(a):
    runs = [variant_summary(d) for d in [a.results, *a.compare]]
    best = min(range(len(runs)), key=lambda k: runs[k]["gap"])
    df = pd.DataFrame({"": list(runs[0]["cols"])} | {r["run"]: list(r["cols"].values()) for r in runs})
    body = (f"<p>{esc('Each variant is the same procedure: the choice of w out of fold by state with the BARC guard, then a final model scored once. Only the inputs or the starting weights differ; every variant uses the same sampled counties and state groups.')}</p>")
    body += table_html(df)
    body += note(f"Rule fixed before the results: the headline model is the variant with the lowest out-of-fold county "
                 f"gap at its chosen w (every chosen w passed the BARC guard). By that rule: {runs[best]['run']}. "
                 "Each variant's BARC-only column is its own refit (same months and starting weights).")
    return body


def section_census_prior(path):
    """The first table and the paired comparison of census_prior.md (census_prior.py) for this run."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    rows = [ln for ln in lines if ln.startswith("|")]
    head = [c.strip() for c in rows[0].strip("|").split("|")]
    end = next((k for k in range(2, len(rows)) if [c.strip() for c in rows[k].strip("|").split("|")] == head), len(rows))
    df = pd.DataFrame([[c.strip() for c in r.strip("|").split("|")] for r in rows[2:end]], columns=head)
    intro = next((ln for ln in lines if "counties scored" in ln), "")
    paired = next((ln for ln in lines if ln.startswith("Last Census + satellite vs last Census alone")), "")
    body = (f"<p>{esc('The question a USDA reader asks first: does the satellite estimate beat simply reusing the last Census (2017) for each county? All estimates are scored against the 2022 Census-only county share; the fitted rows are fitted on the other state groups only.')}</p>")
    body += table_html(df)
    body += note(" ".join(s for s in (intro, paired) if s))
    return body


def section_states(D):
    st = D.states_check
    if st is None:
        return note("state_check.csv is not in this results folder (older run), so the state table is skipped.")
    st = st.copy()
    for c in st.columns:
        if c in ("state", "counties"):
            continue
        st[c] = st[c].map(lambda v: "–" if pd.isna(v) else num(v))
    body = note("Caveat: sampled counties are drawn by share band, not in proportion to corn acres, so a state's "
                "sampled share is not an estimate of the state; the state's own 2022 shares (survey and Census) are "
                "context. The last row is the US: Census over all eligible counties, and the band-stratified estimate "
                "for the models.")
    body += table_html(st, bold_last=True)
    body += note("Shares are silage share of harvested corn acres; the sampled columns are corn-acre weighted over "
                 "the state's sampled counties.")
    return body


def draw_blocks(ax, geo, preds, pcol, title):
    d = geo.merge(preds[["field_id", pcol]].drop_duplicates("field_id"), on="field_id", how="inner").dropna(subset=[pcol])
    if not len(d):
        ax.set_axis_off()
        ax.set_title(f"{title}\n(no blocks)")
        return
    p = d[pcol].to_numpy(float)
    polys = None
    if shp_wkt is not None and "wkt" in d.columns:
        try:
            polys = []
            for w in d.wkt:
                g = shp_wkt.loads(w)
                if g.geom_type != "Polygon":
                    g = max(getattr(g, "geoms", [g]), key=lambda q: q.area) if hasattr(g, "geoms") else g.envelope
                polys.append(np.asarray(g.exterior.coords)[:, :2])
        except Exception as e:
            log("wkt unreadable, using centroids:", e)
            polys = None
    if polys:
        cxy = np.array([q.mean(0) for q in polys])  # a 90 m block is under 1% of the square: drawn enlarged
        ax.scatter(cxy[:, 0], cxy[:, 1], c=p, cmap=P_CMAP, norm=SHARE_NORM, marker="s", s=34, edgecolor="#444444",
                   linewidth=0.3, zorder=3)
        xy = np.concatenate(polys)
        (x0, y0), (x1, y1) = xy.min(0), xy.max(0)
        pad = 0.04 * max(x1 - x0, y1 - y0, 1)
        ax.set_xlim(x0 - pad, x1 + pad)
        ax.set_ylim(y0 - pad, y1 + pad)
        ax.set_aspect("equal")
        L = 1000.0                                  # 1 km scale bar (UTM metres)
        ax.plot([x0, x0 + L], [y0 - pad * 0.6] * 2, color=INK, lw=2)
        ax.text(x0 + L / 2, y0 - pad * 0.45, "1 km", ha="center", va="bottom", fontsize=8, color=INK)
    else:
        lon, lat = pd.to_numeric(d.lon, errors="coerce"), pd.to_numeric(d.lat, errors="coerce")
        ax.scatter(lon, lat, c=p, cmap=P_CMAP, norm=SHARE_NORM, marker="s", s=14, edgecolor="none")
        ax.set_aspect(1 / max(math.cos(math.radians(float(lat.mean()))), 0.2))
    ax.set_facecolor("#e4e4e4")
    ax.set_xticks([])
    ax.set_yticks([])
    called = float((p >= 0.5).mean())
    ax.set_title(f"{title}: {pct(called)} of {len(d)} blocks called silage", color=INK)


def section_zoom(D):
    if D.blocks is None:
        return note("block_predictions.csv is missing, so the zoom-in is skipped.")
    bp = D.blocks
    cands = []
    if D.ct is not None and {"barc_called", "llp_oof_called", "census_share"} <= set(D.ct.columns):
        c = D.ct
        imp = (c.barc_called - c.census_share).abs() - (c.llp_oof_called - c.census_share).abs()
        have = set(bp.loc[bp.kind == "national", "fips"].dropna())
        for f, v in imp.dropna().sort_values(ascending=False).items():
            if f in have:
                name = f"{c.loc[f, 'county']}, {c.loc[f, 'state']}" if {"county", "state"} <= set(c.columns) else f
                cands.append({"fips": f, "test": False, "imp": float(v), "name": f"{name} (sampled county)",
                              "g0": abs(float(c.loc[f, "barc_called"] - c.loc[f, "census_share"])),
                              "g1": abs(float(c.loc[f, "llp_oof_called"] - c.loc[f, "census_share"])),
                              "census": float(c.loc[f, "census_share"]), "pcol": "p_llp_oof",
                              "mlabel": f"Proportion-trained (w {D.w_txt}), out of fold"})
    tc = []
    if D.tct is not None and {"barc_called", "llp_called", "census_share"} <= set(D.tct.columns):
        t = D.tct
        imp = (t.barc_called - t.census_share).abs() - (t.llp_called - t.census_share).abs()
        have = set(bp.loc[bp.kind == "test_county", "fips"].dropna())
        for f, v in imp.dropna().sort_values(ascending=False).items():
            if f in have and D.county_blocks(f, test=True)[0] is not None:
                tc.append({"fips": f, "test": True, "imp": float(v), "name": f"{t.loc[f, 'name']} (test county)",
                           "g0": abs(float(t.loc[f, "barc_called"] - t.loc[f, "census_share"])),
                           "g1": abs(float(t.loc[f, "llp_called"] - t.loc[f, "census_share"])),
                           "census": float(t.loc[f, "census_share"]), "pcol": "p_llp",
                           "mlabel": f"Proportion-trained (w {D.w_txt}), final model"})
    nat = [c for c in cands if D.county_blocks(c["fips"], test=False)[0] is not None]
    pick = (tc[:1] + nat[:1]) if tc else nat[:2]
    if len(pick) < 2:
        pick = (pick + [c for c in nat + tc if c not in pick])[:2]
    if not pick:
        return note("No county has both block predictions and block coordinates, so the zoom-in is skipped.")
    fig = plt.figure(figsize=(12, 5.7 * len(pick)))
    subs = np.atleast_1d(fig.subfigures(len(pick), 1, hspace=0.06))
    for sf, c in zip(subs, pick):
        geo, _ = D.county_blocks(c["fips"], c["test"])
        geo = geo.assign(field_id=geo.field_id.astype(str))
        pr = bp[(bp.fips == c["fips"]) & (bp.kind == ("test_county" if c["test"] else "national"))].copy()
        pr["field_id"] = pr.field_id.astype(str)
        axes = sf.subplots(1, 2)
        sf.subplots_adjust(top=0.84, bottom=0.04, wspace=0.08)
        draw_blocks(axes[0], geo, pr, "p_barc", "BARC-only")
        draw_blocks(axes[1], geo, pr, c["pcol"], c["mlabel"])
        sf.suptitle(f"{c['name']}: Census 2022 silage share {pct(c['census'])}\n"
                    f"off the Census share by: BARC-only {pts(c['g0'])}, proportion-trained {pts(c['g1'])}",
                    fontsize=12, fontweight="bold", color=INK)
    sm = plt.cm.ScalarMappable(cmap=P_CMAP, norm=SHARE_NORM)
    last = subs[-1]
    cb = last.colorbar(sm, ax=list(last.axes), orientation="horizontal", fraction=0.05, pad=0.06, shrink=0.6, aspect=40)
    cb.set_label("block probability of silage (≥ 0.5 is called silage)")
    body = fig_img(fig, "Zoom-in: one county's 90 m corn blocks colored by each model's silage probability")
    body += note("Each square is one all-corn 90 m block from the 2022 Cropland Data Layer inside the county's 10 km "
                 "square, drawn larger than scale. Counties are the ones where the proportion-trained model's share moves most toward the "
                 "Census share (BARC-only gap minus proportion-trained gap); a test county is preferred when one is "
                 "available. The Census share is for the whole county, not this square, and blocks have no labels: "
                 "the picture shows how calls shift, not which block is right.")
    return body


def section_method(D):
    s = D.setting
    chk = D.run.get("checks") or {}
    items = [
        "How the proportion-trained model learns: the same fine-tuning on BARC field labels, plus a loss that makes "
        "the mean predicted silage probability of each sampled county's corn blocks match that county's 2022 Census "
        "silage share (learning from label proportions). Latitude/longitude of every non-BARC pixel is set to BARC's, "
        "so location cannot stand in for the share.",
        "County totals cannot show which blocks are wrong. A county share can be right with the wrong blocks called "
        "silage; the field-level tests (Maryland, Québec) are the check on individual calls.",
        "One 10 km square of 100 random all-corn blocks stands in for each county; the square may not match the "
        "county's mix of silage and grain.",
        "The corn map (2022 Cropland Data Layer) has errors, so some blocks may not be corn, or not one field.",
        "2022 was a drought year in parts of the US; drought changes how much corn is chopped for silage and how crops "
        "look from space, so results may not carry over to other years.",
        "The out-of-fold numbers were also used to choose w from the grid, so they are slightly optimistic. The test "
        "counties and the field-level tests were not used for any choice. In-sample predictions of the final model "
        "(llp_in_sample columns) are not a test and are not shown.",
    ]
    body = "<ul>" + "".join(f"<li>{esc(t)}</li>" for t in items) + "</ul>"
    facts = []
    if s:
        facts.append(f"epochs {s.get('epochs', '?')}, seeds {s.get('seeds', '?')}, folds {s.get('folds', '?')}, "
                     f"w grid {s.get('w_grid', '?')}, chosen w {D.w_txt}")
    if chk:
        facts.append("reproduction check of the BARC-only refit against the official runs: same calls on BARC "
                     f"{chk.get('barc_same_calls', 'n/a')}" + (f", Québec {chk['quebec_same_calls']}"
                                                                 if "quebec_same_calls" in chk else ""))
    if D.run.get("finished_utc"):
        facts.append(f"run finished {D.run['finished_utc']} UTC")
    miss = D.run.get("counties_missing_pixels") or []
    if miss:
        facts.append(f"{len(miss)} selected counties left out for missing pixels")
    if facts:
        body += note("Run: " + "; ".join(facts) + ".")
    return body


# ---------------------------------------------------------------- page

CSS = """
:root{--ink:#1d1d1f;--muted:#5f6368;--line:#dadce0;--bg:#ffffff;--card:#f5f6f8;--accent:#2f6db5;color-scheme:light}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif}
main{max-width:1200px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:27px;line-height:1.25;margin:0 0 6px}
h2{font-size:20px;margin:40px 0 10px;padding-top:22px;border-top:1px solid var(--line)}
h3{font-size:16px;margin:22px 0 6px}
p{max-width:78ch}
img{max-width:100%;height:auto;display:block;margin:10px 0}
nav{font-size:13.5px;margin:8px 0 4px;display:flex;flex-wrap:wrap;gap:4px 14px}
a{color:var(--accent)}
.sub{color:var(--muted);margin:0 0 10px}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px;margin:16px 0}
.tile{background:var(--card);border-radius:8px;padding:10px 12px}
.tile .k{font-size:12.5px;color:var(--muted)}
.tile .v{font-size:21px;font-weight:650;font-variant-numeric:tabular-nums;margin:2px 0}
.tile .s{font-size:12px;color:var(--muted)}
.tablewrap{overflow-x:auto;max-width:100%}
table{border-collapse:collapse;font-size:13.5px;font-variant-numeric:tabular-nums;margin:8px 0}
th,td{padding:5px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{background:var(--card);font-weight:600;white-space:normal;vertical-align:bottom}
tr.hl td{font-weight:700;background:#fff4df}
.note{color:var(--muted);font-size:13.5px}
.smoke{background:#c62828;color:#fff;font-size:30px;font-weight:800;text-align:center;padding:20px;border-radius:6px;margin:14px 0;letter-spacing:.02em}
.err{background:#fdecea;border-left:4px solid #c62828;padding:8px 12px;font-size:13.5px}
details{margin-top:12px}
details pre{white-space:pre-wrap;font-size:12px;background:var(--card);padding:12px;border-radius:6px;overflow-x:auto}
footer{margin-top:40px;color:var(--muted);font-size:12.5px}
"""


def build(a):
    out = Path(a.out)
    D = Data(a)
    states = counties = None
    map_note = None
    try:
        states, counties, map_note = load_boundaries(a, out)
    except Exception as e:
        map_note = f"Basemap failed ({e}); plain longitude/latitude axes."
        log(map_note)
    secs = [("headline", "Headline", lambda: section_headline(D)),
            ("map", "Where the counties are, and what each model predicts", lambda: section_map(D, states, counties, map_note)),
            ("scatter", "Predicted vs Census county share", lambda: section_scatter(D)),
            ("county-cm", "County classes: Census vs predicted", lambda: section_county_cm(D)),
            ("field-cm", "Field-level tests: Maryland and Québec", lambda: section_field_cm(D)),
            ("weight", "How the loss weight w was chosen", lambda: section_choice(D)),
            *([("variants", "Variants tried: months and starting weights", lambda: section_variants(a))]
              if a.compare else []),
            *([("census", "Against the last Census", lambda: section_census_prior(cp))]
              if (cp := Path(a.results) / "census_prior" / "census_prior.md").exists() else []),
            ("states", "States: sampled counties next to state totals", lambda: section_states(D)),
            ("zoom", "Zoom-in: the blocks behind a county share", lambda: section_zoom(D)),
            ("method", "Method and limits", lambda: section_method(D))]
    parts, skipped = [], []
    for sid, title, fn in secs:
        log("section", sid)
        try:
            body = fn()
        except Exception as e:
            traceback.print_exc()
            skipped.append(sid)
            body = f'<p class="err">This section could not be drawn from the files ({esc(type(e).__name__)}: {esc(e)}).</p>'
        parts.append(f'<section id="{sid}"><h2>{esc(title)}</h2>{body}</section>')
    smoke = bool(D.run.get("smoke"))
    banner = '<div class="smoke">SMOKE RUN — not a result</div>' if smoke else ""
    n = D.run.get("counties_used", len(D.ct) if D.ct is not None else "?")
    sub = (f"Hack the Harvest 2026 · Presto fine-tuned on BARC, compared with a county-share (label proportion) "
           f"trained version · {n} sampled US counties, 2022")
    nav = "<nav>" + "".join(f'<a href="#{sid}">{esc(t.split(":")[0])}</a>' for sid, t, _ in secs) + "</nav>"
    md = ""
    if D.metrics_md:
        md = (f'<section id="metrics"><h2>Full metrics report</h2><details><summary>metrics.md from the run '
              f'(click to open)</summary><pre>{esc(D.metrics_md)}</pre></details></section>')
    foot = (f"<footer>Generated by national_page.py from {esc(Path(a.results).name)}"
            + (f"; run finished {esc(D.run.get('finished_utc'))} UTC" if D.run.get("finished_utc") else "")
            + ". All numbers are read or computed from the run's files.</footer>")
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>Silage share across the US</title><style>{CSS}</style></head><body><main>"
            f"{banner}<h1>Silage or grain corn: county shares across the US</h1><p class=\"sub\">{esc(sub)}</p>{nav}"
            + "".join(parts) + md + banner + foot + "</main></body></html>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    log(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {page.count('<img')} figures"
        + (f"; sections with errors: {', '.join(skipped)}" if skipped else "") + ")")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="results folder of presto_national.py")
    ap.add_argument("--national", required=True, help="folder with selection.csv, population.csv, fy/<fips>.csv")
    ap.add_argument("--test-counties", default=None, help="folder with county_squares.csv")
    ap.add_argument("--out", required=True, help="output HTML path")
    ap.add_argument("--boundaries-cache", default=None, help="where to keep the Census boundary files "
                                                              "(default: boundaries/ next to --out)")
    ap.add_argument("--states-url", default=STATES_URL)
    ap.add_argument("--counties-url", default=COUNTIES_URL)
    ap.add_argument("--no-county-polygons", action="store_true", help="skip the county file (eligible-county shading)")
    ap.add_argument("--compare", nargs="*", default=[], help="other results folders of the same counties (variants table)")
    a = ap.parse_args(argv)
    build(a)


if __name__ == "__main__":
    main()
