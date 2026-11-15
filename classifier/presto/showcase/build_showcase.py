"""Pictures of fine-tuned Presto's pixel-by-pixel calls on the 44 test field-years (2021-2023), from the output of
presto_pixel_map.py.

  python classifier/presto/showcase/build_showcase.py --data classifier/presto/showcase/results \
      --field-years classifier/labels/results/field_years_fixed.csv --out classifier/presto/showcase

Writes presto_pixels.html (interactive, one self-contained file from template.html), season_strip.png and
pixel_map.png. True color is Sentinel-2 red, green and blue (B4, B3, B2) as Presto received them: the monthly median
of clear readings.
"""
import argparse
import base64
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from shapely import wkt as shapely_wkt

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
GRAIN, MID, SILAGE = "#2a78d6", "#c3c2b7", "#e34948"
SCORE = LinearSegmentedColormap.from_list("score", [GRAIN, MID, SILAGE])
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def true_color(v):
    """Display color for reflectance (..., 3) in B4, B3, B2 order."""
    return np.clip(v * 3.2, 0, 1) ** 0.8


def load(data, field_years):
    z = np.load(Path(data) / "pixel_map.npz")
    bands = list(z["bands"])
    off = z["offsets"]
    n = np.diff(off)
    vals = z["values"].astype(np.float32)
    p = z["p"].astype(np.float32).mean(1)
    owner = np.repeat(np.arange(len(n)), n)
    fy = pd.read_csv(field_years)
    f = pd.DataFrame({"field_id": z["field_id"], "year": z["year"]}).merge(
        fy[["field_id", "year", "label", "farm", "field", "wkt"]], on=["field_id", "year"], how="left", validate="1:1")
    f["n_pix"] = n
    f["p"] = np.bincount(owner, p) / n
    f["silage_share"] = np.bincount(owner, p >= 0.5) / n
    f["called"] = np.where(f.p >= 0.5, "Silage", "Grain")
    f["right"] = f.called == f.label
    f["name"] = f.farm + " " + f.field.astype(str)
    px = {"off": off, "x": z["x"].astype(np.float64), "y": z["y"].astype(np.float64), "p": p,
          "rgb": vals[:, :, [bands.index("B4"), bands.index("B3"), bands.index("B2")]],
          "ndvi": vals[:, :, bands.index("NDVI")]}
    return f, px, z["month_used"][:, :, 0]


def raster(xs, ys, values, pad=1):
    """Values (n, k) on their 10 m grid: image (H, W, k), NaN off the field, and its imshow extent."""
    c = np.rint((xs - xs.min()) / 10).astype(int) + pad
    r = np.rint((ys.max() - ys) / 10).astype(int) + pad
    img = np.full((r.max() + 1 + pad, c.max() + 1 + pad, values.shape[1]), np.nan)
    img[r, c] = values
    e = 5 + 10 * pad
    return img, (xs.min() - e, xs.max() + e, ys.min() - e, ys.max() + e)


def outline(ax, wkt, **kw):
    g = shapely_wkt.loads(wkt)
    for poly in getattr(g, "geoms", [g]):
        ax.plot(*poly.exterior.xy, **kw)


def class_ndvi(f, px, used):
    """Median over field-years of each field-year's median NDVI, per label and month (months with a clear view)."""
    med = np.full((len(f), 12), np.nan)
    for i in range(len(f)):
        sl = slice(px["off"][i], px["off"][i + 1])
        med[i, used[i]] = np.median(px["ndvi"][sl][:, used[i]], 0)
    return {lab: np.nanmedian(med[(f.label == lab).to_numpy()], 0) for lab in ("Silage", "Grain")}, med


def pick_pair(f, used, only_right=True, pair=None):
    """A silage and a grain field-year Presto called right, same farm and year, most clear months June-December;
    or the two given as FIELD_ID:YEAR."""
    if pair:
        key = list(zip(f.field_id, f.year))
        return tuple(key.index(tuple(int(v) for v in p.split(":"))) for p in pair)
    g = f.assign(late=used[:, 5:].sum(1))
    g = g[g.right] if only_right else g
    best = None
    for _, d in g.groupby(["year", "farm"]):
        s = d[d.label == "Silage"].sort_values(["late", "p"], ascending=[False, False])
        r = d[d.label == "Grain"].sort_values(["late", "p"], ascending=[False, True])
        if len(s) and len(r) and (best is None or s.late.iloc[0] + r.late.iloc[0] > best[0]):
            best = (s.late.iloc[0] + r.late.iloc[0], s.index[0], r.index[0])
    if best is None:
        return pick_pair(f, used, False)
    return best[1], best[2]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def season_strip(f, px, used, out, pair=None):
    pair = pick_pair(f, used, pair=pair)
    cls, med = class_ndvi(f, px, used)
    cols = list(range(5, 12))
    fig = plt.figure(figsize=(13, 6.0), dpi=150)
    gs = fig.add_gridspec(3, len(cols) + 1, height_ratios=[1, 1, 1.25], hspace=0.3, wspace=0.08)
    for row, i in enumerate(pair):
        sl = slice(px["off"][i], px["off"][i + 1])
        xs, ys = px["x"][sl], px["y"][sl]
        r = f.loc[i]
        for k, m in enumerate(cols + [None]):
            ax = fig.add_subplot(gs[row, k])
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if m is None:
                img, ext = raster(xs, ys, px["p"][sl, None])
                ax.imshow(SCORE(img[..., 0]), extent=ext, interpolation="nearest")
                title = "Presto score"
            elif used[i, m]:
                img, ext = raster(xs, ys, true_color(px["rgb"][sl, m]))
                ax.imshow(np.nan_to_num(img, nan=1.0), extent=ext, interpolation="nearest")
                title = MONTHS[m]
            else:
                img, ext = raster(xs, ys, np.ones((len(xs), 3)))
                ax.imshow(np.where(np.isnan(img), 1.0, 0.86), extent=ext, interpolation="nearest")
                ax.text(0.5, 0.5, "no clear\nview", ha="center", va="center", fontsize=8, color=INK2,
                        transform=ax.transAxes)
                title = MONTHS[m]
            outline(ax, r.wkt, color=MUTED, lw=0.6)
            ax.set_xlim(ext[:2]); ax.set_ylim(ext[2:])
            if row == 0:
                ax.set_title(title, fontsize=10, color=INK2)
            if k == 0:
                ax.set_ylabel(f"{r.label} (farm record)\n{r['name']}, {r.year}\nPresto score {r.p:.2f}", fontsize=10,
                              color=INK, rotation=0, ha="right", va="center", labelpad=8)
    ax = fig.add_subplot(gs[2, 1:len(cols)])
    style(ax)
    for lab, color in (("Silage", SILAGE), ("Grain", GRAIN)):
        ax.plot(range(12), cls[lab], color=color, lw=1.2, ls=(0, (4, 3)), label=f"median {lab.lower()} field, 2021-23")
    for i, color in zip(pair, (SILAGE, GRAIN)):
        ax.plot(range(12), med[i], color=color, lw=2, marker="o", ms=4, label=f"{f.loc[i, 'name']} {f.loc[i, 'year']}")
    ax.set_xticks(range(12), MONTHS)
    ax.set_ylim(-0.05, 1)
    ax.set_ylabel("NDVI (field median)", fontsize=9, color=INK2)
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle("What Presto sees: each 10 m pixel's color, month by month", fontsize=13, color=INK, x=0.5, y=0.995)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [f.loc[i, "name"] + f" {f.loc[i, 'year']}" for i in pair]


def pixel_map(f, px, out):
    years = sorted(f.year.unique())
    fig, axes = plt.subplots(len(years), 1, figsize=(13, 4.2 * len(years)), dpi=150)
    for ax, yr in zip(axes, years):
        idx = np.flatnonzero(f.year == yr)
        sl = np.concatenate([np.arange(px["off"][i], px["off"][i + 1]) for i in idx])
        img, ext = raster(px["x"][sl], px["y"][sl], px["p"][sl, None], pad=4)
        ax.imshow(SCORE(img[..., 0]), extent=ext, interpolation="nearest")
        for i in idx:
            r = f.loc[i]
            outline(ax, r.wkt, color=INK if not r.right else MUTED, lw=1.4 if not r.right else 0.5)
            if not r.right:
                c = shapely_wkt.loads(r.wkt).centroid
                ax.annotate(f"{r['name']}: {r.label.lower()}, called {r.called.lower()}", (c.x, c.y),
                            xytext=(0, 14), textcoords="offset points", ha="center", fontsize=8, color=INK,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        right = int(f.right[idx].sum())
        ax.set_title(f"{yr}: {right} of {len(idx)} field-years called right", fontsize=11, color=INK, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlim(ext[:2]); ax.set_ylim(ext[2:])
    cax = fig.add_axes([0.35, 0.06, 0.3, 0.012])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=SCORE), cax=cax, orientation="horizontal")
    cb.set_ticks([0, 0.5, 1], labels=["0  grain", "0.5", "silage  1"])
    cb.ax.tick_params(labelsize=9, colors=INK2)
    cb.outline.set_visible(False)
    cb.set_label("Presto silage score of each 10 m pixel", fontsize=9, color=INK2)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def page_data(f, px, used):
    """Field list and one base64 blob: per pixel col, row (uint16), score, then 12 months of RGB and NDVI (uint8)."""
    cls, med = class_ndvi(f, px, used)
    years, fields, col, row = {}, [], [], []
    for yr in sorted(f.year.unique()):
        idx = np.flatnonzero(f.year == yr)
        sl = np.concatenate([np.arange(px["off"][i], px["off"][i + 1]) for i in idx])
        x0, y0 = px["x"][sl].min() - 5, px["y"][sl].max() + 5
        w = int(np.rint((px["x"][sl].max() + 5 - x0) / 10))
        h = int(np.rint((y0 - px["y"][sl].min() + 5) / 10))
        years[int(yr)] = {"w": w, "h": h, "right": int(f.right[idx].sum()), "n": len(idx)}
        for i in idx:
            r = f.loc[i]
            s = slice(px["off"][i], px["off"][i + 1])
            col.append(np.rint((px["x"][s] - x0 - 5) / 10)); row.append(np.rint((y0 - 5 - px["y"][s]) / 10))
            g = shapely_wkt.loads(r.wkt).simplify(2)
            rings = [[[round((a - x0) / 10, 2), round((y0 - b) / 10, 2)] for a, b in poly.exterior.coords]
                     for poly in getattr(g, "geoms", [g])]
            q = np.percentile(px["ndvi"][s], [25, 50, 75], axis=0)
            fields.append({"i": int(i), "year": int(yr), "name": r["name"], "label": r.label, "called": r.called,
                           "right": bool(r.right), "p": round(float(r.p), 2), "share": round(float(r.silage_share), 3),
                           "n": int(r.n_pix), "start": int(px["off"][i]), "used": used[i].astype(int).tolist(),
                           "ndvi": [[round(float(v), 3) if used[i, m] else None for v in q[:, m]] for m in range(12)],
                           "rings": rings})
    col, row = np.concatenate(col).astype("<u2"), np.concatenate(row).astype("<u2")
    score = np.rint(px["p"] * 255).astype(np.uint8)
    rgb = np.rint(true_color(px["rgb"]) * 255).astype(np.uint8).reshape(len(score), 36)
    ndvi = np.rint((np.clip(px["ndvi"], -1, 1) + 1) * 127.5).astype(np.uint8)
    blob = col.tobytes() + row.tobytes() + score.tobytes() + rgb.tobytes() + ndvi.tobytes()
    return {"years": years, "fields": fields, "n": int(len(score)), "blob": base64.b64encode(blob).decode(),
            "classNdvi": {k: [None if np.isnan(v) else round(float(v), 3) for v in a] for k, a in cls.items()}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder written by presto_pixel_map.py")
    ap.add_argument("--field-years", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pair", nargs=2, metavar="FIELD_ID:YEAR", help="silage and grain field-year for the strip")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f, px, used = load(a.data, a.field_years)
    pair = season_strip(f, px, used, out / "season_strip.png", a.pair)
    pixel_map(f, px, out / "pixel_map.png")
    html = (HERE / "template.html").read_text(encoding="utf-8")
    html = html.replace("/*__DATA__*/null", json.dumps(page_data(f, px, used), separators=(",", ":")))
    (out / "presto_pixels.html").write_text(html, encoding="utf-8")
    print(f"{int(f.right.sum())}/{len(f)} field-years right, {len(px['p'])} pixels; season strip: {' and '.join(pair)}")


if __name__ == "__main__":
    main()
