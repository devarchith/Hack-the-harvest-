"""Figures for the grain vs silage README. Run after grain_silage.py (same inputs, same defaults).

  python research/grain_vs_silage/make_figures.py [--s2-series P --landsat-series P]

Writes results/fig_ndvi_trajectories.png and results/fig_key_features.png.
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grain_silage as gs  # noqa: E402

# Reference palette, light mode (validated: CVD dE 24.7, normal-vision dE 33.6, both >= 3:1 on the surface)
COL = {"Grain": "#2a78d6", "Silage": "#eb6834"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
MONTHS = [(121, "May"), (152, "Jun"), (182, "Jul"), (213, "Aug"), (244, "Sep"), (274, "Oct"), (305, "Nov"),
          (335, "Dec")]


def daily_curves(series, labels):
    """Each field-year's NDVI interpolated to every 3rd day, Apr to early Dec."""
    grid = np.arange(100, 341, 3)
    rows = []
    lab = labels[labels.label.notna()]
    for (fid, yr), g in series.groupby(["field_id", "year"]):
        hit = lab[(lab.field_id == fid) & (lab.year == yr)]
        if hit.empty:
            continue
        g = g.sort_values("doy")
        v = np.interp(grid, g.doy.values, g.ndvi.values, left=np.nan, right=np.nan)
        rows.append(pd.DataFrame({"doy": grid, "ndvi": v, "label": hit.label.iloc[0], "key": f"{fid}-{yr}"}))
    return pd.concat(rows)


def panel(ax, curves, title):
    for cls in ("Grain", "Silage"):
        c = curves[curves.label == cls]
        q = c.groupby("doy").ndvi.quantile([0.25, 0.5, 0.75]).unstack()
        n = c.key.nunique()
        ax.fill_between(q.index, q[0.25], q[0.75], color=COL[cls], alpha=0.15, linewidth=0)
        ax.plot(q.index, q[0.5], color=COL[cls], linewidth=2, label=f"{cls} (n={n})")
        last = q[0.5].dropna()
        ax.annotate(cls, (last.index[-1], last.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    color=INK2, fontsize=9, va="center")
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xticks([d for d, _ in MONTHS], [m for _, m in MONTHS])
    ax.set_xlim(100, 352)
    ax.set_ylim(0, 0.95)
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel("NDVI (median, band = middle 50%)")
    ax.legend(frameon=False, loc="upper left", fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    data = gs.ROOT / "data"
    ap.add_argument("--s2-series", default=data / "series_barc.parquet")
    ap.add_argument("--landsat-series", default=data / "series_barc.parquet")
    a = ap.parse_args()
    s2, ls = gs.load_series(a.s2_series, a.landsat_series)
    ser = gs.merged_series(s2, ls, gs.sensor_offsets(s2, ls))
    lab = gs.load_labels()
    panels = [(daily_curves(ser, lab), "2021-2023 evaluation labels")]
    extra = gs.load_extra()
    if extra is not None:
        xs2, xls, _, xlab = extra
        xser = gs.merged_series(xs2, xls, gs.sensor_offsets(xs2, xls))
        panels.insert(0, (daily_curves(xser, xlab), "2017-2020 farm records (training years)"))

    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.2), sharey=True, squeeze=False)
    for ax, (cur, title) in zip(axes[0], panels):
        panel(ax, cur, title)
    fig.suptitle("Silage fields are cut green in late Aug to early Sep; grain fields dry down slowly",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "results" / "fig_ndvi_trajectories.png", dpi=150)

    # key features: one dot per labeled field-year
    feats = [("ndti_post_drop", "Residue index after the drop\n(lower = barer soil)"),
             ("drop_size", "Size of the largest NDVI drop"),
             ("ndvi_sep15", "NDVI on Sep 15"),
             ("ndvi_pre_drop", "NDVI just before the drop")]
    sets = [("2021-2023", pd.read_csv(HERE / "results" / "features_all_field_years.csv"))]
    p1720 = HERE / "results" / "features_2017_2020.csv"
    if p1720.exists():
        sets.insert(0, ("2017-2020", pd.read_csv(p1720)))
    fig, axes = plt.subplots(len(sets), len(feats), figsize=(3.1 * len(feats), 2.9 * len(sets)), squeeze=False)
    rng = np.random.default_rng(0)
    for r, (nm, df) in enumerate(sets):
        df = df[df.label.isin(["Grain", "Silage"])]
        for c, (col, title) in enumerate(feats):
            ax = axes[r][c]
            for i, cls in enumerate(("Grain", "Silage")):
                v = df.loc[df.label == cls, col].dropna().values
                ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=30, color=COL[cls],
                           edgecolor=SURFACE, linewidth=1, zorder=3)
                ax.hlines(np.median(v), i - 0.3, i + 0.3, color=INK, linewidth=1.5, zorder=4)
            ax.set_xticks([0, 1], [f"Grain\n(n={int((df.label == 'Grain').sum())})",
                                   f"Silage\n(n={int((df.label == 'Silage').sum())})"])
            ax.grid(axis="x", visible=False)
            ax.set_xlim(-0.6, 1.6)
            if r == 0:
                ax.set_title(title, fontsize=10, color=INK, loc="left")
            if c == 0:
                ax.set_ylabel(nm, fontsize=11, color=INK)
    fig.suptitle("The four strongest pre-chosen features, one dot per field-year (black bar = median)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "results" / "fig_key_features.png", dpi=150)
    print("wrote", HERE / "results" / "fig_ndvi_trajectories.png", "and fig_key_features.png")


if __name__ == "__main__":
    main()
