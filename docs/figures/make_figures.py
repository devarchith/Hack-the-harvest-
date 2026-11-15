"""Slide charts for the silage corn vs grain corn submission.

  uv run --no-project --with pandas --with matplotlib --with scikit-learn python -I docs/figures/make_figures.py
  (optional: --data-json <file> also writes every plotted value as JSON, for checking)

Reads only committed result files under classifier/counties/national/ and writes a PNG (200 dpi) and an SVG per
chart next to this script. No network. Definitions follow classifier/counties/national_metrics.py,
census_prior.py and presto_national.py:
  - a field or block is called silage when its predicted silage probability is >= 0.5;
  - a county's predicted share is the share of its blocks called silage (scatter chart, the three-runs chart);
    the two charts that set the models beside the last Census use the mean predicted probability instead, because
    that is the estimate census_prior.py compares with the last Census (columns maryland_only, satellite);
  - county classes by silage share of corn acres: under 5%, 5-40%, 40% and over (lower edge included);
  - county gap: mean absolute difference between the predicted share and the 2022 Census share.
Every count is checked against the run's committed metrics_usda.md / metrics.md / census_prior.md; a difference
is printed.

Names: "Maryland-only model" = columns p_barc / barc_*; "county-share model" = columns p_llp / llp_*.
"""
import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from matplotlib.transforms import ScaledTranslation  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NAT = REPO / "classifier" / "counties" / "national"
REL = "classifier/counties/national"
RUNS = {"results_final": "Jan-Dec imagery,\noriginal weights\n(headline)",
        "results_final_janoct": "Jan-Oct imagery,\noriginal weights",
        "results_final_barconly": "Jan-Dec imagery,\nadapted starting weights"}
HEAD = "results_final"

BLUE, GRAY, ORANGE, BROWN = "#1f6fb2", "#6f6f6f", "#d9822b", "#8a4b12"
BLUE_LIGHT = "#a9cbe8"
INK, INK2, INK3, GRID = "#1a1a1a", "#444444", "#6b6b6b", "#dddddd"
EDGES = [-1, 0.05, 0.40, 2]
CLASSES = ["under 5%", "5-40%", "40%+"]
FIGSIZE = (12.8, 7.2)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 14, "axes.edgecolor": INK3, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "svg.hashsalt": "figures"})

DATA, PROBLEMS = {}, []


# ---------------------------------------------------------------- data
def fields(run, kind):
    f = pd.read_csv(NAT / run / "field_predictions.csv")
    return f[f.kind == kind]


def cm(f, col):
    """2x2 counts, rows = true label, columns = call, order Grain then Silage."""
    y = (f.label == "Silage").to_numpy()
    s = f[col].to_numpy() >= 0.5
    return np.array([[(~y & ~s).sum(), (~y & s).sum()], [(y & ~s).sum(), (y & s).sum()]], dtype=int)


def share_class(v):
    return pd.cut(pd.Series(np.asarray(v, float)), EDGES, right=False, labels=False).to_numpy()


def class_matrix(truth, est):
    m = np.zeros((3, 3), dtype=int)
    for a, b in zip(share_class(truth), share_class(est)):
        m[int(a), int(b)] += 1
    return m


def county_stats(truth, est):
    t, e = pd.Series(np.asarray(truth, float)), pd.Series(np.asarray(est, float))
    return {"mae": float((e - t).abs().mean()), "bias": float((e - t).mean()),
            "spearman": float(t.corr(e, method="spearman")),
            "class_right": int((share_class(t) == share_class(e)).sum()), "n": int(len(t))}


def counties(run):
    return pd.read_csv(NAT / run / "county_predictions.csv", dtype={"fips": str}).set_index("fips")


def prior(run):
    return pd.read_csv(NAT / run / "census_prior" / "census_prior_counties.csv", dtype={"fips": str}).set_index("fips")


# ---------------------------------------------------------------- checks against the committed metrics files
def expect(what, got, want):
    if got != want:
        PROBLEMS.append({"what": what, "computed": got, "committed": want})
        print(f"DIFFERENCE {what}: computed {got}, committed file says {want}")


def check_run(run):
    """Compare this script's counts with metrics_usda.md (field matrices, county table) and metrics.md (3x3)."""
    usda = (NAT / run / "metrics_usda.md").read_text(encoding="utf-8")
    sil = [tuple(map(int, m)) for m in re.findall(r"^\| Silage \| (\d+) \| (\d+) \|$", usda, re.M)]
    gra = [tuple(map(int, m)) for m in re.findall(r"^\| Grain \| (\d+) \| (\d+) \|$", usda, re.M)]
    order = [("barc", "p_barc"), ("barc", "p_llp"), ("quebec", "p_barc"), ("quebec", "p_llp")]
    for (kind, col), (tp, fn), (fp, tn) in zip(order, sil, gra):
        expect(f"{run} {kind} {col} matrix [[TN,FP],[FN,TP]]", cm(fields(run, kind), col).tolist(), [[tn, fp], [fn, tp]])
    if len(sil) != 4 or len(gra) != 4:
        PROBLEMS.append({"what": f"{run} metrics_usda.md field matrices found", "computed": 4, "committed": len(sil)})
    ct, pr = counties(run), prior(run)
    est = {"BARC-only": ct.barc_called, "County-share (out of fold)": ct.llp_oof_called,
           "Last Census reused": pr.last_census.reindex(ct.index),
           "Last Census + satellite (other folds)": pr.census_plus_satellite.reindex(ct.index)}
    for name, e in est.items():
        m = re.search(r"^\| " + re.escape(name) + r" \| ([\d.]+) \(.*?\| (\d+)/(\d+) \|", usda, re.M)
        st = county_stats(ct.census_share, e)
        expect(f"{run} county table, {name}: MAE, class right", [f"{st['mae']:.3f}", st["class_right"], st["n"]],
               [m.group(1), int(m.group(2)), int(m.group(3))] if m else None)
    md = (NAT / run / "metrics.md").read_text(encoding="utf-8")
    for head, col in [(r"BARC-only \(official setting\)", "barc_called"), (r"Proportion-trained, w [\d.]+, out of fold",
                                                                         "llp_oof_called")]:
        m = re.search(r"### County confusion matrix: " + head + r"\n\n.*\n.*\n((?:\|.*\n){3})", md)
        want = [[int(x) for x in ln.strip().strip("|").split("|")[1:]] for ln in m.group(1).strip().splitlines()] if m else None
        expect(f"{run} 3x3 county matrix {col}", class_matrix(ct.census_share, ct[col]).tolist(), want)
    cpm = (NAT / run / "census_prior" / "census_prior.md").read_text(encoding="utf-8")
    for name, col in [("Last Census (2017) alone", "last_census"), ("Satellite, Maryland-only model", "maryland_only"),
                      ("Satellite, county-share model", "satellite"),
                      ("Last Census + satellite (fitted on other folds)", "census_plus_satellite")]:
        m = re.search(r"^\| " + re.escape(name) + r" \| \d+ \| ([\d.]+) \|.*\| (\d+)/(\d+) \|$", cpm, re.M)
        st = county_stats(pr.truth, pr[col])
        # the file prints the gap with up to 4 decimals and drops trailing zeros, so compare within 0.0001
        near = bool(m) and abs(st["mae"] - float(m.group(1))) <= 0.0001
        expect(f"{run} census_prior.md, {name}: mean gap, class right",
               [m.group(1) if near else round(st["mae"], 4), st["class_right"], st["n"]],
               [m.group(1), int(m.group(2)), int(m.group(3))] if m else None)
    expect(f"{run} census_prior truth equals the run's census_share (max difference)",
           float((pr.truth.reindex(ct.index) - ct.census_share).abs().max()), 0.0)
    expect(f"{run} census_class column vs class bands",
           int((pd.Series(share_class(ct.census_share)).map(dict(enumerate(["under 5%", "5-40%", "40% and over"])))
                .to_numpy() == ct.census_class.to_numpy()).sum()), len(ct))


# ---------------------------------------------------------------- drawing helpers
def header(title, subtitle):
    """New 16:9 figure with a left-aligned title and subtitle; returns (fig, y of the free area's top)."""
    fig = plt.figure(figsize=FIGSIZE)
    t = textwrap.wrap(title, 72)
    s = textwrap.wrap(subtitle, 122)
    y = 0.962
    fig.text(0.04, y, "\n".join(t), size=23, weight="bold", color=INK, va="top", ha="left", linespacing=1.15)
    y -= 0.058 * len(t) + 0.012
    fig.text(0.04, y, "\n".join(s), size=14, color=INK2, va="top", ha="left", linespacing=1.25)
    y -= 0.037 * len(s)
    return fig, y


def save(fig, name, source):
    fig.text(0.04, 0.022, "\n".join(textwrap.wrap("Source: " + source, 172)), size=10.5, color=INK3, va="bottom",
             ha="left", linespacing=1.3)
    fig.savefig(HERE / f"{name}.png", dpi=200)
    fig.savefig(HERE / f"{name}.svg", metadata={"Date": None})
    plt.close(fig)
    print("wrote", f"docs/figures/{name}.png and .svg")


def panel_head(ax, color, title, note, size=18):
    """Color swatch, bold panel title and one note line above an axes, at fixed distances in points."""
    up = lambda pts: ax.transAxes + ScaledTranslation(0, pts / 72, ax.figure.dpi_scale_trans)  # noqa: E731
    ax.text(0, 1, note, transform=up(12), size=14.5, color=INK2, va="bottom", ha="left")
    ax.text(0, 1, title, transform=up(34), size=size, weight="bold", color=INK, va="bottom", ha="left")
    if color:
        ax.plot([0, 0.09], [1, 1], transform=up(66), color=color, lw=6, clip_on=False, solid_capstyle="butt")


def pct_text(v):
    """Whole percent, but never 0 or 100 for a cell that is not exactly empty or full."""
    return "<1" if 0 < v < 1 else ">99" if 99 < v < 100 else f"{v:.0f}"


def text_on(rgb):
    r, g, b = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb[:3]]
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.30 else INK


def draw_matrix(ax, m, labels, color, title, note, ylabel=True, count_size=30, pct_size=15, pct_fmt="{}% of row"):
    """Counts with row percentages; shading = row percentage, white to the model's color."""
    cmap = LinearSegmentedColormap.from_list("m", ["#ffffff", color])
    pct = m / m.sum(axis=1, keepdims=True)
    n = len(m)
    for i in range(n):
        for j in range(n):
            rgb = cmap(0.06 + 0.94 * pct[i, j])
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=rgb, edgecolor="white", linewidth=4))
            c = text_on(rgb)
            ax.text(j, i - 0.07, f"{m[i, j]:,}", ha="center", va="center", size=count_size, weight="bold", color=c)
            ax.text(j, i + 0.27, pct_fmt.format(pct_text(100 * pct[i, j])), ha="center", va="center", size=pct_size, color=c)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(n), labels, size=15)
    ax.set_yticks(range(n), labels if ylabel else [""] * n, size=15)
    ax.tick_params(length=0, pad=7)
    for s in ax.spines.values():
        s.set_visible(False)
    panel_head(ax, color, title, note)


def acc(m):
    return int(np.trace(m)), int(m.sum())


# ---------------------------------------------------------------- charts 1 and 2: field confusion matrices
def field_cms(name, kind, title_fn, subtitle_fn, source):
    f = fields(HEAD, kind)
    mb, ml = cm(f, "p_barc"), cm(f, "p_llp")
    fig, top = header(title_fn(mb, ml), subtitle_fn(mb, ml))
    gs = fig.add_gridspec(1, 2, left=0.13, right=0.93, top=top - 0.155, bottom=0.165, wspace=0.28)
    for k, (m, color, model) in enumerate([(mb, GRAY, "Maryland-only model"), (ml, BLUE, "County-share model")]):
        ax = fig.add_subplot(gs[0, k])
        ok, n = acc(m)
        draw_matrix(ax, m, ["Grain", "Silage"], color, model, f"{ok:,} of {n:,} correct ({100 * ok / n:.0f}%)",
                    ylabel=(k == 0))
        ax.set_xlabel("Model call", size=15, labelpad=8)
        if k == 0:
            ax.set_ylabel("True label", size=15, labelpad=10)
    save(fig, name, source)
    DATA[name] = {"rows": "true label (Grain, Silage)", "columns": "model call (Grain, Silage)", "n": int(len(f)),
                  "maryland_only": {"matrix": mb.tolist(), "correct": acc(mb)[0],
                                    "row_pct": (100 * mb / mb.sum(1, keepdims=True)).round(1).tolist()},
                  "county_share": {"matrix": ml.tolist(), "correct": acc(ml)[0],
                                   "row_pct": (100 * ml / ml.sum(1, keepdims=True)).round(1).tolist()},
                  "source": f"{REL}/{HEAD}/field_predictions.csv kind={kind}"}


def chart_cm_maryland():
    def title(mb, ml):
        a, b = acc(mb)[0], acc(ml)[0]
        if a == b:
            return f"Maryland intact: both models get {a} of {acc(mb)[1]} held-out field-years right"
        return f"Maryland held-out field-years: Maryland-only {a}, county-share {b} of {acc(mb)[1]} right"

    field_cms("cm_maryland", "barc", title,
              lambda mb, ml: "Fields at the Beltsville, Maryland research farm (BARC), 2021-2023. Both models were "
                             "trained on 2017-2020 labels only, so these years are held out. A field is called "
                             "silage when its predicted silage probability is 0.5 or more.",
              f"{REL}/{HEAD}/field_predictions.csv (kind = barc), headline run. Rows: true label. Columns: model call.")


def chart_cm_quebec():
    field_cms("cm_quebec", "quebec",
              lambda mb, ml: f"In Québec the county-share model catches {ml[1, 1]:,} of {ml[1].sum():,} silage "
                             f"fields; the Maryland-only model catches {mb[1, 1]:,}",
              lambda mb, ml: f"{int(mb.sum()):,} Québec field-years, 2021-2023. No Québec label was used in training. "
                             f"The price: {ml[0, 1]:,} of {ml[0].sum():,} grain fields are now called silage "
                             f"({mb[0, 1]:,} before).",
              f"{REL}/{HEAD}/field_predictions.csv (kind = quebec), headline run. Threshold 0.5. Rows: true label. "
              "Columns: model call.")


# ---------------------------------------------------------------- chart 3: county class matrices
def chart_county_classes():
    """All three panels from census_prior_counties.csv, so they are like for like: same counties, same truth, and
    the models' county estimate is the mean predicted probability (the estimate census_prior.py uses)."""
    ct = counties(HEAD)
    pr = prior(HEAD).reindex(ct.index)
    t = pr.truth
    panels = [("Maryland-only model", GRAY, pr.maryland_only), ("County-share model", BLUE, pr.satellite),
              ("Last Census reused", ORANGE, pr.last_census)]
    ms = [class_matrix(t, e) for _, _, e in panels]
    right = [acc(m)[0] for m in ms]
    called = {"maryland_only": county_stats(ct.census_share, ct.barc_called)["class_right"],
              "county_share_oof": county_stats(ct.census_share, ct.llp_oof_called)["class_right"]}
    title = (f"Counties in the right silage class: Maryland-only {right[0]}, county-share {right[1]}, "
             f"last Census reused {right[2]} (of {len(pr)})")
    fig, top = header(title, "Classes by silage share of harvested corn acres. A county's predicted share is the mean "
                             "predicted silage probability of its 100 sampled corn blocks. Out of fold: each group of "
                             "states is predicted by models that never saw it. Last Census: the earlier Census share, "
                             "unchanged. Small figures: percent of the row.")
    gs = fig.add_gridspec(1, len(panels), left=0.115, right=0.975, top=top - 0.15, bottom=0.20, wspace=0.10)
    for k, ((model, color, _), m) in enumerate(zip(panels, ms)):
        ax = fig.add_subplot(gs[0, k])
        draw_matrix(ax, m, CLASSES, color, model, f"{right[k]} of {int(m.sum())} right" + (", out of fold" if k == 1 else ""),
                    ylabel=(k == 0), count_size=27, pct_size=14.5, pct_fmt="{}%")
        ax.set_xlabel("Predicted class", size=15, labelpad=8)
        if k == 0:
            ax.set_ylabel("2022 Census class", size=15, labelpad=10)
    save(fig, "county_class_matrices",
         f"{REL}/{HEAD}/census_prior/census_prior_counties.csv (truth, maryland_only, satellite, last_census). "
         f"Counting the share of blocks called silage instead, the models get {called['maryland_only']} and "
         f"{called['county_share_oof']} right.")
    DATA["county_class_matrices"] = {
        "rows": "2022 Census class (under 5%, 5-40%, 40% and over)", "columns": "predicted class, same order",
        "n": int(len(pr)), "model_estimate": "mean predicted probability (census_prior_counties.csv)",
        "class_right_with_share_of_blocks_called": called,
        **{key: {"matrix": m.tolist(), "class_right": r}
           for key, m, r in zip(["maryland_only", "county_share_oof", "last_census_reused"], ms, right)}}


# ---------------------------------------------------------------- chart 4: predicted vs Census scatter
def chart_county_2x2():
    """The 3x3 county matrices merged at the 40% class edge: high-silage county (40% and over) or not. Same file,
    counties, truth and county estimate as chart_county_classes. Exploratory: the two-class view was added after
    the results; the edge itself is one of the two fixed in advance."""
    pr = prior(HEAD)
    panels = [("Maryland-only model", GRAY, pr.maryland_only), ("County-share model", BLUE, pr.satellite),
              ("Last Census reused", ORANGE, pr.last_census)]
    ms = []
    for _, _, e in panels:
        m3 = class_matrix(pr.truth, e)
        ms.append(np.array([[m3[:2, :2].sum(), m3[:2, 2].sum()], [m3[2, :2].sum(), m3[2, 2]]], dtype=int))
    right = [acc(m)[0] for m in ms]
    n_high = int(ms[0][1].sum())
    title = (f"High-silage counties found: Maryland-only {ms[0][1, 1]}, county-share {ms[1][1, 1]}, "
             f"last Census reused {ms[2][1, 1]} (of {n_high})")
    fig, top = header(title, f"{len(pr)} counties in two classes by silage share of harvested corn acres: under 40%, "
                             "or 40% and over. A county's predicted share is the mean predicted silage probability of "
                             "its 100 sampled corn blocks, out of fold by state. Exploratory: this two-class view "
                             "merges the three classes fixed in advance and was added after the results.")
    gs = fig.add_gridspec(1, len(panels), left=0.115, right=0.975, top=top - 0.15, bottom=0.20, wspace=0.10)
    labels = ["under 40%", "40%+"]
    for k, ((model, color, _), m) in enumerate(zip(panels, ms)):
        ax = fig.add_subplot(gs[0, k])
        draw_matrix(ax, m, labels, color, model, f"{right[k]} of {int(m.sum())} right" + (", out of fold" if k == 1 else ""),
                    ylabel=(k == 0), count_size=30, pct_size=15)
        ax.set_xlabel("Predicted class", size=15, labelpad=8)
        if k == 0:
            ax.set_ylabel("2022 Census class", size=15, labelpad=10)
    save(fig, "county_2x2",
         f"{REL}/{HEAD}/census_prior/census_prior_counties.csv (truth, maryland_only, satellite, last_census). "
         "Rows: 2022 Census class. Columns: predicted class. Small figures: percent of the row.")
    DATA["county_2x2"] = {
        "rows": "2022 Census class (under 40%, 40% and over)", "columns": "predicted class, same order",
        "n": int(len(pr)), "model_estimate": "mean predicted probability (census_prior_counties.csv)",
        "exploratory": True,
        **{key: {"matrix": m.tolist(), "right": r}
           for key, m, r in zip(["maryland_only", "county_share_oof", "last_census_reused"], ms, right)}}


def chart_scatter():
    ct = counties(HEAD)
    t = ct.census_share
    sb, sl = county_stats(t, ct.barc_called), county_stats(t, ct.llp_oof_called)
    fig, top = header("Trained on county shares, the model follows the Census far more closely "
                      "than the Maryland-only model",
                      f"{len(ct)} sampled counties in {ct.state.nunique()} states, 2022. Each dot is a county: the "
                      "share of its sampled corn blocks called silage against the Census silage share. County-share "
                      "model scored out of fold (each group of states predicted by models that never saw it). "
                      "Dashed line: prediction equals Census.")
    gs = fig.add_gridspec(1, 2, left=0.16, right=0.90, top=top - 0.15, bottom=0.165, wspace=0.30)
    for k, (model, color, col, st, alpha) in enumerate([
            ("Maryland-only model", GRAY, "barc_called", sb, 0.55),
            ("County-share model, out of fold", BLUE, "llp_oof_called", sl, 0.75)]):
        ax = fig.add_subplot(gs[0, k])
        ax.plot([0, 1], [0, 1], color=INK2, lw=1.4, ls=(0, (5, 4)), zorder=1)
        ax.scatter(t, ct[col], s=58, color=color, alpha=alpha, edgecolor="white", linewidth=0.9, zorder=3,
                   clip_on=False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_xticks(np.arange(0, 1.01, 0.25))
        ax.set_yticks(np.arange(0, 1.01, 0.25))
        ax.grid(color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, labelsize=14, pad=6)
        ax.set_xlabel("2022 Census silage share", size=15, labelpad=8)
        ax.set_ylabel("Predicted silage share", size=15, labelpad=8)
        panel_head(ax, color, model, f"Mean absolute gap {st['mae']:.3f}, rank correlation {st['spearman']:.2f}")
    save(fig, "pred_vs_census_scatter",
         f"{REL}/{HEAD}/county_predictions.csv (census_share, barc_called, llp_oof_called). Rank correlation: Spearman.")
    DATA["pred_vs_census_scatter"] = {
        "x": "census_share", "y": {"maryland_only": "barc_called", "county_share_oof": "llp_oof_called"},
        "points_per_panel": int(len(ct)), "states": int(ct.state.nunique()),
        "maryland_only": {k: round(v, 4) if isinstance(v, float) else v for k, v in sb.items()},
        "county_share_oof": {k: round(v, 4) if isinstance(v, float) else v for k, v in sl.items()}}


# ---------------------------------------------------------------- chart 5: the three runs
def chart_variants():
    rows = {}
    for run in RUNS:
        ct = counties(run)
        mm, mq = cm(fields(run, "barc"), "p_llp"), cm(fields(run, "quebec"), "p_llp")
        rows[run] = {"county_gap_oof": round(county_stats(ct.census_share, ct.llp_oof_called)["mae"], 4),
                     "w_chosen": json.loads((NAT / run / "run.json").read_text(encoding="utf-8"))["setting"]["w_chosen"],
                     "maryland_correct": acc(mm)[0], "maryland_n": acc(mm)[1],
                     "quebec_silage_caught": int(mq[1, 1]), "quebec_silage_n": int(mq[1].sum()),
                     "quebec_grain_called_silage": int(mq[0, 1]), "quebec_grain_n": int(mq[0].sum())}
    h = rows[HEAD]
    best = min(rows, key=lambda r: rows[r]["county_gap_oof"])
    lead = (f"The headline run has the lowest out-of-fold county gap ({h['county_gap_oof']:.3f}), the measure fixed "
            "in advance" if best == HEAD else "Three runs of the county-share model compared")
    fig, top = header(lead,
                      "Three runs of the county-share model. The headline was picked by a rule set before the results: "
                      "lowest county gap, Jan-Dec against Jan-Oct imagery. The adapted-weights run gets one more "
                      "Maryland field right and calls fewer Québec grain fields silage, but catches fewer Québec "
                      "silage fields; it was never a headline candidate.")
    panels = [("County gap,\nout of fold", "lower is better",
               lambda r: r["county_gap_oof"], 0.20, lambda r: f"{r['county_gap_oof']:.3f}"),
              ("Maryland fields\nright", "held-out 2021-2023",
               lambda r: r["maryland_correct"] / r["maryland_n"], 1.0,
               lambda r: f"{r['maryland_correct']} of {r['maryland_n']}"),
              ("Québec silage\ncaught", "no Québec labels used",
               lambda r: r["quebec_silage_caught"] / r["quebec_silage_n"], 1.0,
               lambda r: f"{r['quebec_silage_caught']:,} of {r['quebec_silage_n']:,}"),
              ("Québec grain\ncalled silage", "lower is better",
               lambda r: r["quebec_grain_called_silage"] / r["quebec_grain_n"], 1.0,
               lambda r: f"{r['quebec_grain_called_silage']:,} of {r['quebec_grain_n']:,}")]
    gs = fig.add_gridspec(1, 4, left=0.235, right=0.965, top=top - 0.185, bottom=0.10, wspace=0.20)
    ys = np.arange(len(RUNS))
    for k, (name, note, val, xmax, lab) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        for y, run in zip(ys, RUNS):
            r = rows[run]
            ax.barh(y, xmax, height=0.30, color="#f0f0f0", zorder=1)
            ax.barh(y, val(r), height=0.30, color=BLUE if run == HEAD else BLUE_LIGHT, zorder=2)
            ax.text(0, y - 0.25, lab(r), size=19, weight="bold" if run == HEAD else "normal", color=INK, va="bottom",
                    ha="left")
        ax.set_xlim(0, xmax)
        ax.set_ylim(len(RUNS) - 0.45, -0.75)
        ax.set_xticks([])
        ax.set_yticks(ys, list(RUNS.values()) if k == 0 else [""] * len(RUNS), size=15)
        ax.tick_params(length=0, pad=14)
        if k == 0:
            ax.get_yticklabels()[0].set_weight("bold")
            ax.get_yticklabels()[0].set_color(INK)
        for s in ax.spines.values():
            s.set_visible(False)
        panel_head(ax, None, name, note, size=17)
    save(fig, "variants_bar",
         f"{REL}/results_final, results_final_janoct, results_final_barconly: county_predictions.csv "
         "(llp_oof_called), field_predictions.csv (p_llp). County gap: mean absolute difference between predicted and "
         "2022 Census silage share, 163 counties. Gray track = full scale (gap panel: 0 to 0.20; others: 0 to 100%).")
    DATA["variants_bar"] = {"model": "county-share model in every run", "headline": HEAD,
                            "bar_scale": {"county_gap_oof": [0, 0.20], "other panels": "share of the total, 0 to 1"},
                            "runs": rows}


# ---------------------------------------------------------------- chart 6: against the last Census
def chart_last_census():
    """Every bar from census_prior_counties.csv: same counties, same truth (2022 Census-only share). The models'
    county estimate is the mean predicted probability, which is also what the 'Last Census + model' bar is fitted
    on. The share-of-blocks gaps of metrics_usda.md are in the footnote."""
    ct = counties(HEAD)
    pr = prior(HEAD).reindex(ct.index)
    e = pd.DataFrame({"truth": pr.truth, "maryland_only": pr.maryland_only, "county_share_oof": pr.satellite,
                      "last_census": pr.last_census})
    est = [("Maryland-only model", "maryland_only", GRAY), ("County-share model\n(out of fold)", "county_share_oof", BLUE),
           ("Last Census reused", "last_census", ORANGE)]
    if "census_plus_satellite" in pr:
        e["census_plus_satellite"] = pr.census_plus_satellite
        est.append(("Last Census + county-share\nmodel (fitted on other states)", "census_plus_satellite", BROWN))
    gap = {c: float((e[c] - e.truth).abs().mean()) for _, c, _ in est}
    called = {"maryland_only": float((ct.barc_called - ct.census_share).abs().mean()),
              "county_share_oof": float((ct.llp_oof_called - ct.census_share).abs().mean())}
    src = pr.prior_source.astype(str).value_counts().to_dict()
    sub = (f"Mean absolute gap to the 2022 Census silage share, the same {len(e)} counties for every bar. A model's "
           f"county estimate is the mean predicted silage probability of its blocks. The county-share model cuts the "
           f"Maryland-only gap from {gap['maryland_only']:.3f} to {gap['county_share_oof']:.3f} but does not beat "
           f"the last Census.")
    extra = {}
    if "census_plus_satellite" in e:
        ec, es = (e.last_census - e.truth).abs(), (e.census_plus_satellite - e.truth).abs()
        extra = {"plus_satellite_better_in": int((es < ec).sum()), "plus_satellite_worse_in": int((es > ec).sum())}
        sub += (f" Adding the model to the last Census changes little: better in {extra['plus_satellite_better_in']} "
                f"counties, worse in {extra['plus_satellite_worse_in']}.")
    fig, top = header(f"Reusing the last Census is still closer on county totals: gap "
                      f"{gap['last_census']:.3f} against {gap['county_share_oof']:.3f} for the county-share model", sub)
    ax = fig.add_axes([0.335, 0.20, 0.60, top - 0.05 - 0.20])
    ys = np.arange(len(est))
    for y, (name, c, color) in zip(ys, est):
        ax.barh(y, gap[c], height=0.58, color=color, zorder=2)
        ax.text(gap[c] + 0.004, y, f"{gap[c]:.3f}", size=22, weight="bold", color=INK, va="center", ha="left")
    ax.set_yticks(ys, [n for n, _, _ in est], size=16)
    ax.set_ylim(len(est) - 0.5, -0.5)
    ax.set_xlim(0, max(gap.values()) * 1.16)
    ax.set_xticks(np.arange(0, max(gap.values()) * 1.16, 0.05))
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=10)
    ax.tick_params(axis="x", labelsize=14, pad=6)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color(INK3)
    ax.set_xlabel("Mean absolute gap to the 2022 Census share (0 = perfect; lower is better)", size=15, labelpad=10,
                  loc="left")
    save(fig, "last_census_comparison",
         f"{REL}/{HEAD}/census_prior/census_prior_counties.csv (truth, maryland_only, satellite, last_census, "
         f"census_plus_satellite). Counting the share of blocks called silage instead, the model gaps are "
         f"{called['maryland_only']:.3f} and {called['county_share_oof']:.3f}. Last Census = 2017 share for "
         f"{src.get('2017', 0)} counties, 2012 for {src.get('2012', 0)}, the state's 5-year share for "
         f"{src.get('state 5-yr', 0)}.")
    DATA["last_census_comparison"] = {
        "measure": "mean absolute gap to truth (2022 Census-only share); models = mean predicted probability, "
                   "as in census_prior.md",
        "n": int(len(e)), "bars": {c: round(gap[c], 4) for _, c, _ in est},
        "class_right": {c: int((share_class(e[c]) == share_class(e.truth)).sum()) for _, c, _ in est},
        "footnote_gap_with_share_of_blocks_called_as_in_metrics_usda_md": {k: round(v, 4) for k, v in called.items()},
        "last_census_source_counts": {str(k): int(v) for k, v in src.items()}, **extra}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-json", help="also write every plotted value to this JSON file")
    a = ap.parse_args()
    for run in RUNS:
        check_run(run)
    chart_cm_maryland()
    chart_cm_quebec()
    chart_county_classes()
    chart_county_2x2()
    chart_scatter()
    chart_variants()
    chart_last_census()
    DATA["differences_from_committed_metrics_files"] = PROBLEMS
    print("checks against metrics_usda.md / metrics.md / census_prior.md:", "all equal" if not PROBLEMS else f"{len(PROBLEMS)} differ")
    if a.data_json:
        out = Path(a.data_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(DATA, indent=2, ensure_ascii=False), encoding="utf-8")
        print("wrote plotted values to", out.name)


if __name__ == "__main__":
    main()
