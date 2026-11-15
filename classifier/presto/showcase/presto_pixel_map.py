"""Per-pixel silage probabilities of fine-tuned Presto on the 44 test field-years (2021-2023), with each pixel's map
position and the monthly values Presto was given, for the maps in this folder.

  python classifier/presto/showcase/presto_pixel_map.py \
      --field-years classifier/labels/results/field_years_fixed.csv --pixels /home/keeboh/work/presto/pixels \
      --presto-dir /home/keeboh/ext/presto \
      --predictions classifier/presto/results/standard_test_fixed_labels/test_predictions.csv \
      --out classifier/presto/showcase/results

Refits the setting chosen in the standard test (lr 3e-4, 12 epochs, seeds 0-4) on the 2017-2020 field-years with
presto_standard_test.train_curve, unchanged, and keeps every test pixel's probability for each seed. The mean over
pixels and seeds of a field-year must match p_finetune in --predictions. Pixel positions are recomputed from the
polygons of the table the pixels were pulled with (presto/field_years.csv), on the same 10 m grid and with the same
10 m inward buffer as presto_pixels.py.
"""
import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PRESTO = HERE.parent
ROOT = PRESTO.parents[1]
sys.path[:0] = [str(PRESTO), str(ROOT)]
import presto_standard_test as pst  # noqa: E402
from presto_train import NORMED_BANDS, batches, field_inputs, load_presto, sha256  # noqa: E402
from preprocessing.loader import _field_pixels  # noqa: E402
from odc.geo import xy_  # noqa: E402
from odc.geo.geobox import GeoBox  # noqa: E402

CRS = "EPSG:32618"  # as presto_pixels.py
PULL_TABLE = PRESTO / "field_years.csv"  # the table presto_pixels.py read
KEEP = ["B2", "B3", "B4", "B8", "B11", "NDVI", "VH"]  # bands written for the maps


def pixel_positions(table, years):
    """{(field_id, year): (x, y) pixel centres, in the order presto_pixels.py wrote them}."""
    fy = pd.read_csv(table)
    fy = gpd.GeoDataFrame(fy, geometry=gpd.GeoSeries.from_wkt(fy.wkt), crs=CRS)
    out = {}
    for yr in years:
        d = fy[fy.year == yr].sort_values("field_id")
        ids = d.field_id.tolist()
        x0, y0, x1, y1 = d.total_bounds
        gbox = GeoBox.from_bbox((x0 - 30, y0 - 30, x1 + 30, y1 + 30), crs=CRS, resolution=10, anchor=xy_(0, 0))
        pixels, _ = _field_pixels(d.geometry, ids, gbox, 10.0)
        t = gbox.transform
        for fid, idx in pixels.items():
            r, c = np.divmod(idx, gbox.shape[1])
            out[(int(fid), yr)] = (t.c + (c + 0.5) * t.a, t.f + (r + 0.5) * t.e)
    return out


@torch.no_grad()
def pixel_probs(model, data, rows, bs=8192):
    X, M, LL, offs, key_pix = data
    model.eval()
    pix = np.concatenate([np.arange(offs[r], offs[r + 1]) for r in rows])
    out = np.zeros(offs[-1], np.float32)
    for b in batches(pix, key_pix, bs):
        bt = torch.from_numpy(b).to(X.device)
        out[b] = torch.sigmoid(model(X[bt], M[bt], LL[bt])).float().cpu().numpy()
    return out[pix]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field-years", required=True)
    ap.add_argument("--pixels", required=True, help="folder written by presto_pixels.py")
    ap.add_argument("--presto-dir", required=True)
    ap.add_argument("--predictions", required=True, help="test_predictions.csv of the standard-test run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--tolerance", type=float, default=0.01, help="largest allowed field-year difference")
    a = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    fy = pd.read_csv(a.field_years)
    y = (fy.label == "Silage").to_numpy(int)
    tr, te, _ = pst.standard_split(fy)
    enc, groups = load_presto(a.presto_dir, pretrained=True)
    X, M, LL, offs, key, cov, pixel_hashes = field_inputs(fy, a.pixels, groups)
    data = (torch.from_numpy(X).to(device), torch.from_numpy(M).to(device), torch.from_numpy(LL).to(device),
            offs, np.repeat(key, np.diff(offs)))

    # train_curve scores the test rows after every epoch through pst.predict; keep the model it scores last
    seen, predict = [], pst.predict
    pst.predict = lambda model, d, rows: (seen.append(model), predict(model, d, rows))[1]
    p_seed, p_field = [], []
    for s in a.seeds:
        p_field.append(pst.train_curve(enc, data, y, tr, te, device, s, a.lr, a.epochs)[-1])
        p_seed.append(pixel_probs(seen[-1], data, te))
        seen.clear()
        print(f"seed {s} done", flush=True)
    pst.predict = predict
    p_seed = np.stack(p_seed, 1)

    n_pix = np.diff(offs)[te]
    owner = np.repeat(np.arange(len(te)), n_pix)
    mean_pix = np.bincount(owner, p_seed.mean(1)) / n_pix
    ref = pd.read_csv(a.predictions)
    chk = fy.loc[te, ["field_id", "year", "label"]].reset_index(drop=True).assign(
        n_pix=n_pix, p_pixels=mean_pix.round(4), p_curve=np.mean(p_field, 0).round(4))
    chk = chk.merge(ref[["field_id", "year", "p_finetune"]], on=["field_id", "year"], how="left", validate="1:1")
    chk["diff"] = (chk.p_pixels - chk.p_finetune).abs()
    chk.to_csv(out / "field_check.csv", index=False)
    worst = float(chk["diff"].max())
    same_calls = int(((chk.p_pixels >= 0.5) == (chk.p_finetune >= 0.5)).sum())
    print(f"largest field-year difference from {Path(a.predictions).name}: {worst:.4f}; "
          f"same call on {same_calls}/{len(chk)}", flush=True)
    if not worst <= a.tolerance:
        sys.exit(f"pixel means do not reproduce p_finetune (largest difference {worst:.4f} > {a.tolerance})")

    pos = pixel_positions(PULL_TABLE, sorted(fy.year.iloc[te].unique()))
    xs, ys = [], []
    for r, n in zip(te, n_pix):
        px, py = pos[(int(fy.field_id.iloc[r]), int(fy.year.iloc[r]))]
        if len(px) != n:
            sys.exit(f"field {fy.field_id.iloc[r]} {fy.year.iloc[r]}: {len(px)} recomputed pixels, {n} pulled")
        xs.append(px); ys.append(py)
    pix = np.concatenate([np.arange(offs[r], offs[r + 1]) for r in te])
    bands = [NORMED_BANDS.index(b) for b in KEEP]
    month_used = M[offs[te]][:, :, [NORMED_BANDS.index("B4"), NORMED_BANDS.index("VH")]] == 0
    npz = out / "pixel_map.npz"
    np.savez_compressed(
        npz, field_id=fy.field_id.iloc[te].to_numpy(), year=fy.year.iloc[te].to_numpy(),
        label=fy.label.iloc[te].to_numpy(str), offsets=np.concatenate([[0], np.cumsum(n_pix)]),
        x=np.concatenate(xs).astype(np.float32), y=np.concatenate(ys).astype(np.float32),
        p=p_seed.astype(np.float16), bands=np.array(KEEP), values=X[pix][:, :, bands].astype(np.float16),
        month_used=month_used)
    json.dump({"args": vars(a), "device": device, "torch": torch.__version__, "crs": CRS,
               "test_field_years": len(te), "pixels": int(n_pix.sum()), "largest_difference": round(worst, 4),
               "same_calls": same_calls, "values": "Presto inputs: S2 reflectance, NDVI, S1 VH as (dB + 25) / 25",
               "month_used": "per field-year and month: [Sentinel-2 used, Sentinel-1 used]",
               "hashes": {"field_years": sha256(a.field_years), "pull_table": sha256(PULL_TABLE),
                          "predictions": sha256(a.predictions), "pixel_map.npz": sha256(npz),
                          "presto_weights": sha256(Path(a.presto_dir) / "data" / "default_model.pt"),
                          **{f"pixels/{k}": v for k, v in pixel_hashes.items()}}},
              open(out / "run.json", "w"), indent=2)
    print(f"wrote {npz} ({int(n_pix.sum())} pixels)", flush=True)


if __name__ == "__main__":
    main()
