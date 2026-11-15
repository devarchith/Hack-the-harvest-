"""Frozen Prithvi-EO-2.0 embeddings per field-year, in the table format of the team's shared evaluator (evaluate.py).

  HF_HOME=/home/keeboh/hf python classifier/prithvi/prithvi_embed.py --chips /home/keeboh/work/prithvi/chips \
      --out DIR --size 300M --scale 2 --frames 8

Uses the field-centred 48 x 48 chips of prithvi_train.py with one of two frame sets:
  4 frames: Jul 1-31, Aug 10-Sep 5, Sep 6-Oct 5, Oct 6-Nov 15 (as in prithvi_train.py)
  8 frames: two-week windows from Aug 1 to Nov 20, to follow harvest timing more closely
A pixel with no clear reading in a frame takes the value of the nearest frame in time that has one. The encoder is
frozen; its patch tokens are averaged per frame, weighted by the share of each patch inside the field polygon, then
averaged over the 8 rotations and flips of the chip.

Writes <out>/prithvi_<size>_s<scale>_f<frames>.csv (field_id, year, E0000...; frames x embedding width columns) and a
.json next to it with the checkpoint, frame windows and gap-fill shares.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import prithvi_train as pt

HERE = Path(__file__).resolve().parent
FRAMES = {4: pt.WINDOWS,
          8: [("08-01", "08-14"), ("08-15", "08-28"), ("08-29", "09-11"), ("09-12", "09-25"),
              ("09-26", "10-09"), ("10-10", "10-23"), ("10-24", "11-06"), ("11-07", "11-20")]}


def fill_gaps(X):
    """Fill each pixel's empty frames (NaN) from the nearest frame in time with a reading; earlier frame on ties."""
    out, T = X.copy(), X.shape[2]
    for t in range(T):
        miss = np.isnan(out[:, :, t])
        for d in range(1, T):
            for s in (t - d, t + d):
                if 0 <= s < T and miss.any():
                    take = miss & ~np.isnan(X[:, :, s])
                    out[:, :, t][take] = X[:, :, s][take]
                    miss &= ~take
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field-years", default=str(HERE / "field_years.csv"))
    ap.add_argument("--chips", required=True, help="folder written by prithvi_chips.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="300M", choices=list(pt.REPOS))
    ap.add_argument("--scale", type=int, default=1, help="upsampling of the 30 m chip; 2 makes one patch 240 m")
    ap.add_argument("--frames", type=int, default=4, choices=sorted(FRAMES))
    a = ap.parse_args()
    t0 = time.time()
    pt.WINDOWS = FRAMES[a.frames]  # field_inputs reads the frame windows from here
    fy = pd.read_csv(a.field_years)
    X, M, TC, LC = pt.field_inputs(fy, a.chips)
    inside = (M > 0.5)[:, None]
    empty = (np.isnan(X[:, 0]) & inside).sum((0, 2, 3)) / inside.sum()
    X = fill_gaps(X)
    left = float((np.isnan(X[:, 0]) & inside).any(1).sum() / inside.sum())
    print("share of field pixels with no clear reading, per frame:", np.round(empty, 3).tolist(),
          "| still empty in every frame:", round(left, 4), flush=True)

    enc, cfg, info = pt.load_encoder(a.size, a.frames, pt.CHIP * a.scale, "cuda")
    model = pt.FieldClassifier(enc, a.frames, a.scale).to("cuda")
    E = pt.embed_all(model, (pt.normalise(X, cfg), M, TC, LC), np.arange(len(fy)), "cuda")
    cols = [f"E{i:04d}" for i in range(E.shape[1])]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"prithvi_{a.size}_s{a.scale}_f{a.frames}"
    pd.concat([fy[["field_id", "year"]].reset_index(drop=True), pd.DataFrame(E.round(5), columns=cols)],
              axis=1).to_csv(out / f"{name}.csv", index=False)
    (out / f"{name}.json").write_text(json.dumps(
        {"args": vars(a), "model": info, "frames": FRAMES[a.frames], "empty_share_per_frame": empty.round(4).tolist(),
         "empty_after_fill": left, "field_years": len(fy), "columns": len(cols),
         "field_years_sha256": pt.sha256(a.field_years), "runtime_s": round(time.time() - t0, 1)}, indent=1))
    print(f"wrote {out / name}.csv: {len(fy)} field-years x {len(cols)} columns, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
