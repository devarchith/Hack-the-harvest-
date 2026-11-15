"""Frozen Presto embedding of every labeled field-year, for the team's shared evaluator (../prithvi/evaluate.py).

  python classifier/presto/presto_embed.py --pixels /home/keeboh/work/presto/pixels \
      --presto-dir /home/keeboh/ext/presto --out /home/keeboh/work/presto/presto_frozen.csv

Same inputs as presto_train.py, pretrained encoder not fine-tuned: the 128-d embedding of each pixel, averaged per
field-year. Writes field_id, farm, field, year, label, n_pix, Q000..Q127.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from presto_train import PixelClassifier, field_inputs, field_outputs, load_presto  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field-years", default=str(HERE / "field_years.csv"))
    ap.add_argument("--pixels", required=True, help="folder written by presto_pixels.py")
    ap.add_argument("--presto-dir", required=True, help="nasaharvest/presto files: single_file_presto.py, data/")
    ap.add_argument("--out", required=True, help="CSV to write")
    ap.add_argument("--last-month", type=int, default=12, choices=range(1, 13), help="mask later months (10 = October)")
    a = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fy = pd.read_csv(a.field_years)
    enc, groups = load_presto(a.presto_dir, pretrained=True)
    X, M, LL, offs, key, _, _ = field_inputs(fy, a.pixels, groups, a.last_month)
    data = (torch.from_numpy(X).to(device), torch.from_numpy(M).to(device), torch.from_numpy(LL).to(device),
            offs, np.repeat(key, np.diff(offs)))
    Z = field_outputs(PixelClassifier(enc).to(device), data, np.arange(len(fy)),
                      lambda mdl, x, m, ll: mdl.embed(x, m, ll))
    out = pd.concat([fy[["field_id", "farm", "field", "year", "label"]].assign(n_pix=np.diff(offs)),
                     pd.DataFrame(Z, columns=[f"Q{i:03d}" for i in range(Z.shape[1])])], axis=1)
    out.round(6).to_csv(a.out, index=False)
    print(f"wrote {len(out)} field-years x {Z.shape[1]} to {a.out}")


if __name__ == "__main__":
    main()
