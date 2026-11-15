#!/usr/bin/env bash
# Score frozen Prithvi-EO-2.0 embeddings with the team's shared evaluator (evaluate.py), the same way the Presto,
# AlphaEarth and TESSERA embeddings were scored.
#   bash classifier/prithvi/prithvi_evaluate.sh
# evaluate.py imports research/grain_vs_silage/model_comparison.py and reads results/aef_embeddings.csv from the team
# repository's main branch, which this workspace copy predates, so it runs inside a fresh copy of main under $WORK.
# The four tables: Prithvi-EO-2.0 300M, chip at 30 m or upsampled to 15 m, 4 frames or 8 two-week frames.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-/home/keeboh/venvs/prithvi/bin/python}"
WORK="${WORK:-/home/keeboh/work/eval}"
CHIPS="${CHIPS:-/home/keeboh/work/prithvi/chips}"
export HF_HOME="${HF_HOME:-/home/keeboh/hf}"
TREE="$WORK/Hack-the-harvest--main"
mkdir -p "$WORK"
if [ ! -d "$TREE" ]; then
  curl -sSL https://codeload.github.com/devarchith/Hack-the-harvest-/zip/refs/heads/main -o "$WORK/main.zip"
  "$PY" -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$WORK/main.zip" "$WORK"
fi
EVAL_DIR="$TREE/research/pretrained_classifier"
mkdir -p "$EVAL_DIR/results/prithvi_emb"
cp "$HERE/evaluate.py" "$EVAL_DIR/evaluate.py"
for f in 4 8; do
  for s in 1 2; do
    name="prithvi_300M_s${s}_f${f}"
    "$PY" "$HERE/prithvi_embed.py" --chips "$CHIPS" --out "$EVAL_DIR/results/prithvi_emb" --size 300M --scale "$s" \
      --frames "$f"
    "$PY" "$EVAL_DIR/evaluate.py" --features "$EVAL_DIR/results/prithvi_emb/$name.csv" --prefix E --name "$name" \
      --title "Prithvi-EO-2.0 300M, $((30 / s)) m chip, $f frames"
  done
done
"$PY" "$HERE/prithvi_select.py" --runs "$EVAL_DIR/results" --out "$HERE/results/evaluate"
