#!/usr/bin/env bash
# ---------------------------------------------------------------
# Run the full DWPC ablation matrix (SSL + UDA) over seeds.
#
# The rows are descriptive (no a0/a1 codes). Integer indices 0-7 still
# work on the CLI and map to the descriptive config names:
#
#   0 off                 (matrix baseline, DWPC disabled)
#   1 proda_symmetric      (ProDA-style symmetric flip)
#   2 witnessA_only        (appearance witness only)
#   3 witnessB_nogate      (structure witness, no rare-gate)
#   4 dualwitness_noflip   (both witnesses, flips disabled)
#   5 full                 (full DWPC)
#   6 full_static_struct   (full, target_adaptive=False)
#   7 full_no_flipveto     (full, flip_veto=False)
#
# Usage:
#   bash tools/run_dwpc_all.sh [GPU] [MODES] [ROWS] [SEEDS]
#
#   GPU    CUDA device id            (default: 0)
#   MODES  space list: ssl uda      (default: "ssl uda")
#   ROWS   space list of indices    (default: "0 1 2 3 4 5 6 7")
#          OR descriptive tags (e.g. "off full")
#   SEEDS  space list               (default: "0 1 2")
#
# Examples:
#   bash tools/run_dwpc_all.sh                       # everything, 3 seeds
#   bash tools/run_dwpc_all.sh 0 ssl "witnessA_only dualwitness_noflip full"
#   bash tools/run_dwpc_all.sh 0 ssl "2 4 5"         # same via indices
#
# Each run lands in work_dirs/dwpc/<mode>_<tag>_s<seed>/.
# ---------------------------------------------------------------
set -u

PY=/home/ubuntu/SatelliteImageSegSupevised/.venv/bin/python
GPU="${1:-0}"
MODES="${2:-ssl uda}"
ROWS="${3:-0 1 2 3 4 5 6 7}"
SEEDS="${4:-0 1 2}"
CFG_DIR=configs/daformer
OUT_ROOT=work_dirs/dwpc

# index -> descriptive tag
declare -A TAG=(
  [0]=off [1]=proda_symmetric [2]=witnessA_only [3]=witnessB_nogate
  [4]=dualwitness_noflip [5]=full [6]=full_static_struct [7]=full_no_flipveto
)

mkdir -p "$OUT_ROOT"
echo "GPU=$GPU MODES=[$MODES] ROWS=[$ROWS] SEEDS=[$SEEDS]"

for mode in $MODES; do
  for row in $ROWS; do
    # accept either an integer index or a descriptive tag
    tag="${TAG[$row]:-$row}"
    cfg="$CFG_DIR/${mode}_oem_dwpc_${tag}_mitb5.py"
    if [ ! -f "$cfg" ]; then
      echo "[skip] missing $cfg"; continue
    fi
    for seed in $SEEDS; do
      wd="$OUT_ROOT/${mode}_${tag}_s${seed}"
      echo "=== ${mode} ${tag} seed ${seed} -> ${wd} ==="
      CUDA_VISIBLE_DEVICES="$GPU" "$PY" tools/train.py "$cfg" \
        --work-dir "$wd" --seed "$seed" --deterministic
    done
  done
done

echo "All runs dispatched. Evaluate with:"
echo "  $PY tools/dwpc_eval.py iou <cfg> <wd>/latest.pth --rare 0 1 6"
