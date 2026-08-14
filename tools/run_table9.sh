#!/usr/bin/env bash
# Table IX grid: {DAPCN-SSL, SupOnly} x {UNetFormer, PyramidMamba} x {5,1,10}%
# on ResNeXt101. Runs sequentially on one GPU.
#
#   bash tools/run_table9.sh              # full 12-run grid, 5% first
#   bash tools/run_table9.sh 5            # only the 5% ratios (4 runs)
#   bash tools/run_table9.sh "5 10"       # 5% then 10%
#
# 5% runs first because every ablation in the paper (Tables IV-VIII, XI) is
# defined on OEM 5% -- it is the anchor for everything else.
#
# ~21h per run on ResNeXt101. Each run is resumable: re-running skips any
# work_dir that already holds a final checkpoint.
set -u
cd "$(dirname "$0")/.."

RATIOS=${1:-"5 1 10"}
GPU=${GPU:-0}
ITERS=${ITERS:-40000}

source /home/ubuntu/SatelliteImageSegSupevised/.venv/bin/activate
export PYTHONPATH=/home/ubuntu/S4_SelfTraining:${PYTHONPATH:-}

for R in $RATIOS; do
  for DEC in unetformer pyramidmamba; do
    for METHOD in dapcn sup; do

      if [ "$METHOD" = dapcn ]; then
        CFG=configs/daformer/ssl_oem_dapcn_${DEC}_resnext101.py
        # SSL: labeled + disjoint unlabeled pool
        SPLIT_OPTS="data.train.labeled.split=train_${R}pct_labeled.txt \
                    data.train.unlabeled.split=train_${R}pct_unlabeled.txt"
      else
        CFG=configs/daformer/sup_oem_${DEC}_resnext101.py
        # SupOnly: labeled only
        SPLIT_OPTS="data.train.split=train_${R}pct_labeled.txt"
      fi

      NAME=t9_${METHOD}_${DEC}_rx101_${R}pct
      WD=work_dirs/$NAME

      if [ -f "$WD/iter_${ITERS}.pth" ]; then
        echo "[skip] $NAME already complete"
        continue
      fi

      # The EC2 instance has been stopped mid-run before, which kills training.
      # Checkpoints land every 4000 iters, so pick up from the last one rather
      # than throwing away a day of compute.
      RESUME=""
      if [ -f "$WD/latest.pth" ]; then
        RESUME="--resume-from $WD/latest.pth"
        echo "[rsum] $NAME from $(readlink -f "$WD/latest.pth" | xargs basename)"
      fi

      echo "[run ] $NAME  ($(date '+%F %T'))"
      mkdir -p "$WD"
      python tools/train.py "$CFG" \
        --work-dir "$WD" --seed 0 --gpu-ids "$GPU" $RESUME \
        --options runner.max_iters=$ITERS $SPLIT_OPTS \
        >> "$WD/train.log" 2>&1

      if [ -f "$WD/iter_${ITERS}.pth" ]; then
        echo "[done] $NAME"
      else
        # Do not silently continue past a failure -- a missing baseline makes
        # the whole row of the table meaningless.
        echo "[FAIL] $NAME -- see $WD/train.log"; tail -5 "$WD/train.log"
      fi
    done
  done
done
echo "TABLE9_GRID_DONE"
