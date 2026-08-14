#!/usr/bin/env bash
# Non-DAPCN queue for aws_dapcn: complete the SelfTrain (FixMatch-style, all DAPCN
# components off) row at the remaining ratios. 1/8 is done = 69.43, which beats
# SupOnly 69.17 and DAPCN 65.06 — so this row is the baseline DAPCN must actually
# clear. 1/16 and 1/4 confirm the gap is not ratio-specific.
# NO DAPCN config here; DAPCN runs last on aws_temp.
set -u
cd /home/ubuntu/S4_SelfTraining
source /home/ubuntu/SatelliteImageSegSupevised/.venv/bin/activate
export PYTHONPATH=/home/ubuntu/S4_SelfTraining:${PYTHONPATH:-}
PRE=model.pretrained=pretrained/mit_b5.pth
CFG=configs/daformer/ssl_oem_daformer_selftrain_mitb5.py

for R in 1_16 1_4; do
  WD=work_dirs/T_selftrain_oem_mitb5_$R
  [ -s "$WD/iter_40000.pth" ] && { echo "[skip] selftrain $R"; continue; }
  RESUME=""; [ -f "$WD/latest.pth" ] && RESUME="--resume-from $WD/latest.pth"
  mkdir -p "$WD"; echo "[run ] selftrain $R $(date '+%m-%d %H:%M')"
  python tools/train.py $CFG --work-dir "$WD" --seed 0 --gpu-ids 0 $RESUME \
    --options runner.max_iters=40000 $PRE \
      data.train.labeled.split=train_${R}_labeled.txt \
      data.train.unlabeled.split=train_${R}_unlabeled.txt >>"$WD/train.log" 2>&1
  echo "[done] selftrain $R -> $(grep -ohE 'mIoU: [0-9.]+' "$WD"/*.log | grep -oE '[0-9.]+' | sort -g | tail -1)"
done
echo "SELFTRAIN_ROWS_DONE $(date)"
