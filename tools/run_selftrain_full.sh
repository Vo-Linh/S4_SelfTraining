#!/usr/bin/env bash
# SelfTrain (in-codebase FixMatch, no DAPCN) at 100% labels — the "Full" row.
# labeled = unlabeled = all 3500 (unlabeled=copy-of-labeled, matching the baseline Full convention).
set -u
cd /home/ubuntu/S4_SelfTraining
source /home/ubuntu/SatelliteImageSegSupevised/.venv/bin/activate
export PYTHONPATH=/home/ubuntu/S4_SelfTraining:${PYTHONPATH:-}
WD=work_dirs/T_selftrain_oem_mitb5_full
mkdir -p "$WD"
RESUME=""; [ -f "$WD/latest.pth" ] && RESUME="--resume-from $WD/latest.pth"
python tools/train.py configs/daformer/ssl_oem_daformer_selftrain_mitb5.py \
  --work-dir "$WD" --seed 0 --gpu-ids 0 $RESUME \
  --options runner.max_iters=40000 model.pretrained=pretrained/mit_b5.pth \
    data.train.labeled.split=train_3500_fixed.txt \
    data.train.unlabeled.split=train_3500_fixed.txt >>"$WD/train.log" 2>&1
echo "SELFTRAIN_FULL_DONE $(date)"
