#!/usr/bin/env bash
# Non-DAPCN experiment queue for aws_dapcn (L40S 45GB).
# DAPCN configs are deliberately NOT here — those run last, on aws_temp.
#   tmux new -As nondapcn 'bash tools/run_nondapcn_queue.sh'
set -u
cd /home/ubuntu/S4_SelfTraining
source /home/ubuntu/SatelliteImageSegSupevised/.venv/bin/activate
export PYTHONPATH=/home/ubuntu/S4_SelfTraining:${PYTHONPATH:-}

PRE=model.pretrained=pretrained/mit_b5.pth

run(){ NAME=$1; shift; WD=work_dirs/$NAME
  [ -f "$WD/iter_40000.pth" ] && { echo "[skip] $NAME (already at 40k)"; return; }
  mkdir -p "$WD"
  R=""; [ -f "$WD/latest.pth" ] && R="--resume-from $WD/latest.pth"
  echo "[run ] $NAME  $(date '+%m-%d %H:%M')"
  python tools/train.py "$@" --work-dir "$WD" --seed 0 --gpu-ids 0 $R >>"$WD/train.log" 2>&1
  echo "[done] $NAME  $(date '+%m-%d %H:%M')  best=$(grep -ohE 'mIoU: [0-9.]+' "$WD"/*.log 2>/dev/null | grep -oE '[0-9.]+' | sort -g | tail -1)"
}

# 1. FixMatch-style SSL baseline @ 1/8 — the diagnostic row.
#    Same codebase/schedule as DAPCN 1/8 (0.6506), all DAPCN components off.
#    Tells us whether DAPCN's deficit vs SupOnly (0.6917) is the prototypes
#    or the self-training path as a whole.
run T_selftrain_oem_mitb5_1_8 configs/daformer/ssl_oem_daformer_selftrain_mitb5.py \
  --options runner.max_iters=40000 $PRE \
    data.train.labeled.split=train_1_8_labeled.txt \
    data.train.unlabeled.split=train_1_8_unlabeled.txt

# 2. REMOVED 2026-07-29: T_sup_oem_mitb5_1_4 is already running on aws_temp via
#    tools/run_native_rows.sh (for R in 1_8 1_16 1_4; do run_sup $R). Running it here
#    too would duplicate ~7 h of GPU for an identical result. Do not re-add.

echo "NON-DAPCN QUEUE DONE $(date)"
