# DAPCN-SSL Experiments — commands & status

Paper: `STIC_semi.pdf` (DAPCN). Target: OpenEarthMap (9-class), semi-supervised.
Everything below runs on `aws_temp` (EC2 `i-0724fc3ab947e1f27`, g6e.2xlarge, 46 GB GPU).

---

## 0. Setup (every session)

```bash
ssh aws_temp
cd /home/ubuntu/S4_SelfTraining
source /home/ubuntu/SatelliteImageSegSupevised/.venv/bin/activate
export PYTHONPATH=/home/ubuntu/S4_SelfTraining:$PYTHONPATH   # REQUIRED
```

`PYTHONPATH` is not optional: the venv has an *editable* `mmseg` pointing at
`SatelliteImageSegSupevised`, which **lacks DAPCN**. Without this, the wrong
`mmseg` wins and `DAPCN_SSL` won't resolve.

Utilities:

| | |
|---|---|
| watch | `tail -f work_dirs/<name>/train.log` |
| kill | `pkill -f tools/train.py` |
| GPU | `watch nvidia-smi` |
| smoke any config | append `--no-validate --options runner.max_iters=25 log_config.interval=5` |

**The instance stops.** A stopped instance kills training; checkpoints only land
every 4000 iters, so anything before iter 4000 is lost. `nohup` survives an SSH
drop, *not* an instance stop.

---

## 1. Ready to run (all smoke-tested, 25 iters, no errors)

All 40k iters, val + ckpt every 4k. Replace `<CFG>` and run:

```bash
mkdir -p work_dirs/<CFG>
nohup python tools/train.py configs/daformer/<CFG>.py \
  --work-dir work_dirs/<CFG> --seed 0 --gpu-ids 0 \
  > work_dirs/<CFG>/train.log 2>&1 &
```

| `<CFG>` | Encoder | Decoder | Peak GPU | s/iter | ETA 40k |
|---|---|---|---|---|---|
| `ssl_oem_dapcn_unetformer_resnext101` | ResNeXt101-32x16d | UNetFormer GLA | 30.1 GB | 1.9 | ~21 h |
| `ssl_oem_dapcn_unetformer_r18` | ResNet18 | UNetFormer GLA | 5.5 GB | 0.17 | ~2 h |
| `ssl_oem_dapcn_pyramidmamba_r18` | ResNet18 | PyramidMamba | 10.3 GB | 0.16 | ~2 h |
| `ssl_oem_dapcn_daformer_mitb5` | MiT-B5 (random init) | DAFormer | 20.8 GB | 1.0 | ~11 h |

`ssl_oem_dapcn_daformer_mitb5` runs from **random init** — a baseline sanity run,
not a paper number.

The two `*_r18` configs are a **matched decoder pair**: same encoder, same
DAPCN-SSL loop, same optimizer/iters. Both derive `feature_dim=512` and
`proto_to_decoder Linear(512->64)` — the decoder is the only variable. This is
the comparison for Table XIII.

**Caveat:** all four use the legacy `train_500_fixed` / `train_3500_fixed` pair,
where the labeled images **reappear** in the unlabeled pool. Fine for an
architecture comparison (all share it), wrong for the paper protocol — use the
disjoint splits in §2.

---

## 2. Splits

Disjoint + nested labeled/unlabeled splits, generated from the 3500-image train
pool (verified: exact ratios, zero overlap, full coverage, 1% ⊂ 5% ⊂ 10%):

```bash
python tools/make_ssl_splits.py \
  --pool /home/ubuntu/data/OpenEarthMap/OpenEarthMap_flat/train_3500_fixed.txt \
  --out-dir /home/ubuntu/data/OpenEarthMap/OpenEarthMap_flat \
  --ratios 1 5 10 --seed 0
```

| ratio | labeled | unlabeled |
|---|---|---|
| 1% | 35 | 3465 |
| 5% | 175 | 3325 |
| 10% | 350 | 3150 |

Already on disk as `train_{1,5,10}pct_{labeled,unlabeled}.txt`.

---

## 3. BLOCKED — P1, the paper's headline (Table IX)

`ssl_oem_dapcn_mitb5_{1,5,10}pct` are written and **build cleanly**, splits
resolve. They need one file that does not exist:

```
pretrained/mit_b5.pth      # ImageNet-1K MiT-B5 encoder (paper §V-A)
```

Without it `init_weights()` hard-fails (`FileNotFoundError`) — it does **not**
silently random-init, so there is no risk of quietly producing junk numbers.

Fix: convert HF `nvidia/mit-b5` to DAFormer's `mix_transformer` key naming.
Then:

```bash
python tools/train.py configs/daformer/ssl_oem_dapcn_mitb5_5pct.py \
  --work-dir work_dirs/ssl_oem_dapcn_mitb5_5pct --seed 0 --gpu-ids 0
```

---

## 4. Paper experiment backlog

Protocol (§V-A): OEM + LoveDA, 512×512, ratios 1/5/10%, SegFormer MiT-B5
(ImageNet-1K), AdamW 6e-5 + poly(0.9), batch 8 = 4 labeled + 4 unlabeled
(`samples_per_gpu=4`). Metrics: per-class IoU, mIoU, mF1, Kappa.

| Pri | Table | Experiment | Setting | Status |
|---|---|---|---|---|
| **P1** | IX | DAPCN vs SupOnly + SoTA baselines, OEM | 1/5/10% | configs ready, blocked on `mit_b5.pth` |
| P1 | IX | SupOnly baseline | 1/5/10% | **config not written** (needs a supervised OEM dataset base) |
| P2 | XI | Module ablation: DA / PLC / L_aff | OEM 5% | not started |
| P2 | IV–VIII | Hyperparams: K, η, α_c, (T_s×T_w), (λ_b, λ_p) | OEM 5% | not started |
| P3 | XII | Prototype update: static / EMA-teacher / dynamic anchor | OEM 5% | not started |
| P3 | XIII | Decoder/backbone-agnostic | OEM 5% | **UNetFormer + PyramidMamba pair ready** |
| P3 | XIV | Training complexity (s/epoch, peak GPU, params) | OEM 5% | partly measured (table in §1) |
| P4 | X | Same as IX on LoveDA | 1/5/10% | LoveDA not set up |
| P4 | III | Cross-dataset OEM↔LoveDA direct transfer, 6 aligned classes | full-sup | not started |
| P4 | Fig 1-3 | Qualitative; pseudo-label correction before/after; t-SNE prototypes | — | not started |

Order: everything in IV–VIII and XI is defined **on OEM 5%**, so get 5% working
first — it is the anchor for the whole paper.

---

## 5. Notes

- `mamba_ssm 1.2.0.post1` + `causal_conv1d` are installed (PyramidMamba needs them).
- PyramidMamba: keep `anchor_after_fusion=False` (Solution 1). Its fused map is at
  **full input resolution** `(64,512,512)`, so anchoring after fusion would run EM
  clustering over B×512×512 pixels. UNetFormer's is `(64,128,128)`.
- `default_runtime.py` emails on run completion; `unset SMTP_HOST NOTIFY_RECIPIENT`
  to silence for throwaway runs.
- `max_iters=40000` is inherited, not a translation of the paper's "80 epochs"
  (the epoch→iter mapping is ambiguous under SSL). What matters for a fair table
  is that the budget is *equal across ratios* — it is.
