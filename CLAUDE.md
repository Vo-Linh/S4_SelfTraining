# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **DAFormer fork specialized for satellite/aerial-imagery semantic segmentation**. The upstream DAFormer codebase (CVPR 2022, MMSegmentation 0.16.0 based) has been extended with a custom **DAPCN** (Dynamic-Anchor Prototype Clustering Network) family that supports both Unsupervised Domain Adaptation (UDA) and Semi-Supervised Learning (SSL). The original `dacs.py` UDA module and ImageNet Feature Distance have been removed because they are inappropriate for overhead satellite imagery.

On top of DAPCN sits **DWPC** (Dual-Witness Pseudo-label Correction) — a pseudo-label corrector that cross-examines every teacher pseudo-label with two independent witnesses (an *appearance* witness over class-bound prototypes, and a *structure* witness over label-map topology), flipping a pixel to a rare class only when both agree and protecting it whenever either vouches. See `DWPC_PROPOSAL.md` for the full method/experiment design.

See `DISCUSSION_SUMMARY.md` for an in-depth design document — it is more authoritative than the upstream `README.md` for everything DAPCN/SSL related.

## Environment Setup

The canonical setup is:

```shell
python -m venv ~/venv/daformer && source ~/venv/daformer/bin/activate
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7   # must be installed after the rest
```

**On this machine the actual working interpreter is**
`/home/ubuntu/SatelliteImageSegSupevised/.venv/bin/python` (mmcv 1.7.2, torch 2.0.1+cu118). The `~/venv/daformer` path above does **not** exist here. Use the Satellite venv to run anything in this repo.

> Caveat: that venv has its own editable-installed `mmseg` (pointing at the `SatelliteImageSegSupevised` repo) which **lacks** DAPCN. Entry-point scripts must put this repo's root on `sys.path` *before* importing `mmseg` so the local package (with `DAPCN`/`DAPCN_SSL` registered) wins. `tools/train.py`, `tools/dwpc_eval.py`, and `tools/rare_class_audit.py` already do this `sys.path.insert(0, repo_root)`.

Pretrained MiT-B5 encoder weights belong in `pretrained/mit_b5.pth` (download link in upstream README).

## Common Commands

```shell
# Single training run from a hand-written config (no auto-generated wrapper)
python tools/train.py configs/daformer/satellite_ssl_dapcn_daformer_mitb5.py

# Same, but with auto-generated work_dir, timestamped name, git rev capture
python run_experiments.py --config configs/daformer/satellite_ssl_dapcn_daformer_mitb5.py

# Batch experiments defined in experiments.py by integer ID
python run_experiments.py --exp <ID>

# Evaluation on a finished work_dir (expects work_dir/latest.pth + matching .json)
sh test.sh work_dirs/<exp_dir>

# Single-config test with custom flags
python -m tools.test path/to/config path/to/ckpt --eval mIoU

# Ablation suite (groups: all|baseline|component|hyperparam|boundary)
bash tools/run_ablation_ssl.sh <group> <gpu_id> <seed>
bash tools/run_ablation_proto_lambda.sh <gpu_id> <seed>
python tools/collect_ablation_results.py   # aggregate -> console + CSV

# DWPC: full A0-A7 matrix (SSL + UDA) over seeds -> work_dirs/dwpc/
bash tools/run_dwpc_all.sh [GPU] [MODES] [ROWS] [SEEDS]
bash tools/run_dwpc_all.sh 0 ssl "2 4 5"     # MVE: Witness-A / Witness-B / full

# DWPC eval: stratified rare/head IoU (H2) + flip precision (H5)
python tools/dwpc_eval.py iou <config> <ckpt> --rare 0 1 6
python tools/dwpc_eval.py flip --dump-dir <work_dir>/dwpc_dumps

# Lint (configured via .pre-commit-config.yaml — flake8/isort/yapf)
pre-commit run --all-files
```

There is no Python unit-test suite in this repo; "testing" means evaluating a checkpoint via `tools/test.py` or `test.sh`.

## Architecture: How the Pieces Fit

### Config-driven, MMSegmentation 0.16.0 style

Every training run is fully defined by a config that inherits a list of `_base_` files. The four canonical bases used by SSL configs are:

- `configs/_base_/models/<arch>.py` — model architecture (e.g. `daformer_sepaspp_mitb5.py`, `deeplabv3plus_r101-d8.py`)
- `configs/_base_/datasets/<dataset>.py` — pipelines + dataset wrapper (e.g. `ssl_satellite_512x512.py`)
- `configs/_base_/ssl/dapcn_ssl.py` (or `_base_/uda/dapcn.py`) — DAPCN training-loop hyperparameters
- `configs/_base_/schedules/{adamw,poly10warm}.py` — optimizer + LR schedule

Ablation configs in `configs/daformer/ablation_ssl/` exist only to override one knob each (e.g. `A1_no_boundary.py` sets `boundary_lambda=0`). To change a default everywhere, edit the base file in `_base_/ssl/` — to change only one experiment, override in the leaf config.

`tools/train.py` calls `build_train_model` (custom wrapper around `build_segmentor`) which delegates to `mmseg/models/builder.py`. When the config contains a top-level `uda = dict(type=...)`, the segmentor is built as the named UDA/SSL module from `mmseg/models/uda/` rather than as a plain `EncoderDecoder`.

### DAPCN module family (`mmseg/models/uda/`)

All training-loop variants inherit from `UDADecorator` (student + EMA teacher wrapper):

- `dapcn.py` → `DAPCN` — cross-domain UDA (source labeled, target unlabeled). For DWPC it now also builds a `PrototypeMemory` (Witness A bank) updated from source GT + target-confident pixels.
- `dapcn_ssl.py` → `DAPCN_SSL` — single-domain SSL (labeled/unlabeled split). Adds pseudo-label warmup + prototype-based pseudo-label correction.
- `dapcn_ssl_dlv3plus.py` → `DAPCN_SSL_DLV3Plus` — SSL variant using a DeepLabV3+ / ResNet-101 backbone instead of MiT-B5.
- `dynamic_anchor.py` → `DynamicAnchorModule` — `nn.Parameter`-backed prototype bank with differentiable EM refinement (E-step softmax assignment → M-step weighted mean), MLP quality gate, optional EMA smoothing.
- `dwpc_mixin.py` → `DWPCMixin` — shared DWPC orchestration mixed into both `DAPCN` and `DAPCN_SSL`. `_dwpc_init` reads the `dwpc=dict(...)` config block and registers the Witness-B EMA buffers (`wb_*`); `_dwpc_correct` runs both witnesses, applies the dual-witness flip (budgeted top-K) and the protective per-pixel weight `w_i`, with an `omega(t)` warmup ramp that degrades exactly to the prior scalar-weight behaviour at `dwpc_start_iter`. Witness A reuses `PrototypeMemory` (no new optimization); witnesses are `@torch.no_grad()` so there are no new learnable params.

**Six-step `forward_train` loop** (identical structure across `DAPCN` and `DAPCN_SSL`):

1. Supervised CE on labeled/source image (saves `src_feat`, calls `.backward(retain_graph=True)`).
2. DAPCN loss (boundary + prototype) on labeled features.
3. EMA teacher inference on unlabeled/target → softmax → pseudo-label.
   - **3b — legacy soft correction:** if `proto_correction` and `local_iter >= proto_correction_start_iter`, blend `ema_softmax` with `f_theta(PT) · a` (prototypes projected through `decode_head.conv_seg`).
   - **3b — DWPC (preferred):** if `dwpc.enabled` and `local_iter >= dwpc_start_iter`, `_dwpc_correct` produces a *corrected hard label* (dual-witness flip) and a *genuine per-pixel* `pseudo_weight` (replacing the scalar broadcast). The two 3b paths are mutually exclusive — DWPC configs set `proto_correction=False`. `pseudo_weight` is already a `(B,H,W)` tensor that survives ClassMix unchanged, so no ClassMix code changes were needed.
4. ClassMix: paste labeled patches into unlabeled image.
5. Supervised CE on mixed image with `seg_weight = pseudo_weight` (saves `tgt_feat`, `retain_graph=True`).
6. DAPCN loss on mixed features.

Each step calls `.backward()` separately, accumulating into one optimizer step. `retain_graph=True` is required on steps 1 and 5 because steps 2 and 6 reuse the feature graph. See `DISCUSSION_SUMMARY.md` §11 for the full gradient accumulation table.

### Custom loss functions (`mmseg/models/losses/`)

- `affinity_boundary_loss.py` — 4-neighbour pairwise affinity (temperature-scaled cosine sim → BCE vs GT discontinuity map). Orientation-invariant; the default boundary mode.
- `dapg_loss.py` — composite `L_intra + L_inter + L_quality` over prototypes from `DynamicAnchorModule`.

### DWPC witness modules (`mmseg/models/utils/`)

Both are `@torch.no_grad()` helpers consumed by `DWPCMixin` — they inform the flip decision and the per-pixel weight only, so neither carries a gradient.

- `witness_appearance.py` → `witness_appearance(...)` — Witness A. GMM prototype-density log-likelihood ratio over the class-bound `PrototypeMemory` components, with an inverse-frequency rare prior (`build_rare_prior_logits`), asymmetric flip-to-rare (or `symmetric=True` to reproduce ProDA), robust per-image standardization, and a cold-start guard (`proto_memory.is_initialised()`). Returns `a_map` (rare-evidence in [0,1]) and `cstar_map` (best rare candidate, -1 if none).
- `witness_structure.py` → `witness_structure(...)` — Witness B. Non-differentiable area / connectivity / co-occurrence / containment plausibility on the integer label map, with target-adaptive EMA buffers (`wb_*`) updated from the target pseudo-label stream and per-class warmup. Connected components are computed with iterative max-pool label propagation (approximate, GPU-friendly).

### Datasets (`mmseg/datasets/`)

- `uda_dataset.py` — `UDADataset` pairs one source + one target sample per `__getitem__`, supports Rare Class Sampling (RCS).
- `ssl_dataset.py` — `SSLDataset` partitions a **single** domain via `splits/labeled.txt`; same five output keys as `UDADataset` (`img`, `img_metas`, `gt_semantic_seg`, `target_img`, `target_img_metas`) so the training loop is unchanged.
- `builder.py` — factory routes `type='SSLDataset'` / `'UDADataset'` correctly.

The unlabeled pool intentionally includes the labeled images — they still benefit from consistency regularization through pseudo-labels and ClassMix.

### Two-solution architecture for DynamicAnchor placement

The DynamicAnchorModule can operate either on raw encoder features (e.g. 2048-d for R101) or on the decoder's fused space (e.g. 512-d). This is controlled by `anchor_after_fusion`:

- `anchor_after_fusion=False` (Solution 1, default) — anchors on encoder features. A learned `proto_to_decoder = Linear(encoder_dim → decoder_dim)` projects prototypes into the decoder's space before they pass through `conv_seg` for pseudo-label correction.
- `anchor_after_fusion=True` (Solution 2) — anchors directly on fused decoder features; no projection layer.

See `configs/daformer/ssl_s1_deeplabv3plus_r101.py` and `configs/daformer/ssl_s2_after_fusion.py` for examples.

## Important Configuration Conventions

- **`paramwise_cfg.custom_keys`** is used heavily. `dynamic_anchor.prototypes` and `proto_to_decoder` must run with `lr_mult=10.0, decay_mult=0.0` (random init + EM gradient attenuation + cluster-centre semantics). `head` also gets `lr_mult=10.0`. Drop these settings and DynAnchor will train extremely slowly or collapse.
- **`num_classes`** is set per-config at the leaf level (e.g. `model = dict(decode_head=dict(num_classes=7))`); satellite taxonomy is project-specific.
- **`pseudo_weight_ignore_top/bottom = 0`** is correct for overhead satellite imagery (no hood/sky to mask). The original Cityscapes configs use nonzero values — do not copy blindly.
- **`boundary_loss_mode='affinity'`** is the default for satellite. `'sobel'`, `'laplacian'`, `'hybrid'` exist for ablation Group C only.
- **DWPC is configured via a `uda=dict(dwpc=dict(...))` block** (parsed by `DWPCMixin._dwpc_init`). Key toggles: `enabled`, `start_iter`, `rare_class_ids` (OEM bottom-3 = `[0,1,6]`), `witness_a_enabled`/`witness_a_symmetric`/`witness_a_rare_prior_gamma`, `witness_b_enabled`/`target_adaptive`, `rare_gate`, `flip_veto`, `tau_flip`, `flip_budget`, `beta_s`. **Always set `proto_correction=False` when `dwpc.enabled=True`** (running both 3b correctors is incoherent). On SSL, Witness A needs target coverage in the bank, so DWPC SSL configs also set `contrastive_lambda>0` + `contrastive_use_teacher=True`.
- The DWPC ablation matrix lives in `configs/daformer/{ssl,uda}_oem_dwpc_a{0..7}_daformer_mitb5.py` (one toggle-set per row; SSL rows inherit `ssl_oem_dapcn_daformer_mitb5_v2.py`, UDA rows inherit `uda_oem_dapcn_daformer_mitb5.py`). `ssl_oem_dwpc_full_daformer_mitb5.py` is the self-contained "everything on" config (== row A5, all knobs spelled out).

## Dataset Layout (Satellite)

```
data/satellite/
├── images/{train,val}/
├── labels/{train,val}/         # uint8 class IDs, 255 = ignore
├── splits/labeled.txt          # basenames (no suffix) of labeled training images
├── sample_class_stats.json     # per-image class pixel counts (RCS)
└── samples_with_class.json     # class -> [(file, pixel_count), ...] (RCS)
```

The two JSON files must be generated before training — see `DISCUSSION_SUMMARY.md` §4.3 for the generator script. RCS will silently disable itself if they're missing.

## Where to Look First

| Need to understand... | Read |
|-----------------------|------|
| The full SSL training pipeline (6 steps, gradient flow, warmup) | `mmseg/models/uda/dapcn_ssl.py` + `DISCUSSION_SUMMARY.md` §3–4, §11 |
| Prototype correction math (`p^c_j = Σ f_θ(PT_i) · a_ij`) | `_correct_pseudo_labels()` in `dapcn_ssl.py` + `DISCUSSION_SUMMARY.md` §8 |
| DWPC dual-witness flip + weight | `_dwpc_correct()` in `dwpc_mixin.py` + `DWPC_PROPOSAL.md` §4 |
| DWPC witness math (appearance / structure) | `witness_appearance.py`, `witness_structure.py` + `DWPC_PROPOSAL.md` §4.2–4.3 |
| The DWPC ablation matrix (A0–A7) | `DWPC_PROPOSAL.md` §7 + `configs/daformer/{ssl,uda}_oem_dwpc_a*.py` |
| Adding a new ablation | Existing files under `configs/daformer/ablation_ssl/` (each overrides one knob) |
| The annotated reference config | `configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0_dapcn.py` |
| MMSegmentation conventions | https://mmsegmentation.readthedocs.io/en/v0.16.0/ — this codebase is locked to that version |
