# Confounds, pitfalls & known failure modes

Read this when triaging a training-run regression in this repo. It is the
checklist that turns "the new thing looks worse" into a defensible verdict.

## Table of contents
1. The confound checklist (run this before any code reading)
2. Repo-specific facts (venv, paths, log format, taxonomy)
3. Catalogue of known DWPC/DAPCN failure modes
4. The empirical forward-probe recipe (when buffers aren't enough)

---

## 1. The confound checklist

A regression is only real after **all** of these are cleared. Most "regressions"
die at step 1 or 2.

- **Iteration mismatch.** Is the treatment run *finished*? A run read at iter
  20k against a 40k baseline understates it. `compare_runs.py` reports
  `COMPLETE` vs `INCOMPLETE (cur/max)` and `[TRAIN PROCESS ALIVE]`. Compare only
  at matched epochs. Re-check after the run finishes — early leads can reverse
  (DWPC led on rare classes at 20k, then *lost* them by 40k).
- **Wrong baseline / conflated interventions.** Does the baseline differ from the
  treatment by *more than the one knob under test*? `compare_runs.py`'s CONFIG
  DIFF lists every differing `uda` key. If the baseline is vanilla self-training
  (`boundary_lambda=0, proto_lambda=0, contrastive_lambda=0, dynamic_anchor=None`)
  but the treatment is full-stack-plus-DWPC, the delta measures *five* things, not
  DWPC. The isolating comparison is **A5 vs A0** (same DAPCN stack, only the
  `dwpc` block toggled).
- **The "off" anchor isn't clean.** Even A0 may leave a *different* corrector on:
  A0 resolves to `proto_correction=True` (legacy soft corrector) while DWPC rows
  set it `False`. So A5−A0 measures (DWPC) vs (legacy correction), not vs
  (nothing). For a true zero-point, add an A0′ with **both** `dwpc.enabled=False`
  **and** `proto_correction=False`.
- **Single-seed noise.** A mIoU delta inside ~±0.01 is noise. A per-class IoU on a
  rare class that oscillates across validations (e.g. class_1: 0.00→0.05→0.15→0.13)
  is not a credible "win/loss" even directionally. Claiming a per-class effect
  needs ≥3 seeds.
- **No runtime observability.** DWPC logs no flip counts or `w_i` stats, so an
  inert corrector looks identical to an active one in the loss curves. If the
  mechanism's effect can't be seen in the logs, **probe it** (§4) before
  attributing anything to it.
- **Config vs spec drift.** The on-disk ablation configs may diverge from the
  preregistered matrix in `DWPC_PROPOSAL.md` §7. Verify by *resolving* the config
  (MMCV), not by reading the leaf file. Known drift: A4 has `witness_a_enabled=True`
  though the spec lists A4 as Witness-A-OFF; A3→A4 toggles two knobs at once.

## 2. Repo-specific facts

- **Working venv:** `/home/ubuntu/SatelliteImageSegSupevised/.venv/bin/python`
  (mmcv 1.7.2, torch 2.0.1+cu118). CLAUDE.md's `~/venv/daformer` does **not**
  exist here.
- **Import precedence:** entry scripts must `sys.path.insert(0, repo_root)` before
  importing `mmseg`, else the venv's editable `mmseg` (no DAPCN) wins.
- **GPU etiquette:** training may be live on the single L4 (~20/24 GB). Any probe
  must be CPU-only (`CUDA_VISIBLE_DEVICES=""`). Never kill processes.
- **Log format:** `work_dir/*.log.json` is JSON-lines; `mode:"val"` rows carry
  `mIoU`, `aAcc`, `IoU.class_N`; `mode:"train"` rows carry `iter` + losses.
  Resumed runs produce multiple `*.log.json` files — merge them. Validation fires
  every `evaluation.interval` iters; align by `epoch` (logged, monotonic).
- **Resolved config:** the baseline-style `*.json` meta dump has a line with the
  full `uda` dict; DWPC runs instead dump a fully-resolved `*.py` (parse via
  `mmcv.Config.fromfile`). `compare_runs.py` handles both.
- **OpenEarthMap taxonomy:** 9 classes. Rare set (bottom-3 by frequency) =
  **{0, 1, 6}**. Class freqs ≈ [0.0065, 0.0149, 0.225, 0.160, 0.067, 0.200,
  0.032, 0.138, 0.156].
- **Where runs live:** `work_dirs/local-ssl_oem/` (vanilla baselines),
  `work_dirs/dwpc/` (DWPC rows). `bash tools/run_dwpc_all.sh` dispatches the
  A0–A7 matrix.

## 3. Catalogue of known DWPC/DAPCN failure modes

These were found by the first full diagnosis; check them first on any DWPC run.

- **Flip saturation (zero flips).** `F_map = a_map * s_cstar` fuses a calibrated
  appearance prob (`a_map`, reaches ~1) with a structural co-occurrence score
  (`s_cstar`) that is **structurally capped ≈0.10–0.23 for rare classes** (because
  `wb_cooc` rows are diagonal-dominant — a rare class is rarely adjacent to itself
  in the teacher map). The product can never clear `tau_flip=0.5`, so the
  dual-witness flip — DWPC's headline mechanism — **never fires**. Symptom in
  `probe_checkpoint.py`: `wb_cooc` diagonals 0.6–0.9. Fix options: conjunctive
  thresholds (`a_map>τ_a & s_cstar>τ_s` with per-class τ_s from `wb_cooc` rows),
  normalise `s_cstar` by its achievable max, or drop `tau_flip` below ~0.12.
- **Weight misconception (it amplifies, not suppresses).** The baseline scalar
  `pseudo_weight` is the *fraction* of pixels above `pseudo_threshold=0.968`,
  broadcast flat — small while the teacher is weak, NOT ~1.0. DWPC's
  `w_i = σ(β_s·s_map)·[1−(1−a)(1−r_conf)]` (β_s=5) has a floor of `0.5·r_conf`
  and uses full continuous confidence, so mean(w_i) is typically **2–350× the
  scalar** early/mid. The CE uses `reduction='mean'` (no sum-of-weights renorm),
  so effective loss strength ∝ mean(w_i): DWPC trains on *more* pseudo-label loss
  mass, plausibly driving early-training drag and head-class regression — the
  opposite of "weight suppresses signal".
- **Lopsided rare prior.** `dwpc_rare_prior_logits` with `gamma=1.0` over-boosts
  the rarest class (e.g. [5.03, 4.21, …, 3.43]); `cstar` collapses to that single
  class for ~every pixel. `probe_checkpoint.py` flags spreads >1.5.
- **proto_inter pinned / quality≈0.** `proto_loss_inter` saturated at the margin
  and `proto_loss_quality≈0` suggests the DAPG prototype stack is partly inert —
  tangential to DWPC but relevant when asking "why is full-stack slower".
- **Mutually-exclusive correctors not enforced.** `proto_correction` and
  `dwpc.enabled` are independent `if` blocks (`dapcn_ssl.py` ~831/850); a config
  setting both silently double-corrects. There is no assert.

## 4. The empirical forward-probe recipe

When buffers (`probe_checkpoint.py`) show the witnesses are *warm* but the run is
still flat, the question is whether the corrector actually changes labels/weights.
Run this inside the diagnosis workflow's empirical-probe agent (it has code
context); keep it CPU-only and non-intrusive:

1. `CUDA_VISIBLE_DEVICES=""`, `sys.path.insert(0, repo_root)`.
2. `Config.fromfile(work_dir/<dumped>.py)`, `build_train_model`, load checkpoint
   (`map_location='cpu'`, `strict=False` — expect 0 missing/unexpected).
3. One real val image → EMA-teacher forward → `ema_softmax` + student feat.
4. Call `model._dwpc_correct(ema_softmax, student_feat, local_iter=<ckpt iter>,
   scalar_weight=<baseline fraction>)`.
5. Report: `(corrected_label != pseudo_label).sum()` (flip count + per-class
   histogram), `w_i` mean/std vs the scalar, and `F_map` percentiles vs `tau_eff`.

A flip count of 0 with warm witnesses ⇒ the flip math is saturated (§3), not a
coverage problem.
