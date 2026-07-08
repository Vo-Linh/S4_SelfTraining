---
name: diagnose-training-regression
description: >-
  This skill should be used when the user reports that a DAPCN / DWPC / SSL /
  UDA training run "underperforms", "is lower than baseline", "doesn't match
  expectation", "regressed", "isn't working", or wants to compare two runs in
  work_dirs/ and find out WHY one is worse. Trigger it for phrases like "dwpc is
  worse than baseline", "why is my run lower", "compare full_s0 to the
  baseline", "the ablation didn't help", "performance dropped after I added X",
  or any request to debug/triage a segmentation training experiment in this
  repo. Use it even when the user just pastes two work_dir paths and asks what
  happened. Prefer this over ad-hoc log grepping — it removes the iteration and
  baseline confounds mechanically before any conclusion.
version: 0.1.0
---

# Diagnose a training-run regression (DAPCN / DWPC / SSL)

## Purpose

Turn "the new thing looks worse" into a defensible, mechanism-level verdict for
this DAFormer/DAPCN segmentation repo. The recurring trap is concluding from a
final-number-vs-final-number comparison when the runs differ in how long they
trained and in which components are enabled — and when the new component is
silently inert so the loss curves can't reveal it. This skill enforces the
order that avoids that trap: **establish ground truth → triage confounds →
audit + probe in parallel → synthesize**, leaning on two deterministic scripts
and an adversarially-verified workflow rather than re-deriving everything by
hand each time.

Why this matters: the first DWPC investigation found that the headline gap was
*mostly missing training + a wrong baseline*, and that underneath it the flip
mechanism was producing **zero flips** — a fact invisible in the logs and
discoverable only by probing the checkpoint. Skipping the ground-truth/confound
steps would have chased a phantom; skipping the probe would have missed the real
bug.

## The method

Work the four phases in order. Do not read corrector code before Phase 1–2 —
most "regressions" are resolved as confounds and never reach a code audit.

### Phase 1 — Establish ground truth (always, cheap)

Run the comparison script under the working venv:

```
/home/ubuntu/SatelliteImageSegSupevised/.venv/bin/python \
  .claude/skills/diagnose-training-regression/scripts/compare_runs.py \
  <baseline_work_dir> <treatment_work_dir> --rare 0 1 6 --labels baseline dwpc
```

It reports, per run: COMPLETE vs INCOMPLETE and whether a train process is still
alive; the **matched-epoch** mIoU curve (Δ aligned by epoch, not by "whatever
each reached"); per-class IoU with rare classes flagged and rare/head means; and
a **CONFIG DIFF** of the resolved `uda` blocks. Read the matched-epoch table and
the config diff before forming any hypothesis.

### Phase 2 — Triage confounds

Walk the checklist in `references/confounds-and-pitfalls.md` §1. The four that
kill most "regressions": (a) iteration mismatch — is the treatment finished?;
(b) wrong baseline — does the config diff show more than the one knob under test
differing?; (c) single-seed noise — deltas inside ~±0.01 mIoU, or oscillating
rare-class IoU, are noise; (d) no observability — if the component's effect
can't be seen in the logs, it must be probed (Phase 3) before being credited or
blamed. State explicitly what the available comparison *can* and *cannot*
establish. If the comparison is confounded, the primary recommendation is often
to run the isolating baseline (e.g. A0′ with `dwpc.enabled=False` AND
`proto_correction=False`), not to change code.

### Phase 3 — Audit + probe in parallel

Two complementary moves; do both.

**Deterministic checkpoint probe** (fast, CPU-only, safe alongside live
training):

```
CUDA_VISIBLE_DEVICES="" /home/ubuntu/SatelliteImageSegSupevised/.venv/bin/python \
  .claude/skills/diagnose-training-regression/scripts/probe_checkpoint.py \
  <work_dir>/latest.pth --rare 0 1 6 --min-count 5
```

It reports whether Witness A's `PrototypeMemory` bank is initialised (cold ⇒
Witness A inert), whether Witness B's `wb_*` buffers are warmed per class,
whether `dwpc_rare_prior_logits` boosts only the rare ids (and flags a lopsided
prior), and scans for NaN/Inf. Diagonal-dominant `wb_cooc` (0.6–0.9) is the
fingerprint of the flip-saturation bug (see reference §3).

**Fan-out audit workflow** (when a code-level verdict is needed). First paste
the Phase-1/Phase-3 numbers into the `GROUND_TRUTH` block of
`scripts/diagnose_workflow.js`, then launch it with the Workflow tool:
`Workflow({scriptPath: ".claude/skills/diagnose-training-regression/scripts/diagnose_workflow.js"})`.
It runs four independent audit axes in parallel — **code-correctness** (sign /
scale / budget / warmup), **mechanism** (does the per-pixel weight suppress or
amplify the signal?), **empirical-probe** (build the model CPU-only, run
`_dwpc_correct` on a real image, count flips + measure `w_i` vs the scalar), and
**design-confound** — and adversarially re-verifies each axis the moment it
finishes before a synthesis agent writes the ranked verdict. The adversarial
verify step is load-bearing: the first run's audits made two sign-claim errors
that only the verifiers caught.

### Phase 4 — Synthesize

Produce the verdict in this structure (the synthesis agent already does, but
apply it for inline diagnoses too):

1. **Headline** — the single most important reason, plainly.
2. **Is there a real bug?** yes/no/partial, citing only claims that *survived*
   verification; separate genuine bugs from benign-but-suspicious from pure
   confounds; rank by severity.
3. **The symptom's mechanism** (drag/regression), with probe numbers.
4. **What the fair comparison shows** — matched-iteration, isolating-baseline;
   quantify; flag seed noise.
5. **Ranked next actions** — concrete, with `file:line` and the exact change for
   code fixes, and the expected payoff. Distinguish "not yet demonstrated" from
   "demonstrably harmful". No cheerleading — the user wants skeptical analysis.

## Scaling effort

For a quick "is this even a regression?" check, Phase 1 + Phase 2 alone often
settle it. For "is there a bug / write the paper-grade verdict", run the full
workflow with the adversarial-verify pass. Match the finder/verifier depth to
how thorough the user asks to be.

## Resources

### Scripts (`scripts/`)
- **`compare_runs.py`** — iteration-aligned metric comparison + liveness +
  resolved-config diff across runs. Stdlib only (mmcv used only to parse a dumped
  `.py` config). Start here, every time.
- **`probe_checkpoint.py`** — CPU-only checkpoint buffer inspection: bank/witness
  warmth, rare-prior sanity, NaN/Inf scan. The deterministic 80% of "is the
  corrector inert?".
- **`diagnose_workflow.js`** — the parallel-audit + adversarial-verify + synthesize
  Workflow template. Fill its `GROUND_TRUTH` block from the scripts, then launch.

### References (`references/`)
- **`confounds-and-pitfalls.md`** — the confound checklist (§1), repo facts (venv,
  paths, log format, taxonomy — §2), the catalogue of known DWPC failure modes
  (flip saturation, weight misconception, lopsided prior, unenforced corrector
  exclusivity — §3), and the empirical forward-probe recipe (§4). Read §1 in
  Phase 2; consult §3 whenever a DWPC run is involved.
