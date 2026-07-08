// =============================================================================
// Diagnosis workflow template — parallel audits + adversarial verify + synth.
//
// HOW TO USE: this is a TEMPLATE. Before launching:
//   1. Run compare_runs.py and probe_checkpoint.py FIRST (cheap, deterministic).
//   2. Paste their key numbers into GROUND_TRUTH below (replace the <<...>>),
//      so the agents build on established facts instead of re-deriving them.
//   3. Launch via the Workflow tool:  Workflow({scriptPath: "<this file>"})
//      (or copy the script inline). Edit + re-launch to iterate.
//
// The shape mirrors the proven DWPC diagnosis: four independent audit axes run
// in parallel, EACH is adversarially re-verified the moment it finishes
// (pipeline, no barrier), then one synthesis agent writes the ranked verdict.
// Scale the finder/verifier counts to how thorough the user asked to be.
// =============================================================================

export const meta = {
  name: 'training-regression-diagnosis',
  description: 'Diagnose why a DAPCN/DWPC/SSL run underperforms: code audit + mechanism + empirical probe + design confound, each adversarially verified',
  phases: [
    { title: 'Audit', detail: 'parallel code/mechanism/empirical/design audits' },
    { title: 'Verify', detail: 'adversarially re-check each axis' },
    { title: 'Synthesize', detail: 'ranked diagnosis + recommendations' },
  ],
}

// ---- FILL THIS IN from compare_runs.py + probe_checkpoint.py output ----------
const GROUND_TRUTH = `
ESTABLISHED FACTS (verified from logs/checkpoint — treat as given, build on them):
- Runs compared: <<BASELINE work_dir + what it is>> vs <<TREATMENT work_dir + what it is>>.
- Matched-iteration mIoU (from compare_runs.py): <<table or key rows>>.
- Completion status: <<is either run still training? matched length?>>.
- Config diff (uda block): <<the knobs that ACTUALLY differ — names the confounds>>.
- Checkpoint probe (probe_checkpoint.py): <<bank initialised? wb_count warm? rare_prior sane?>>.

ENVIRONMENT (critical):
- venv python: /home/ubuntu/SatelliteImageSegSupevised/.venv/bin/python (mmcv 1.7.2).
- Repo root: /home/ubuntu/S4_SelfTraining. Any script MUST sys.path.insert(0, repo_root)
  BEFORE importing mmseg (the venv's editable mmseg lacks DAPCN).
- DO NOT disturb a running training job: never kill processes; any model run MUST be
  CPU-only (CUDA_VISIBLE_DEVICES=""), the GPU may be occupied.

KEY FILES: mmseg/models/uda/dwpc_mixin.py, mmseg/models/utils/witness_appearance.py,
  mmseg/models/utils/witness_structure.py, mmseg/models/uda/dapcn_ssl.py (+ dapcn.py),
  DWPC_PROPOSAL.md (the spec), the dumped config in each work_dir.
`

const FINDINGS_SCHEMA = {
  type: 'object',
  properties: {
    summary: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          claim: { type: 'string', description: 'specific, falsifiable claim' },
          severity: { type: 'string', enum: ['bug', 'likely-cause', 'design-smell', 'confound', 'benign', 'info'] },
          evidence: { type: 'string', description: 'file:line, code/log quotes, or numbers' },
          mechanism: { type: 'string', description: 'how it would cause underperformance, or why benign' },
          confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
        },
        required: ['claim', 'severity', 'evidence', 'mechanism', 'confidence'],
      },
    },
  },
  required: ['summary', 'findings'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    axis: { type: 'string' },
    upheld: { type: 'array', items: { type: 'string' } },
    refuted: { type: 'array', items: { type: 'string' } },
    missed: { type: 'array', items: { type: 'string' } },
    netAssessment: { type: 'string' },
  },
  required: ['axis', 'upheld', 'refuted', 'netAssessment'],
}

phase('Audit')

const AXES = [
  {
    key: 'code-correctness',
    label: 'audit:code',
    prompt: `${GROUND_TRUTH}
AXIS: correctness of the corrector's decision logic — sign conventions, thresholds,
budget/top-K, warmup degradation. Read the corrector (dwpc_mixin.py _dwpc_correct) and
the witnesses, cross-check against the spec (DWPC_PROPOSAL.md). Be ADVERSARIAL: assume
there IS an inverted gate / wrong-scale fusion / off-by-one and try to find it. Verify:
(1) any gate raises protection in the intended direction; (2) fused scores combine on
COMPATIBLE scales (a common bug: multiplying a calibrated [0,1] prob by a structurally
capped score, then thresholding at 0.5 -> never fires); (3) budget can't be bypassed or
always-zero; (4) warmup degrades exactly to prior behaviour at start_iter; (5) corrected
labels can never inject invalid class ids. Quote exact lines.`,
  },
  {
    key: 'mechanism',
    label: 'audit:mechanism',
    prompt: `${GROUND_TRUTH}
AXIS: does the per-pixel weight / loss reweighting SUPPRESS or AMPLIFY the learning signal
vs the baseline? Find how the BASELINE pseudo_weight is computed (often a thresholded
FRACTION broadcast flat — NOT ~1.0) and the treatment's per-pixel weight formula. Determine
mean(w_i) vs the baseline scalar across plausible confidence/structure regimes. Check the CE
reduction (mean vs sum-of-weights) so you know whether effective loss strength scales with
mean(weight). State a quantitative best-estimate ratio and whether it explains the observed
drag (and at which training phase). Quote exact lines.`,
  },
  {
    key: 'empirical-probe',
    label: 'probe:forward',
    prompt: `${GROUND_TRUTH}
AXIS: EMPIRICAL forward probe — the decisive, non-speculative test. You may write+run a short
script. CONSTRAINTS: use the venv python, prefix CUDA_VISIBLE_DEVICES="" (CPU only), sys.path.insert
the repo root, do NOT kill anything. Build the model from the work_dir's dumped config, load the
checkpoint, take ONE real val image, run the EMA teacher -> ema_softmax + student feat, and call the
module's correction method (_dwpc_correct) at the relevant local_iter. REPORT NUMBERS: flip volume
(#flipped, fraction, per-target-class histogram), mean/std of the per-pixel weight w_i AND the baseline
scalar on the same batch (is the corrector inert? is weight suppressing or amplifying?), and the
score-map stats (a_map, s_map, s_cstar, F_map percentiles vs the flip threshold). If the full build is
impractical on CPU, fall back to exercising the witness functions on small synthetic inputs and label it
as synthetic. Numbers over adjectives.`,
  },
  {
    key: 'design-confound',
    label: 'audit:design',
    prompt: `${GROUND_TRUTH}
AXIS: experimental-design validity. From the config diff, state exactly what the comparison can and
cannot establish. Identify: (a) wrong baseline (does it differ by more than the one knob under test?),
(b) iteration mismatch, (c) single-seed noise, (d) whether an isolating baseline run actually exists on
disk (ls work_dirs), (e) whether the ablation configs match the preregistered matrix in the spec.
Name the ONE comparison that isolates the component under test, confirm whether its anchor is clean
(e.g. does the "off" row leave a DIFFERENT corrector on?), and recommend the minimal runs (which rows,
how many seeds, to what iter) to make the hypotheses decidable. Be a skeptical research scientist.`,
  },
]

const audited = await pipeline(
  AXES,
  (ax) => agent(ax.prompt, { label: ax.label, phase: 'Audit', schema: FINDINGS_SCHEMA })
            .then(r => ({ axis: ax.key, ...r })),
  (res, ax) => {
    if (ax.key === 'empirical-probe') return { ...res, verdict: { axis: ax.key, upheld: [], refuted: [], netAssessment: 'empirical — self-evidencing' } }
    return agent(
      `${GROUND_TRUTH}
You are an adversarial verifier for axis "${ax.key}". Another agent produced these findings:
${JSON.stringify(res, null, 2)}
Independently re-read the relevant code/configs and verify each claim. Default to skepticism: a claim
survives only if the evidence actually backs it. For any "bug"/"likely-cause", try hard to REFUTE it by
finding the line that makes it benign. Flag anything the audit MISSED. Return your verdict.`,
      { label: `verify:${ax.key}`, phase: 'Verify', schema: VERDICT_SCHEMA }
    ).then(v => ({ ...res, verdict: v }))
  }
)

phase('Synthesize')
const synthesis = await agent(
  `${GROUND_TRUTH}
You are the lead investigator. Verified audit findings across all axes:
${JSON.stringify(audited.filter(Boolean), null, 2)}

Write the definitive diagnosis for a research-scientist user. Structure:
1. HEADLINE (2-4 sentences): the single most important reason, stated plainly.
2. IS THERE A REAL BUG? yes/no/partial, citing only claims that SURVIVED verification; separate genuine
   bugs from benign-but-suspicious from pure confounds; rank by severity.
3. THE OBSERVED SYMPTOM (e.g. drag / regression): best-supported mechanism, with probe numbers.
4. WHAT THE FAIR COMPARISON SHOWS: matched-iteration, isolating-baseline reading; quantify; flag seed noise.
5. RANKED NEXT ACTIONS: concrete, with file:line for code fixes, the exact change, and expected payoff.
   Distinguish "not yet demonstrated" from "demonstrably harmful". No cheerleading. Return markdown.`,
  { label: 'synthesize', phase: 'Synthesize' }
)

return { synthesis, audited: audited.filter(Boolean) }
