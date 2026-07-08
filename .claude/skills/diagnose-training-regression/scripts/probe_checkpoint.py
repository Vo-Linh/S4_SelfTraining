#!/usr/bin/env python3
"""CPU-only probe of a DAPCN/DWPC checkpoint's buffers.

When a self-training run underperforms, the decisive question is usually
mechanistic: is the corrector actually *doing* anything, or is it silently
inert? Training logs rarely answer this (DWPC logs no flip/weight stats), so
this script opens the checkpoint and inspects the buffers that govern the
correctors directly — no GPU, no training disruption, no model build needed.

It answers, from buffers alone:
  - PrototypeMemory (Witness A bank): initialised? per-class norms + update
    counts (cold-start => Witness A is inert regardless of config).
  - Witness B EMA buffers (wb_*): per-class warmup counts vs min_count;
    co-occurrence matrix sanity; any NaN/Inf.
  - dwpc_rare_prior_logits: which classes are boosted (should be ONLY the
    rare ids) and by how much (a lopsided prior collapses the flip candidate).
  - generic scan: any buffer/param that is all-zero, NaN, or Inf.

This is the deterministic 80%. The full empirical test — building the model
and calling ``_dwpc_correct`` on a real image to measure flip volume and
mean(w_i) vs the scalar baseline — is inherently module-specific; run that
inside the diagnosis workflow's empirical-probe agent (see
references/diagnosis-workflow.md), which has full code context.

Usage:
  CUDA_VISIBLE_DEVICES="" \
  /home/ubuntu/SatelliteImageSegSupevised/.venv/bin/python probe_checkpoint.py \
      <work_dir>/iter_XXXX.pth [--rare 0 1 6] [--min-count 5]

Forces CPU; only depends on torch.
"""
import argparse
import os

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')  # never grab the busy GPU

import torch  # noqa: E402


def _tensor_health(t):
    nan = bool(torch.isnan(t).any())
    inf = bool(torch.isinf(t).any())
    allzero = bool((t == 0).all())
    return nan, inf, allzero


def find_keys(sd, *substrings):
    out = []
    for k in sd:
        if all(s in k for s in substrings):
            out.append(k)
    return out


def report_proto_memory(sd, rare):
    keys = [k for k in sd if 'proto_memory' in k]
    if not keys:
        print('  (no proto_memory buffers — Witness A bank absent in this '
              'checkpoint)')
        return
    print('  proto_memory keys:', keys)
    # prototypes: usually (C*K, D) or (C, K, D); counts: (C*K,) or (C,)
    proto_k = next((k for k in keys if 'proto' in k.split('.')[-1]
                    or k.endswith('prototypes')), None)
    count_k = next((k for k in keys if 'count' in k), None)
    if proto_k is not None:
        p = sd[proto_k].float()
        flat = p.reshape(-1, p.shape[-1])
        norms = flat.norm(dim=1)
        init = bool((norms > 1e-6).any())
        print(f'    {proto_k}: shape {tuple(p.shape)}  '
              f'initialised={init}')
        print('    per-component L2 norm:',
              [round(float(x), 3) for x in norms[:16]],
              ('...' if norms.numel() > 16 else ''))
        nan, inf, _ = _tensor_health(p)
        if nan or inf:
            print(f'    !! NaN={nan} Inf={inf}')
    if count_k is not None:
        c = sd[count_k].float().reshape(-1)
        print(f'    {count_k}:', [int(x) for x in c])
        cold = [i for i, x in enumerate(c) if x == 0]
        if cold:
            print(f'    !! classes with ZERO updates (cold): {cold} '
                  f'-> Witness A inert for these')
        rare_cold = [i for i in cold if i in rare]
        if rare_cold:
            print(f'    !! RARE classes never updated: {rare_cold} '
                  f'(Witness A cannot recognise them)')


def report_witness_b(sd, rare, min_count):
    wb = {k: sd[k] for k in sd if '.wb_' in k or k.startswith('wb_')
          or k.endswith('wb_count')}
    # be permissive about prefix
    wb = {k: sd[k] for k in sd if 'wb_' in k}
    if not wb:
        print('  (no wb_* buffers — Witness B disabled or not registered)')
        return
    count_k = next((k for k in wb if 'wb_count' in k), None)
    if count_k is not None:
        c = sd[count_k].float().reshape(-1)
        print(f'  {count_k}:', [int(x) for x in c])
        warm = [i for i, x in enumerate(c) if x >= min_count]
        cold = [i for i, x in enumerate(c) if x < min_count]
        print(f'    warmed (>= {min_count}): {warm}')
        if cold:
            print(f'    !! still cold (< {min_count}): {cold} '
                  f'-> Witness B inert (terms forced to 1.0) for these')
    for k in sorted(wb):
        if k == count_k:
            continue
        t = sd[k].float()
        nan, inf, allzero = _tensor_health(t)
        flag = ''
        if nan or inf:
            flag = f'  !! NaN={nan} Inf={inf}'
        elif allzero:
            flag = '  !! all-zero (uninitialised?)'
        if t.dim() == 2 and t.shape[0] == t.shape[1]:
            rowsum = t.sum(dim=1)
            print(f'  {k}: shape {tuple(t.shape)} '
                  f'row-sums~{[round(float(x), 2) for x in rowsum[:9]]}'
                  f' diag~{[round(float(t[i, i]), 2) for i in range(min(9, t.shape[0]))]}{flag}')
        else:
            print(f'  {k}: shape {tuple(t.shape)} '
                  f'vals={[round(float(x), 4) for x in t.reshape(-1)[:9]]}{flag}')


def report_rare_prior(sd, rare):
    k = next((k for k in sd if 'rare_prior' in k), None)
    if k is None:
        print('  (no dwpc_rare_prior_logits buffer)')
        return
    t = sd[k].float().reshape(-1)
    boosted = [(i, round(float(x), 3)) for i, x in enumerate(t) if x != 0]
    print(f'  {k}:', [round(float(x), 3) for x in t])
    print(f'    boosted classes: {boosted}')
    nonrare_boost = [i for i, _ in boosted if i not in rare]
    if nonrare_boost:
        print(f'    !! NON-rare classes boosted: {nonrare_boost} '
              f'(prior should boost only {sorted(rare)})')
    if len(boosted) > 1:
        vals = [v for _, v in boosted]
        if max(vals) - min(vals) > 1.5:
            print(f'    note: prior is lopsided (spread {max(vals)-min(vals):.2f}) '
                  f'-> flip candidate c* will collapse to the most-boosted class')


def generic_scan(sd):
    bad = []
    for k, v in sd.items():
        if not torch.is_tensor(v) or v.dtype not in (
                torch.float16, torch.float32, torch.float64):
            continue
        nan, inf, allzero = _tensor_health(v)
        if nan or inf:
            bad.append((k, 'NaN' if nan else 'Inf'))
    if bad:
        print('  !! tensors with NaN/Inf:')
        for k, why in bad[:40]:
            print(f'     {k}: {why}')
    else:
        print('  no NaN/Inf in any float tensor.')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ckpt')
    ap.add_argument('--rare', type=int, nargs='*', default=[0, 1, 6])
    ap.add_argument('--min-count', type=int, default=5)
    args = ap.parse_args()
    rare = set(args.rare)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    meta = ckpt.get('meta', {}) if isinstance(ckpt, dict) else {}
    sd = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt

    print('=' * 70)
    print(f'CHECKPOINT: {args.ckpt}')
    print(f"  meta.iter={meta.get('iter')}  meta.epoch={meta.get('epoch')}  "
          f"#tensors={len(sd)}")
    print('=' * 70)
    print('\n[Witness A — PrototypeMemory bank]')
    report_proto_memory(sd, rare)
    print('\n[Witness B — structural EMA buffers]')
    report_witness_b(sd, rare, args.min_count)
    print('\n[Rare-class prior]')
    report_rare_prior(sd, rare)
    print('\n[Generic health scan]')
    generic_scan(sd)
    print('\nInterpretation:')
    print('  - Cold bank / cold wb_count for rare classes => the corrector is')
    print('    INERT (not a tuning problem). Fix coverage before tuning.')
    print('  - Warm everything but the run still flat => the flip/weight math')
    print('    or the experiment design is the issue: run the full forward')
    print('    probe + config audit in the diagnosis workflow.')


if __name__ == '__main__':
    main()
