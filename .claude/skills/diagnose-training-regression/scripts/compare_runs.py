#!/usr/bin/env python3
"""Compare two (or more) MMSeg/DAPCN training runs *fairly*.

The #1 mistake when triaging a regression is comparing final-number vs
final-number when the runs differ in (a) how many iterations they have
trained and (b) which components are enabled. This script removes both
confounds mechanically:

  - parses every ``*.log.json`` in each work_dir (handles resumed runs:
    multiple log files are merged and de-duplicated),
  - aligns validation rows by EPOCH (logged, monotonic, robust) so mIoU is
    compared at matched training progress, not at "whatever each run reached",
  - prints per-class IoU with rare classes flagged, and the rare/head means,
  - reports whether each run is COMPLETE or still mid-training (max iter
    reached vs ``max_iters``) and whether a training process is still alive,
  - diffs the resolved ``uda`` config block across runs so you immediately
    see which knobs actually differ (the "wrong baseline" confound).

Usage:
  python compare_runs.py <work_dir_A> <work_dir_B> [more...] \
      [--rare 0 1 6] [--labels baseline dwpc]

No heavy deps (stdlib only). Safe to run anywhere; never touches the GPU.
"""
import argparse
import glob
import json
import os
import os.path as osp
import re
import subprocess


def _load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def load_run(work_dir):
    """Return dict(val=[...], train=[...], cfg=<resolved config dict or None>,
    name, max_iter, max_iters, eval_interval)."""
    log_jsons = sorted(glob.glob(osp.join(work_dir, '*.log.json')))
    val, train = [], []
    seen_val = set()
    for lj in log_jsons:
        for r in _load_jsonl(lj):
            mode = r.get('mode')
            if mode == 'val':
                # de-dup resumed overlap on (epoch, mIoU)
                key = (r.get('epoch'), round(r.get('mIoU', -1), 6))
                if key in seen_val:
                    continue
                seen_val.add(key)
                val.append(r)
            elif mode == 'train':
                train.append(r)
    val.sort(key=lambda r: (r.get('epoch', 0)))
    max_iter = max([r.get('iter', 0) for r in train], default=0)

    # Resolved config, tried in two ways:
    #  (1) the *.json meta dump (NOT *.log.json) carries, on one line, the
    #      fully-resolved config with a top-level 'uda' dict (stdlib only).
    #  (2) fallback: the dumped *.py config (cfg.dump writes the FULLY
    #      resolved, _base_-free config) parsed via mmcv — used when a run
    #      only dumped a .py (no meta json). Lazy-imports mmcv.
    cfg = None
    for mj in glob.glob(osp.join(work_dir, '*.json')):
        if mj.endswith('.log.json'):
            continue
        for r in _load_jsonl(mj):
            if isinstance(r, dict) and isinstance(r.get('uda'), dict):
                cfg = r
                break
        if cfg:
            break
    if cfg is None:
        for py in glob.glob(osp.join(work_dir, '*.py')):
            try:
                from mmcv import Config
                c = Config.fromfile(py)
                if isinstance(c.get('uda'), dict):
                    cfg = dict(c)  # plain dict view with 'uda','runner',...
                    break
            except Exception:
                continue

    name = osp.basename(osp.normpath(work_dir))
    max_iters = eval_interval = None
    if cfg:
        name = cfg.get('name', name)
        runner = cfg.get('runner') or {}
        max_iters = runner.get('max_iters') or (cfg.get('uda') or {}).get(
            'max_iters')
        ev = cfg.get('evaluation') or {}
        eval_interval = ev.get('interval')
    return dict(work_dir=work_dir, val=val, train=train, cfg=cfg, name=name,
                max_iter=max_iter, max_iters=max_iters,
                eval_interval=eval_interval)


def process_alive(work_dir):
    try:
        out = subprocess.run(['ps', 'aux'], capture_output=True, text=True,
                             timeout=10).stdout
    except Exception:
        return None
    needle = osp.basename(osp.normpath(work_dir))
    hits = [ln for ln in out.splitlines()
            if 'train.py' in ln and needle in ln and 'grep' not in ln]
    return len(hits) > 0


def class_keys(row):
    ks = sorted([k for k in row if k.startswith('IoU.class_')],
                key=lambda k: int(k.split('_')[-1]))
    return ks


def fmt_run_header(run, rare):
    alive = process_alive(run['work_dir'])
    complete = (run['max_iters'] is not None
                and run['max_iter'] >= run['max_iters'])
    status = 'COMPLETE' if complete else (
        f"INCOMPLETE ({run['max_iter']}/{run['max_iters']})"
        if run['max_iters'] else f"iter {run['max_iter']}")
    alive_s = ' [TRAIN PROCESS ALIVE]' if alive else ''
    print(f"  {run['name']}")
    print(f"    dir          : {run['work_dir']}")
    print(f"    status       : {status}{alive_s}")
    print(f"    val points   : {len(run['val'])}"
          f"  (eval interval {run['eval_interval']})")
    if run['val']:
        last = run['val'][-1]
        print(f"    last val     : epoch {last.get('epoch')} "
              f"mIoU {last.get('mIoU')}")


def uda_flat(cfg):
    """Flatten the uda block one level (dwpc sub-dict -> dotted keys)."""
    if not cfg:
        return {}
    uda = cfg.get('uda', {})
    flat = {}
    for k, v in uda.items():
        if k == 'model':
            continue
        if isinstance(v, dict):
            for kk, vv in v.items():
                if not isinstance(vv, (dict, list)):
                    flat[f'{k}.{kk}'] = vv
            flat[k] = f'<dict:{len(v)}keys>'
        elif isinstance(v, list):
            flat[k] = f'<list:{len(v)}>'
        else:
            flat[k] = v
    return flat


def diff_configs(runs):
    flats = [uda_flat(r['cfg']) for r in runs]
    allkeys = sorted(set().union(*[set(f) for f in flats])) if flats else []
    rows = []
    for k in allkeys:
        vals = [f.get(k, '—') for f in flats]
        if len(set(map(repr, vals))) > 1:
            rows.append((k, vals))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('work_dirs', nargs='+')
    ap.add_argument('--rare', type=int, nargs='*', default=[0, 1, 6],
                    help='rare class ids (OpenEarthMap bottom-3 = 0 1 6)')
    ap.add_argument('--labels', nargs='*', default=None)
    args = ap.parse_args()

    runs = [load_run(wd) for wd in args.work_dirs]
    labels = args.labels or [r['name'][:24] for r in runs]
    rare = set(args.rare)

    print('=' * 78)
    print('RUN SUMMARY')
    print('=' * 78)
    for r in runs:
        fmt_run_header(r, rare)
        print()

    # ---- config diff (the "wrong baseline / conflated interventions" check) --
    print('=' * 78)
    print('CONFIG DIFF (uda block) — keys that differ across runs')
    print('=' * 78)
    drows = diff_configs(runs)
    if not drows:
        print('  (no differences found in resolved uda block — or configs '
              'unavailable)')
    else:
        w = max(len(k) for k, _ in drows)
        print('  ' + 'key'.ljust(w) + '  ' + '  |  '.join(labels))
        for k, vals in drows:
            print('  ' + k.ljust(w) + '  ' +
                  '  |  '.join(str(v) for v in vals))
    print()

    # ---- iteration-aligned metric comparison ------------------------------
    print('=' * 78)
    print('MATCHED-EPOCH COMPARISON  (Δ = run2 - run1; aligned by epoch)')
    print('=' * 78)
    # align on epochs present in ALL runs
    epoch_sets = [set(v.get('epoch') for v in r['val']) for r in runs]
    common = sorted(set.intersection(*epoch_sets)) if epoch_sets and all(
        epoch_sets) else []
    if not common:
        print('  No common epochs across runs — cannot align. Per-run curves:')
        for lab, r in zip(labels, runs):
            curve = ', '.join(f"e{v.get('epoch')}:{v.get('mIoU'):.4f}"
                              for v in r['val'])
            print(f'    {lab}: {curve}')
        return

    by_epoch = [{v.get('epoch'): v for v in r['val']} for r in runs]
    hdr = 'epoch | ' + ' | '.join(f'{lab[:10]:>10}' for lab in labels)
    if len(runs) == 2:
        hdr += ' |     Δ'
    print('  ' + hdr)
    print('  ' + '-' * (len(hdr)))
    for ep in common:
        rowvals = [by_epoch[i][ep].get('mIoU') for i in range(len(runs))]
        line = f'{ep:>5} | ' + ' | '.join(f'{m:>10.4f}' for m in rowvals)
        if len(runs) == 2:
            line += f' | {rowvals[1] - rowvals[0]:+.4f}'
        print('  ' + line)

    # rare/head breakdown at the LAST common epoch
    ep = common[-1]
    print()
    print(f'  Per-class IoU at matched epoch {ep} '
          f'(rare classes {sorted(rare)} marked *):')
    ks = class_keys(by_epoch[0][ep])
    hdr2 = 'class      | ' + ' | '.join(f'{lab[:10]:>10}' for lab in labels)
    if len(runs) == 2:
        hdr2 += ' |     Δ'
    print('  ' + hdr2)
    print('  ' + '-' * len(hdr2))
    rare_means = [[] for _ in runs]
    head_means = [[] for _ in runs]
    for k in ks:
        cid = int(k.split('_')[-1])
        mark = '*' if cid in rare else ' '
        vals = [by_epoch[i][ep].get(k) for i in range(len(runs))]
        for i, vv in enumerate(vals):
            (rare_means if cid in rare else head_means)[i].append(vv)
        line = f'class_{cid}{mark} | ' + ' | '.join(f'{v:>10.3f}'
                                                    for v in vals)
        if len(runs) == 2:
            line += f' | {vals[1] - vals[0]:+.3f}'
        print('  ' + line)

    def _mean(xs):
        return sum(xs) / len(xs) if xs else float('nan')
    print('  ' + '-' * len(hdr2))
    rm = [_mean(x) for x in rare_means]
    hm = [_mean(x) for x in head_means]
    line = 'mIoU_rare* | ' + ' | '.join(f'{v:>10.3f}' for v in rm)
    if len(runs) == 2:
        line += f' | {rm[1] - rm[0]:+.3f}'
    print('  ' + line)
    line = 'mIoU_head  | ' + ' | '.join(f'{v:>10.3f}' for v in hm)
    if len(runs) == 2:
        line += f' | {hm[1] - hm[0]:+.3f}'
    print('  ' + line)

    print()
    print('  READ THIS BEFORE CONCLUDING:')
    print('  - If a run is INCOMPLETE, its last mIoU is NOT its final score.')
    print('    Compare only at matched epochs above.')
    print('  - If the CONFIG DIFF shows more than the intended knob differing,')
    print('    the comparison conflates multiple interventions (wrong')
    print('    baseline). Isolate by diffing only the one knob under test.')
    print('  - Single-seed deltas inside ~±0.01 mIoU are noise; need >=3 seeds')
    print('    to claim a per-class effect is real.')


if __name__ == '__main__':
    main()
