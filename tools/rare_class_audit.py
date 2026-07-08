#!/usr/bin/env python
# ---------------------------------------------------------------
# Rare-class audit of the SSL unlabelled pool.
#
# Purpose (the Path-A-vs-Path-B diagnostic):
#   Runs the trained EMA teacher over the unlabelled split and, for
#   every class, measures whether the rare classes actually appear in
#   the pool and how confidently / at what entropy the teacher predicts
#   them.  Cross-references against the on-disk GT to report the *true*
#   rarity of each class.
#
# It answers two questions empirically rather than by guessing:
#   Q3  Are rare-class regions high-entropy? (mean entropy, pred vs class)
#   Q6  Does the rare class exist in the unlabelled pool, just unlabelled?
#        -> many high-confidence rare-class pixels  => Path A (mine it)
#        -> almost none                             => Path B (generate it)
#
# Side effect: writes a per-image rare-class "mining score" CSV — the
# exact ranking an inverted-S5 miner would consume.
#
# Run (compatible env = mmcv 1.7.2):
#   /home/ubuntu/SatelliteImageSegSupevised/.venv/bin/python \
#       tools/rare_class_audit.py \
#       configs/daformer/ssl_oem_daformer_selftrain_mitb5.py \
#       work_dirs/local-ssl_oem/260527_0253_ssl_oem_daformer_selftrain_mitb5_f0630/iter_4000.pth \
#       --max-images 300        # smoke test; drop for the full 3500
# ---------------------------------------------------------------
import argparse
import csv
import json
import os
import os.path as osp
import sys

# Ensure the LOCAL S4 mmseg (with DAFormerHead/DAPCN registered) wins over
# any editable-installed mmseg in the venv's site-packages.
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import mmcv
import numpy as np
import torch
import torch.nn.functional as F

from mmseg.apis import init_segmentor


def parse_args():
    p = argparse.ArgumentParser(description='Rare-class audit of SSL pool')
    p.add_argument('config', help='training config (model = EncoderDecoder)')
    p.add_argument('checkpoint', help='DAPCN_SSL checkpoint (.pth)')
    p.add_argument('--split', default='train_3500_fixed.txt',
                   help='split file (relative to data_root) to audit')
    p.add_argument('--weights', choices=['ema', 'student'], default='ema',
                   help='which sub-model to load from the checkpoint')
    p.add_argument('--conf-thresh', type=float, default=None,
                   help='confidence threshold; default = cfg pseudo_threshold')
    p.add_argument('--max-images', type=int, default=0,
                   help='cap #images for a quick smoke test (0 = all)')
    p.add_argument('--pad-multiple', type=int, default=32)
    p.add_argument('--img-scale', type=int, nargs=2, default=[1024, 1024])
    p.add_argument('--out-dir', default=None,
                   help='output dir (default: alongside checkpoint)')
    p.add_argument('--device', default='cuda:0')
    return p.parse_args()


def load_teacher(config, checkpoint, which, device):
    """Build the inner EncoderDecoder and load student/EMA weights."""
    model = init_segmentor(config, checkpoint=None, device=device)
    sd = torch.load(checkpoint, map_location='cpu')
    sd = sd.get('state_dict', sd)
    prefix = 'ema_model.' if which == 'ema' else 'model.'
    sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    missing, unexpected = model.load_state_dict(sub, strict=False)
    # backbone/decode_head should match exactly; report anything odd
    real_missing = [k for k in missing if not k.startswith('ema_')]
    if real_missing:
        print(f'[warn] {len(real_missing)} missing keys, e.g. {real_missing[:3]}')
    if unexpected:
        print(f'[warn] {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}')
    model.eval()
    return model


@torch.no_grad()
def predict_logits(model, img_path, img_scale, pad_mult, device, mean, std):
    """Whole-image inference -> logits (1, C, h, w) at padded resolution."""
    img = mmcv.imread(img_path)                       # BGR HxWx3
    img = mmcv.imrescale(img, tuple(img_scale))       # keep ratio
    img = mmcv.imnormalize(img, mean, std, to_rgb=True)
    img = mmcv.impad_to_multiple(img, pad_mult, pad_val=0)
    h, w = img.shape[:2]
    img_t = torch.from_numpy(img.transpose(2, 0, 1))[None].float().to(device)
    meta = [dict(ori_shape=(h, w, 3), img_shape=(h, w, 3),
                 pad_shape=(h, w, 3), scale_factor=1.0, flip=False)]
    return model.encode_decode(img_t, meta)           # (1, C, h, w)


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    data_root = cfg.data_root
    num_classes = cfg.model.decode_head.num_classes
    classes = list(getattr(__import__('mmseg.datasets', fromlist=['OpenEarthMapDataset']),
                           'OpenEarthMapDataset').CLASSES)
    thresh = args.conf_thresh
    if thresh is None:
        thresh = cfg.get('uda', {}).get('pseudo_threshold', 0.968)
    mean = np.array(cfg.img_norm_cfg['mean'], dtype=np.float32)
    std = np.array(cfg.img_norm_cfg['std'], dtype=np.float32)
    ignore = 255

    out_dir = args.out_dir or osp.dirname(args.checkpoint)
    mmcv.mkdir_or_exist(out_dir)

    # ---- read split ----
    suffix = '.tif'
    with open(osp.join(data_root, args.split)) as f:
        names = []
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            base = osp.basename(ln)
            if not base.endswith(suffix):
                base += suffix
            names.append(base)
    if args.max_images:
        names = names[:args.max_images]
    img_dir = osp.join(data_root, 'images/train')
    ann_dir = osp.join(data_root, 'annotations/train')
    print(f'Auditing {len(names)} images | classes={num_classes} '
          f'| conf-thresh={thresh} | weights={args.weights}')

    model = load_teacher(args.config, args.checkpoint, args.weights, args.device)

    C = num_classes
    sweep = [0.5, 0.7, 0.9, thresh]              # confidence sweep
    gt_pixels = np.zeros(C, dtype=np.int64)      # native-res true pixels
    gt_images = np.zeros(C, dtype=np.int64)
    gt_pix_al = np.zeros(C, dtype=np.int64)      # pred-res true pixels (recall denom)
    pred_pixels = np.zeros(C, dtype=np.int64)
    tp_argmax = np.zeros(C, dtype=np.int64)      # pred==c & gt==c
    tp_sweep = np.zeros((len(sweep), C), dtype=np.int64)  # tp & conf>=t
    conf_true_sum = np.zeros(C, dtype=np.float64)   # conf on true-c pixels
    ent_true_sum = np.zeros(C, dtype=np.float64)    # entropy on true-c pixels
    ent_pred_sum = np.zeros(C, dtype=np.float64)    # entropy on pred-c pixels
    ent_pred_cnt = np.zeros(C, dtype=np.int64)
    logC = float(np.log(C))

    per_image_rows = []                          # mining-score CSV

    prog = mmcv.ProgressBar(len(names))
    for name in names:
        stem = name
        img_path = osp.join(img_dir, stem)
        ann_path = osp.join(ann_dir, stem)

        # ----- Prediction side (teacher pseudo-label view) -----
        logits = predict_logits(model, img_path, args.img_scale,
                                 args.pad_multiple, args.device, mean, std)
        prob = F.softmax(logits, dim=1)[0]            # (C, h, w)
        conf, pred = prob.max(dim=0)                  # (h, w)
        ent = -(prob * (prob + 1e-12).log()).sum(0) / logC   # (h,w) in [0,1]
        h, w = pred.shape
        pred_np = pred.cpu().numpy()
        conf_np = conf.cpu().numpy()
        ent_np = ent.cpu().numpy()

        pb = np.bincount(pred_np.reshape(-1), minlength=C)[:C]
        pred_pixels += pb
        for c in range(C):
            m = pred_np == c
            if m.any():
                ent_pred_sum[c] += ent_np[m].sum()
                ent_pred_cnt[c] += int(m.sum())

        # ----- GT side: native rarity + pred-aligned recall -----
        row = {'image': stem}
        if osp.exists(ann_path):
            gt = mmcv.imread(ann_path, flag='unchanged')
            if gt.ndim == 3:
                gt = gt[..., 0]
            gtb = np.bincount(gt[gt != ignore].reshape(-1), minlength=C)[:C]
            gt_pixels += gtb
            gt_images += (gtb > 0).astype(np.int64)
            # align GT -> prediction resolution (nearest) for recall
            gt_al = mmcv.imresize(gt, (w, h), interpolation='nearest')
            for c in range(C):
                tc = gt_al == c
                ntc = int(tc.sum())
                if ntc == 0:
                    continue
                gt_pix_al[c] += ntc
                hit = tc & (pred_np == c)
                tp_argmax[c] += int(hit.sum())
                conf_true_sum[c] += float(conf_np[tc].sum())
                ent_true_sum[c] += float(ent_np[tc].sum())
                for si, t in enumerate(sweep):
                    tp_sweep[si, c] += int((hit & (conf_np >= t)).sum())
                row[f'true_px_c{c}'] = ntc
                row[f'recall_argmax_c{c}'] = float(hit.sum() / ntc)
        per_image_rows.append(row)
        prog.update()

    # ---- assemble summary ----
    tot_gt = max(int(gt_pixels.sum()), 1)
    tot_pred = max(int(pred_pixels.sum()), 1)
    denom_al = np.maximum(gt_pix_al, 1)
    recall_argmax = tp_argmax / denom_al
    recall_sweep = tp_sweep / denom_al[None, :]
    mean_conf_true = conf_true_sum / denom_al
    mean_ent_true = ent_true_sum / denom_al
    mean_ent = np.where(ent_pred_cnt > 0,
                        ent_pred_sum / np.maximum(ent_pred_cnt, 1), np.nan)
    gt_frac = gt_pixels / tot_gt
    # rare = low GT fraction, but exclude classes absent from the pool
    present = gt_images > 0
    order = sorted([c for c in range(C) if present[c]], key=lambda c: gt_frac[c])
    rare_classes = order[:3]

    summary = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'weights': args.weights,
        'split': args.split,
        'n_images': len(names),
        'conf_thresh': thresh,
        'num_classes': C,
        'class_names': classes,
        'rare_classes_by_gt': rare_classes,
        'per_class': [],
    }
    summary['conf_sweep'] = sweep
    for c in range(C):
        summary['per_class'].append({
            'class': c,
            'name': classes[c] if c < len(classes) else f'class_{c}',
            'gt_pixel_frac': float(gt_frac[c]),
            'gt_images': int(gt_images[c]),
            'pred_pixel_frac': float(pred_pixels[c] / tot_pred),
            'recall_argmax': float(recall_argmax[c]),
            'recall_at_sweep': [float(recall_sweep[si, c]) for si in range(len(sweep))],
            'mean_conf_on_true': float(mean_conf_true[c]),
            'mean_entropy_on_true': float(mean_ent_true[c]),
            'mean_entropy_on_pred': None if np.isnan(mean_ent[c]) else float(mean_ent[c]),
        })

    json_path = osp.join(out_dir, 'rare_class_audit.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    csv_path = osp.join(out_dir, 'rare_class_mining_scores.csv')
    fields = ['image']
    for c in range(C):
        fields += [f'true_px_c{c}', f'recall_argmax_c{c}']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, restval=0)
        w.writeheader()
        w.writerows(per_image_rows)

    # ---- console report ----
    print('\n\n================ RARE-CLASS AUDIT ================')
    hdr = (f'{"cls":>3} {"name":<9} {"gt_frac":>8} {"gtImg":>5} '
           f'{"recall@am":>9} {"r@.968":>7} {"conf@true":>9} '
           f'{"ent@true":>8} {"ent@pred":>8}')
    print(hdr)
    print('-' * len(hdr))
    last = len(sweep) - 1
    for c in range(C):
        mep = '   nan' if np.isnan(mean_ent[c]) else f'{mean_ent[c]:>8.3f}'
        print(f'{c:>3} {classes[c][:9]:<9} {gt_frac[c]:>8.4f} '
              f'{gt_images[c]:>5d} {recall_argmax[c]:>9.3f} '
              f'{recall_sweep[last, c]:>7.3f} {mean_conf_true[c]:>9.3f} '
              f'{mean_ent_true[c]:>8.3f} {mep}')
    freq_classes = order[-3:]
    rare_ent = float(np.mean([mean_ent_true[c] for c in rare_classes]))
    freq_ent = float(np.mean([mean_ent_true[c] for c in freq_classes]))
    print('-' * len(hdr))
    print(f'rare (low GT frac, present): {rare_classes} '
          f'{[classes[c] for c in rare_classes]}')
    print(f'mean entropy@true  rare={rare_ent:.3f}  frequent={freq_ent:.3f} '
          f'-> {"rare HIGHER (S5 filter would discard them)" if rare_ent > freq_ent else "rare NOT higher"}')
    print('\nPer rare/hard class — Path A (mine) vs B (generate):')
    for c in rare_classes:
        exists = gt_images[c] >= 0.02 * len(names)
        localizes = recall_argmax[c] >= 0.10
        trustable = recall_sweep[last, c] >= 0.02
        if not exists:
            verp = 'near-ABSENT -> Path B (GENERATE)'
        elif localizes and not trustable:
            verp = ('EXISTS + model localizes it at argmax but NEVER confident '
                    '-> Path A (MINE) blocked only by the fixed threshold '
                    '=> per-class adaptive threshold')
        elif localizes and trustable:
            verp = 'EXISTS + confidently found -> Path A (MINE), already feasible'
        else:
            verp = ('EXISTS in GT but model cannot localize it (recall~0) '
                    '-> representation problem (Path B or stronger features)')
        print(f'  c{c} {classes[c]:<9} gtImg={gt_images[c]:>4d} '
              f'recall@am={recall_argmax[c]:.3f} recall@.968={recall_sweep[last,c]:.3f} '
              f'conf@true={mean_conf_true[c]:.3f}\n       -> {verp}')
    print('==================================================')
    print(f'\nJSON : {json_path}\nCSV  : {csv_path}')


if __name__ == '__main__':
    main()
