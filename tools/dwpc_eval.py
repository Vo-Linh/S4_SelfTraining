#!/usr/bin/env python
# ---------------------------------------------------------------
# DWPC evaluation — stratified rare/head IoU (H2) + flip precision (H5).
#
# Two independent modes:
#
#  (1) Stratified IoU on a labelled split (default). Runs the EMA teacher
#      over a split with GT, accumulates a confusion matrix, and reports
#      per-class IoU split into a rare set and a head set. Compare two
#      runs to get Delta_rare / Delta_head for hypothesis H2.
#
#        python tools/dwpc_eval.py iou CONFIG CKPT \
#            --split val_2000_fixed.txt --rare 0 1 6
#
#  (2) Flip precision from training-time dumps (H5). During training with
#      ``dwpc_debug``, dump npz files holding (flip_mask, corrected_label,
#      gt) on a labelled batch; this mode aggregates per-rare-class flip
#      precision/recall over those dumps.
#
#        python tools/dwpc_eval.py flip --dump-dir work_dirs/<run>/dwpc_dumps
#
# Compatible env = mmcv 1.7.2 (same as tools/rare_class_audit.py).
# ---------------------------------------------------------------
import argparse
import glob
import json
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import mmcv
import numpy as np
import torch
import torch.nn.functional as F

from mmseg.apis import init_segmentor


# ------------------------------------------------------------------
# Core metric — pure function, unit-testable without a model.
# ------------------------------------------------------------------
def stratified_iou(conf_mat, rare_ids):
    """Per-class IoU from a confusion matrix, split into rare vs head.

    Args:
        conf_mat (np.ndarray): (C, C) integer matrix, rows = GT class,
            cols = predicted class.
        rare_ids (list[int]): indices forming the rare set.

    Returns:
        dict: per-class IoU plus ``mIoU``, ``mIoU_rare``, ``mIoU_head``.
    """
    conf = conf_mat.astype(np.float64)
    C = conf.shape[0]
    inter = np.diag(conf)
    union = conf.sum(axis=1) + conf.sum(axis=0) - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    rare = sorted(set(rare_ids))
    head = [c for c in range(C) if c not in rare]

    def _mean(ids):
        vals = [iou[c] for c in ids if not np.isnan(iou[c])]
        return float(np.mean(vals)) if vals else float('nan')

    return dict(
        per_class={c: (None if np.isnan(iou[c]) else float(iou[c]))
                   for c in range(C)},
        mIoU=_mean(range(C)),
        mIoU_rare=_mean(rare),
        mIoU_head=_mean(head),
        rare_ids=rare, head_ids=head)


def load_teacher(config, checkpoint, which, device):
    """Build the inner EncoderDecoder and load student/EMA weights."""
    model = init_segmentor(config, checkpoint=None, device=device)
    sd = torch.load(checkpoint, map_location='cpu')
    sd = sd.get('state_dict', sd)
    prefix = 'ema_model.' if which == 'ema' else 'model.'
    sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    model.load_state_dict(sub, strict=False)
    model.eval()
    return model


@torch.no_grad()
def predict(model, img_path, img_scale, pad_mult, device, mean, std):
    img = mmcv.imread(img_path)
    img = mmcv.imrescale(img, tuple(img_scale))
    img = mmcv.imnormalize(img, mean, std, to_rgb=True)
    img = mmcv.impad_to_multiple(img, pad_mult, pad_val=0)
    h, w = img.shape[:2]
    img_t = torch.from_numpy(img.transpose(2, 0, 1))[None].float().to(device)
    meta = [dict(ori_shape=(h, w, 3), img_shape=(h, w, 3),
                 pad_shape=(h, w, 3), scale_factor=1.0, flip=False)]
    logits = model.encode_decode(img_t, meta)
    return logits.argmax(dim=1)[0].cpu().numpy(), (h, w)


def run_iou(args):
    cfg = mmcv.Config.fromfile(args.config)
    data_root = cfg.data_root
    C = cfg.model.decode_head.num_classes
    mean = np.array(cfg.img_norm_cfg['mean'], dtype=np.float32)
    std = np.array(cfg.img_norm_cfg['std'], dtype=np.float32)
    ignore = 255

    with open(osp.join(data_root, args.split)) as f:
        names = [osp.basename(ln.strip()) for ln in f if ln.strip()]
    names = [n if n.endswith('.tif') else n + '.tif' for n in names]
    if args.max_images:
        names = names[:args.max_images]
    img_dir = osp.join(data_root, 'images/train')
    ann_dir = osp.join(data_root, 'annotations/train')

    model = load_teacher(args.config, args.checkpoint, args.weights,
                         args.device)
    conf = np.zeros((C, C), dtype=np.int64)
    prog = mmcv.ProgressBar(len(names))
    for name in names:
        ann_path = osp.join(ann_dir, name)
        if not osp.exists(ann_path):
            prog.update()
            continue
        pred, (h, w) = predict(model, osp.join(img_dir, name),
                               args.img_scale, args.pad_multiple,
                               args.device, mean, std)
        gt = mmcv.imread(ann_path, flag='unchanged')
        if gt.ndim == 3:
            gt = gt[..., 0]
        gt = mmcv.imresize(gt, (w, h), interpolation='nearest')
        valid = gt != ignore
        g = gt[valid].reshape(-1).astype(np.int64)
        p = pred[valid].reshape(-1).astype(np.int64)
        conf += np.bincount(g * C + p, minlength=C * C).reshape(C, C)
        prog.update()
    print()

    res = stratified_iou(conf, args.rare)
    print(json.dumps(res, indent=2))
    out = args.out or osp.join(osp.dirname(args.checkpoint),
                               'dwpc_stratified_iou.json')
    mmcv.dump(res, out)
    print(f'\nwrote {out}')


def run_flip(args):
    """Aggregate per-rare-class flip precision/recall over training dumps.

    Each dump is an .npz with arrays ``flip_mask`` (bool), ``corrected``
    (int label after flip) and ``gt`` (int GT label), all same shape.
    """
    files = sorted(glob.glob(osp.join(args.dump_dir, '*.npz')))
    if not files:
        print(f'no .npz dumps in {args.dump_dir}')
        return
    rare = sorted(set(args.rare))
    tp = {c: 0 for c in rare}      # flipped to c AND gt==c
    fp = {c: 0 for c in rare}      # flipped to c AND gt!=c
    missed = {c: 0 for c in rare}  # gt==c but teacher wrong & not flipped
    for fpath in files:
        d = np.load(fpath)
        flip, corr, gt = d['flip_mask'].astype(bool), d['corrected'], d['gt']
        for c in rare:
            fc = flip & (corr == c)
            tp[c] += int((fc & (gt == c)).sum())
            fp[c] += int((fc & (gt != c)).sum())
            missed[c] += int(((gt == c) & ~flip & (corr != c)).sum())
    rows = {}
    for c in rare:
        denom_p = tp[c] + fp[c]
        denom_r = tp[c] + missed[c]
        rows[c] = dict(
            flips=denom_p,
            precision=(tp[c] / denom_p) if denom_p else None,
            recall=(tp[c] / denom_r) if denom_r else None)
    overall_tp = sum(tp.values())
    overall_fp = sum(fp.values())
    out = dict(per_rare_class=rows, n_dumps=len(files),
               flip_precision=(overall_tp / (overall_tp + overall_fp))
               if (overall_tp + overall_fp) else None)
    print(json.dumps(out, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description='DWPC evaluation')
    sub = p.add_subparsers(dest='mode', required=True)

    pi = sub.add_parser('iou', help='stratified rare/head IoU')
    pi.add_argument('config')
    pi.add_argument('checkpoint')
    pi.add_argument('--split', default='val_2000_fixed.txt')
    pi.add_argument('--rare', type=int, nargs='+', default=[0, 1, 6])
    pi.add_argument('--weights', choices=['ema', 'student'], default='ema')
    pi.add_argument('--max-images', type=int, default=0)
    pi.add_argument('--pad-multiple', type=int, default=32)
    pi.add_argument('--img-scale', type=int, nargs=2, default=[1024, 1024])
    pi.add_argument('--device', default='cuda:0')
    pi.add_argument('--out', default=None)

    pf = sub.add_parser('flip', help='flip precision from training dumps')
    pf.add_argument('--dump-dir', required=True)
    pf.add_argument('--rare', type=int, nargs='+', default=[0, 1, 6])
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == 'iou':
        run_iou(args)
    else:
        run_flip(args)


if __name__ == '__main__':
    main()
