#!/usr/bin/env python
# ---------------------------------------------------------------
# DAPCN-SSL prototype-correction probe (offline, read-only).
#
# Two modes:
#
# --space encoder (default): faithfully reproduces the live training
#   loop's class_prototype correction using the checkpoint's
#   proto_memory bank. Measures, per valid val pixel:
#     1. nearest-proto accuracy  = argmax_c cos(f, P_c) == GT
#     2. teacher accuracy        = argmax_c teacher_softmax == GT
#     3. own-vs-other margin     = cos(f, P_gt) - max_{c!=gt} cos(f, P_c)
#     4. blended corrected label -> pl_acc.raw / .delta / flip_correct
#        (sanity check against the training log).
#
# --space decoder: tests the FIX hypothesis (root cause #1: the bank
#   lives in raw stage-4 encoder space which is not class-separable by
#   cosine). Builds a FRESH class-conditioned centroid bank from the
#   decoder's fused feature space (``_fuse_features``, 256-d at 1/4
#   res) using the first ``--bank-images`` labeled images, then
#   evaluates on the rest:
#     a. pure cosine margin / nearest-centroid acc  (is the space
#        separable at all?)
#     b. centroids projected through ``conv_seg`` (the trained
#        decision boundary) -> prototype class distributions, blended
#        with the teacher exactly like the live correction, at BOTH
#        the config params (alpha/temp) and the recommended softened
#        params (alpha=0.2, temp=1.0). Positive delta = fix works.
#
# IMPORTANT: features are centered per-BATCH (like the live loop's
# ``center()`` over dim (0,2,3)), so images are processed in batches
# of ``--batch-size`` (default 4, matching training). Processing at
# batch=1 changes the centering frame and deflates the cosine margin.
#
# Usage:
#   python tools/proto_correction_probe.py \
#       work_dirs/ssl_oem_dapcn_mitb5_1_16/ssl_oem_dapcn_mitb5_1_16.py \
#       work_dirs/ssl_oem_dapcn_mitb5_1_16/iter_4000.pth \
#       --max-images 32                 # encoder mode (repro)
#   python tools/proto_correction_probe.py \
#       work_dirs/ssl_oem_dapcn_mitb5_1_16/ssl_oem_dapcn_mitb5_1_16.py \
#       work_dirs/ssl_oem_dapcn_mitb5_1_16/iter_4000.pth \
#       --space decoder --bank-images 16 --max-images 48
# ---------------------------------------------------------------
import argparse
import copy
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import mmcv
import numpy as np
import torch
import torch.nn.functional as F

from mmseg.models.builder import build_segmentor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('checkpoint')
    p.add_argument('--max-images', type=int, default=32)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--split', default='val_2000_fixed.txt')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--space', choices=['encoder', 'decoder'],
                   default='encoder',
                   help='encoder = checkpoint proto_memory bank (repro); '
                        'decoder = fresh bank in fused decoder space (fix)')
    p.add_argument('--bank-images', type=int, default=16,
                   help='decoder mode: images used to build the centroid '
                        'bank (first N of the split)')
    return p.parse_args()


def load_submodel(cfg, sd, prefix):
    m = copy.deepcopy(cfg)
    m.pop('pretrained', None)
    if isinstance(m.get('backbone'), dict):
        m['backbone'].pop('pretrained', None)
    model = build_segmentor(m)
    sub = {k[len(prefix):]: v for k, v in sd.items()
           if k.startswith(prefix)}
    model.load_state_dict(sub, strict=False)
    return model.eval()


def _load_batches(names, img_dir, ann_dir, mean, std, batch_size):
    """Yield (imgs, gts) batches of resized 512x512 images + GT."""
    batch, batch_gt = [], []
    for name in names:
        ann_path = osp.join(ann_dir, name)
        img_path = osp.join(img_dir, name)
        if not (osp.exists(ann_path) and osp.exists(img_path)):
            continue
        img = mmcv.imread(img_path)
        img = mmcv.imresize(img, (512, 512))
        img = mmcv.imnormalize(img, mean, std, to_rgb=True)
        gt = mmcv.imread(ann_path, flag='unchanged')
        if gt.ndim == 3:
            gt = gt[..., 0]
        gt = mmcv.imresize(gt, (512, 512), interpolation='nearest')
        batch.append(torch.from_numpy(img.transpose(2, 0, 1)))
        batch_gt.append(gt)
        if len(batch) == batch_size:
            yield batch, batch_gt
            batch, batch_gt = [], []
    if batch:
        yield batch, batch_gt


def run_decoder_mode(args, cfg, sd):
    """Fix-hypothesis probe: fresh centroid bank in fused decoder space.

    Builds class centroids from the decoder's fused features (256-d at
    1/4 res, ``_fuse_features``) of the first ``--bank-images`` images,
    then on held-out images measures:
      a. pure cosine margin + nearest-centroid acc (is the space
         class-separable at all?)
      b. the live correction reproduced with centroids projected
         through ``conv_seg``, at the config params and the softened
         params (alpha=0.2, temp=1.0). Positive delta => fix works.
    """
    dev = args.device
    C = cfg.model.decode_head.num_classes
    mean = np.array(cfg.img_norm_cfg['mean'], dtype=np.float32)
    std = np.array(cfg.img_norm_cfg['std'], dtype=np.float32)
    ignore = 255
    alpha_cfg = cfg.uda.proto_correction_alpha
    temp_cfg = cfg.uda.proto_correction_temperature
    alpha_soft, temp_soft = 0.2, 1.0
    D = cfg.model.decode_head.channels      # 256 (fused space)

    student = load_submodel(cfg.model, sd, 'model.')
    teacher = load_submodel(cfg.model, sd, 'ema_model.')
    student.to(dev).train()
    teacher.to(dev)
    decode_head = student.decode_head

    with open(osp.join(cfg.data_root, args.split)) as f:
        names = [osp.basename(ln.strip()) for ln in f if ln.strip()]
    names = [n if n.endswith('.tif') else n + '.tif'
             for n in names][:args.max_images]
    img_dir = osp.join(cfg.data_root, 'images/val')
    ann_dir = osp.join(cfg.data_root, 'annotations/val')

    bank_names = names[:args.bank_images]
    eval_names = names[args.bank_images:]

    # ---- build centroids in fused decoder space (per-batch centered) ----
    acc = torch.zeros(C, D, device=dev)
    cnt = torch.zeros(C, device=dev)
    for batch, gts in _load_batches(bank_names, img_dir, ann_dir, mean,
                                    std, args.batch_size):
        x = torch.stack(batch).to(dev)
        with torch.no_grad():
            feats = student.extract_feat(x)
            fused = decode_head._fuse_features(feats)   # (B, D, Hf, Wf)
        fused = fused - fused.mean(dim=(0, 2, 3), keepdim=True)
        B, _, Hf, Wf = fused.shape
        q = F.normalize(fused.permute(0, 2, 3, 1).reshape(-1, D), dim=1)
        gs = np.concatenate([
            mmcv.imresize(g, (Wf, Hf),
                          interpolation='nearest').astype(np.int64)
            .reshape(-1) for g in gts])
        for c in range(C):
            n = int((gs == c).sum())
            if n:
                acc[c] += q[gs == c].sum(dim=0)
                cnt[c] += n
    P = torch.nan_to_num(F.normalize(acc, dim=1), nan=0.0)  # (C, D)
    missing = (cnt == 0).nonzero().flatten().tolist()
    print(f'[decoder] bank: {len(bank_names)} build imgs, '
          f'{len(eval_names)} eval imgs, '
          f'classes w/o support: {missing or "none"}')

    # ---- project centroids through the trained classifier ----
    with torch.no_grad():
        proto_logits = decode_head.conv_seg(
            P.unsqueeze(-1).unsqueeze(-1))               # (C, 9, 1, 1)
    proto_probs = torch.softmax(proto_logits.squeeze(-1).squeeze(-1),
                                dim=1)                   # (C, 9)
    print('[decoder] conv_seg(centroid) class distributions:')
    print('         ' + '  '.join(
        f'c{c}:{proto_probs[c].argmax().item()}' for c in range(C)))

    # ---- evaluate on held-out images ----
    n_total = n_teacher = n_nn = n_margin = 0
    own_sum = other_sum = margin_sum = 0.0
    stats = {tag: {'n_corr': 0, 'n_flip': 0, 'n_flip_ok': 0}
             for tag in ('config', 'soft', 'gated09', 'gated0968')}
    per_class = {c: {'n': 0, 'teacher': 0,
                     'nn': 0, 'nn_n': 0} for c in range(C)}
    # Per-class flip analysis (config blend): for each class c, how many
    # pixels FLIPPED INTO c (teacher->c via prototype) and how many were
    # correct, and how many flipped OUT OF c.
    flip_into = np.zeros(C, dtype=np.int64)
    flip_into_ok = np.zeros(C, dtype=np.int64)
    flip_out = np.zeros(C, dtype=np.int64)
    flip_out_ok = np.zeros(C, dtype=np.int64)

    for batch, gts in _load_batches(eval_names, img_dir, ann_dir, mean,
                                    std, args.batch_size):
        x = torch.stack(batch).to(dev)
        with torch.no_grad():
            feats = student.extract_feat(x)
            fused = decode_head._fuse_features(feats)
            meta = [dict(ori_shape=(512, 512, 3), img_shape=(512, 512, 3),
                         pad_shape=(512, 512, 3), scale_factor=1.0,
                         flip=False)] * len(batch)
            tsoft = torch.softmax(
                teacher.encode_decode(x, meta).detach(), dim=1)
        fused = fused - fused.mean(dim=(0, 2, 3), keepdim=True)
        B, _, Hf, Wf = fused.shape
        q = F.normalize(fused.permute(0, 2, 3, 1).reshape(-1, D), dim=1)
        sim = q @ P.t()                                   # (N, C)
        nn_label = sim.argmax(1)

        # margin + nearest-centroid acc at feature res (1/4)
        gs_all = [mmcv.imresize(g, (Wf, Hf),
                                interpolation='nearest').astype(np.int64)
                  .reshape(-1) for g in gts]
        gs = np.concatenate(gs_all)
        val = gs != ignore
        gv = gs[val]
        sv = sim[val]
        n_nn += int((nn_label.cpu().numpy()[val] == gv).sum())
        n_margin += len(gv)
        own = sv[torch.arange(len(gv), device=dev), gv]
        mask = torch.ones_like(sv, dtype=torch.bool)
        mask[torch.arange(len(gv), device=dev), gv] = False
        other = sv[mask].view(len(gv), C - 1).max(dim=1).values
        own_sum += own.sum().item()
        other_sum += other.sum().item()
        margin_sum += (own - other).sum().item()
        nn_np = nn_label.cpu().numpy()[val]
        for c in range(C):
            m = gv == c
            if m.sum():
                per_class[c]['nn_n'] += int(m.sum())
                per_class[c]['nn'] += int((nn_np[m] == c).sum())

        # teacher accuracy + valid-pixel count at 512 res (per image)
        t_label = tsoft.argmax(1).cpu().numpy()
        t_max = tsoft.max(dim=1).values          # (B, 512, 512)
        for j in range(B):
            g = gts[j]
            valid = g != ignore
            gv = g[valid]
            n_total += int(valid.sum())
            n_teacher += int((t_label[j][valid] == gv).sum())
            for c in range(C):
                m = gv == c
                if m.sum():
                    per_class[c]['n'] += int(m.sum())
                    per_class[c]['teacher'] += int(
                        (t_label[j][valid][m] == c).sum())

        # correction: assignment-weighted conv_seg probs, blended w/ teacher
        tag_specs = [('config', alpha_cfg, temp_cfg, None),
                     ('soft', alpha_soft, temp_soft, None),
                     ('gated09', alpha_cfg, temp_cfg, 0.9),
                     ('gated0968', alpha_cfg, temp_cfg, 0.968)]
        for tag, alpha, temp, gate in tag_specs:
            a = torch.softmax(sim / temp, dim=1)          # (N, C)
            p_proto = (a @ proto_probs).reshape(
                B, Hf, Wf, C).permute(0, 3, 1, 2)         # (B, 9, Hf, Wf)
            p_proto = F.interpolate(p_proto, size=(512, 512),
                                    mode='bilinear', align_corners=False)
            blended = (1 - alpha) * tsoft + alpha * p_proto
            c_label = blended.argmax(1)
            if gate is not None:   # only touch low-confidence pixels
                c_label = torch.where(
                    t_max < gate, c_label, tsoft.argmax(1))
            c_label = c_label.cpu().numpy()
            s = stats[tag]
            for j in range(B):
                g = gts[j]
                valid = g != ignore
                gv = g[valid]
                tt = t_label[j][valid]
                tc = c_label[j][valid]
                s['n_corr'] += int((tc == gv).sum())
                fl = tc != tt
                s['n_flip'] += int(fl.sum())
                s['n_flip_ok'] += int((fl & (tc == gv)).sum())
            if tag == 'config':
                # per-class flip breakdown for the reference blend
                pl = p_proto.argmax(1).cpu().numpy()
                bl = c_label
                for j in range(B):
                    g = gts[j]
                    valid = g != ignore
                    gv = g[valid]
                    tl_j = t_label[j][valid]
                    pl_j = pl[j][valid]
                    bl_j = bl[j][valid]
                    flip = bl_j != tl_j
                    for c in range(C):
                        into = flip & (pl_j == c)
                        if into.any():
                            flip_into[c] += int(into.sum())
                            flip_into_ok[c] += int(
                                (into & (gv == c)).sum())
                        out = flip & (tl_j == c)
                        if out.any():
                            flip_out[c] += int(out.sum())
                            flip_out_ok[c] += int(
                                (out & (bl_j == gv)).sum())

    print('\n===== decoder-space fix probe =====')
    print(f'images={len(eval_names)}  valid_pixels={n_total} '
          f'(512-res; nearest-centroid is at 1/4 res)')
    print(f'[a] nearest-centroid acc (cosine, 1/4 res): '
          f'{n_nn / max(n_margin, 1):.4f}')
    print(f'[a] own-proto cos: {own_sum / max(n_margin, 1):.4f}   '
          f'other-proto cos: {other_sum / max(n_margin, 1):.4f}')
    print(f'    MARGIN (own-other): '
          f'{margin_sum / max(n_margin, 1):.4f}   '
          f'(>0 => fused space IS separable by cosine)')
    print(f'[b] teacher acc            : {n_teacher / max(n_total, 1):.4f}')
    for tag, alpha, temp, gate in tag_specs:
        s = stats[tag]
        acc = s['n_corr'] / max(n_total, 1)
        gate_s = f', gate={gate}' if gate else ''
        print(f'[b] corrected acc (alpha={alpha}, temp={temp}{gate_s}, '
              f'{tag}): {acc:.4f}   delta: '
              f'{acc - n_teacher / max(n_total, 1):+.4f} '
              f'  flip_rate: {s["n_flip"] / max(n_total, 1):.4f} '
              f'flip_correct: '
              f'{s["n_flip_ok"] / max(s["n_flip"], 1):.4f}')
    print('\nper-class (nearest-centroid @1/4 res vs teacher @512):')
    print('  class |  n_pixels |  nn_acc | teacher_acc')
    for c in range(C):
        pc = per_class[c]
        if pc['n']:
            print(f'  {c:5d} | {pc["n"]:9d} | '
                  f'{pc["nn"] / max(pc["nn_n"], 1):7.4f} | '
                  f'{pc["teacher"] / pc["n"]:10.4f}')

    print('\nper-class FLIP analysis (config blend alpha=0.5 temp=0.1):')
    print('  flip_INTO c  | n_flips | n_correct | flip_correct')
    good_classes = []
    for c in range(C):
        if flip_into[c] == 0:
            continue
        fc = flip_into_ok[c] / flip_into[c]
        flag = '  <-- net win' if fc > 0.5 else ''
        print(f'  {c:5d}       | {flip_into[c]:7d} | '
              f'{flip_into_ok[c]:9d} | {fc:5.3f}{flag}')
        if fc > 0.5 and flip_into[c] >= 50:
            good_classes.append(c)

    def policy_delta(keep_flips, keep_ok):
        keep_wrong = keep_flips - keep_ok
        return (keep_ok - keep_wrong) / max(n_total, 1)

    oracle = stats['config']['n_flip_ok'] / max(n_total, 1)
    w = {4, 6}
    k64 = sum(flip_into[c] for c in w)
    ok64 = sum(flip_into_ok[c] for c in w)
    ksel = sum(flip_into[c] for c in good_classes)
    oksel = sum(flip_into_ok[c] for c in good_classes)
    kt64 = sum(flip_out[c] for c in w)
    okt64 = sum(flip_out_ok[c] for c in w)
    ref_delta = (stats['config']['n_corr'] / max(n_total, 1)
                 - n_teacher / max(n_total, 1))

    print('\n=== per-class-conditional policy deltas (vs teacher) ===')
    print(f'oracle ceiling (keep only correct flips)          : '
          f'{oracle:+.5f}')
    print(f'config blend reference delta                      : '
          f'{ref_delta:+.5f}')
    print(f'cond_proto   flip only toward c in {sorted(w)}        : '
          f'{policy_delta(k64, ok64):+.5f}   (keeps {k64} flips)')
    print(f'cond_proto   flip toward {sorted(good_classes)} : '
          f'{policy_delta(ksel, oksel):+.5f}   (keeps {ksel} flips)')
    print(f'cond_teacher flip only away from c in {sorted(w)}    : '
          f'{policy_delta(kt64, okt64):+.5f}   (keeps {kt64} flips)')
    print('\npositive delta above 0 = per-class-conditional has measurable '
          'upside; magnitude vs oracle ceiling tells how much headroom '
          'remains.')


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    sd = torch.load(args.checkpoint, map_location='cpu')['state_dict']
    if args.space == 'decoder':
        run_decoder_mode(args, cfg, sd)
        return
    data_root = cfg.data_root
    C = cfg.model.decode_head.num_classes
    mean = np.array(cfg.img_norm_cfg['mean'], dtype=np.float32)
    std = np.array(cfg.img_norm_cfg['std'], dtype=np.float32)
    ignore = 255
    temp = cfg.uda.proto_correction_temperature
    alpha = cfg.uda.proto_correction_alpha
    dev = args.device

    student = load_submodel(cfg.model, sd, 'model.')
    teacher = load_submodel(cfg.model, sd, 'ema_model.')
    student.to(dev).train()          # live loop keeps the student in train()
    teacher.to(dev)
    print(f'[mode] student train_mode={student.training}')

    protos = sd['proto_memory.prototypes'].float().to(dev)   # (C, D)
    P = F.normalize(protos, dim=1)

    with open(osp.join(data_root, args.split)) as f:
        names = [osp.basename(ln.strip()) for ln in f if ln.strip()]
    names = [n if n.endswith('.tif') else n + '.tif'
             for n in names][:args.max_images]
    img_dir = osp.join(data_root, 'images/val')
    ann_dir = osp.join(data_root, 'annotations/val')

    n_proto = n_teacher = n_corr = n_total = 0
    n_flip = n_flip_ok = 0
    own_sum = other_sum = margin_sum = n_margin = 0.0
    per_class = {c: {'n': 0, 'proto': 0, 'teacher': 0} for c in range(C)}
    pred_counts = np.zeros(C, dtype=np.int64)
    gt_counts = np.zeros(C, dtype=np.int64)

    def run_batch(imgs):
        x = torch.stack(imgs).to(dev)
        with torch.no_grad():
            feats = student.extract_feat(x)
            f = feats[-1]
            fc = f - f.mean(dim=(0, 2, 3), keepdim=True)  # center()
            B, D, Hf, Wf = fc.shape
            q = F.normalize(fc.permute(0, 2, 3, 1).reshape(-1, D), dim=1)
            sim = q @ P.t()                               # (N, C)
            meta = [dict(ori_shape=(512, 512, 3), img_shape=(512, 512, 3),
                         pad_shape=(512, 512, 3), scale_factor=1.0,
                         flip=False)] * B
            tsoft = torch.softmax(
                teacher.encode_decode(x, meta).detach(), dim=1)
        pp = torch.softmax(sim / temp, dim=1).reshape(
            B, Hf, Wf, C).permute(0, 3, 1, 2)
        pp = F.interpolate(pp, size=(512, 512), mode='bilinear',
                           align_corners=False)
        blended = (1 - alpha) * tsoft + alpha * pp
        return sim, B, Hf, Wf, pp, blended, tsoft

    batch = []
    batch_meta = []
    for i, name in enumerate(names):
        ann_path = osp.join(ann_dir, name)
        img_path = osp.join(img_dir, name)
        if not (osp.exists(ann_path) and osp.exists(img_path)):
            continue
        img = mmcv.imread(img_path)
        img = mmcv.imresize(img, (512, 512))
        img = mmcv.imnormalize(img, mean, std, to_rgb=True)
        gt = mmcv.imread(ann_path, flag='unchanged')
        if gt.ndim == 3:
            gt = gt[..., 0]
        gt = mmcv.imresize(gt, (512, 512), interpolation='nearest')
        batch.append(torch.from_numpy(img.transpose(2, 0, 1)))
        batch_meta.append(gt)

        if len(batch) == args.batch_size or i == len(names) - 1:
            sim, B, Hf, Wf, pp, blended, tsoft = run_batch(batch)
            proto_label = pp.argmax(1).cpu().numpy()
            teacher_label = tsoft.argmax(1).cpu().numpy()
            corr_label = blended.argmax(1).cpu().numpy()

            for j in range(B):
                g = batch_meta[j]
                valid = g != ignore
                gv = g[valid]
                tp = proto_label[j][valid]
                tt = teacher_label[j][valid]
                tc = corr_label[j][valid]
                n_proto += int((tp == gv).sum())
                n_teacher += int((tt == gv).sum())
                n_corr += int((tc == gv).sum())
                n_total += len(gv)
                fl = tc != tt
                n_flip += int(fl.sum())
                n_flip_ok += int((fl & (tc == gv)).sum())
                for c in range(C):
                    m = gv == c
                    if m.sum():
                        per_class[c]['n'] += int(m.sum())
                        per_class[c]['proto'] += int((tp[m] == c).sum())
                        per_class[c]['teacher'] += int((tt[m] == c).sum())

            # margin at the 16x16 feature grid
            gs_all = [mmcv.imresize(g, (Wf, Hf),
                                    interpolation='nearest').astype(np.int64)
                      .reshape(-1) for g in batch_meta]
            gs = np.concatenate(gs_all)
            val = gs != ignore
            sv = sim[val]
            gv = gs[val]
            pred_counts += np.bincount(sim.argmax(1).cpu().numpy()[val],
                                       minlength=C)
            gt_counts += np.bincount(gv, minlength=C)
            mask = torch.ones_like(sv, dtype=torch.bool)
            mask[torch.arange(len(gv), device=sv.device), gv] = False
            other = sv[mask].view(len(gv), C - 1).max(dim=1).values
            own = sv[torch.arange(len(gv), device=sv.device), gv]
            own_sum += own.sum().item()
            other_sum += other.sum().item()
            margin_sum += (own - other).sum().item()
            n_margin += len(gv)

            batch, batch_meta = [], []

    print('\n===== root-cause probe results =====')
    print(f'images={len(names)}  valid_pixels={n_total}')
    print(f'[1] nearest-proto acc : {n_proto / n_total:.4f}')
    print(f'[2] teacher acc       : {n_teacher / n_total:.4f}')
    print(f'[3] corrected (blend) : {n_corr / n_total:.4f}')
    print(f'[4] flip rate         : {n_flip / n_total:.4f}   '
          f'flip_correct: {n_flip_ok / max(n_flip, 1):.4f}')
    print(f'[5] own-proto cos     : {own_sum / n_margin:.4f}   '
          f'other-proto cos: {other_sum / n_margin:.4f}')
    print(f'    MARGIN (own-other): {margin_sum / n_margin:.4f}   '
          f'(< 0 => pixel is closer to a WRONG class prototype)')
    print(f'[6] GT  class dist    : {gt_counts.tolist()}')
    print(f'    pred class dist   : {pred_counts.tolist()}')
    print('\nper-class:')
    print(f'  class |  n_pixels | proto_acc | teacher_acc')
    for c in range(C):
        pc = per_class[c]
        if pc['n']:
            print(f'  {c:5d} | {pc["n"]:9d} | '
                  f'{pc["proto"] / pc["n"]:9.4f} | '
                  f'{pc["teacher"] / pc["n"]:9.4f}')


if __name__ == '__main__':
    main()
