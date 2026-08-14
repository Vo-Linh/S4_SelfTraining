#!/usr/bin/env python
"""DAPCN model analysis for the paper figures.

Three subcommands, one shared model loader:

  correction   Fig 2 -- teacher pseudo-label vs DAPCN-corrected vs GT, with the
               flipped pixels highlighted and flip precision reported.
  prototypes   Fig 3 -- t-SNE of the learned prototypes over pixel features,
               plus intra-class compactness / inter-class separation.
  assign       Sec IV-G -- soft assignment maps (which pixels attend to which
               prototype).

Unlike the supervised repo's tools/visualize_*.py, which read the anchor from
``model.decode_head.dynamic_anchor``, in this fork DAPCN lives in the *UDA
wrapper*: prototypes are on ``DAPCN_SSL.dynamic_anchor`` and the anchor feature
comes from ``DAPCN_SSL._get_anchor_features()``. That is the only structural
difference, and it is why those scripts cannot simply be copied here.

Usage:
  python tools/dapcn_analysis.py correction  CONFIG CKPT [-n 4] [--out-dir DIR]
  python tools/dapcn_analysis.py prototypes  CONFIG CKPT [-n 8] [--out-dir DIR]
  python tools/dapcn_analysis.py assign      CONFIG CKPT [-n 2] [-k 6] [--out-dir DIR]
"""
import argparse
import os
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.parallel import collate
from mmcv.runner import load_checkpoint
from mmcv.utils import Config

from mmseg.datasets import build_dataset
from mmseg.models.builder import build_train_model

IGNORE = 255


# --------------------------------------------------------------------------
# shared
# --------------------------------------------------------------------------
def load(config, ckpt, device='cuda'):
    """Build DAPCN_SSL + the unlabeled train dataset (images that carry GT)."""
    cfg = Config.fromfile(config)
    model = build_train_model(
        cfg, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, ckpt, map_location='cpu')
    model.to(device).eval()

    if getattr(model, 'dynamic_anchor', None) is None:
        raise SystemExit(
            'This checkpoint has no DynamicAnchorModule (proto_lambda=0 -> a '
            'non-DAPCN baseline). Point at a DAPCN run.')

    # The unlabeled split still loads GT from disk, which is exactly what we
    # need to score the correction against ground truth.
    ds = build_dataset(cfg.data.train.unlabeled)
    return model, ds, cfg


def sample(ds, i, device='cuda'):
    """One sample -> (img, gt, img_metas)."""
    b = collate([ds[i]], samples_per_gpu=1)
    img = b['img'].data[0].to(device)
    gt = b['gt_semantic_seg'].data[0].to(device).squeeze(1)  # (1,H,W)
    return img, gt, b['img_metas'].data[0]


def denorm(img, cfg):
    """Undo Normalize for display."""
    nrm = next(t for t in cfg.data.train.unlabeled.pipeline
               if t['type'] == 'Normalize')
    mean = np.array(nrm['mean']).reshape(1, 1, 3)
    std = np.array(nrm['std']).reshape(1, 1, 3)
    x = img[0].cpu().numpy().transpose(1, 2, 0) * std + mean
    return np.clip(x / 255.0, 0, 1)


def colorize(lbl, palette):
    """(H,W) label -> (H,W,3) RGB, ignore_index rendered black."""
    pal = np.array(palette, dtype=np.uint8)
    out = np.zeros((*lbl.shape, 3), dtype=np.uint8)
    valid = lbl != IGNORE
    out[valid] = pal[lbl[valid]]
    return out


@torch.no_grad()
def teacher_probs(model, img, metas):
    """EMA-teacher softmax at input resolution."""
    return torch.softmax(model.get_ema_model().encode_decode(img, metas), dim=1)


# --------------------------------------------------------------------------
# Fig 2 -- pseudo-label correction
# --------------------------------------------------------------------------
@torch.no_grad()
def cmd_correction(model, ds, cfg, args):
    pal = ds.PALETTE
    n_flip = n_fix = n_break = 0
    tea_ok = cor_ok = n_valid = 0

    fig, axes = plt.subplots(args.n, 5, figsize=(20, 4 * args.n))
    axes = np.atleast_2d(axes)

    for r in range(args.n):
        img, gt, metas = sample(ds, r * args.stride)
        p_tea = teacher_probs(model, img, metas)
        p_cor = model._correct_pseudo_labels(img, metas, p_tea)

        pl_t = p_tea.argmax(1)[0]
        pl_c = p_cor.argmax(1)[0]
        g = gt[0]

        valid = g != IGNORE
        flip = (pl_t != pl_c) & valid
        # A flip is a "fix" when the teacher was wrong and the correction is
        # right; a "break" when it was right and the correction ruins it.
        fixed = flip & (pl_t != g) & (pl_c == g)
        broke = flip & (pl_t == g) & (pl_c != g)

        n_flip += flip.sum().item()
        n_fix += fixed.sum().item()
        n_break += broke.sum().item()
        tea_ok += ((pl_t == g) & valid).sum().item()
        cor_ok += ((pl_c == g) & valid).sum().item()
        n_valid += valid.sum().item()

        im = denorm(img, cfg)
        ov = im.copy()
        ov[flip.cpu().numpy()] = [1, 1, 0]  # flipped pixels in yellow

        for c, (pic, ttl) in enumerate([
                (im, 'Input'),
                (colorize(pl_t.cpu().numpy(), pal), 'Teacher pseudo-label'),
                (colorize(pl_c.cpu().numpy(), pal), 'DAPCN-corrected'),
                (colorize(g.cpu().numpy(), pal), 'Ground truth'),
                (ov, f'Flipped ({flip.sum().item()} px)')]):
            axes[r, c].imshow(pic)
            axes[r, c].set_axis_off()
            if r == 0:
                axes[r, c].set_title(ttl, fontsize=13)

    plt.tight_layout()
    out = osp.join(args.out_dir, 'fig2_correction.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')

    prec = n_fix / max(n_flip, 1)
    print(f'saved {out}')
    print(f'  flipped pixels     : {n_flip}')
    print(f'  fixed (wrong->right): {n_fix}')
    print(f'  broke (right->wrong): {n_break}')
    print(f'  flip precision     : {prec:.3f}   (>0.5 => correction helps)')
    print(f'  teacher   pixel acc: {tea_ok / max(n_valid,1):.4f}')
    print(f'  corrected pixel acc: {cor_ok / max(n_valid,1):.4f}')


# --------------------------------------------------------------------------
# Fig 3 -- prototype t-SNE + feature-space metrics
# --------------------------------------------------------------------------
@torch.no_grad()
def cmd_prototypes(model, ds, cfg, args):
    from sklearn.manifold import TSNE

    feats, labels = [], []
    for i in range(args.n):
        img, gt, _ = sample(ds, i * args.stride)
        f = model._get_anchor_features(model.get_model().extract_feat(img))
        B, C, Hf, Wf = f.shape
        # GT must be brought to the anchor resolution to colour the pixels.
        g = F.interpolate(gt[:, None].float(), size=(Hf, Wf),
                          mode='nearest')[0, 0].long()
        feats.append(f[0].permute(1, 2, 0).reshape(-1, C).cpu())
        labels.append(g.reshape(-1).cpu())

    X = torch.cat(feats)
    y = torch.cat(labels)
    keep = y != IGNORE
    X, y = X[keep], y[keep]

    if len(X) > args.max_points:  # t-SNE is O(n^2); subsample
        idx = torch.randperm(len(X))[:args.max_points]
        X, y = X[idx], y[idx]

    assign, proto, _ = model.dynamic_anchor(
        model._get_anchor_features(
            model.get_model().extract_feat(sample(ds, 0)[0])))
    assign, proto = assign.cpu(), proto.cpu()

    Xn = F.normalize(X, dim=1)
    Pn = F.normalize(proto, dim=1)

    # Paper's claim: DAPCN yields compact intra-class, separated inter-class.
    cls = sorted(set(y.tolist()))

    def class_metrics(Z):
        Zn = F.normalize(Z, dim=1)
        cents = F.normalize(
            torch.stack([Zn[y == c].mean(0) for c in cls]), dim=1)
        intra = float(np.mean([(1 - (Zn[y == c] @ cents[j])).mean().item()
                               for j, c in enumerate(cls)]))
        sim = cents @ cents.T
        inter = float((1 - sim[~torch.eye(len(cls), dtype=bool)]).mean())
        return intra, inter

    intra, inter = class_metrics(X)
    # Deep transformer features are strongly anisotropic: a large shared mean
    # direction swamps the per-pixel residual, so RAW cosine is ~1 between every
    # pair and both metrics degenerate to 0. Centering exposes whether class
    # structure actually exists underneath.
    intra_c, inter_c = class_metrics(X - X.mean(0, keepdim=True))

    # --- collapse diagnostics (the EM is what consumes the raw features) ---
    ent = float((-(assign * assign.clamp_min(1e-9).log()).sum(1)).mean()
                / np.log(assign.shape[1]))
    psim = Pn @ Pn.T
    proto_cos = float(psim[~torch.eye(len(Pn), dtype=bool)].mean())
    pix_cos = float((Xn @ Xn.T)[~torch.eye(len(Xn), dtype=bool)].mean())
    collapsed = proto_cos > 0.99 and ent > 0.99

    emb = TSNE(n_components=2, perplexity=30, init='pca',
               random_state=0).fit_transform(
                   torch.cat([Xn, Pn]).numpy().astype(np.float64))
    ep, eq = emb[:len(Xn)], emb[len(Xn):]

    plt.figure(figsize=(9, 8))
    pal = np.array(ds.PALETTE) / 255.0
    for j, c in enumerate(cls):
        m = (y == c).numpy()
        plt.scatter(ep[m, 0], ep[m, 1], s=3, alpha=.35,
                    color=pal[c], label=ds.CLASSES[c])
    plt.scatter(eq[:, 0], eq[:, 1], s=190, c='black', marker='*',
                edgecolors='white', linewidths=1.2,
                label=f'prototypes (K={len(eq)})', zorder=5)
    plt.legend(markerscale=2, fontsize=8, loc='best')
    plt.title(f'DAPCN prototypes over pixel features (t-SNE)\n'
              f'intra-class dist {intra:.3f} | inter-class dist {inter:.3f}')
    plt.axis('off')
    plt.tight_layout()

    out = osp.join(args.out_dir, 'fig3_prototypes_tsne.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'saved {out}')
    print(f'  pixels {len(Xn)} | prototypes {len(eq)} | classes {len(cls)}')
    print('  --- class structure (intra low / inter high = good) ---')
    print(f'    RAW      (what the EM consumes): intra={intra:.4f} inter={inter:.4f}')
    print(f'    CENTERED (latent structure)    : intra={intra_c:.4f} inter={inter_c:.4f}')
    print('  --- prototype collapse diagnostics ---')
    print(f'    mean pairwise cosine BETWEEN pixels    : {pix_cos:.5f}  (1.0 = anisotropic)')
    print(f'    mean pairwise cosine BETWEEN prototypes: {proto_cos:.5f}  (1.0 = all identical)')
    print(f'    assignment entropy / max entropy       : {ent:.4f}  (1.0 = uniform)')
    if collapsed:
        print('\n  ** PROTOTYPE COLLAPSE DETECTED **')
        print('     All prototypes are identical and every pixel assigns uniformly')
        print('     to all K. The EM M-step recomputes each prototype as a weighted')
        print('     mean of near-identical (anisotropic) features, so they converge')
        print('     to one point; the correction then adds the SAME class prior to')
        print('     every pixel instead of a per-pixel one.')
        print('     RAW metrics are ~0 while CENTERED are not => the class structure')
        print('     exists but is hidden behind a large shared mean direction, which')
        print('     DynamicAnchorModule L2-normalises but never CENTERS.')


# --------------------------------------------------------------------------
# Sec IV-G -- assignment maps
# --------------------------------------------------------------------------
@torch.no_grad()
def cmd_assign(model, ds, cfg, args):
    for r in range(args.n):
        img, _, _ = sample(ds, r * args.stride)
        f = model._get_anchor_features(model.get_model().extract_feat(img))
        B, C, Hf, Wf = f.shape
        assign, proto, quality = model.dynamic_anchor(f)
        A = assign.reshape(B, Hf, Wf, -1)[0]                    # (Hf,Wf,K)

        # Rank prototypes by how much total mass they attract.
        top = A.sum((0, 1)).argsort(descending=True)[:args.k]
        im = denorm(img, cfg)

        fig, axes = plt.subplots(1, args.k + 1, figsize=(4 * (args.k + 1), 4))
        axes[0].imshow(im)
        axes[0].set_title('Input')
        axes[0].set_axis_off()
        for j, p in enumerate(top):
            heat = F.interpolate(
                A[None, ..., p][None], size=im.shape[:2],
                mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
            axes[j + 1].imshow(im)
            axes[j + 1].imshow(heat, cmap='jet', alpha=.55)
            axes[j + 1].set_title(
                f'proto {p.item()} (q={quality[p].item():.2f})')
            axes[j + 1].set_axis_off()
        plt.tight_layout()
        out = osp.join(args.out_dir, f'assign_{r}.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'saved {out}  (active prototypes: {proto.shape[0]})')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('correction', 'prototypes', 'assign'):
        s = sub.add_parser(name)
        s.add_argument('config')
        s.add_argument('checkpoint')
        s.add_argument('-n', type=int, default=4, help='num images')
        s.add_argument('--stride', type=int, default=1,
                       help='step between sampled images')
        s.add_argument('--out-dir', default='work_dirs/analysis')
        s.add_argument('-k', type=int, default=6,
                       help='assign: prototypes to show')
        s.add_argument('--max-points', type=int, default=6000,
                       help='prototypes: t-SNE subsample cap')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model, ds, cfg = load(args.config, args.checkpoint)
    dict(correction=cmd_correction,
         prototypes=cmd_prototypes,
         assign=cmd_assign)[args.cmd](model, ds, cfg, args)


if __name__ == '__main__':
    main()
