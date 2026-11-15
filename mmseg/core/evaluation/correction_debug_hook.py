# ---------------------------------------------------------------
# ProtoCorrectionDebugHook: diagnostics for prototype-based
# pseudo-label correction in DAPCN-SSL.
#
# DIAGNOSTIC ONLY.  This hook NEVER touches the training loss, the
# gradients, or checkpoint selection.  It reads the held-out ground
# truth of nominally *unlabeled* images purely to answer three
# questions that cannot be answered from the training log alone:
#
#   1. Is the EMA teacher any good on its own?
#        -> debug/teacher_mIoU, debug/ce_teacher
#   2. Does the prototype signal p^c carry usable class information
#      on its own?
#        -> debug/proto_mIoU, debug/ce_proto
#   3. Does blending the two beat BOTH components, or merely dilute
#      the better one?
#        -> debug/blend_mIoU, debug/ce_blend
#
# Plus a targeted look at the pixels the correction actually changes:
#        -> debug/flip_ratio, debug/flip_acc_before, debug/flip_acc_after
#
# IMPORTANT (experimental hygiene): the ground truth used here is, by
# construction, hidden from training.  Using these numbers to pick a
# checkpoint or to early-stop would leak it back in.  Read them to
# understand the mechanism; report final results from the proper
# validation split.
# ---------------------------------------------------------------

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.parallel import collate
from mmcv.runner import HOOKS, Hook, get_dist_info

from .metrics import intersect_and_union


def _unwrap(model):
    """Strip MMDataParallel / DDP(-Wrapper) layers off a model."""
    while hasattr(model, 'module'):
        model = model.module
    return model


@HOOKS.register_module()
class ProtoCorrectionDebugHook(Hook):
    """Monitor whether prototype correction improves pseudo-labels.

    Runs on a fixed subset of images so that numbers are comparable
    across iterations, at two cadences:

      * every ``ce_log_interval`` iters — cheap: cross-entropy against
        the hidden GT plus flipped-pixel statistics, on
        ``ce_num_samples`` images.
      * every ``miou_eval_interval`` iters — full: additionally mIoU,
        accumulated over all ``num_debug_samples`` images.

    mIoU is the metric to judge by: this data is class-imbalanced
    (that is why RCS exists), so pixel accuracy would be dominated by
    the common classes.  The CE values are the cheap high-frequency
    signal used to spot instability between mIoU evaluations —
    especially around ``proto_correction_start_iter`` and the end of
    ``pseudo_label_warmup_iters``.

    Only runs on rank 0 under distributed training.

    Args:
        dataset (dict): Config of the dataset holding the debug images
            AND their ground truth. Built with ``build_dataset``. Its
            pipeline must be deterministic (no random crop/flip) and
            must ``Collect`` both ``img`` and ``gt_semantic_seg``.
        num_debug_samples (int): Size of the fixed subset used for the
            mIoU evaluation. Default: 64.
        ce_num_samples (int): Size of the (prefix) subset used for the
            cheap CE/flip logging. Default: 8.
        ce_log_interval (int): Iteration interval for the cheap pass.
            0 disables it. Default: 200.
        miou_eval_interval (int): Iteration interval for the full pass.
            0 disables it. Default: 1000.
        batch_size (int): Images per forward pass. Default: 2.
        subset_seed (int): Seed for picking the fixed subset, so the
            same images are used across runs. Default: 0.
        ignore_index (int): Label value to ignore. Default: 255.
    """

    EPS = 1e-8

    def __init__(self,
                 dataset,
                 num_debug_samples=64,
                 ce_num_samples=8,
                 ce_log_interval=200,
                 miou_eval_interval=1000,
                 batch_size=2,
                 subset_seed=0,
                 ignore_index=255):
        self.dataset_cfg = dataset
        self.num_debug_samples = num_debug_samples
        self.ce_num_samples = ce_num_samples
        self.ce_log_interval = ce_log_interval
        self.miou_eval_interval = miou_eval_interval
        self.batch_size = batch_size
        self.subset_seed = subset_seed
        self.ignore_index = ignore_index

        self.dataset = None
        self.indices = []
        self.enabled = True

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def before_run(self, runner):
        rank, _ = get_dist_info()
        if rank != 0:
            self.enabled = False
            return

        model = _unwrap(runner.model)
        if not hasattr(model, 'compute_correction_diagnostics'):
            runner.logger.warning(
                'ProtoCorrectionDebugHook: model %s has no '
                'compute_correction_diagnostics(); disabling the hook.',
                type(model).__name__)
            self.enabled = False
            return

        # Imported here to avoid a circular import at module load time
        # (mmseg.datasets imports from mmseg.core).
        from mmseg.datasets import build_dataset
        self.dataset = build_dataset(self.dataset_cfg)

        n_total = len(self.dataset)
        n_pick = min(self.num_debug_samples, n_total)
        rng = np.random.RandomState(self.subset_seed)
        self.indices = sorted(
            rng.choice(n_total, size=n_pick, replace=False).tolist())

        runner.logger.info(
            'ProtoCorrectionDebugHook: %d/%d debug images '
            '(CE every %d iters on %d imgs, mIoU every %d iters on %d imgs). '
            'Hidden GT is used for MONITORING ONLY — never for training '
            'or checkpoint selection.', n_pick, n_total, self.ce_log_interval,
            min(self.ce_num_samples, n_pick), self.miou_eval_interval, n_pick)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def after_train_iter(self, runner):
        if not self.enabled:
            return

        do_miou = (self.miou_eval_interval > 0
                   and self.every_n_iters(runner, self.miou_eval_interval))
        do_ce = (self.ce_log_interval > 0
                 and self.every_n_iters(runner, self.ce_log_interval))
        if not (do_miou or do_ce):
            return

        model = _unwrap(runner.model)
        # The EMA teacher only exists after the first forward_train.
        if getattr(model, 'local_iter', 0) < 1:
            return

        n = len(self.indices) if do_miou else min(self.ce_num_samples,
                                                  len(self.indices))
        metrics = self._evaluate(model, self.indices[:n], compute_miou=do_miou)

        # Logged twice on purpose: through the log buffer so the values
        # reach TensorBoard/the JSON log, and directly so they are never
        # dropped when our interval does not align with log_config's.
        for key, val in metrics.items():
            runner.log_buffer.output[key] = val
        runner.log_buffer.ready = True

        runner.logger.info(
            'ProtoCorrectionDebug @ iter %d [%d imgs]: %s', runner.iter + 1, n,
            ', '.join('%s=%.4f' % (k.replace('debug/', ''), v)
                      for k, v in sorted(metrics.items())))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def _evaluate(self, model, indices, compute_miou):
        """Run the three signals over ``indices`` and reduce to scalars."""
        device = next(model.parameters()).device
        num_classes = model.get_model().decode_head.num_classes

        was_training = model.training
        # eval() matters: DynamicAnchorModule mutates its persistent
        # prototypes in-place when ema_decay > 0 and self.training is
        # True, which would let diagnostics perturb training state.
        model.eval()

        ce_sums = dict(teacher=0.0, proto=0.0, blend=0.0)
        ce_batches = 0
        flip_n = flip_correct_after = flip_correct_before = 0
        valid_n = 0
        area_acc = {}
        correction_active = False
        has_proto = False

        try:
            for start in range(0, len(indices), self.batch_size):
                chunk = indices[start:start + self.batch_size]
                img, gt, img_metas = self._load_batch(chunk, device)

                out = model.compute_correction_diagnostics(img, img_metas)
                correction_active = out['correction_active']
                has_proto = out['proto'] is not None

                signals = {'teacher': out['teacher'], 'blend': out['blend']}
                if has_proto:
                    signals['proto'] = out['proto']

                for name, probs in signals.items():
                    ce_sums[name] += self._nll(probs, gt).item()
                ce_batches += 1

                preds = {
                    name: probs.argmax(dim=1)
                    for name, probs in signals.items()
                }

                # --- Flipped-pixel statistics ---------------------------
                # The pixels where correction actually changes the label
                # are the only ones it can help or hurt on.
                valid = gt != self.ignore_index
                flipped = (preds['blend'] != preds['teacher']) & valid
                flip_n += int(flipped.sum().item())
                valid_n += int(valid.sum().item())
                if flipped.any():
                    flip_correct_after += int(
                        (preds['blend'][flipped] == gt[flipped]).sum().item())
                    flip_correct_before += int(
                        (preds['teacher'][flipped] == gt[flipped]).sum().item())

                # --- mIoU accumulation ----------------------------------
                if compute_miou:
                    gt_np = gt.cpu().numpy()
                    for name, pred in preds.items():
                        pred_np = pred.cpu().numpy()
                        for b in range(pred_np.shape[0]):
                            areas = intersect_and_union(
                                pred_np[b], gt_np[b], num_classes,
                                self.ignore_index)
                            prev = area_acc.get(name)
                            area_acc[name] = areas if prev is None else tuple(
                                p + a for p, a in zip(prev, areas))
        finally:
            if was_training:
                model.train()
            # Processing dozens of images here leaves the caching allocator
            # fragmented, and training then resumes with roughly 1 GB of
            # headroom on a 24 GB card.  A run died with CUDA OOM at exactly
            # this boundary (iter 1000, immediately after this hook and the
            # eval).  Returning the cached blocks costs a little speed once
            # per invocation and changes nothing about training.
            torch.cuda.empty_cache()

        metrics = {'debug/correction_active': float(correction_active)}

        for name in ('teacher', 'proto', 'blend'):
            if name == 'proto' and not has_proto:
                continue
            metrics['debug/ce_%s' % name] = ce_sums[name] / max(ce_batches, 1)

        if flip_n:
            metrics['debug/flip_acc_after'] = flip_correct_after / flip_n
            metrics['debug/flip_acc_before'] = flip_correct_before / flip_n
        metrics['debug/flip_ratio'] = flip_n / max(valid_n, 1)

        if compute_miou:
            for name, areas in area_acc.items():
                area_intersect, area_union = areas[0], areas[1]
                iou = area_intersect / torch.clamp(area_union, min=self.EPS)
                # Classes absent from both prediction and GT have union 0
                # and must not be counted as 0 IoU.
                present = area_union > 0
                metrics['debug/%s_mIoU' % name] = float(
                    iou[present].mean().item()) if present.any() else 0.0

        return metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_batch(self, indices, device):
        """Fetch and collate a batch straight from the dataset.

        Deliberately synchronous and worker-free: this runs on at most a
        few dozen images at wide intervals, and avoiding a DataLoader
        keeps the hook free of sampler/worker state that could interact
        with training.
        """
        samples = [self.dataset[i] for i in indices]
        batch = collate(samples, samples_per_gpu=len(samples))

        img = batch['img'].data[0].to(device)
        gt = batch['gt_semantic_seg'].data[0].to(device)
        if gt.dim() == 4:
            gt = gt.squeeze(1)
        img_metas = batch['img_metas'].data[0]
        return img, gt.long(), img_metas

    def _nll(self, probs, gt):
        """Cross-entropy of an already-normalised probability map.

        ``probs`` holds probabilities, not logits, so take the log and
        use NLL rather than ``F.cross_entropy`` (which would apply a
        second softmax).
        """
        log_probs = torch.log(probs.clamp_min(self.EPS))
        return F.nll_loss(log_probs, gt, ignore_index=self.ignore_index)
