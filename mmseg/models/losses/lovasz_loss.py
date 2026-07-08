# Lovasz-Softmax loss, ported to this v0.16-era fork's loss interface.
# Reference: Berman, Triki, Blaschko, "The Lovasz-Softmax loss" (CVPR 2018).
# Adapted from open-mmlab/mmsegmentation lovasz_loss.py.
#
# This is a direct (sub-)differentiable surrogate of the mean-IoU (Jaccard)
# metric, which is exactly what the evaluation reports. It is the most
# targeted lever for lifting rare/volatile-class IoU (here: Bareland/Road).
#
# Wiring note: BaseDecodeHead.losses() in this fork builds a SINGLE
# loss_decode, so this loss is NOT used as loss_decode. It is injected as a
# parallel, gradient-bearing auxiliary term inside
# DAPCNHeadMixin.dapcn_forward_train (gated by lovasz_lambda > 0), computed on
# the FULL-resolution upsampled logits so its IoU surrogate matches the CE /
# eval resolution.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import get_class_weight, weight_reduce_loss


def lovasz_grad(gt_sorted):
    """Compute the gradient of the Lovasz extension w.r.t. sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def flatten_probs(probs, labels, ignore_index=None):
    """Flatten predictions and labels; drop ignored pixels."""
    if probs.dim() == 3:
        probs = probs.unsqueeze(1)
    B, C, H, W = probs.size()
    probs = probs.permute(0, 2, 3, 1).contiguous().reshape(-1, C)
    labels = labels.reshape(-1)
    if ignore_index is None:
        return probs, labels
    valid = labels != ignore_index
    return probs[valid], labels[valid]


def lovasz_softmax_flat(probs, labels, classes='present', class_weight=None):
    """Multi-class Lovasz-Softmax loss on flat (N, C) probabilities.

    Returns a 0-d scalar tensor in all cases (including the empty-input case),
    so it is safe to place under a key containing 'loss' for _parse_losses.
    """
    if probs.numel() == 0:
        # No valid pixels: return a real 0-d scalar (NOT a (0, C) tensor) so
        # downstream .mean()/sum() over the loss dict does not choke.
        return probs.sum() * 0.0
    C = probs.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ('all', 'present') else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground mask for class c
        if classes == 'present' and fg.sum() == 0:
            continue
        class_pred = probs[:, 0] if C == 1 else probs[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        fg_sorted = fg[perm.data]
        loss = torch.dot(errors_sorted, lovasz_grad(fg_sorted))
        if class_weight is not None:
            loss = loss * class_weight[c]
        losses.append(loss)
    if len(losses) == 0:
        return probs.sum() * 0.0
    return torch.stack(losses).mean()


def lovasz_softmax(probs,
                   labels,
                   classes='present',
                   per_image=False,
                   class_weight=None,
                   reduction='mean',
                   avg_factor=None,
                   ignore_index=255):
    """Multi-class Lovasz-Softmax loss.

    Args:
        probs (Tensor): (B, C, H, W) class probabilities (post-softmax).
        labels (Tensor): (B, H, W) ground-truth labels.
    """
    if per_image:
        out = [
            lovasz_softmax_flat(
                *flatten_probs(
                    prob.unsqueeze(0), label.unsqueeze(0), ignore_index),
                classes=classes,
                class_weight=class_weight)
            for prob, label in zip(probs, labels)
        ]
        loss = weight_reduce_loss(
            torch.stack(out), None, reduction, avg_factor)
    else:
        loss = lovasz_softmax_flat(
            *flatten_probs(probs, labels, ignore_index),
            classes=classes,
            class_weight=class_weight)
    return loss


@LOSSES.register_module()
class LovaszLoss(nn.Module):
    """Lovasz-Softmax loss (multi-class mean-IoU surrogate).

    Args:
        loss_type (str): 'multi_class' (softmax). Default: 'multi_class'.
        classes (str | list): 'present' (default) | 'all' | explicit list.
        per_image (bool): per-image vs. whole-batch surrogate. Default: False.
        reduction (str): used only when per_image=True. Default: 'mean'.
        class_weight (list[float] | str | None): per-class weight. Default: None.
        loss_weight (float): module-level scale. Default: 1.0.
    """

    def __init__(self,
                 loss_type='multi_class',
                 classes='present',
                 per_image=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        super().__init__()
        assert loss_type == 'multi_class', \
            "LovaszLoss only supports loss_type='multi_class'"
        assert classes in ('all', 'present') or isinstance(
            classes, (list, tuple))
        self.classes = classes
        self.per_image = per_image
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):
        """Forward.

        Args:
            cls_score (Tensor): (B, C, H, W) raw logits.
            label (Tensor): (B, H, W) ground-truth labels.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        # Lovasz-Softmax expects probabilities.
        probs = F.softmax(cls_score, dim=1)
        loss = self.loss_weight * lovasz_softmax(
            probs,
            label,
            self.classes,
            self.per_image,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=ignore_index)
        return loss
