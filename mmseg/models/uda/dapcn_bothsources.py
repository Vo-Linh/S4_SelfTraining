# ---------------------------------------------------------------
# DAPCNBothSources — Step 2 (KE_HOACH_HOAN_CHINH.md, Quy trinh 1).
#
# Supervised DAPG fed by BOTH sources through a SINGLE DAPG call:
# features of X_L and X_U are concatenated BEFORE dynamic_anchor,
# so the EM refinement and the one EMA bank write per step already
# mix both distributions. No teacher, no ClassMix, no pseudo-labels.
#
#   X_L -> encoder -> F_L -+-> conv_seg -> CE + boundary (labeled only)
#                          |
#   X_U -> encoder -> F_U -+-> cat([F_L, F_U]) -> dynamic_anchor
#                                                 (update_bank=True)
#                                              -> DAPGLoss (3 terms)
#
# Loss weights live in the inner DAFormerDAPCNHead (same values as
# the Step-1 config); this wrapper owns none, so the Step1-vs-Step2
# gap is exactly "U features inside DAPG".
#
# Optional diagnostic dual banks: when the head's dynamic_anchor sets
# aux_banks=('labeled', 'unlabeled'), each step also refines F_L / F_U
# separately under no_grad into those buffers (logged as
# diag_cos_<name>). They never seed EM and never carry gradient.
#
# NEW file — no existing module is modified. It is registered under
# UDA and imported from the Step-2 config, so
# mmseg/models/uda/__init__.py stays untouched.
# ---------------------------------------------------------------

import torch
import torch.nn.functional as F

from mmseg.models import UDA
from mmseg.models.uda.uda_decorator import UDADecorator


@UDA.register_module()
class DAPCNBothSources(UDADecorator):
    """Two-source supervised DAPG with concatenated features.

    Built via the ``uda`` config block; the inner model is a plain
    EncoderDecoder + DAFormerDAPCNHead. The training batch comes from
    SSLDataset (``img``/``gt_semantic_seg`` = X_L, ``target_img`` =
    X_U); val/test paths are inherited unchanged from UDADecorator.
    """

    def __init__(self, **cfg):
        super(DAPCNBothSources, self).__init__(**cfg)

    def forward_train(self, img, img_metas, gt_semantic_seg, target_img,
                      target_img_metas):
        model = self.get_model()
        head = model.decode_head

        # --- Labeled branch: CE + boundary only (DAPG comes later,
        #     once, on the concatenated features) ---
        feats_L = model.extract_feat(img)
        seg_logits = head.forward(feats_L)
        losses = head.losses(seg_logits, gt_semantic_seg, None)

        _, _, H, W = seg_logits.shape
        gt_resized = F.interpolate(
            gt_semantic_seg.float(), size=(H, W),
            mode='nearest').long().squeeze(1)

        if head.boundary_lambda > 0:
            losses.update(
                head._boundary_loss(seg_logits, gt_resized, feats_L))

        # --- Concat branch: ONE DAPG call on [F_L; F_U] ---
        # _dapg_loss only reads inputs[-1] (encoder space, Solution 1),
        # so the other scales are passed through untouched.
        # Gradients flow to BOTH encoders (shared backbone).
        feats_U = model.extract_feat(target_img)
        if head.proto_lambda > 0:
            inputs_cat = list(feats_L)
            inputs_cat[-1] = torch.cat([feats_L[-1], feats_U[-1]], dim=0)
            losses.update(head._dapg_loss(inputs_cat, update_bank=True))

        # --- Diagnostic per-source banks (no gradient, no training
        #     effect): the SAME trained E-step refines F_L / F_U
        #     separately under no_grad; only the named aux EMA buffers
        #     are written. Convention: aux_banks[0] <- X_L,
        #     aux_banks[1] <- X_U (extra names are ignored).
        da = head.dynamic_anchor
        aux_names = list(getattr(da, 'aux_banks', ()))
        if head.proto_lambda > 0 and len(aux_names) >= 2:
            with torch.no_grad():
                da(feats_L[-1], update_bank=False, aux_bank=aux_names[0])
                da(feats_U[-1], update_bank=False, aux_bank=aux_names[1])
            for _n in aux_names:
                losses[f'diag_cos_{_n}'] = da.bank_cos_mean(_n).detach()

        # --- Contrastive bank: labeled only (needs GT) ---
        if head.contrastive_lambda > 0:
            losses.update(
                head._contrastive_loss(seg_logits, gt_resized, feats_L))

        # Forwarded once per step (we bypass head.forward_train, which
        # would do this itself); drives the contrastive warmup schedule.
        head._iter += 1
        return losses

    def train_step(self, data_batch, optimizer, **kwargs):
        """Manual backward/step (no OptimizerHook): needs
        ``optimizer_config = None`` in the config, same as DAPCN_SSL."""
        optimizer.zero_grad()
        losses = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)
        loss.backward()
        optimizer.step()

        log_vars.pop('loss', None)
        return dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
