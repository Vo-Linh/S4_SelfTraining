# ---------------------------------------------------------------
# DAPCN-CPS: Dual-Student Cross Pseudo Supervision + DAPCN-SSL
#
# Extension of DAPCN_SSL, not a modification: this module only
# subclasses and composes existing code (Open/Closed Principle).
# No file outside this one is touched; registration is triggered
# by importing this module (done from the new config files).
#
# Design (see KE_HOACH_CPS_DAPCN.md, Final):
#   D1  Two students, each running the FULL DAPCN_SSL pipeline
#       (supervised CE + DAPCN losses on labeled, EMA-teacher
#       pseudo-labels + ClassMix on unlabeled), plus a cross
#       pseudo-supervision loss between the students (CPS,
#       arXiv:2106.01226).
#   D2  The cross loss runs on BOTH labeled and unlabeled images,
#       both directions (S1<-S2 and S2<-S1), like CPS.
#   D3  Each student keeps its OWN EMA teacher (untouched per-branch
#       machinery: proto-correction, warmup, threshold).
#   D4  DAPCN-on-unlabeled is selected per config via the existing
#       apply_boundary_on_target / apply_proto_on_target flags
#       (normal config: on; fallback config: off).  Note the
#       pre-existing caveat, unchanged here: the boundary loss
#       accepts confidence weights, DAPGLoss does not, so on the
#       mixed branch the proto term is unweighted.
#   D5  The cross weight ramps linearly from 0 over
#       ``cps_warmup_iters`` (same pattern as pseudo_label_warmup).
#   D6  Validation/inference use student 1 (inherited val path).
#
# Branch 2 initialisation:
#   Branch 2 is built under a different torch RNG seed, so every
#   randomly-initialised parameter (decode head, DynamicAnchorModule
#   prototypes, ...) differs from branch 1.  Pretrained backbone
#   weights load deterministically and are therefore identical by
#   design; the different random parts plus per-forward stochasticity
#   (dropout/DropPath, ClassMix draws) give CPS the perturbation it
#   needs to avoid collapse.  The global RNG state is restored
#   afterwards, so runs stay reproducible.
#
# Memory discipline:
#   Each branch backward()s its own graphs internally (unchanged).
#   The cross terms forward one image batch through one student at a
#   time and backward immediately, so the cross stage never holds
#   more than a single forward graph.  Peer targets are computed
#   under no_grad (stop-gradient, as in CPS).
# ---------------------------------------------------------------

from copy import deepcopy

import torch
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.models import UDA
from mmseg.models.uda.dapcn_ssl import DAPCN_SSL


class DAPCN_SSL_CPSBranch(DAPCN_SSL):
    """Branch base: 3D confidence weight for the affinity loss.

    Latent-shape fix confined to this new file (old files untouched).
    ``AffinityBoundaryLoss`` documents ``pseudo_weight`` as (B, H, W)
    and unsqueezes it internally, but
    ``DAPCN_SSL._compute_dapcn_losses`` hands ``_resize_to_feature``
    a 4D (B, 1, H, W) tensor.  The labeled path always passes None
    and every shipped SSL config keeps
    ``apply_boundary_on_target=False``, so this never fired —
    enabling DAPCN-on-unlabeled (D4=b) is the first to hit it.
    Squeezing the channel dim here restores the documented contract.
    Unaffected: None weights (labeled path) and the binary/hybrid
    modes (which do not route weights through this method the same
    way / already squeeze).
    """

    def _resize_to_feature(self, feat, seg_label, pseudo_weight):
        seg_label, pseudo_weight = super()._resize_to_feature(
            feat, seg_label, pseudo_weight)
        if pseudo_weight is not None and pseudo_weight.dim() == 4:
            pseudo_weight = pseudo_weight.squeeze(1)
        return seg_label, pseudo_weight


@UDA.register_module()
class DAPCN_CPS(DAPCN_SSL_CPSBranch):
    """Dual-student CPS on top of DAPCN-SSL.

    ``self`` IS branch 1 (all inherited machinery: student, EMA
    teacher, DynamicAnchorModule, losses).  Branch 2 is a second,
    independently-initialised :class:`DAPCN_SSL_CPSBranch` held as a
    submodule, so its parameters join the optimizer and checkpoints
    automatically, and the inherited val/test path keeps reporting
    student 1 (decision D6).

    Args:
        cps_lambda (float): Weight of the cross pseudo-supervision
            loss. Default: 1.0.
        cps_warmup_iters (int): Iterations over which the cross
            weight ramps linearly from 0 to ``cps_lambda``.
            Default: 0 (full weight from the first step).
        branch2_seed (int): Torch manual seed used ONLY while
            building branch 2, so its random inits differ from
            branch 1. Default: 1.
        **cfg: All :class:`DAPCN_SSL` kwargs, shared by both
            branches (each branch deep-copies what it builds from).
    """

    def __init__(self,
                 cps_lambda=1.0,
                 cps_warmup_iters=0,
                 branch2_seed=1,
                 **cfg):
        super(DAPCN_CPS, self).__init__(**cfg)

        self.cps_lambda = cps_lambda
        self.cps_warmup_iters = cps_warmup_iters
        self.branch2_seed = branch2_seed

        # Build branch 2 under a different RNG state so its random
        # inits differ; restore afterwards for reproducibility.
        cpu_rng_state = torch.get_rng_state()
        try:
            torch.manual_seed(branch2_seed)
            self.branch2 = DAPCN_SSL_CPSBranch(**deepcopy(cfg))
        finally:
            torch.set_rng_state(cpu_rng_state)

    def _get_cps_scale(self):
        """Linear warmup scale for the cross loss (cf. D5)."""
        if self.cps_warmup_iters <= 0:
            return 1.0
        return min(1.0, self.local_iter / self.cps_warmup_iters)

    def _cross_terms(self, img, img_metas, target_img, target_img_metas,
                     scale):
        """Cross pseudo-supervision on labeled + unlabeled (cf. D2).

        For each of the two clean inputs, both students predict; each
        student is supervised by the peer's detached argmax, both
        directions.  Each term is backwarded immediately so only one
        forward graph is alive at a time.

        Args:
            img (Tensor): Labeled images.
            img_metas (list[dict]): Labeled image info.
            target_img (Tensor): Unlabeled images.
            target_img_metas (list[dict]): Unlabeled image info.
            scale (float): Warmup scale in [0, 1].

        Returns:
            dict: Unweighted per-input cross losses for logging.
        """
        s1 = self.get_model()
        s2 = self.branch2.get_model()
        weight = self.cps_lambda * scale / 4.0
        log_vars = {}
        for x, m, tag in ((img, img_metas, 'src'),
                          (target_img, target_img_metas, 'tgt')):
            with torch.no_grad():
                tgt_for_s2 = s1.encode_decode(x, m).argmax(dim=1)
                tgt_for_s1 = s2.encode_decode(x, m).argmax(dim=1)
            loss_s1 = F.cross_entropy(s1.encode_decode(x, m), tgt_for_s1)
            (weight * loss_s1).backward()
            loss_s2 = F.cross_entropy(s2.encode_decode(x, m), tgt_for_s2)
            (weight * loss_s2).backward()
            log_vars['loss_cps_%s' % tag] = (
                (loss_s1 + loss_s2) / 2.0).item()
        return log_vars

    def forward_train(self, img, img_metas, gt_semantic_seg, target_img,
                      target_img_metas):
        """Dual-branch forward: both pipelines + cross loss.

        Branch 1 runs first (inherited pipeline, backwarded
        internally), then branch 2 identically; the cross terms run
        last on the two CLEAN inputs so both students see exactly the
        same pixels, as CPS requires (the branches' own ClassMix
        draws intentionally differ — extra perturbation).
        """
        # === Branch 1: inherited full pipeline (backward inside) ===
        log_vars = super(DAPCN_CPS, self).forward_train(
            img, img_metas, gt_semantic_seg, target_img,
            target_img_metas)

        # === Branch 2: identical pipeline, own teacher (D3) ===
        log_vars_b2 = self.branch2.forward_train(
            img, img_metas, gt_semantic_seg, target_img,
            target_img_metas)
        log_vars.update(add_prefix(log_vars_b2, 'b2'))

        # === Cross pseudo-supervision (D2, D5) ===
        # self.local_iter was already incremented by branch 1 above;
        # both branches stay in lockstep (one step each per call).
        scale = self._get_cps_scale()
        if self.cps_lambda > 0 and scale > 0:
            log_vars.update(
                self._cross_terms(
                    img, img_metas, target_img, target_img_metas,
                    scale))
        log_vars['cps_scale'] = scale

        return log_vars

    def train_step(self, data_batch, optimizer, **kwargs):
        """Single training iteration, both branches.

        Mirrors the inherited DynAnchor Adam-state reset for branch 2
        (same activation boundary, same corrector exemption), then
        delegates everything else to branch 1's step.
        """
        b2 = self.branch2
        if (b2.dynamic_anchor is not None
                and b2.corrector_lambda == 0
                and b2.local_iter == b2.proto_correction_start_iter):
            for p in b2.dynamic_anchor.parameters():
                if p in optimizer.state:
                    del optimizer.state[p]
            if b2.proto_to_decoder is not None:
                for p in b2.proto_to_decoder.parameters():
                    if p in optimizer.state:
                        del optimizer.state[p]

        return super(DAPCN_CPS, self).train_step(
            data_batch, optimizer, **kwargs)
