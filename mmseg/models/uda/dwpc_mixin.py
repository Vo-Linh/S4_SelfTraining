# ---------------------------------------------------------------
# DWPC — Dual-Witness Pseudo-label Correction (shared orchestration)
#
# A mixin reused by both DAPCN (UDA) and DAPCN_SSL so the dual-witness
# flip + per-pixel weighting logic stays identical across the two
# six-step training loops.  The host module must provide:
#   - self.proto_memory          (PrototypeMemory or None)
#   - self._get_anchor_features  (encoder features -> witness-A space)
#   - the wb_* buffers registered by _dwpc_init (when Witness B is on)
#
# Decision rule (DWPC_PROPOSAL.md §4.4):
#   F_i = a_i * s_i(c*)                         (veto: both must agree)
#   flip to c* iff F_i > tau_flip AND in per-image top-K(F) AND c* rare
#   w_i = sigmoid(beta_s * s_i(y))
#         * [1 - (1 - a_i)(1 - r_conf)]         (either witness vouches)
# A warmup ramp omega(t) degrades DWPC to the prior scalar-weight
# behaviour exactly at t = dwpc_start_iter.
# ---------------------------------------------------------------

import torch
import torch.nn.functional as F

from mmseg.models.utils.witness_appearance import (
    witness_appearance,
    build_rare_prior_logits,
)
from mmseg.models.utils.witness_structure import witness_structure


class DWPCMixin:
    """Dual-Witness Pseudo-label Correction, shared by DAPCN / DAPCN_SSL."""

    def _dwpc_init(self, dwpc_cfg, num_classes, class_freq=None):
        """Read DWPC config, register Witness-B EMA buffers + rare prior.

        Args:
            dwpc_cfg (dict | None): the ``dwpc=dict(...)`` config block.
            num_classes (int): C.
            class_freq (list[float] | None): per-class pixel frequency for
                the inverse-frequency rare prior.
        """
        cfg = dict(dwpc_cfg) if dwpc_cfg else {}
        self.dwpc_enabled = cfg.get('enabled', False)
        self.dwpc_start_iter = cfg.get('start_iter', 1500)
        self.dwpc_ramp_iters = cfg.get('ramp_iters', 1000)

        # Witness A (appearance)
        self.witness_a_enabled = cfg.get('witness_a_enabled', True)
        self.witness_a_symmetric = cfg.get('witness_a_symmetric', False)
        self.witness_a_beta = cfg.get('witness_a_beta', 1.0)
        self.witness_a_sigma2 = cfg.get('witness_a_sigma2', 0.5)
        self.witness_a_rare_prior_gamma = cfg.get('witness_a_rare_prior_gamma',
                                                  0.0)
        self.witness_a_standardize = cfg.get('witness_a_standardize', True)

        # Witness B (structure)
        self.witness_b_enabled = cfg.get('witness_b_enabled', True)
        self.witness_b_ema = cfg.get('witness_b_ema', 0.99)
        self.witness_b_cc_iters = cfg.get('witness_b_cc_iters', 64)
        self.witness_b_resolution = cfg.get('witness_b_resolution', 128)
        self.witness_b_enable_containment = cfg.get(
            'witness_b_enable_containment', True)
        self.witness_b_min_count = cfg.get('witness_b_min_count', 5)
        self.target_adaptive = cfg.get('target_adaptive', True)

        # Fusion / decision
        self.rare_gate = cfg.get('rare_gate', True)
        self.flip_veto = cfg.get('flip_veto', True)
        self.rare_class_ids = list(cfg.get('rare_class_ids', []))
        self.dwpc_tau_flip = cfg.get('tau_flip', 0.5)
        self.dwpc_flip_budget = cfg.get('flip_budget', 0.005)
        self.dwpc_beta_s = cfg.get('beta_s', 5.0)
        self.dwpc_num_classes = num_classes

        if class_freq is None:
            class_freq = cfg.get('class_freq', None)

        # Rare-class prior (None when gamma == 0)
        rp = build_rare_prior_logits(
            num_classes, self.rare_class_ids, class_freq=class_freq,
            gamma=self.witness_a_rare_prior_gamma)
        if rp is not None:
            self.register_buffer('dwpc_rare_prior_logits', rp)
        else:
            self.dwpc_rare_prior_logits = None

        # Witness-B target-adaptive EMA buffers
        if self.dwpc_enabled and self.witness_b_enabled:
            C = num_classes
            self.register_buffer('wb_area_mean', torch.zeros(C))
            self.register_buffer('wb_conn_mean', torch.zeros(C))
            self.register_buffer('wb_contain_mean', torch.zeros(C))
            self.register_buffer('wb_cooc', torch.zeros(C, C))
            self.register_buffer('wb_count', torch.zeros(C))

    def _dwpc_needed(self, local_iter):
        """True when DWPC should run this iteration."""
        return self.dwpc_enabled and local_iter >= self.dwpc_start_iter

    def _dwpc_ramp(self, local_iter):
        return min(max((local_iter - self.dwpc_start_iter) /
                       max(self.dwpc_ramp_iters, 1), 0.0), 1.0)

    @torch.no_grad()
    def _dwpc_correct(self, ema_softmax, student_feat, local_iter,
                      scalar_weight):
        """Run both witnesses, flip, and produce a per-pixel weight.

        Args:
            ema_softmax (Tensor): (B, C, H, W) teacher softmax (full res).
            student_feat: encoder features on the unlabeled/target image
                (list of multi-scale tensors), already extracted no_grad.
            local_iter (int): current iteration.
            scalar_weight (float): baseline pseudo_weight (fraction of
                high-confidence pixels) to blend toward during warmup.

        Returns:
            tuple: corrected_label (B, H, W) long, w_i (B, H, W) float.
        """
        B, C, H, W = ema_softmax.shape
        device = ema_softmax.device
        pseudo_prob, pseudo_label = ema_softmax.max(dim=1)   # (B,H,W)
        r_conf = pseudo_prob
        omega = self._dwpc_ramp(local_iter)

        # ---------- Witness A (appearance) ----------
        if self.witness_a_enabled and getattr(self, 'proto_memory', None) \
                is not None:
            # Witness-A features must live in the SAME space as the
            # PrototypeMemory bank. DAPCN_SSL keeps the bank in the fused
            # decoder space (`_get_proto_features`); the UDA DAPCN falls
            # back to its own `_get_anchor_features`.
            getter = getattr(self, '_get_proto_features',
                             self._get_anchor_features)
            feat = getter(student_feat)   # (B,D,Hf,Wf)
            Hf, Wf = feat.shape[-2:]
            tl = F.interpolate(pseudo_label.float().unsqueeze(1),
                               size=(Hf, Wf),
                               mode='nearest').long().squeeze(1)
            a_lo, cstar_lo = witness_appearance(
                feat, tl, self.proto_memory, self.rare_class_ids,
                beta=self.witness_a_beta, sigma2=self.witness_a_sigma2,
                rare_prior_logits=self.dwpc_rare_prior_logits,
                symmetric=self.witness_a_symmetric,
                standardize=self.witness_a_standardize)
            a_map = F.interpolate(a_lo.unsqueeze(1), size=(H, W),
                                  mode='bilinear',
                                  align_corners=False).squeeze(1)
            cstar_map = F.interpolate(
                cstar_lo.float().unsqueeze(1), size=(H, W),
                mode='nearest').long().squeeze(1)
        else:
            a_map = torch.zeros(B, H, W, device=device)
            cstar_map = torch.full((B, H, W), -1, dtype=torch.long,
                                   device=device)

        # ---------- Witness B (structure) ----------
        if self.witness_b_enabled:
            res = self.witness_b_resolution
            if res > 0 and (H, W) != (res, res):
                pl_lo = F.interpolate(
                    pseudo_label.float().unsqueeze(1), size=(res, res),
                    mode='nearest').long().squeeze(1)
                cs_lo = F.interpolate(
                    cstar_map.float().unsqueeze(1), size=(res, res),
                    mode='nearest').long().squeeze(1)
            else:
                pl_lo, cs_lo = pseudo_label, cstar_map
            bufs = dict(
                area_mean=self.wb_area_mean, conn_mean=self.wb_conn_mean,
                contain_mean=self.wb_contain_mean, cooc=self.wb_cooc,
                count=self.wb_count)
            s_map_lo, s_cstar_lo, new_bufs = witness_structure(
                pl_lo, cs_lo, self.dwpc_num_classes, self.rare_class_ids,
                bufs, target_adaptive=self.target_adaptive,
                cc_iters=self.witness_b_cc_iters, ema=self.witness_b_ema,
                min_count=self.witness_b_min_count,
                enable_containment=self.witness_b_enable_containment)
            if self.target_adaptive:
                self.wb_area_mean.copy_(new_bufs['area_mean'])
                self.wb_conn_mean.copy_(new_bufs['conn_mean'])
                self.wb_contain_mean.copy_(new_bufs['contain_mean'])
                self.wb_cooc.copy_(new_bufs['cooc'])
                self.wb_count.copy_(new_bufs['count'])
            s_map = F.interpolate(s_map_lo.unsqueeze(1), size=(H, W),
                                  mode='bilinear',
                                  align_corners=False).squeeze(1)
            s_cstar = F.interpolate(s_cstar_lo.unsqueeze(1), size=(H, W),
                                    mode='bilinear',
                                    align_corners=False).squeeze(1)
        else:
            s_map = torch.ones(B, H, W, device=device)
            s_cstar = torch.ones(B, H, W, device=device)

        # ---------- Flip decision (dual-witness veto + budget) ----------
        F_map = a_map * s_cstar if self.flip_veto else a_map
        tau_eff = self.dwpc_tau_flip + (1 - omega) * (1 - self.dwpc_tau_flip)
        flip_mask = (F_map > tau_eff) & (cstar_map >= 0)

        budget = int(self.dwpc_flip_budget * H * W)
        if budget > 0:
            Ff = F_map.reshape(B, -1)
            k = min(budget, Ff.shape[1])
            kth = Ff.topk(k, dim=1).values[:, -1].view(B, 1, 1)
            flip_mask = flip_mask & (F_map >= kth)
        else:
            flip_mask = torch.zeros_like(flip_mask)

        corrected_label = torch.where(flip_mask, cstar_map, pseudo_label)

        # ---------- Per-pixel weight ----------
        a_gate = a_map if self.rare_gate else torch.zeros_like(a_map)
        w_struct = torch.sigmoid(self.dwpc_beta_s * s_map)
        w_i = w_struct * (1.0 - (1.0 - a_gate) * (1.0 - r_conf))

        # warmup blend toward scalar baseline (exact degradation at t0)
        w_i = (1 - omega) * scalar_weight + omega * w_i

        return corrected_label, w_i
