# ---------------------------------------------------------------
# LabeledDualBankAnchorModule: DualBank with a CLASS-AWARE bank_L.
#
# Extension of DualBankAnchorModule (subclass only, nothing else is
# touched).  Registration is triggered by importing this module from a
# config.
#
# bank_L is split into ``num_classes`` blocks of ``K / num_classes``
# anchors; block c belongs to class c.  It is built with the labels:
#
#   for every class c present in the batch (GT at feature resolution):
#       EM (num_iters) over the class-c pixels and the class-c anchors
#       only, seeded from the class-c block of the bank (or from a
#       farthest-point sample of class-c pixels the first time);
#       EMA-write the result into the class-c block.
#   classes absent from the batch keep their block untouched.
#
# The assignment returned to L_dapcn^L is therefore zero outside the
# pixel's own class block: intra pulls each labeled pixel towards the
# anchors of ITS class, inter keeps all 32 anchors apart.
#
# Reading bank_L (P2^u in cross mode) can use the anchor's own class as
# value (``anchor_onehot``): V_k = onehot(class(k)), so P2 is the
# attention mass per class, with no learned anchor head in between.
#
# bank_U is unchanged (label-free EM, see DualBankAnchorModule).
# ---------------------------------------------------------------

import torch
import torch.nn.functional as F_func

from mmseg.models.builder import MODELS
from mmseg.models.uda.dynamic_anchor_dualbank import DualBankAnchorModule


@MODELS.register_module(force=True)
class LabeledDualBankAnchorModule(DualBankAnchorModule):
    """DualBankAnchorModule whose first bank is built per class.

    Args:
        num_classes (int): Number of classes; ``max_groups`` must be a
            multiple of it.
        labeled_bank (str | None): Name of the class-aware bank; default
            ``bank_names[0]``.
        min_class_pixels (int): A class needs at least this many valid
            pixels in the batch to be refined / written. Default: 16.
        **kwargs: Everything :class:`DualBankAnchorModule` takes.
    """

    def __init__(self,
                 num_classes,
                 labeled_bank=None,
                 min_class_pixels=16,
                 **kwargs):
        super(LabeledDualBankAnchorModule, self).__init__(**kwargs)
        assert self.max_groups % num_classes == 0, (
            'max_groups (%d) must be a multiple of num_classes (%d)' %
            (self.max_groups, num_classes))
        self.num_classes = num_classes
        self.per_class = self.max_groups // num_classes
        self.labeled_bank = (self.bank_names[0] if labeled_bank is None
                             else labeled_bank)
        assert self.labeled_bank in self.bank_names, self.labeled_bank
        self.min_class_pixels = max(int(min_class_pixels), self.per_class)
        self.register_buffer(
            'anchor_class',
            torch.arange(self.max_groups) // self.per_class)
        self.register_buffer(
            '_cls_init', torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer(
            '_cls_writes', torch.zeros(num_classes, dtype=torch.long))

    # ------------------------------------------------------------------
    def _block(self, c):
        return slice(c * self.per_class, (c + 1) * self.per_class)

    def valid_anchors(self, bank):
        """(K,) bool: anchors that hold data (all True for label-free)."""
        ema, initialised = self._bank(bank)
        if bank != self.labeled_bank:
            return torch.full((self.max_groups, ), bool(initialised.item()),
                              dtype=torch.bool, device=ema.device)
        return self._cls_init[self.anchor_class]

    @torch.no_grad()
    def _fps(self, feats, k, pool=4096):
        """k farthest-point-sampled rows (cosine) of ``feats`` (N, C)."""
        if feats.shape[0] > pool:
            feats = feats[torch.randperm(feats.shape[0],
                                         device=feats.device)[:pool]]
        norm = F_func.normalize(feats, dim=1)
        picked = [0]
        best = 1.0 - norm @ norm[0]
        for _ in range(k - 1):
            nxt = int(best.argmax())
            picked.append(nxt)
            best = torch.minimum(best, 1.0 - norm @ norm[nxt])
        return norm[picked]

    # ------------------------------------------------------------------
    def refine_labeled(self, features, labels, update_bank=True):
        """Per-class EM on the labeled bank.

        Args:
            features (Tensor): (B, C, H, W), with gradient.
            labels (Tensor): (B, H, W) GT at feature resolution; values
                outside [0, num_classes) are ignored.

        Returns:
            tuple: ``feats`` (N_valid, C) -- the valid pixels, in the
                order the assignment rows refer to; ``assign`` (N_valid,
                K), zero outside each pixel's class block; ``proto``
                (K, C), refined for present classes and the (detached)
                bank anchors for absent ones; ``quality`` (K,) ones;
                ``present`` list of refined classes.
        """
        B, C, H, W = features.shape
        feats = features.permute(0, 2, 3, 1).reshape(-1, C)
        lab = labels.reshape(-1).long()
        valid = (lab >= 0) & (lab < self.num_classes)
        feats, lab = feats[valid], lab[valid]
        feats_norm = F_func.normalize(feats, dim=1)
        bad = torch.isnan(feats_norm).any(dim=1, keepdim=True)
        feats_norm = torch.where(bad, torch.zeros_like(feats_norm),
                                 feats_norm)

        ema, _ = self._bank(self.labeled_bank)
        bank_norm = F_func.normalize(ema.detach(), dim=1)
        rows = [bank_norm[self._block(c)] for c in range(self.num_classes)]
        assign = feats.new_zeros(feats.shape[0], self.max_groups)
        present = []
        counts = torch.bincount(lab, minlength=self.num_classes).tolist()

        for c in range(self.num_classes):
            if counts[c] < self.min_class_pixels:
                continue
            idx = (lab == c).nonzero(as_tuple=False).squeeze(1)
            f_c, fn_c = feats[idx], feats_norm[idx]
            if bool(self._cls_init[c]):
                proto = rows[c]
            else:
                proto = self._fps(f_c.detach(), self.per_class)
            a = None
            for _ in range(self.num_iters):
                sim = torch.clamp(self._similarity(fn_c, proto),
                                  min=-50.0, max=50.0)
                a = torch.softmax(sim, dim=1)              # (n_c, P)
                sizes = a.sum(dim=0).clamp(min=self.EPS * 100)
                p_new = torch.mm(a.t(), f_c) / sizes.unsqueeze(1)
                norms = torch.norm(p_new, dim=1, keepdim=True).clamp(
                    min=self.MIN_PROTO_NORM, max=self.MAX_PROTO_NORM)
                proto = F_func.normalize(p_new / norms, dim=1)
            if a is None:                                  # num_iters == 0
                a = torch.softmax(torch.clamp(
                    self._similarity(fn_c, proto), -50.0, 50.0), dim=1)
            cols = torch.arange(c * self.per_class,
                                (c + 1) * self.per_class,
                                device=feats.device)
            assign = assign.index_put(
                (idx.unsqueeze(1), cols.unsqueeze(0)), a)
            rows[c] = proto
            present.append(c)

        proto = torch.cat(rows, dim=0)
        if update_bank and self.training and present:
            self._write_labeled(proto.detach(), present)
        quality = torch.ones(self.max_groups, device=features.device)
        return feats, assign, proto, quality, present

    @torch.no_grad()
    def _write_labeled(self, proto, present):
        ema, initialised = self._bank(self.labeled_bank)
        for c in present:
            sl = self._block(c)
            if not bool(self._cls_init[c]):
                ema[sl] = proto[sl]
                self._cls_init[c] = True
            else:
                decay = self.ema_decay
                if self.ema_warmup:
                    n = float(self._cls_writes[c])
                    decay = min(decay, 1.0 - 1.0 / (n + 1.0))
                ema[sl] = ema[sl] * decay + proto[sl] * (1.0 - decay)
            self._cls_writes[c] += 1
        initialised.fill_(True)
        getattr(self, '_writes_%s' % self.labeled_bank).add_(1)

    # ------------------------------------------------------------------
    def lookup(self, features, bank):
        """Parent lookup, with anchors that hold no data masked out."""
        out = super(LabeledDualBankAnchorModule, self).lookup(features, bank)
        if out is None or bank != self.labeled_bank:
            return out
        valid = self.valid_anchors(bank)
        if bool(valid.all()):
            return out
        # Recompute from the logits so masked anchors get exactly zero.
        assign, anchors = out
        logp = torch.log(assign.clamp_min(1e-30))
        logp = logp.masked_fill(~valid.unsqueeze(0), float('-inf'))
        return torch.softmax(logp, dim=1), anchors

    def predict_onehot(self, features, bank):
        """P2 with V = onehot(anchor class): attention mass per class."""
        out = self.lookup(features, bank)
        if out is None:
            return None
        assign, _ = out
        B, _, H, W = features.shape
        onehot = F_func.one_hot(self.anchor_class,
                                self.num_classes).to(assign.dtype)
        probs = torch.mm(assign, onehot)                   # (N, cls)
        probs = probs.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return probs, assign
