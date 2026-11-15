# ---------------------------------------------------------------
# DAPCN-SSL DualBank, two optimizers + class-aware bank_L.
#
# Extension of DAPCN_SSL_DualBank (subclass only).  Registration is
# triggered by importing this module from a config.
#
# Two changes over DAPCN_SSL_DualBank:
#
# 1. bank_L is built WITH the labels (LabeledDualBankAnchorModule): one
#    block of anchors per class, EM run inside each class, and P2^u
#    reads it with V = onehot(anchor class) (``labeled_bank_value``).
#
# 2. The losses are split into two groups, each owned by an optimizer:
#
#      'main'  L_seg, L_boundary, L_dapcn^L, L_dapcn^U    (everything else)
#      'aux'   L_sup2, L_con1, L_con2                     (``aux_loss_keys``)
#
#    Each group's gradient is taken ONLY w.r.t. the parameters of its
#    own optimizer (torch.autograd.grad with explicit inputs) and kept
#    in a separate buffer, so a parameter that belongs to one optimizer
#    never sees the other group's gradient, and a parameter that
#    belongs to both receives two separate Adam updates.  Which
#    parameters each optimizer holds is decided by the config
#    (``optimizer`` / ``optimizer2``, see mmseg/apis/train.py); this
#    class reads it from the optimizers themselves.
#
#      A (isolated): aux holds only the P2 branch (W_q, W_k,
#        conv_seg_proto): L_con1 then has no parameter to update and the
#        shared encoder / head / Psi_1 learn from 'main' only.
#      B (joint):    aux also holds backbone + decode head, with its own
#        lr multipliers.
# ---------------------------------------------------------------

import torch
import torch.nn.functional as F

from mmseg.models import UDA
from mmseg.models.uda.dapcn_dualbank import DAPCN_SSL_DualBank
from mmseg.models.uda.dynamic_anchor_dualbank_labeled import \
    LabeledDualBankAnchorModule


@UDA.register_module()
class DAPCN_SSL_DualBank2Opt(DAPCN_SSL_DualBank):
    """DualBank with a class-aware bank_L and two optimizers.

    Args:
        aux_loss_keys (tuple[str]): Losses optimised by the 'aux'
            optimizer (matched against the key without the ``u.``
            prefix). Everything else whose key contains ``'loss'`` goes
            to 'main'.
        labeled_bank_value (str): How P2 reads the class-aware bank:
            'onehot' (V = onehot of the anchor's class) or 'head' (the
            parent's anchor head, conv_seg_proto / head_chung).
        **cfg: Everything :class:`DAPCN_SSL_DualBank` takes.
    """

    def __init__(self,
                 aux_loss_keys=('loss_sup2', 'loss_con1', 'loss_con2'),
                 labeled_bank_value='onehot',
                 **cfg):
        super(DAPCN_SSL_DualBank2Opt, self).__init__(**cfg)
        assert labeled_bank_value in ('onehot', 'head'), labeled_bank_value
        self.aux_loss_keys = tuple(aux_loss_keys)
        self.labeled_bank_value = labeled_bank_value
        self.class_bank = isinstance(self.dynamic_anchor,
                                     LabeledDualBankAnchorModule)
        self._opt_params = None
        self._stash = None

    # ------------------------------------------------------------------
    # class-aware bank_L
    # ------------------------------------------------------------------
    def _dapg(self, fused, bank, lam, logits=None, log_vars=None,
              labels=None):
        da = self.dynamic_anchor
        if not (self.class_bank and bank == da.labeled_bank):
            return super(DAPCN_SSL_DualBank2Opt, self)._dapg(
                fused, bank, lam, logits=logits, log_vars=log_vars)
        assert labels is not None, 'class-aware bank needs the labels'
        feats, assign, proto, quality, present = da.refine_labeled(
            fused, labels, update_bank=True)
        loss, parts = self.dapg_loss_fn(feats, assign, proto, quality)
        if log_vars is not None:
            log_vars['lbank_present'] = float(len(present))
            log_vars['lbank_ready'] = float(da._cls_init.sum())
        return lam * loss, parts

    def _predict_p2(self, fused, bank):
        da = self.dynamic_anchor
        if (self.class_bank and bank == da.labeled_bank
                and self.labeled_bank_value == 'onehot'):
            return da.predict_onehot(fused, bank)
        return super(DAPCN_SSL_DualBank2Opt, self)._predict_p2(fused, bank)

    # ------------------------------------------------------------------
    # two optimizers
    # ------------------------------------------------------------------
    def _is_aux(self, key):
        k = key[2:] if key.startswith('u.') else key
        return k in self.aux_loss_keys

    def _backward_losses(self, losses):
        if self._opt_params is None:          # single optimizer: parent
            return super(DAPCN_SSL_DualBank2Opt,
                         self)._backward_losses(losses)
        _, lv = self._parse_losses(losses)    # logging only
        lv.pop('loss', None)

        groups = {'aux': [], 'main': []}
        for k, v in losses.items():
            if 'loss' not in k:
                continue
            v = v if torch.is_tensor(v) else sum(v)
            groups['aux' if self._is_aux(k) else 'main'].append(v)
        terms = [(name, sum(vs)) for name, vs in groups.items()
                 if vs and name in self._opt_params]
        terms = [(n, t) for n, t in terms if t.requires_grad]
        for i, (name, total) in enumerate(terms):
            params = self._opt_params[name]
            grads = torch.autograd.grad(
                total, params, retain_graph=i < len(terms) - 1,
                allow_unused=True)
            stash = self._stash[name]
            for p, g in zip(params, grads):
                if g is None:
                    continue
                key = id(p)
                stash[key] = g if key not in stash else stash[key] + g
        return lv

    @staticmethod
    def _step_with_grad(opt):
        """``opt.step()`` over the param groups that have a gradient.

        torch 1.8's AdamW raises UnboundLocalError ('beta1') when a group
        has no gradient at all, which happens whenever an aux loss is
        still off (e.g. before ``con_start_iter``).  Skipping those
        groups is exactly what Adam does for a parameter without grad.
        The full group list is restored, so the lr hook (which zips over
        ``param_groups``) sees the original order.
        """
        groups = opt.param_groups
        live = [g for g in groups
                if any(p.grad is not None for p in g['params'])]
        if not live:
            return
        if len(live) == len(groups):
            opt.step()
            return
        opt.param_groups = live
        try:
            opt.step()
        finally:
            opt.param_groups = groups

    def train_step(self, data_batch, optimizer, **kwargs):
        if not isinstance(optimizer, dict):
            return super(DAPCN_SSL_DualBank2Opt, self).train_step(
                data_batch, optimizer, **kwargs)
        assert set(optimizer) == {'main', 'aux'}, set(optimizer)
        self._opt_params = {
            name: [p for g in opt.param_groups for p in g['params']
                   if p.requires_grad]
            for name, opt in optimizer.items()}
        self._stash = {name: {} for name in optimizer}
        for opt in optimizer.values():
            opt.zero_grad()
        try:
            log_vars = self(**data_batch)
            for name in ('main', 'aux'):
                stash = self._stash[name]
                params = self._opt_params[name]
                for p in params:
                    p.grad = stash.get(id(p))
                self._step_with_grad(optimizer[name])
                for p in params:
                    p.grad = None
                log_vars['opt_%s_nparam_grad' % name] = float(
                    sum(1 for p in params if id(p) in stash))
        finally:
            self._opt_params = None
            self._stash = None

        log_vars.pop('loss', None)
        return dict(log_vars=log_vars,
                    num_samples=len(data_batch['img_metas']))
