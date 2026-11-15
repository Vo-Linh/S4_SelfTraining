# ---------------------------------------------------------------
# DAPCN-SSL DualBank: two anchor banks, cross-attention read-out.
#
# Extension of DAPCN_SSL, not a modification: this module subclasses
# and composes existing code (Open/Closed).  Registration is triggered
# by importing it from the config.
#
#   x^l -> Encoder -> F^l --+-> Head psi1 ------------------> P1^l -> CE(y)
#                           |-> bank_L  (EM+EMA, L_dapcn^L)
#                           '-> Q=F^l . bank_U (cross-attn) -> P2^l -> CE(y)
#   x^u -> Encoder -> F^u --+-> Head psi1 ------------------> P1^u
#                           |-> bank_U  (EM+EMA, L_dapcn^U)
#                           '-> Q=F^u . bank_L (cross-attn) -> P2^u
#   L_con,1 = CE(P1^u, y2_hat) * w      L_con,2 = CE(P2^u, y1_hat) * w
#
# * F is the fused decoder feature (256-d), the space conv_seg lives
#   in (anchor_after_fusion=True).
# * Banks are built without class labels.  Class information lives in
#   the anchor head (``conv_seg_proto``) and in W_q / W_k only.
# * F is detached into the cross-attention (DualBankAnchorModule
#   .lookup), so P2 never shapes the encoder directly; it only does so
#   through y2_hat -> L_con,1 -> P1.
# * No EMA teacher and no ClassMix: the two branches of ONE student
#   supervise each other (CPS-style, but between P1 and P2).
# * ``dualbank_mode``:  'cross' reads the OTHER stream's bank (the
#   method),  'own' reads the stream's own bank (ablation).
#
# Memory discipline: each stream backwards its own graph before the
# next stream starts.  Streams are independent within a step because
# banks are EMA buffers, not graph nodes.
# ---------------------------------------------------------------

import contextlib

import torch
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from mmseg.models import UDA
from mmseg.models.uda.dapcn_ssl import DAPCN_SSL
from mmseg.models.uda.dynamic_anchor_dualbank import (DualBankAnchorModule,
                                                      HeadChung)


@UDA.register_module()
class DAPCN_SSL_DualBank(DAPCN_SSL):
    """Two-bank, cross-attention semi-supervised DAPCN.

    Required config (checked): ``dynamic_anchor.type =
    'DualBankAnchorModule'``, ``anchor_after_fusion=True`` and
    ``corrector_lambda > 0`` (this builds ``conv_seg_proto``, the head
    that turns anchors into class distributions).  The corrector loss
    itself is NOT used; ``sup2_lambda`` replaces it.

    Args:
        dualbank_mode (str): 'cross' or 'own'. Default: 'cross'.
        sup2_lambda (float): Weight of L_sup,2 (P2 on labeled).
        proto_lambda_u (float | None): Weight of L_dapcn^U; defaults to
            ``proto_lambda``.
        con_lambda (float): Weight of the cross pseudo-supervision
            (L_con,1 + L_con,2). 0 disables the unlabeled P1/P2
            supervision entirely (labeled-only phase).
        con1_lambda, con2_lambda (float | None): Separate weights for
            L_con,1 (P1 <- y2, reaches the encoder) and L_con,2
            (P2 <- y1, reaches only W_q / W_k / the anchor head). None
            (default) means "use ``con_lambda``". They exist to tell
            which of the two losses drives the collapse seen when both
            are on: set one to 0 and keep the other.
        con_start_iter (int): Iteration the cross loss switches on;
            the banks need time to leave their cold start first.
        con_warmup_iters (int): Linear ramp of the cross weight.
        con_threshold (float): ``con_select='threshold'``: a pixel's
            pseudo-label counts when the producing branch's
            max-probability reaches this value (w = 1[conf >= t]).
        con_select (str): 'threshold' or 'percentile'. P2 is a mixture
            of anchor class distributions, so its max-probability
            rarely reaches the P1-style 0.9 (measured: 0% of pixels)
            and L_con,1 would never fire. 'percentile' keeps, per
            branch and per batch, the ``con_percentile`` most confident
            pixels instead.
        con_percentile (float): Share of pixels kept per branch under
            'percentile'. Default: 0.4.
        con_min_conf (float): Floor applied on top of the percentile
            cut, so a branch with no confident pixel contributes none.
        head_mode (str): 'legacy' reads banks with W_q / W_k and the
            separate ``conv_seg_proto`` (cosine x 1/tau).  'slide' follows
            the method slide: Q = F, K and V from one shared 1x1 ``Head
            chung`` applied to the bank, scaled dot-product / sqrt(D).
        k_init_gain (float): Initial gain of the Head chung key map
            (see :class:`HeadChung`). Used by 'slide' only.
        bn_freeze_unlabeled (bool): Do not let the unlabeled stream's
            forward write BatchNorm running statistics.  Without it the
            stream that runs last each step (unlabeled) dominates the
            statistics used at eval time (measured: labeled-image
            accuracy 0.72 as saved vs 0.93 after re-estimating BN).
        clean_pseudo_labels (bool): Produce y1 / conf1 from P1 without
            dropout (the training-mode logits carry Dropout2d noise).
        nll_norm (str): 'all' divides P2 losses by every pixel (the
            original behaviour); 'valid' by the non-ignored pixels, like
            the cross-entropy used for P1.
        con_select='classwise': per-class confidence threshold in the
            style of FlexMatch.  A class whose predictions are rarely
            confident gets a lower threshold, so a class cannot starve
            (the collapse seen at 24k+ absorbed whole classes).
        resid_start_iter (int): From this iteration on, the residual
            anchors of the unlabeled bank (``dynamic_anchor.resid_k``)
            are fitted on pixels the LABELED bank covers poorly.
            Before it (and while bank_L is empty) all anchors run the
            plain EM.
        resid_cov_quantile (float): A pixel counts as "poorly covered"
            when its max cosine to bank_L is below this quantile of the
            batch (smoothed by an EMA, ``_cov_c0``).
        resid_scale (float): Width of the sigmoid around that cut.
        resid_floor (float): Smallest pixel weight, so a residual
            anchor never loses all its mass.
        resid_boundary_radius (int): Pixels within this many feature
            cells of a change in P1's label map get weight ~0 (54% of
            poorly covered pixels are class boundaries, whose feature
            mixes two classes). 0 disables.
        diag_interval (int): Iterations between bank diagnostics.
    """

    def __init__(self,
                 dualbank_mode='cross',
                 sup2_lambda=1.0,
                 proto_lambda_u=None,
                 con_lambda=1.0,
                 con1_lambda=None,
                 con2_lambda=None,
                 con_start_iter=1000,
                 con_warmup_iters=1000,
                 con_threshold=0.9,
                 con_select='threshold',
                 con_percentile=0.4,
                 con_min_conf=0.0,
                 head_mode='legacy',
                 k_init_gain=10.0,
                 bn_freeze_unlabeled=False,
                 clean_pseudo_labels=False,
                 nll_norm='all',
                 cls_status_momentum=0.99,
                 resid_start_iter=1000,
                 resid_cov_quantile=0.2,
                 resid_scale=0.02,
                 resid_floor=0.05,
                 resid_boundary_radius=1,
                 diag_interval=50,
                 **cfg):
        super(DAPCN_SSL_DualBank, self).__init__(**cfg)
        assert dualbank_mode in ('cross', 'own'), dualbank_mode
        assert con_select in ('threshold', 'percentile', 'classwise'), con_select
        assert head_mode in ('legacy', 'slide'), head_mode
        assert nll_norm in ('all', 'valid'), nll_norm
        assert 0.0 < con_percentile <= 1.0, con_percentile
        assert isinstance(self.dynamic_anchor, DualBankAnchorModule), (
            "uda.dynamic_anchor.type must be 'DualBankAnchorModule'")
        assert self.anchor_after_fusion, (
            'DualBank needs anchor_after_fusion=True (fused 256-d F).')
        if head_mode == 'legacy':
            assert self.conv_seg_proto is not None, (
                'set corrector_lambda > 0 so conv_seg_proto is built.')

        self.dualbank_mode = dualbank_mode
        self.sup2_lambda = sup2_lambda
        self.proto_lambda_u = (
            self.proto_lambda if proto_lambda_u is None else proto_lambda_u)
        self.con_lambda = con_lambda
        self.con1_lambda = con_lambda if con1_lambda is None else con1_lambda
        self.con2_lambda = con_lambda if con2_lambda is None else con2_lambda
        self.con_start_iter = con_start_iter
        self.con_warmup_iters = con_warmup_iters
        self.con_threshold = con_threshold
        self.con_select = con_select
        self.con_percentile = con_percentile
        self.con_min_conf = con_min_conf
        self.diag_interval = diag_interval
        self.head_mode = head_mode
        self.bn_freeze_unlabeled = bn_freeze_unlabeled
        self.clean_pseudo_labels = clean_pseudo_labels
        self.nll_norm = nll_norm
        self.cls_status_momentum = cls_status_momentum
        self.resid_start_iter = resid_start_iter
        self.resid_cov_quantile = resid_cov_quantile
        self.resid_scale = resid_scale
        self.resid_floor = resid_floor
        self.resid_boundary_radius = resid_boundary_radius
        self.register_buffer('_cov_c0', torch.zeros(()))
        self.register_buffer('_cov_c0_init', torch.zeros((), dtype=torch.bool))

        num_classes = self.get_model().decode_head.num_classes
        if head_mode == 'slide':
            self.head_chung = HeadChung(
                self.dynamic_anchor.feature_dim, num_classes, k_init_gain)
            # the legacy anchor head is not used in this mode
            self.conv_seg_proto = None
        # learning status of each class, per branch (P1, P2)
        self.register_buffer('_cls_status', torch.zeros(2, num_classes))

        # No EMA teacher in this method: free the second backbone that
        # DAPCN_SSL.__init__ built.
        self.ema_model = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _read_bank(self, stream_bank):
        """Bank a stream reads under the configured mode."""
        if self.dualbank_mode == 'cross':
            return self.dynamic_anchor.other_bank(stream_bank)
        return stream_bank

    def _con_scale(self):
        if self.local_iter < self.con_start_iter:
            return 0.0
        if self.con_warmup_iters <= 0:
            return 1.0
        return min(1.0, (self.local_iter - self.con_start_iter) /
                   self.con_warmup_iters)

    def _confident(self, conf, y=None, branch=0):
        """0/1 mask of the pseudo-label pixels a branch may teach with.

        Args:
            conf (Tensor): Max-probability of the producing branch.
            y (Tensor | None): Its predicted class (needed by 'classwise').
            branch (int): 0 for P1, 1 for P2 (selects the class status).
        """
        if self.con_select == 'threshold':
            return (conf >= self.con_threshold).float()
        if self.con_select == 'classwise':
            return self._confident_classwise(conf, y, branch)
        cut = torch.quantile(
            conf.flatten().float(), 1.0 - self.con_percentile)
        cut = torch.clamp(cut, min=self.con_min_conf)
        return (conf >= cut).float()

    def _confident_classwise(self, conf, y, branch):
        """FlexMatch-style class-conditional threshold (the slide's w_c).

        status_c   EMA of the share of pixels predicted as class c with
                   conf >= con_threshold  (how well c is being learned)
        beta_c   = status_c / max_c status_c
        thr_c    = con_threshold * beta_c / (2 - beta_c), clamped to
                   [con_min_conf, con_threshold]

        A pixel of predicted class c is kept when conf >= thr_c.
        """
        assert y is not None, "con_select='classwise' needs the labels"
        num_classes = self._cls_status.shape[1]
        sigma = torch.bincount(
            y[conf >= self.con_threshold].flatten(),
            minlength=num_classes).float() / max(conf.numel(), 1)
        status = self._cls_status[branch]
        mom = self.cls_status_momentum
        status.mul_(mom).add_(sigma.to(status.device), alpha=1.0 - mom)
        beta = status / status.max().clamp_min(1e-6)
        thr = (self.con_threshold * beta / (2.0 - beta)).clamp(
            min=self.con_min_conf, max=self.con_threshold)
        self._last_thr = thr.detach()
        return (conf >= thr[y]).float()

    def _nll(self, probs, label, ignore_index=255, weight=None):
        """NLL on probabilities (P2 is a distribution, not logits)."""
        loss = F.nll_loss(
            torch.log(probs.clamp_min(1e-6)), label,
            ignore_index=ignore_index, reduction='none')
        if weight is not None:
            loss = loss * weight
        if getattr(self, 'nll_norm', 'all') == 'valid':
            denom = (label != ignore_index).sum().clamp_min(1)
        else:
            denom = loss.numel()
        return loss.sum() / denom

    def _predict_p2(self, fused, bank):
        """P2 for ``fused`` read from ``bank`` under the configured head."""
        if self.head_mode == 'slide':
            return self.dynamic_anchor.attend(fused, bank, self.head_chung)
        return self.dynamic_anchor.predict(fused, bank, self.conv_seg_proto)

    @contextlib.contextmanager
    def _frozen_bn_stats(self):
        """Forward in train mode without writing BN running statistics."""
        if not self.bn_freeze_unlabeled:
            yield
            return
        bns = [m for m in self.get_model().modules()
               if isinstance(m, _BatchNorm)]
        old = [m.momentum for m in bns]
        for m in bns:
            m.momentum = 0.0          # running = (1 - 0) * running + 0 * batch
        try:
            yield
        finally:
            for m, o in zip(bns, old):
                m.momentum = o

    def _p1_for_labels(self, head, fused, logits):
        """P1 probabilities used to make y1 / conf1 (no gradient)."""
        with torch.no_grad():
            if self.clean_pseudo_labels:
                return torch.softmax(head.conv_seg(fused.detach()), dim=1)
            return torch.softmax(logits.detach(), dim=1)

    @torch.no_grad()
    def _resid_weights(self, fused, logits, bank, log_vars):
        """Pixel weights for the residual anchors of the unlabeled bank.

            w_i = floor + (1 - floor) * sigmoid((c0 - cov_i) / s) * interior_i

        cov_i is the max cosine of the pixel to the OTHER bank's anchors
        (bank_L); c0 its running ``resid_cov_quantile`` quantile, so the
        poorest-covered ~20% of pixels get w close to 1.  ``interior_i``
        is 0 within ``resid_boundary_radius`` cells of a change in P1's
        label map.  Returns None when the split is inactive (the plain
        EM then runs for every anchor).
        """
        da = self.dynamic_anchor
        other = da.other_bank(bank)
        if (da.resid_k == 0 or bank != da.resid_bank
                or self.local_iter < self.resid_start_iter
                or not da.bank_ready(other)):
            return None
        B, C, H, W = fused.shape
        f = F.normalize(fused.detach().permute(0, 2, 3, 1).reshape(-1, C),
                        dim=1)
        anchors = F.normalize(da._bank(other)[0].detach(), dim=1)
        cov = (f @ anchors.t()).max(dim=1)[0]              # (N,)
        q = torch.quantile(cov, self.resid_cov_quantile)
        if not bool(self._cov_c0_init):
            self._cov_c0.copy_(q)
            self._cov_c0_init.fill_(True)
        else:
            self._cov_c0.mul_(0.99).add_(q, alpha=0.01)
        w = torch.sigmoid((self._cov_c0 - cov) / self.resid_scale)

        interior = torch.ones_like(w)
        r = self.resid_boundary_radius
        if r > 0:
            y = logits.detach().argmax(dim=1, keepdim=True).float()
            k = 2 * r + 1
            hi = F.max_pool2d(y, k, stride=1, padding=r)
            lo = -F.max_pool2d(-y, k, stride=1, padding=r)
            interior = (hi == lo).float().reshape(-1)
        w = self.resid_floor + (1.0 - self.resid_floor) * w * interior
        log_vars['resid_w_mean'] = float(w.mean())
        log_vars['resid_c0'] = float(self._cov_c0)
        log_vars['resid_interior'] = float(interior.mean())
        return w

    def _backward_losses(self, losses):
        """Sum the losses of one stream, backward, return the log vars.

        Subclasses override this to route groups of losses to different
        optimizers (see ``DAPCN_SSL_DualBank2Opt``).
        """
        loss, lv = self._parse_losses(losses)
        loss.backward()
        lv.pop('loss', None)
        return lv

    def _dapg(self, fused, bank, lam, logits=None, log_vars=None,
              labels=None):
        """L_dapcn on one stream: EM seeded from / written to ``bank``.

        ``labels`` (feature-resolution GT) is ignored here; subclasses with
        a class-aware bank use it.
        """
        B, C, H, W = fused.shape
        flat = fused.permute(0, 2, 3, 1).reshape(-1, C)
        w = None
        if logits is not None:
            w = self._resid_weights(fused, logits, bank, log_vars)
        assign, proto, quality = self.dynamic_anchor.refine(
            fused, bank, update_bank=True, pixel_weight=w)
        loss, parts = self.dapg_loss_fn(flat, assign, proto, quality)
        return lam * loss, parts

    def _resize_label(self, label, hw):
        if label.dim() == 4:
            label = label.squeeze(1)
        return F.interpolate(
            label.float().unsqueeze(1), size=hw,
            mode='nearest').long().squeeze(1)

    @torch.no_grad()
    def _diagnostics(self, log_vars):
        if self.local_iter % self.diag_interval != 0:
            return
        da = self.dynamic_anchor
        for k, v in da.cross_bank_stats().items():
            log_vars['diag.%s' % k] = float(v)
        if da.resid_k > 0 and all(da.bank_ready(n) for n in da.bank_names):
            a = F.normalize(da._bank(da.resid_bank)[0], dim=1)
            b = F.normalize(da._bank(da.other_bank(da.resid_bank))[0], dim=1)
            cov = (a @ b.t()).max(dim=1)[0]                # per U-anchor
            kn = cov.shape[0] - da.resid_k
            log_vars['diag.cov_normal_by_L'] = float(cov[:kn].mean())
            log_vars['diag.cov_resid_by_L'] = float(cov[kn:].mean())

    # ------------------------------------------------------------------
    # Streams
    # ------------------------------------------------------------------
    def _labeled_stream(self, img, gt, log_vars):
        model = self.get_model()
        head = model.decode_head
        da = self.dynamic_anchor

        feats = model.extract_feat(img)
        fused = head._fuse_features(feats)
        logits = head.cls_seg(fused)                       # P1^l
        _, _, H, W = logits.shape

        # L_sup,1 (+ accuracy log) on the upsampled logits, as usual.
        losses = head.losses(logits, gt)
        gt_r = self._resize_label(gt, (H, W))

        if self.boundary_lambda > 0:
            losses['loss_boundary'] = self.boundary_lambda * \
                self._compute_boundary_loss(
                    logits, gt_r, feats, True, None)

        if self.proto_lambda > 0:
            lp, parts = self._dapg(fused, 'labeled', self.proto_lambda,
                                   log_vars=log_vars, labels=gt_r)
            losses['loss_dapcn_l'] = lp
            for k in ('intra', 'inter', 'cos_mean'):
                log_vars['dapcn_l.%s' % k] = float(parts[k])

        # P2^l: cross-attention over the other stream's bank.
        out = self._predict_p2(fused, self._read_bank('labeled'))

        if out is not None and self.sup2_lambda > 0:
            p2, assign = out
            losses['loss_sup2'] = self.sup2_lambda * self._nll(
                p2, gt_r, self.ignore_index)
            with torch.no_grad():
                valid = gt_r != self.ignore_index
                a2 = (p2.argmax(1) == gt_r)[valid].float().mean()
                dis = (p2.argmax(1) != logits.argmax(1))[valid].float() \
                    .mean()
                log_vars['acc_p2_l'] = float(a2)
                # ceiling of the codebook being read (see purity())
                log_vars['purity_l'] = float(da.purity(
                    assign, gt_r.reshape(-1), p2.shape[1],
                    self.ignore_index))
                log_vars['disagree_l'] = float(dis)
                for k, v in da.usage_stats(assign).items():
                    log_vars['%s_l' % k] = float(v)

        log_vars.update(self._backward_losses(losses))

    def _unlabeled_stream(self, target_img, log_vars):
        model = self.get_model()
        head = model.decode_head
        da = self.dynamic_anchor

        with self._frozen_bn_stats():
            feats = model.extract_feat(target_img)
            fused = head._fuse_features(feats)
            logits = head.cls_seg(fused)                   # P1^u

        losses = {}
        if self.proto_lambda_u > 0:
            lp, parts = self._dapg(fused, 'unlabeled', self.proto_lambda_u,
                                   logits=logits, log_vars=log_vars)
            losses['loss_dapcn_u'] = lp
            for k in ('intra', 'inter', 'cos_mean'):
                log_vars['dapcn_u.%s' % k] = float(parts[k])

        out = self._predict_p2(fused, self._read_bank('unlabeled'))
        scale = self._con_scale()
        log_vars['con_scale'] = scale

        if out is not None:
            p2, assign = out
            with torch.no_grad():
                p1 = self._p1_for_labels(head, fused, logits)
                conf1, y1 = p1.max(dim=1)
                conf2, y2 = p2.detach().max(dim=1)
                w1 = self._confident(conf1, y1, 0)           # y1 -> P2
                w2 = self._confident(conf2, y2, 1)           # y2 -> P1
                if self.con_select == 'classwise':
                    thr = self._last_thr
                    log_vars['con_thr_min'] = float(thr.min())
                    log_vars['con_thr_max'] = float(thr.max())
                log_vars['con_cover_p1'] = float(w1.mean())
                log_vars['con_cover_p2'] = float(w2.mean())
                log_vars['disagree_u'] = float((y1 != y2).float().mean())
                for k, v in da.usage_stats(assign).items():
                    log_vars['%s_u' % k] = float(v)

            # L_con,1: P1 learns from P2's pseudo-label (reaches the
            # encoder).  L_con,2: P2 learns from P1's (only W_q, W_k
            # and the anchor head).  Each has its own weight.
            if scale > 0 and self.con1_lambda > 0:
                ce1 = F.cross_entropy(
                    logits, y2, ignore_index=self.ignore_index,
                    reduction='none')
                losses['loss_con1'] = (
                    self.con1_lambda * scale * (ce1 * w2).sum()
                    / ce1.numel())
            if scale > 0 and self.con2_lambda > 0:
                losses['loss_con2'] = self.con2_lambda * scale * self._nll(
                    p2, y1, self.ignore_index, weight=w1)

        if losses:
            log_vars.update(add_prefix_u(self._backward_losses(losses)))
        else:
            # Keep the graph from lingering when nothing is optimised.
            del logits, fused, feats

    # ------------------------------------------------------------------
    def forward_train(self, img, img_metas, gt_semantic_seg, target_img,
                      target_img_metas):
        log_vars = {}
        self._labeled_stream(img, gt_semantic_seg, log_vars)
        self._unlabeled_stream(target_img, log_vars)
        self._diagnostics(log_vars)
        self.local_iter += 1
        return log_vars


def add_prefix_u(d):
    """Tag unlabeled-stream losses so logs stay readable."""
    return {('u.%s' % k): v for k, v in d.items()}
