# ---------------------------------------------------------------
# DualBankAnchorModule: two named EMA anchor banks + direct lookup.
#
# Extension of DynamicAnchorModule, not a modification: this module
# only subclasses it (Open/Closed).  No existing file is touched;
# registration is triggered by importing this module from a config.
#
# What it adds on top of DynamicAnchorModule
# ------------------------------------------
# DynamicAnchorModule already owns the machinery this needs: EM
# refinement, the EMA write rule, the learned Query/Key similarity and
# ``aux_banks`` (named EMA buffers).  But ``aux_banks`` are diagnostic
# only -- they never seed EM and nothing reads them.  Here the two
# named banks are first-class:
#
#   refine(feat, bank)   EM seeded FROM the named bank, result folded
#                        back INTO the same bank (EMA).  Used for
#                        L_dapcn^L (bank 'labeled') and L_dapcn^U
#                        (bank 'unlabeled').
#   lookup(feat, bank)   cross-attention read of a bank: pixels are
#                        the queries, the bank's anchors the keys,
#                        no EM.  assign = softmax(sim(W_q f, W_k a)).
#   predict(feat, bank, head)
#                        P2 = assign . softmax(head(anchors)), i.e. the
#                        anchors' class distributions averaged by the
#                        attention (variant A: cosine-softmax,
#                        single head, V = class-prob of the anchor).
#
# Banks carry no gradient (EMA buffers); features are detached inside
# ``lookup`` so a cross-attention loss can only reach W_q, W_k and the
# class head, never the encoder.
#
# Cold start: both banks are seeded from the SAME ``self.prototypes``
# parameter, so they start identical and diverge only through the data
# each one sees.  ``cross_bank_stats`` measures exactly that.
# ---------------------------------------------------------------

import math

import torch
import torch.nn as nn
import torch.nn.functional as F_func

from mmseg.models.builder import MODELS
from mmseg.models.uda.dynamic_anchor import DynamicAnchorModule


class HeadChung(nn.Module):
    """The slide's "Head chung": one head shared by both banks.

    Maps a bank anchor ``a`` (dim D) to the two things cross-attention
    reads from it:

        K = W_k a              (D)    matched against the pixel query Q = F
        V = softmax(W_v a)     (C)    class distribution of the anchor

    Both are 1x1 convolutions (linear maps on a vector), applied to the
    anchors of whichever bank is being read, so the same weights serve
    bank_L and bank_U.  It is a different head from Psi_1, which
    classifies pixel features.

    ``k_init_gain``: W_k starts as ``gain * I``.  With Q = F and the
    prescribed ``/sqrt(D)`` scaling, an identity K gives almost uniform
    attention (measured: |F| ~ 17-22, so logits have std ~0.25 and the
    mean max-weight over 32 anchors is 0.06 vs 0.031 for uniform), and
    W_k would need tens of thousands of steps to learn the missing
    ~x10.  The gain is an initialisation, not a change of the formula.
    """

    def __init__(self, dim, num_classes, k_init_gain=10.0):
        super().__init__()
        self.k = nn.Conv2d(dim, dim, 1, bias=False)
        self.v = nn.Conv2d(dim, num_classes, 1)
        with torch.no_grad():
            self.k.weight.copy_(
                torch.eye(dim).view(dim, dim, 1, 1) * k_init_gain)

    def forward(self, anchors):
        """anchors (K, D) -> keys (K, D), values (K, C) (rows sum to 1)."""
        a = anchors[:, :, None, None]
        return (self.k(a)[:, :, 0, 0],
                torch.softmax(self.v(a)[:, :, 0, 0], dim=1))


@MODELS.register_module(force=True)
class DualBankAnchorModule(DynamicAnchorModule):
    """DynamicAnchorModule with two named, first-class EMA banks.

    Args:
        bank_names (tuple[str]): Names of the banks. Convention:
            ``bank_names[0]`` is fed by the labeled stream and
            ``bank_names[1]`` by the unlabeled stream. Default:
            ``('labeled', 'unlabeled')``.
        bank_init (str): How a bank is seeded the first time it is
            written. ``'seed'`` (default) starts EM from the shared
            ``self.prototypes`` parameter -- near-orthogonal to real
            features, so the first E-step is almost uniform and every
            anchor collapses to the batch mean (measured: bank cos
            0.99 after 500 iters). ``'feature_sample'`` starts EM from
            K farthest-point-sampled pixel features of the first batch
            that reaches the bank, so anchors are distinct from step 0
            (and the two banks differ because they see different
            images).
        bank_max_cos (float): Write gate. A refined (K, C) prototype
            set whose mean off-diagonal cosine exceeds this is NOT
            folded into the bank (counted in ``bank_skipped_<name>``).
            Early in training the decoder features are nearly identical
            at every pixel, so EM returns K copies of one vector;
            writing that locks the bank into a collapse that the slow
            EMA then takes hundreds of iterations to leave (measured:
            bank cos 0.99 after 500 iters, P2 at chance). 1.0 (default)
            disables the gate.
        ema_warmup (bool): Bias-corrected EMA for the banks: the n-th
            write uses ``min(ema_decay, 1 - 1/(n+1))`` so a young bank
            follows the data closely instead of being dominated by its
            first write. Default: False.
        lookup_center (bool): Centre pixels and anchors on the bank's
            mean anchor before the cross-attention similarity. Decoder
            features share a large common component, so raw cosines
            all sit in a narrow band and the softmax over anchors is
            nearly uniform (measured: usage entropy 0.9, P2 far below
            P1). Centring removes the common part. Parameter-free.
            Only the attention changes; the anchors fed to the class
            head are the original normalised ones. Default: False.
        lookup_learned_scale (bool): Learn a SEPARATE log-scale for the
            lookup similarity (the E-step scale used inside ``refine``
            is left alone, so sharpening attention cannot also change
            how the banks are built). Default: False.
        lookup_temperature (float | None): Initial temperature of the
            lookup similarity; ``None`` reuses ``temperature``.
        resid_k (int): Number of "residual" anchors in ``resid_bank``
            (the LAST ``resid_k`` of its ``max_groups``). Their EM runs
            as its own mixture (softmax over the residual anchors only)
            with each pixel's responsibility multiplied by the pixel
            weight passed to ``refine``; the remaining anchors run the
            plain, unweighted EM among themselves. With weights that
            favour pixels the other bank explains poorly, the residual
            anchors model what the other bank misses. 0 (default)
            disables the split and changes nothing.
        resid_bank (str | None): Bank the residual split applies to;
            default ``bank_names[1]``.
        **kwargs: Everything :class:`DynamicAnchorModule` takes.
            ``ema_decay`` must be > 0 (the banks are EMA buffers) and
            ``proto_source`` must be ``'memory'``.
    """

    def __init__(self,
                 bank_names=('labeled', 'unlabeled'),
                 bank_init='seed',
                 bank_max_cos=1.0,
                 ema_warmup=False,
                 lookup_center=False,
                 lookup_learned_scale=False,
                 lookup_temperature=None,
                 resid_k=0,
                 resid_bank=None,
                 **kwargs):
        assert bank_init in ('seed', 'feature_sample'), bank_init
        kwargs['aux_banks'] = tuple(bank_names)
        super(DualBankAnchorModule, self).__init__(**kwargs)
        assert self.ema_decay > 0, (
            'DualBankAnchorModule needs ema_decay > 0 (banks are EMA).')
        assert self.proto_source == 'memory', (
            "DualBankAnchorModule needs proto_source='memory'.")
        assert len(self.aux_banks) == 2, (
            'exactly two banks expected, got %r' % (self.aux_banks, ))
        self.bank_names = self.aux_banks
        self.bank_init = bank_init
        self.bank_max_cos = bank_max_cos
        self.ema_warmup = ema_warmup
        self.lookup_center = lookup_center
        self.lookup_learned_scale = lookup_learned_scale
        _tau = self.temperature if lookup_temperature is None \
            else lookup_temperature
        self.lookup_temperature = _tau
        if lookup_learned_scale:
            self.lookup_logit_scale = nn.Parameter(
                torch.tensor(math.log(1.0 / _tau)))
        for _name in self.bank_names:
            self.register_buffer(
                '_writes_%s' % _name, torch.zeros((), dtype=torch.long))
            self.register_buffer(
                '_skipped_%s' % _name, torch.zeros((), dtype=torch.long))
        # Which bank the next EM run is seeded from (set by ``refine``),
        # and the sampled seed to use if that bank is still empty.
        self._active_bank = None
        self._seed_override = None
        # residual split (see ``resid_k``)
        assert 0 <= resid_k < self.max_groups, resid_k
        self.resid_k = resid_k
        self.resid_bank = (self.bank_names[1] if resid_bank is None
                           else resid_bank)
        assert self.resid_bank in self.bank_names, self.resid_bank
        self._pixel_w = None

    # ------------------------------------------------------------------
    # Bank access
    # ------------------------------------------------------------------
    def _bank(self, bank):
        if bank not in self.bank_names:
            raise ValueError('Unknown bank %r, expected one of %r.' %
                             (bank, self.bank_names))
        return (getattr(self, '_ema_prototypes_%s' % bank),
                getattr(self, '_ema_initialised_%s' % bank))

    def bank_ready(self, bank):
        return bool(self._bank(bank)[1].item())

    def other_bank(self, bank):
        """The bank a stream reads from under cross lookup."""
        self._bank(bank)
        return self.bank_names[1] if bank == self.bank_names[0] \
            else self.bank_names[0]

    # ------------------------------------------------------------------
    # EM seeding: per-bank instead of the main bank
    # ------------------------------------------------------------------
    def _get_initial_prototypes(self):
        if self._active_bank is None:
            return super(DualBankAnchorModule,
                         self)._get_initial_prototypes()
        ema, initialised = self._bank(self._active_bank)
        if initialised.item():
            source = ema
        elif self._seed_override is not None:
            source = self._seed_override
        else:
            source = self.prototypes
        proto = F_func.normalize(source, dim=1)
        return torch.where(
            torch.isnan(proto).any(dim=1, keepdim=True),
            torch.randn_like(proto) * 0.01, proto)

    # ------------------------------------------------------------------
    # L_dapcn side: EM seeded from, and written into, one bank
    # ------------------------------------------------------------------
    def refine(self, features, bank, update_bank=True, pixel_weight=None):
        """EM refinement seeded from ``bank``; EMA-writes back to it.

        Returns the same ``(assign, proto, quality)`` triple as
        ``DynamicAnchorModule.forward`` so it feeds ``DAPGLoss``
        unchanged.  The write is the parent's aux-bank path: the full
        (K, C) refined prototypes, before any quality gating.

        Args:
            pixel_weight (Tensor | None): (B*H*W,) weights in [0, 1]
                for the residual anchors' M-step (see ``resid_k``).
                Ignored unless ``bank`` is ``resid_bank`` and
                ``resid_k > 0``.
        """
        _, initialised = self._bank(bank)
        self._active_bank = bank
        if (pixel_weight is not None and self.resid_k > 0
                and bank == self.resid_bank):
            self._pixel_w = pixel_weight.detach()
        if self.bank_init == 'feature_sample' and not initialised.item():
            self._seed_override = self._sample_seed(features)
        try:
            return super(DualBankAnchorModule, self).forward(
                features, update_bank=False,
                aux_bank=bank if update_bank else None)
        finally:
            self._active_bank = None
            self._seed_override = None
            self._pixel_w = None

    def _m_assign(self, sim, assign):
        """Two mixtures when pixel weights are set, else the plain EM.

        Normal anchors: softmax among themselves, every pixel counts
        once.  Residual anchors: softmax among themselves, pixel i
        counts ``w_i``.  The two groups do not compete, so the
        residual ones are not starved of the pixels the normal ones
        already explain.
        """
        w = self._pixel_w
        if w is None or self.resid_k == 0:
            return assign
        kn = sim.shape[1] - self.resid_k
        a_n = torch.softmax(sim[:, :kn], dim=1)
        a_r = torch.softmax(sim[:, kn:], dim=1) * w.unsqueeze(1)
        return torch.cat([a_n, a_r], dim=1)

    @torch.no_grad()
    def _sample_seed(self, features, pool=4096):
        """K well-spread pixel features to start a bank from.

        Farthest-point sampling (cosine) over a random pool of pixels:
        plain random sampling would draw K near-duplicates from the
        dominant class and recreate the collapse in a milder form.
        """
        B, C, H, W = features.shape
        feats = features.detach().permute(0, 2, 3, 1).reshape(-1, C)
        feats = feats[torch.randperm(feats.shape[0],
                                     device=feats.device)[:pool]]
        norm = F_func.normalize(feats, dim=1)
        keep = ~torch.isnan(norm).any(dim=1)
        feats, norm = feats[keep], norm[keep]

        k = min(self.max_groups, feats.shape[0])
        picked = [0]
        best = 1.0 - norm @ norm[0]                  # cosine distance
        for _ in range(k - 1):
            nxt = int(best.argmax())
            picked.append(nxt)
            best = torch.minimum(best, 1.0 - norm @ norm[nxt])
        seed = feats[picked]
        if seed.shape[0] < self.max_groups:          # tiny batch: pad
            extra = seed[torch.randint(
                0, seed.shape[0], (self.max_groups - seed.shape[0], ),
                device=seed.device)]
            seed = torch.cat([seed, extra], dim=0)
        return seed

    def forward(self, features, update_bank=False, aux_bank=None):
        # Keep the parent contract (main bank, EM seeded from main) for
        # any caller that still goes through ``forward``.
        self._active_bank = None
        return super(DualBankAnchorModule, self).forward(
            features, update_bank=update_bank, aux_bank=aux_bank)

    # ------------------------------------------------------------------
    # Bank write: spread gate + optional bias-corrected EMA
    # ------------------------------------------------------------------
    @staticmethod
    def _spread(proto):
        """Mean off-diagonal cosine of a prototype set (1.0 = collapsed)."""
        pn = F_func.normalize(proto, dim=1)
        k = pn.shape[0]
        if k < 2:
            return pn.new_zeros(())
        sim = pn @ pn.t()
        return (sim.sum() - sim.diagonal().sum()) / (k * (k - 1))

    @torch.no_grad()
    def _update_memory(self, proto_refined, bank='main'):
        if bank not in self.bank_names or (
                self.bank_max_cos >= 1.0 and not self.ema_warmup):
            return super(DualBankAnchorModule, self)._update_memory(
                proto_refined, bank=bank)

        if self._spread(proto_refined) > self.bank_max_cos:
            getattr(self, '_skipped_%s' % bank).add_(1)
            return

        ema, initialised = self._bank(bank)
        writes = getattr(self, '_writes_%s' % bank)
        if not initialised.item():
            ema.copy_(proto_refined)
            initialised.fill_(True)
        else:
            decay = self.ema_decay
            if self.ema_warmup:
                decay = min(decay, 1.0 - 1.0 / (float(writes) + 1.0))
            ema.mul_(decay).add_(proto_refined, alpha=1.0 - decay)
        writes.add_(1)

    # ------------------------------------------------------------------
    # P2 side: cross-attention read of a bank
    # ------------------------------------------------------------------
    def lookup(self, features, bank):
        """Attention of every pixel over the anchors of ``bank``.

            assign_ik = softmax_k( sim(W_q f_i, W_k a_k) )

        Features are detached: a loss on this path trains W_q / W_k
        (and whatever consumes ``assign``), never the encoder.  The
        bank is an EMA buffer and carries no gradient either.

        Args:
            features (Tensor): (B, C, H, W).
            bank (str): Bank to read.

        Returns:
            tuple[Tensor, Tensor] | None: ``assign`` (N, K) and the
                normalised anchors (K, C); None while the bank has not
                been written yet.
        """
        ema, initialised = self._bank(bank)
        if not initialised.item():
            return None

        B, C, H, W = features.shape
        feats = features.detach().permute(0, 2, 3, 1).reshape(-1, C)
        feats_norm = F_func.normalize(feats, dim=1)
        feats_norm = torch.where(
            torch.isnan(feats_norm).any(dim=1, keepdim=True),
            torch.zeros_like(feats_norm), feats_norm)

        anchors = F_func.normalize(ema.detach(), dim=1)
        q_in, k_in = feats_norm, anchors
        if self.lookup_center:
            mean = anchors.mean(dim=0, keepdim=True)
            q_in = F_func.normalize(feats_norm - mean, dim=1)
            k_in = F_func.normalize(anchors - mean, dim=1)
        sim = torch.clamp(
            self._lookup_similarity(q_in, k_in), min=-50.0, max=50.0)
        return torch.softmax(sim, dim=1), anchors

    def _lookup_similarity(self, q_in, k_in):
        """Scaled similarity for the cross-attention read.

        Same form as ``_similarity`` (optional W_q / W_k, re-normalised
        cosine) but with its own scale, so it can be sharpened without
        touching the EM used to build the banks.
        """
        if self.use_learned_assignment:
            q_in = F_func.normalize(self.query_proj(q_in), dim=1)
            k_in = F_func.normalize(self.key_proj(k_in), dim=1)
        cos = torch.mm(q_in, k_in.t())
        if self.lookup_learned_scale:
            scale = self.lookup_logit_scale.exp().clamp(
                max=self.max_logit_scale)
        else:
            scale = 1.0 / self.lookup_temperature
        return cos * scale

    def attend(self, features, bank, head_chung):
        """Slide-literal cross-attention read of a bank.

            P2 = softmax( Q K^T / sqrt(D) ) V
            Q = F (raw pixel features, detached)
            K, V = head_chung(anchors of ``bank``)

        Unlike ``predict`` there is no W_q, no normalisation and no
        temperature: the only scale is 1/sqrt(D).  Features are detached,
        so a loss on P2 reaches ``head_chung`` only, never the encoder;
        the bank is an EMA buffer and carries no gradient.

        Returns:
            tuple[Tensor, Tensor] | None: probabilities (B, C, H, W) and
                the attention (N, K); None while the bank is empty.
        """
        ema, initialised = self._bank(bank)
        if not initialised.item():
            return None
        B, Cf, H, W = features.shape
        q = torch.nan_to_num(
            features.detach().permute(0, 2, 3, 1).reshape(-1, Cf))
        k, v = head_chung(ema.detach())
        logits = torch.clamp(q @ k.t() / math.sqrt(Cf), min=-50.0, max=50.0)
        assign = torch.softmax(logits, dim=1)
        probs = (assign @ v).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return probs.contiguous(), assign

    def predict(self, features, bank, head):
        """P2: class probabilities from a cross-attention read.

            P2_i = sum_k assign_ik * softmax(head(a_k))

        Args:
            features (Tensor): (B, C, H, W).
            bank (str): Bank to read.
            head (callable): Maps anchors (K, C, 1, 1) -> logits
                (K, num_classes, 1, 1) (e.g. ``conv_seg_proto``).

        Returns:
            tuple[Tensor, Tensor] | None: probabilities
                (B, num_classes, H, W) (rows sum to 1) and ``assign``
                (N, K) for diagnostics; None while the bank is empty.
        """
        out = self.lookup(features, bank)
        if out is None:
            return None
        assign, anchors = out
        B, _, H, W = features.shape

        anchor_logits = head(anchors.unsqueeze(-1).unsqueeze(-1))
        anchor_probs = torch.softmax(
            anchor_logits.squeeze(-1).squeeze(-1), dim=1)   # (K, cls)
        probs = torch.mm(assign, anchor_probs)               # (N, cls)
        # contiguous: torch 1.8's CPU nll_loss2d backward rejects the
        # permuted view (GPU does not care).
        probs = probs.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return probs, assign

    # ------------------------------------------------------------------
    # Diagnostics (no gradient, no training effect)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def cross_bank_stats(self):
        """How different are the two banks?

        Returns a dict of detached scalars, or ``{}`` until both banks
        have been written:

          bank_cos_matched   mean cosine after optimal one-to-one
                             (Hungarian) matching of anchors
          bank_cos_sameidx   mean cosine of anchor k vs anchor k
          bank_cover_a2b     mean over a's anchors of max cosine to b
          bank_cover_b2a     the reverse; a large gap means one bank
                             covers the other but not vice versa
          bank_mean_<name>   off-diagonal mean cosine inside each bank
                             (redundancy / collapse indicator)
        """
        a_name, b_name = self.bank_names
        if not (self.bank_ready(a_name) and self.bank_ready(b_name)):
            return {}
        a = F_func.normalize(self._bank(a_name)[0], dim=1)
        b = F_func.normalize(self._bank(b_name)[0], dim=1)
        sim = a @ b.t()                                      # (K, K)

        try:
            from scipy.optimize import linear_sum_assignment
            r, c = linear_sum_assignment((-sim).cpu().numpy())
            matched = sim[torch.as_tensor(r), torch.as_tensor(c)].mean()
        except ImportError:  # greedy fallback, an optimistic bound
            matched = sim.max(dim=1).values.mean()

        stats = {
            'bank_cos_matched': matched,
            'bank_cos_sameidx': sim.diagonal().mean(),
            'bank_cover_a2b': sim.max(dim=1).values.mean(),
            'bank_cover_b2a': sim.max(dim=0).values.mean(),
        }
        for name in self.bank_names:
            stats['bank_mean_%s' % name] = self.bank_cos_mean(name)
            stats['bank_writes_%s' % name] = getattr(
                self, '_writes_%s' % name).float()
            stats['bank_skipped_%s' % name] = getattr(
                self, '_skipped_%s' % name).float()
        return {k: v.detach().float() for k, v in stats.items()}

    @staticmethod
    @torch.no_grad()
    def usage_stats(assign, min_share=0.01):
        """How the attention spreads over anchors.

        Two different questions, easy to confuse:

        * marginal use (``usage_*``): mean attention per anchor over ALL
          pixels.  High entropy only says the anchors are used evenly
          overall -- it says nothing about any single pixel.
        * per-pixel sharpness (``attn_*``): how peaked each pixel's own
          row is.  This is what decides whether P2 can discriminate.

        Args:
            assign (Tensor): (N, K) attention rows.
            min_share (float): An anchor counts as used when its mean
                attention exceeds this share.

        Returns:
            dict: ``usage_entropy`` (marginal, normalised to [0, 1]),
                ``usage_active`` (fraction of anchors in use),
                ``attn_pixel_entropy`` (mean per-pixel entropy,
                normalised: 1 = uniform row, 0 = one-hot) and
                ``attn_max`` (mean of each pixel's largest weight;
                1/K = uniform, 1 = one-hot).
        """
        share = assign.mean(dim=0)
        share = share / share.sum().clamp_min(1e-12)
        k = share.numel()
        log_k = torch.log(torch.tensor(float(max(k, 2)),
                                       device=assign.device))
        ent = -(share * torch.log(share.clamp_min(1e-12))).sum()
        row_ent = -(assign * torch.log(assign.clamp_min(1e-12))).sum(1)
        return {
            'usage_entropy': ent / log_k,
            'usage_active': (share > min_share).float().mean(),
            'attn_pixel_entropy': (row_ent / log_k).mean(),
            'attn_max': assign.max(dim=1).values.mean(),
        }

    @staticmethod
    @torch.no_grad()
    def purity(assign, labels, num_classes, ignore_index=255):
        """Accuracy ceiling of a codebook: how class-pure are its anchors?

        Each pixel is hard-assigned to its top anchor; every anchor is
        then given the class that dominates its pixels, and the share of
        pixels that this best possible labelling gets right is returned.
        P2 can never beat this on the same pixels, whatever the head or
        the attention learn.

        Args:
            assign (Tensor): (N, K) attention rows.
            labels (Tensor): (N,) ground-truth class ids.
            num_classes (int): Number of classes.
            ignore_index (int): Label to skip.

        Returns:
            Tensor: scalar in [0, 1] (0 if there is no valid pixel).
        """
        valid = (labels != ignore_index) & (labels < num_classes)
        if not bool(valid.any()):
            return assign.new_zeros(())
        k = assign.shape[1]
        top = assign.argmax(dim=1)[valid]
        lab = labels[valid].long()
        counts = torch.zeros(k, num_classes, device=assign.device)
        counts.index_put_((top, lab), torch.ones_like(top, dtype=torch.float),
                          accumulate=True)
        return counts.max(dim=1).values.sum() / counts.sum()
