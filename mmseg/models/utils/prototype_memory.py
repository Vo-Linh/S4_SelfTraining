# ---------------------------------------------------------------
# Prototype Memory Bank for Supervised Segmentation with DAPCN
# Supports: single or multiple prototypes per class, EMA updates,
#           and prototype-based contrastive loss (InfoNCE).
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeMemory(nn.Module):
    """Class-wise prototype memory bank with EMA updates.

    Maintains persistent, cross-batch prototypes for each semantic class.
    Supports multiple prototypes per class to capture intra-class diversity
    (e.g., hard/easy sub-regions, multi-modal distributions).

    Design rationale:
        - DynamicAnchorModule discovers *transient* per-batch prototypes
          from the feature geometry (no class awareness).
        - PrototypeMemory accumulates *persistent* class-conditioned
          prototypes across the entire training trajectory via EMA.
        - The two are complementary: DynamicAnchor captures fine-grained
          structure; PrototypeMemory provides stable, class-aligned anchors
          for contrastive regularisation.

    Args:
        num_classes (int): Number of semantic classes.
        feature_dim (int): Dimension of feature embeddings.
        num_prototypes_per_class (int): Number of prototypes maintained
            per class.  When >1, incoming features are assigned to the
            nearest prototype within the correct class via cosine
            similarity (similarity-based routing). Default: 1.
        ema (float): EMA momentum coefficient. Default: 0.999.
        init_strategy (str): 'zeros' or 'random'. Default: 'zeros'.
    """

    EPS = 1e-6

    def __init__(self,
                 num_classes,
                 feature_dim,
                 num_prototypes_per_class=1,
                 ema=0.999,
                 init_strategy='zeros'):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.K = num_prototypes_per_class
        self.ema = ema

        # Prototype bank: (num_classes * K, feature_dim)
        total = num_classes * self.K
        if init_strategy == 'zeros':
            self.register_buffer('prototypes', torch.zeros(total, feature_dim))
        elif init_strategy == 'random':
            proto = torch.randn(total, feature_dim)
            proto = F.normalize(proto, dim=1)
            self.register_buffer('prototypes', proto)
        else:
            raise ValueError(f"Unknown init strategy: {init_strategy}")

        # Per-prototype update counter (for first-update detection)
        self.register_buffer('update_counts', torch.zeros(total))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _class_slice(self, c):
        """Return (start, end) indices for class c in the flat bank."""
        return c * self.K, (c + 1) * self.K

    def get_class_prototypes(self, c):
        """Return prototypes for class c: shape (K, D)."""
        s, e = self._class_slice(c)
        return self.prototypes[s:e]

    # ------------------------------------------------------------------
    # EMA update
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(self, features, labels, mask=None):
        """Update prototypes using EMA with similarity-based routing.

        Args:
            features (Tensor): (N, D) normalised feature embeddings.
            labels (Tensor): (N,) integer class labels.
            mask (Tensor | None): (N,) boolean confidence mask.
        """
        if mask is None:
            mask = torch.ones(labels.shape, dtype=torch.bool,
                              device=labels.device)

        for c in range(self.num_classes):
            class_mask = (labels == c) & mask
            if class_mask.sum() == 0:
                continue

            class_feats = features[class_mask]           # (M, D)
            class_feats = F.normalize(class_feats, dim=1)
            class_feats = torch.nan_to_num(class_feats, nan=0.0)

            s, e = self._class_slice(c)
            class_protos = self.prototypes[s:e]          # (K, D)

            if self.K == 1:
                # Single prototype per class: straightforward mean EMA
                mean_feat = class_feats.mean(dim=0)
                if self.update_counts[s] == 0:
                    self.prototypes[s] = mean_feat
                else:
                    self.prototypes[s] = (
                        self.ema * self.prototypes[s]
                        + (1 - self.ema) * mean_feat)
                self.update_counts[s] += 1
            else:
                # Multi-prototype: similarity-based routing
                # Assign each feature to its nearest prototype within class
                if self.update_counts[s:e].sum() == 0:
                    # First update: initialise via k-means++ style seeding
                    n_init = min(self.K, class_feats.shape[0])
                    indices = self._fps_init(class_feats, n_init)
                    self.prototypes[s:s + n_init] = class_feats[indices]
                    self.update_counts[s:s + n_init] += 1
                else:
                    # Cosine similarity routing
                    proto_norm = F.normalize(class_protos, dim=1)
                    proto_norm = torch.nan_to_num(proto_norm, nan=0.0)
                    sim = torch.mm(class_feats, proto_norm.t())  # (M, K)
                    assign = sim.argmax(dim=1)                   # (M,)

                    for k in range(self.K):
                        k_mask = assign == k
                        if k_mask.sum() == 0:
                            continue
                        mean_feat = class_feats[k_mask].mean(dim=0)
                        idx = s + k
                        if self.update_counts[idx] == 0:
                            self.prototypes[idx] = mean_feat
                        else:
                            self.prototypes[idx] = (
                                self.ema * self.prototypes[idx]
                                + (1 - self.ema) * mean_feat)
                        self.update_counts[idx] += 1

    @staticmethod
    def _fps_init(feats, k):
        """Farthest-point sampling to pick k diverse initial prototypes."""
        N = feats.shape[0]
        indices = [torch.randint(0, N, (1,), device=feats.device).item()]
        for _ in range(1, k):
            selected = feats[indices]
            dist = torch.cdist(feats, selected)
            min_dist = dist.min(dim=1).values
            indices.append(min_dist.argmax().item())
        return indices

    # ------------------------------------------------------------------
    # Forward / retrieval
    # ------------------------------------------------------------------
    def forward(self):
        """Return the full prototype bank: (num_classes * K, D)."""
        return self.prototypes

    def get_all_normalised(self):
        """Return L2-normalised prototypes: (num_classes * K, D)."""
        protos = F.normalize(self.prototypes, dim=1)
        return torch.nan_to_num(protos, nan=0.0)

    def is_initialised(self):
        """True once every class has at least one updated prototype."""
        for c in range(self.num_classes):
            s, _ = self._class_slice(c)
            if self.update_counts[s] == 0:
                return False
        return True


def prototype_contrastive_loss(features,
                               prototypes,
                               labels,
                               num_classes,
                               num_prototypes_per_class=1,
                               temperature=0.07,
                               ignore_index=255):
    """Prototype-based contrastive loss (InfoNCE) with multi-prototype support.

    For each pixel feature, the *positive set* consists of all prototypes
    belonging to the same class; the *negative set* is all remaining
    prototypes.  When ``num_prototypes_per_class > 1``, we take the
    maximum similarity across prototypes of the correct class as the
    positive logit (hard-positive mining), following the multi-prototype
    NCE formulation of ProtoNCE (Li et al., CVPR 2021).

    L = -log( exp(sim_pos / tau) / sum_j exp(sim_j / tau) )

    Args:
        features (Tensor): (N, D) pixel-level feature embeddings.
        prototypes (Tensor): (C*K, D) prototype bank.
        labels (Tensor): (N,) ground-truth class indices.
        num_classes (int): Total number of classes.
        num_prototypes_per_class (int): K prototypes per class.
        temperature (float): Softmax temperature. Default: 0.07.
        ignore_index (int): Label to ignore. Default: 255.

    Returns:
        Tensor: Scalar contrastive loss.
    """
    K = num_prototypes_per_class

    # Filter ignored pixels
    valid = labels != ignore_index
    if valid.sum() == 0:
        return torch.tensor(0.0, device=features.device, requires_grad=True)

    feats = F.normalize(features[valid], dim=1)          # (M, D)
    feats = torch.nan_to_num(feats, nan=0.0)
    labs = labels[valid]                                   # (M,)
    proto_norm = F.normalize(prototypes, dim=1)            # (C*K, D)
    proto_norm = torch.nan_to_num(proto_norm, nan=0.0)

    # Similarity matrix: (M, C*K)
    sim = torch.mm(feats, proto_norm.t()) / temperature

    if K == 1:
        # Single prototype per class: standard InfoNCE
        log_probs = F.log_softmax(sim, dim=1)              # (M, C)
        loss = -log_probs[torch.arange(len(labs), device=labs.device), labs]
    else:
        # Multi-prototype: reshape to (M, C, K), take max per class
        sim_3d = sim.view(-1, num_classes, K)              # (M, C, K)
        sim_class, _ = sim_3d.max(dim=2)                   # (M, C)
        log_probs = F.log_softmax(sim_class, dim=1)        # (M, C)
        loss = -log_probs[torch.arange(len(labs), device=labs.device), labs]

    return loss.mean()


def prototype_contrastive_loss_extended(features,
                                        class_prototypes,
                                        extra_negatives,
                                        labels,
                                        num_classes,
                                        num_prototypes_per_class=1,
                                        temperature=0.07,
                                        ignore_index=255):
    """InfoNCE with class-conditioned positives and an extra hard-negative bank.

    The standard ``prototype_contrastive_loss`` discriminates each pixel
    against ``num_classes * K`` class-conditioned prototypes only.  With
    9 classes and K=1 this is a 9-way classification task — easy to
    saturate.

    This variant accepts an additional bank of ``M`` *class-agnostic*
    prototypes (e.g. the dataset-level prototypes from
    DynamicAnchorModule).  Those prototypes are placed only in the
    InfoNCE denominator, never as positives, so they act as
    permanently-hard negatives spanning the feature manifold.  The
    discrimination task becomes ``(C*K + M)``-way and stays informative
    long after the class-centroid task plateaus.

    Args:
        features (Tensor): (N, D) pixel features.
        class_prototypes (Tensor): (C*K, D) class-conditioned prototypes.
        extra_negatives (Tensor): (M, D) hard negatives — typically the
            DynamicAnchorModule prototypes ``.detach()``-ed so the
            contrastive loss does not reshape the DA geometry.
        labels (Tensor): (N,) integer class indices.
        num_classes (int): C.
        num_prototypes_per_class (int): K. Default: 1.
        temperature (float): InfoNCE temperature. Default: 0.07.
        ignore_index (int): Label to ignore. Default: 255.

    Returns:
        Tensor: Scalar loss.
    """
    K = num_prototypes_per_class

    valid = labels != ignore_index
    if valid.sum() == 0:
        return torch.tensor(0.0, device=features.device, requires_grad=True)

    feats = F.normalize(features[valid], dim=1)
    feats = torch.nan_to_num(feats, nan=0.0)
    labs = labels[valid]

    cls_p = F.normalize(class_prototypes, dim=1)
    cls_p = torch.nan_to_num(cls_p, nan=0.0)
    ext_p = F.normalize(extra_negatives, dim=1)
    ext_p = torch.nan_to_num(ext_p, nan=0.0)

    # Similarity against class prototypes and extra negatives
    sim_cls = torch.mm(feats, cls_p.t()) / temperature      # (N, C*K)
    sim_ext = torch.mm(feats, ext_p.t()) / temperature      # (N, M)

    if K == 1:
        # Positive logit = the correct class's prototype similarity
        pos_logit = sim_cls[torch.arange(len(labs), device=labs.device), labs]
    else:
        # Multi-prototype hard-positive mining: max over K of the
        # correct class's prototypes
        sim_3d = sim_cls.view(-1, num_classes, K)
        sim_class_max, _ = sim_3d.max(dim=2)                # (N, C)
        pos_logit = sim_class_max[
            torch.arange(len(labs), device=labs.device), labs]

    # Denominator: logsumexp over EVERY prototype (class + extra negatives).
    # No reduction across the K dimension here — each prototype is its
    # own competing answer, giving the loss a much harder negative pool.
    all_sim = torch.cat([sim_cls, sim_ext], dim=1)          # (N, C*K + M)
    log_denom = torch.logsumexp(all_sim, dim=1)             # (N,)

    loss = -(pos_logit - log_denom)
    return loss.mean()
