# ---------------------------------------------------------------
# DAPCN: Dynamic Attention-based Prototype Clustering Network
# Reference: docs/DAPCN.md
# --------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


@LOSSES.register_module(force=True)
class DAPGLoss(nn.Module):
    """Dynamic Attention-based Prototype Grouping Loss.

    Components:
    1. Intra-group compactness (L_intra): Features close to their assigned
       group prototype
    2. Inter-group separation (L_inter): Different prototypes should be
       distinct beyond margin
    3. Quality regularization (L_quality): Encourage well-defined groups

    Args:
        margin (float): Margin for inter-group separation loss. Default: 0.3.
        lambda_inter (float): Weight for L_inter. Default: 0.5.
        lambda_quality (float): Weight for L_quality. Default: 0.1.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    EPS = 1e-6

    def __init__(self, margin=0.3, lambda_inter=0.5, lambda_quality=0.1,
                 loss_weight=1.0):
        super().__init__()
        self.margin = margin
        self.lambda_inter = lambda_inter
        self.lambda_quality = lambda_quality
        self.loss_weight = loss_weight

    def forward(self, features, group_assignments, group_prototypes,
                group_quality):
        """Compute DAPG loss.

        Args:
            features (torch.Tensor): Raw features of shape (N, C)
            group_assignments (torch.Tensor): Soft assignment weights of
                shape (N, K')
            group_prototypes (torch.Tensor): Group prototype vectors of
                shape (K', C)
            group_quality (torch.Tensor): Quality scores of shape (K',)

        Returns:
            tuple: (loss, loss_dict) where loss is scalar tensor and
                loss_dict is dict with individual components
        """
        # Get dimensions
        n_groups = group_prototypes.shape[0]

        # Normalize features and prototypes for cosine similarity
        f_norm = F.normalize(features, p=2, dim=1)
        q_norm = F.normalize(group_prototypes, p=2, dim=1)

        # L_intra: Intra-group compactness
        # Formula: (1/N) * sum_i sum_k A_ik * (1 - cos(F_i, Q_k))
        sim_intra = torch.mm(f_norm, q_norm.t())  # (N, K')
        weighted_sim = (sim_intra * group_assignments).sum(dim=1)
        loss_intra = (1.0 - weighted_sim).mean()

        # L_inter: Inter-group separation with margin
        # Formula: (1/K(K-1)) * sum_{g!=h} max(0, cos(Q_g, Q_h) - margin)
        sim_inter = torch.mm(q_norm, q_norm.t())  # (K', K')
        if n_groups > 1:
            mask = 1.0 - torch.eye(n_groups, device=sim_inter.device)
            margin_violations = F.relu(sim_inter - self.margin) * mask
            loss_inter = margin_violations.sum() / (n_groups *
                                                    (n_groups - 1))
        else:
            loss_inter = torch.tensor(0.0, device=features.device)

        # L_quality: Quality regularization
        # Formula: -mean(log(quality + epsilon))
        quality_clamped = torch.clamp(group_quality, min=self.EPS)
        loss_quality = -torch.log(quality_clamped).mean()

        # Total loss
        loss_proto = self.loss_weight * (loss_intra +
                                          self.lambda_inter * loss_inter +
                                          self.lambda_quality * loss_quality)

        # --- Diagnostics -------------------------------------------------
        # Detached, and deliberately named WITHOUT the substring 'loss'.
        # `BaseSegmentor._parse_losses` sums every entry whose key contains
        # 'loss', so returning the components under names like
        # 'loss_intra' made them land in the optimised total a second time,
        # on top of `loss_proto` — giving intra/inter/quality effective
        # weights of 1.1 / 1.05 / 1.01 instead of the configured
        # 0.1 / 0.05 / 0.01.
        with torch.no_grad():
            if n_groups > 1:
                off_diag = sim_inter[~torch.eye(
                    n_groups, dtype=torch.bool, device=sim_inter.device)]
                cos_mean = off_diag.mean()
                cos_max = off_diag.max()
                # Share of prototype pairs that are closer than the margin
                # allows — a direct read on how redundant the bank is.
                frac_over = (off_diag > self.margin).float().mean()
            else:
                zero = torch.zeros((), device=features.device)
                cos_mean, cos_max, frac_over = zero, zero, zero

        loss_dict = {
            'intra': loss_intra.detach(),
            'inter': loss_inter.detach(),
            'quality': loss_quality.detach(),
            'n_groups': torch.tensor(
                float(n_groups), device=features.device),
            'cos_mean': cos_mean,
            'cos_max': cos_max,
            'frac_over_margin': frac_over,
        }

        return loss_proto, loss_dict


if __name__ == "__main__":
    # Example usage of DAPGLoss
    # Simulate a batch of features with 100 samples, 256 channels
    batch_size = 100
    feature_dim = 256
    n_groups = 5  # Number of prototype groups

    # Create random features (normalized to unit vectors for realistic scenario)
    features = torch.randn(batch_size, feature_dim)
    features = F.normalize(features, p=2, dim=1)

    # Create soft group assignments (N, K') - probabilities sum to 1
    group_assignments = torch.softmax(torch.randn(batch_size, n_groups), dim=1)

    # Create group prototypes (K', C)
    group_prototypes = torch.randn(n_groups, feature_dim)
    group_prototypes = F.normalize(group_prototypes, p=2, dim=1)

    # Create quality scores (K') - values between 0 and 1
    group_quality = torch.sigmoid(torch.randn(n_groups))

    # Initialize loss
    loss_fn = DAPGLoss(margin=0.3, lambda_inter=0.5, lambda_quality=0.1, loss_weight=1.0)

    # Compute loss
    loss, loss_dict = loss_fn(features, group_assignments, group_prototypes, group_quality)

    # Print results
    print("=" * 50)
    print("DAPGLoss Example")
    print("=" * 50)
    print(f"Input shapes:")
    print(f"  Features: {features.shape}")
    print(f"  Group assignments: {group_assignments.shape}")
    print(f"  Group prototypes: {group_prototypes.shape}")
    print(f"  Group quality: {group_quality.shape}")
    print(f"\nLoss components:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.6f}")
    print(f"\nTotal loss: {loss.item():.6f}")
    print("=" * 50)
