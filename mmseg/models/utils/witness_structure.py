# ---------------------------------------------------------------
# DWPC Witness B — Domain-invariant structure (target-adaptive)
#
# The structural witness scores the *topological / geometric
# plausibility* of a label map, independent of appearance.  Land-cover
# topology is (largely) shift-invariant — a river is connected anywhere,
# a building has bounded area anywhere — so these statistics transfer
# across domains where prototype/appearance evidence does not.
#
# Four per-class plausibility terms, each in [0, 1], product-combined:
#   - area         : class area vs target-adaptive EMA expected area
#   - connectivity : fragmentation (#components / area) vs EMA
#   - co-occurrence : per-pixel neighbour compatibility vs EMA adjacency
#   - containment  : how consistently a class is hosted by one neighbour
#
# The EMA statistics are estimated from the *target pseudo-label stream*
# (no target labels needed) and are passed in/out as a buffer dict so
# the owning module can register them for checkpointing.
#
# Deliberately NON-differentiable (connected components / histograms are
# not differentiable anyway): Witness B only informs the flip decision
# and the per-pixel weight.  See DWPC_PROPOSAL.md §4.3 / §10 risk 3.
# ---------------------------------------------------------------

import torch
import torch.nn.functional as F


def _one_hot(label, num_classes):
    """(B,H,W) long -> (B,C,H,W) float one-hot (ignore handled by clamp)."""
    lab = label.clamp(0, num_classes - 1)
    return F.one_hot(lab, num_classes).permute(0, 3, 1, 2).float()


def _connected_components(onehot, iters):
    """Approximate connected-component labelling via label propagation.

    Each foreground pixel is seeded with a unique id; iterated 3x3
    max-pooling (masked back to the foreground) floods the largest id
    through each component.  After enough iterations, pixels of one
    component share a single id.  This is approximate (``iters`` bounds
    the propagation radius) but cheap and GPU-friendly.

    Args:
        onehot (Tensor): (B, C, H, W) binary class masks.
        iters (int): propagation iterations.

    Returns:
        Tensor: (B, C, H, W) component-id map (0 = background).
    """
    B, C, H, W = onehot.shape
    binc = onehot.reshape(B * C, 1, H, W)
    idx = torch.arange(1, H * W + 1, device=onehot.device,
                       dtype=torch.float32).view(1, 1, H, W)
    lab = idx * binc
    for _ in range(iters):
        lab = F.max_pool2d(lab, 3, stride=1, padding=1) * binc
    return lab.reshape(B, C, H, W)


def _count_components(comp):
    """Count distinct positive ids per (B, C). Returns (B, C) float."""
    B, C, H, W = comp.shape
    flat = comp.reshape(B, C, H * W)
    counts = torch.zeros(B, C, device=comp.device)
    for b in range(B):
        for c in range(C):
            u = torch.unique(flat[b, c])
            counts[b, c] = (u > 0).sum()
    return counts


def _neighbor_maps(label):
    """4-neighbour shifted copies of a (B,H,W) label map (edge-wrapped)."""
    return [torch.roll(label, shifts=s, dims=(1, 2))
            for s in [(1, 0), (-1, 0), (0, 1), (0, -1)]]


def _cooc_plausibility(qmap, neighbors, P_cooc, num_classes):
    """Per-pixel mean P_cooc[qmap, neighbour] over 4 neighbours.

    Args:
        qmap (Tensor): (B,H,W) hypothesised class at each pixel.
        neighbors (list[Tensor]): 4 shifted neighbour label maps.
        P_cooc (Tensor): (C,C) row-normalised adjacency.
        num_classes (int): C.

    Returns:
        Tensor: (B,H,W) plausibility in [0, 1].
    """
    q = qmap.clamp(0, num_classes - 1).reshape(-1)
    total = torch.zeros_like(q, dtype=torch.float32)
    for nb in neighbors:
        nbc = nb.clamp(0, num_classes - 1).reshape(-1)
        total = total + P_cooc[q, nbc]
    return (total / len(neighbors)).view_as(qmap)


@torch.no_grad()
def witness_structure(pseudo_label,
                      cstar_map,
                      num_classes,
                      rare_class_ids,
                      buffers,
                      target_adaptive=True,
                      cc_iters=64,
                      ema=0.99,
                      min_count=5,
                      enable_containment=True):
    """Structural plausibility of the assigned label and the rare candidate.

    Args:
        pseudo_label (Tensor): (B, H, W) long teacher argmax.
        cstar_map (Tensor): (B, H, W) long rare candidate from Witness A
            (-1 where none).
        num_classes (int): C.
        rare_class_ids (list[int]): rare class indices (unused directly
            here; kept for signature symmetry / future gating).
        buffers (dict): EMA statistics with keys ``area_mean`` (C,),
            ``conn_mean`` (C,), ``contain_mean`` (C,), ``cooc`` (C,C),
            ``count`` (C,).
        target_adaptive (bool): update the EMA buffers from this batch.
            If False (row A6), buffers are frozen.
        cc_iters (int): connected-component propagation iterations.
        ema (float): EMA momentum for the buffers.
        min_count (int): per-class warmup count below which that class's
            area/conn/contain terms are treated as plausible (=1).
        enable_containment (bool): include the containment term.

    Returns:
        tuple:
            s_map (Tensor): (B,H,W) plausibility of the assigned class.
            s_cstar (Tensor): (B,H,W) neighbourhood plausibility of c*.
            buffers (dict): updated EMA statistics.
    """
    B, H, W = pseudo_label.shape
    device = pseudo_label.device
    C = num_classes
    eps = 1e-6

    onehot = _one_hot(pseudo_label, C)                       # (B,C,H,W)
    area_frac = onehot.mean(dim=(2, 3))                      # (B,C)

    comp = _connected_components(onehot, cc_iters)
    n_comp = _count_components(comp)                         # (B,C)
    area_px = onehot.sum(dim=(2, 3))                         # (B,C)
    frag = n_comp / (area_px + 1.0)                          # (B,C)

    # --- co-occurrence (adjacency) batch statistics ---
    neighbors = _neighbor_maps(pseudo_label)
    cooc_count = torch.zeros(C, C, device=device)
    flat = pseudo_label.clamp(0, C - 1).reshape(-1)
    for nb in neighbors:
        nbc = nb.clamp(0, C - 1).reshape(-1)
        cooc_count += torch.bincount(flat * C + nbc,
                                     minlength=C * C).view(C, C).float()
    row = cooc_count.sum(dim=1, keepdim=True).clamp(min=1.0)
    P_batch = cooc_count / row                               # (C,C)

    # containment: among non-self neighbours of class c, the share of the
    # single most common host class (high => consistently embedded).
    off_diag = cooc_count.clone()
    off_diag[torch.arange(C), torch.arange(C)] = 0.0
    host_total = off_diag.sum(dim=1).clamp(min=1.0)          # (C,)
    host_share = off_diag.max(dim=1).values / host_total     # (C,)

    # --- EMA buffer update (target-adaptive) ---
    area_mean = buffers['area_mean']
    conn_mean = buffers['conn_mean']
    contain_mean = buffers['contain_mean']
    P_cooc = buffers['cooc']
    count = buffers['count']

    if target_adaptive:
        batch_area = area_frac.mean(dim=0)                  # (C,)
        batch_frag = frag.mean(dim=0)                       # (C,)
        present = (area_px.sum(dim=0) > 0).float()          # (C,)

        def _ema_update(buf, val):
            # zero momentum on a class's first observation, else `ema`
            fresh = (count == 0).float()
            m = ema * (1 - fresh)
            new = m * buf + (1 - m) * val
            return torch.where(present.bool(), new, buf)

        area_mean = _ema_update(area_mean, batch_area)
        conn_mean = _ema_update(conn_mean, batch_frag)
        contain_mean = _ema_update(contain_mean, host_share)
        # P_cooc updates only for rows actually observed this batch
        observed = (row.squeeze(1) > 1.0).float().view(C, 1)
        m_cooc = ema if count.sum() > 0 else 0.0
        P_cooc = torch.where(
            observed.bool(),
            m_cooc * P_cooc + (1 - m_cooc) * P_batch,
            P_cooc)
        count = count + present

        buffers = dict(area_mean=area_mean, conn_mean=conn_mean,
                       contain_mean=contain_mean, cooc=P_cooc, count=count)

    # --- per-(b,c) plausibility terms (broadcast to pixels later) ---
    warm = (count >= min_count).float().view(1, C)          # (1,C)

    s_area = torch.exp(-(torch.log((area_frac + eps) /
                                   (area_mean.view(1, C) + eps))).abs())
    s_area = warm * s_area + (1 - warm) * 1.0

    s_conn = torch.exp(-F.relu(frag - conn_mean.view(1, C)) /
                       (conn_mean.view(1, C) + eps))
    s_conn = warm * s_conn + (1 - warm) * 1.0

    if enable_containment:
        s_contain = torch.exp(-(host_share.view(1, C) -
                                contain_mean.view(1, C)).abs())
        s_contain = warm * s_contain + (1 - warm) * 1.0
        s_contain = s_contain.expand(B, C)
    else:
        s_contain = torch.ones(B, C, device=device)

    s_bc = (s_area * s_conn * s_contain).clamp(0.0, 1.0)    # (B,C)

    # gather per-pixel plausibility of the *assigned* class
    lab = pseudo_label.clamp(0, C - 1)
    s_assigned = s_bc.gather(1, lab.reshape(B, -1)).view(B, H, W)

    # co-occurrence per-pixel for assigned class and for the candidate
    cooc_assigned = _cooc_plausibility(pseudo_label, neighbors, P_cooc, C)
    s_map = (s_assigned * cooc_assigned).clamp(0.0, 1.0)

    # s_cstar: neighbourhood support for the hypothesised rare flip.
    cstar_q = cstar_map.clamp(min=0)
    s_cstar = _cooc_plausibility(cstar_q, neighbors, P_cooc, C)
    s_cstar = torch.where(cstar_map >= 0, s_cstar,
                          torch.zeros_like(s_cstar)).clamp(0.0, 1.0)

    return s_map, s_cstar, buffers
