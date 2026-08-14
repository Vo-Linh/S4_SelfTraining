# ---------------------------------------------------------------
# DAPCN-SSL Base Configuration
# Semi-Supervised Learning with Dynamic Attention-based
# Prototype Clustering Network
# Optimised for satellite image segmentation
# ---------------------------------------------------------------

uda = dict(
    type='DAPCN_SSL',
    # --- Self-training (EMA teacher) ---
    alpha=0.999,
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
    mix='class',
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    debug_img_interval=1000,
    # --- SSL-specific ---
    # Warmup: ramp up pseudo-label weight over N iters
    # (teacher is unreliable early on with few labeled samples)
    pseudo_label_warmup_iters=1000,
    # --- Prototype-based pseudo-label correction ---
    # Compare each target pixel against the persistent semantic class
    # prototypes. The nearest prototype per class gives a class distribution
    # that is blended with the EMA teacher. ``dynamic_anchor`` remains
    # available to reproduce earlier class-agnostic correction experiments.
    proto_correction=True,
    proto_correction_mode='class_prototype',
    proto_correction_alpha=0.8,
    proto_correction_temperature=0.1,
    proto_correction_start_iter=1000,
    # --- DAPCN loss weights ---
    boundary_lambda=0.3,
    proto_lambda=0.1,
    # --- Class-conditioned PrototypeMemory + InfoNCE contrastive ---
    # Activated on the labeled CE step only (mixed/pseudo features
    # would corrupt the class-conditioned bank). Set contrastive_lambda
    # to 0 to disable.
    contrastive_lambda=0.1,
    contrastive_temp=0.07,
    num_prototypes_per_class=1,
    prototype_ema=0.999,
    prototype_init_strategy='zeros',
    # --- Boundary configuration ---
    boundary_loss_mode='affinity',
    boundary_mode='sobel',
    apply_boundary_on_target=True,
    apply_proto_on_target=True,
    hybrid_binary_weight=0.5,
    ignore_index=255,
    # --- DynamicAnchorModule ---
    dynamic_anchor=dict(
        type='DynamicAnchorModule',
        max_groups=96,
        temperature=0.1,
        num_iters=3,
        init_method='xavier',
        min_quality=0.1,
        use_quality_gate=True,
        use_mask_predictor=False,
        ema_decay=0.0,
    ),
    # --- DAPGLoss ---
    dapg_loss=dict(
        type='DAPGLoss',
        margin=0.3,
        lambda_inter=0.5,
        lambda_quality=0.1,
        loss_weight=1.0,
    ),
    # --- AffinityBoundaryLoss ---
    affinity_loss=dict(
        type='AffinityBoundaryLoss',
        temperature=0.5,
        scale=2,
        num_neighbors=4,
        ignore_index=255,
        loss_weight=1.0,
    ),
)
use_ddp_wrapper = True
