# ---------------------------------------------------------------
# DAPCN-SSL DualBank (cross) + DAFormer — OpenEarthMap, 1/8 split
#
# Two label-free anchor banks (bank_L from F^l, bank_U from F^u) and a
# cross-attention read-out: the labeled stream reads bank_U, the
# unlabeled stream reads bank_L.  Variant A of the cross-attention:
# cosine-softmax (W_q, W_k), single head, V = class distribution of the
# anchor (conv_seg_proto).  P1 (conv_seg) and P2 (cross-attn) supervise
# each other on unlabeled pixels.  No EMA teacher, no ClassMix.
#
# F = fused decoder feature (256-d, anchor_after_fusion=True).
# Eval reports P1 (the standard decode path).
#
# Siblings (all inherit this file):
#   oem_ssl_1_8_dualbank_p2sup_mitb5.py  con_lambda=0: labeled-only phase
#   oem_ssl_1_8_dualbank_own_mitb5.py    each stream reads its OWN bank
#
# Base chain is the 10k-iter debug run (val on 100 images) so the first
# experiments are cheap; override runner/evaluation for a full run.
# ---------------------------------------------------------------

import mmseg.models.uda.dapcn_dualbank
import mmseg.models.uda.dynamic_anchor_dualbank

del mmseg

_base_ = ['./oem_ssl_1_8_debug_s2_mitb5.py']

uda = dict(
    type='DAPCN_SSL_DualBank',
    anchor_after_fusion=True,
    # --- two-bank cross-attention ---
    dualbank_mode='cross',
    sup2_lambda=1.0,
    # Cross pseudo-supervision between P1 and P2 (unlabeled).  Banks
    # start from xavier noise, so keep it off until they have settled.
    con_lambda=1.0,
    con_start_iter=1000,
    con_warmup_iters=1000,
    con_threshold=0.9,
    diag_interval=50,
    # --- unused teacher machinery, off ---
    proto_correction=False,
    # conv_seg_proto (anchor -> class head, the "Head chung") is only
    # built when corrector_lambda > 0.  The corrector LOSS is not used
    # by this class; sup2_lambda replaces it.
    corrector_lambda=1.0,
    # --- DAPCN losses, one per stream ---
    boundary_lambda=0.5,
    proto_lambda=0.05,
    proto_lambda_u=0.05,
    dynamic_anchor=dict(
        type='DualBankAnchorModule',
        bank_names=('labeled', 'unlabeled'),
        #   bank_max_cos  refuse to write a refined set whose mean
        #                 off-diagonal cosine is above this (collapsed);
        #   ema_warmup    let a young bank follow the data instead of
        #                 being dominated by its first write.
        bank_init='feature_sample',
        bank_max_cos=0.9,
        ema_warmup=True,
        max_groups=32,
        temperature=0.1,
        num_iters=3,
        init_method='xavier',
        use_quality_gate=False,
        use_learned_assignment=True,
        ema_decay=0.999,
    ),
)

custom_hooks = []

optimizer = dict(
    paramwise_cfg=dict(
        custom_keys={
            'head': dict(lr_mult=10.0),
            'dynamic_anchor.prototypes': dict(lr_mult=10.0, decay_mult=0.0),
            'dynamic_anchor.query_proj': dict(lr_mult=10.0, decay_mult=0.0),
            'dynamic_anchor.key_proj': dict(lr_mult=10.0, decay_mult=0.0),
            'conv_seg_proto': dict(lr_mult=10.0),
            'pos_block': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        }))

name = 'oem_ssl_1_8_dualbank_cross_mitb5'
name_uda = 'dapcn_ssl_dualbank_cross'
