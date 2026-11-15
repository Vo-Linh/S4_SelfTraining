# ---------------------------------------------------------------
# DualBank, class-aware bank_L + two optimizers  --  A1 (isolated)
#
# Base of the 4-run study (A1, A2, B1, B2); all inherit this file.
#
#   bank_L   built WITH the labels: 4 anchors x 8 classes, EM inside
#            each class (LabeledDualBankAnchorModule); P2^u reads it with
#            V = onehot(anchor class).  bank_U unchanged (label-free).
#   optimizer  ('main')  L_seg, L_boundary, L_dapcn^L, L_dapcn^U
#                        -> backbone, decode head (fusion + Psi_1)
#   optimizer2 ('aux')   L_sup2, L_con1, L_con2
#                        -> A: ONLY the P2 branch (W_q, W_k, conv_seg_proto)
#
# In A the shared encoder / fusion / Psi_1 never see the aux losses:
# L_con1 has no parameter to update, and the images without labels
# reach P1 only through L_dapcn^U (plus, indirectly, through W_q / W_k,
# which the EM of both banks uses).  W_q / W_k belong to 'aux' only, so
# the L_dapcn gradient on them is dropped.
#
# A1: L_dapcn^L = L_dapcn^U = 0.05 (baseline weights).
# A2: L_dapcn^U x10                 (does bank_U clustering fight overfit?)
# B1: aux also updates backbone + head, same lr multipliers as 'main'
# B2: aux also updates backbone + head, small lr there
# ---------------------------------------------------------------

import mmseg.models.uda.dapcn_dualbank_2opt
import mmseg.models.uda.dynamic_anchor_dualbank_labeled

del mmseg

_base_ = ['./oem_biased_ssl_1_8_dualbank_cross_40k_mitb5.py']

uda = dict(
    type='DAPCN_SSL_DualBank2Opt',
    aux_loss_keys=('loss_sup2', 'loss_con1', 'loss_con2'),
    labeled_bank_value='onehot',
    proto_lambda=0.05,
    proto_lambda_u=0.05,
    dynamic_anchor=dict(
        type='LabeledDualBankAnchorModule',
        num_classes=8,          # max_groups=32 -> 4 anchors per class
        min_class_pixels=16,
    ),
)

# Parameters that only the aux optimizer owns (the P2 branch).
_p2_branch = [
    'dynamic_anchor.query_proj',
    'dynamic_anchor.key_proj',
    'dynamic_anchor.lookup_logit_scale',
    'conv_seg_proto',
    'head_chung',
]

# 'main': inherited AdamW(6e-5) + paramwise (backbone x1, head x10),
# minus the P2 branch.
optimizer = dict(exclude_prefixes=_p2_branch)

# 'aux': the P2 branch only.
optimizer2 = dict(
    type='AdamW', lr=6e-05, betas=(0.9, 0.999), weight_decay=0.01,
    param_prefixes=_p2_branch,
    paramwise_cfg=dict(
        custom_keys={
            'dynamic_anchor.query_proj': dict(lr_mult=10.0, decay_mult=0.0),
            'dynamic_anchor.key_proj': dict(lr_mult=10.0, decay_mult=0.0),
            'dynamic_anchor.lookup_logit_scale': dict(
                lr_mult=10.0, decay_mult=0.0),
            'conv_seg_proto': dict(lr_mult=10.0),
            'head_chung': dict(lr_mult=10.0),
        }))

runner = dict(type='IterBasedRunner', max_iters=12000)

name = 'oem_biased_ssl_1_8_dualbank_lbank2opt_A1_iso_12k_mitb5'
name_uda = 'dapcn_ssl_dualbank_lbank2opt_A1_iso_biased'
