# ---------------------------------------------------------------
# UDA ladder — TIER 3: DAPCN + DWPC (full method).
#
# The full DAPCN UDA stack (inherited from tier 2) with the complete
# DWPC dual-witness pseudo-label correction enabled (Witness A
# appearance + Witness B structure + rare-gate + target-adaptive +
# flip-veto). Same DWPC block as the SSL tier-3 config.
#
# Ladder:
#   tier 1  uda_oem_daformer_selftrain_mitb5     (self-training)
#   tier 2  uda_oem_dapcn_dynanchor_proto_mitb5  (+ DynAnchor + ProtoMem)
#   tier 3  THIS FILE                            (+ DWPC dual-witness)
#
# tier3 - tier2  ==  the effect of the DWPC block, ALONE.
# ---------------------------------------------------------------

_base_ = ['./uda_oem_dapcn_dynanchor_proto_mitb5.py']

uda = dict(
    dwpc=dict(
        enabled=True,
        start_iter=1500,
        ramp_iters=1000,
        rare_class_ids=[0, 1, 6],
        class_freq=[0.00654, 0.01488, 0.22473, 0.16005, 0.06658,
                    0.20029, 0.03246, 0.13799, 0.15648],
        witness_a_enabled=True,
        witness_a_symmetric=False,
        witness_a_rare_prior_gamma=1.0,
        witness_a_beta=1.0,
        witness_a_sigma2=0.5,
        witness_a_standardize=True,
        witness_b_enabled=True,
        witness_b_ema=0.99,
        witness_b_cc_iters=64,
        witness_b_resolution=128,
        witness_b_enable_containment=True,
        witness_b_min_count=5,
        target_adaptive=True,
        rare_gate=True,
        flip_veto=True,
        tau_flip=0.5,
        flip_budget=0.005,
        beta_s=5.0,
    ),
)

name = 'uda_oem_dapcn_dwpc_full_mitb5'
exp = 'uda_oem_dwpc'
name_uda = 'dapcn_dwpc_full'
