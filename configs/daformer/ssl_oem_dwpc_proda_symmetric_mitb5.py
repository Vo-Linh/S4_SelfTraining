# ---------------------------------------------------------------
# DWPC ablation row A1 (SSL) — ProDA-style symmetric Witness A
# Auto-generated from the A0-A7 matrix (DWPC_PROPOSAL.md §7).
# Inherits the full SSL v2 leaf; overrides only DWPC toggles.
# ---------------------------------------------------------------

_base_ = ['./ssl_oem_dapcn_daformer_mitb5_v2.py']

uda = dict(
    proto_correction=False,
    dwpc={   'enabled': True,
    'start_iter': 1500,
    'ramp_iters': 1000,
    'rare_class_ids': [0, 1, 6],
    'class_freq': [   0.00654,
                      0.01488,
                      0.22473,
                      0.16005,
                      0.06658,
                      0.20029,
                      0.03246,
                      0.13799,
                      0.15648],
    'witness_a_enabled': True,
    'witness_a_symmetric': True,
    'witness_a_rare_prior_gamma': 0.0,
    'witness_a_beta': 1.0,
    'witness_a_sigma2': 0.5,
    'witness_a_standardize': True,
    'witness_b_enabled': False,
    'witness_b_ema': 0.99,
    'witness_b_cc_iters': 64,
    'witness_b_resolution': 128,
    'witness_b_enable_containment': True,
    'witness_b_min_count': 5,
    'target_adaptive': True,
    'rare_gate': True,
    'flip_veto': True,
    'tau_flip': 0.5,
    'flip_budget': 0.01,
    'beta_s': 5.0})

name = 'ssl_oem_dwpc_proda_symmetric_mitb5'
exp = 'ssl_oem_dwpc'
name_uda = 'dwpc_proda_symmetric'
