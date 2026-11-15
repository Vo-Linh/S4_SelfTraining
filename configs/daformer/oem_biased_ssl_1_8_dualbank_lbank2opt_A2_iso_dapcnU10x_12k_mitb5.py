# ---------------------------------------------------------------
# A2 (isolated): as A1, L_dapcn^U weight x10 (0.05 -> 0.5).
# Question: does the label-free clustering on bank_U reduce overfitting
# when it is the ONLY unlabeled signal that reaches the encoder?
# Compare against A1 (same everything, 0.05).
# ---------------------------------------------------------------

_base_ = ['./oem_biased_ssl_1_8_dualbank_lbank2opt_A1_iso_12k_mitb5.py']

uda = dict(proto_lambda_u=0.5)

name = 'oem_biased_ssl_1_8_dualbank_lbank2opt_A2_iso_dapcnU10x_12k_mitb5'
name_uda = 'dapcn_ssl_dualbank_lbank2opt_A2_iso_dapcnU10x_biased'
