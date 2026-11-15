# ---------------------------------------------------------------
# A0_iso_dapcnU0 (isolated): as A1, L_dapcn^U weight = 0.0 (A1 0.05, A2 0.5).
# Zero point: does L_dapcn^U at 0.05 help at all? With 0 the loss is skipped (bank_U not updated; P2^u reads bank_L).
# ---------------------------------------------------------------

_base_ = ['./oem_biased_ssl_1_8_dualbank_lbank2opt_A1_iso_12k_mitb5.py']

uda = dict(proto_lambda_u=0.0)

name = 'oem_biased_ssl_1_8_dualbank_lbank2opt_A0_iso_dapcnU0_12k_mitb5'
name_uda = 'dapcn_ssl_dualbank_lbank2opt_A0_iso_dapcnU0_biased'
