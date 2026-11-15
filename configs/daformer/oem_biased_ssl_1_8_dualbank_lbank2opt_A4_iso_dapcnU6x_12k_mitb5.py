# ---------------------------------------------------------------
# A4 (isolated): as A1, L_dapcn^U weight = 0.3 (A0 0, A1 0.05, A3 0.15, A2 0.5).
# Locates the break between 0.15 (= A1) and 0.5 (-10 mIoU).
# ---------------------------------------------------------------

_base_ = ['./oem_biased_ssl_1_8_dualbank_lbank2opt_A1_iso_12k_mitb5.py']

uda = dict(proto_lambda_u=0.3)

name = 'oem_biased_ssl_1_8_dualbank_lbank2opt_A4_iso_dapcnU6x_12k_mitb5'
name_uda = 'dapcn_ssl_dualbank_lbank2opt_A4_iso_dapcnU6x_biased'
