# ---------------------------------------------------------------
# A3_iso_dapcnU3x (isolated): as A1, L_dapcn^U weight = 0.15 (A1 0.05, A2 0.5).
# Dose point between A1 and A2 (log midpoint): is the harm gradual or a threshold?
# ---------------------------------------------------------------

_base_ = ['./oem_biased_ssl_1_8_dualbank_lbank2opt_A1_iso_12k_mitb5.py']

uda = dict(proto_lambda_u=0.15)

name = 'oem_biased_ssl_1_8_dualbank_lbank2opt_A3_iso_dapcnU3x_12k_mitb5'
name_uda = 'dapcn_ssl_dualbank_lbank2opt_A3_iso_dapcnU3x_biased'
