# ---------------------------------------------------------------
# DWPC ablation row A0 (SSL) — DAPCN-SSL/UDA baseline (DWPC off)
# Auto-generated from the A0-A7 matrix (DWPC_PROPOSAL.md §7).
# Inherits the full SSL v2 leaf; overrides only DWPC toggles.
# ---------------------------------------------------------------

_base_ = ['./ssl_oem_dapcn_daformer_mitb5_v2.py']

uda = dict(dwpc={'enabled': False})

name = 'ssl_oem_dwpc_off_mitb5'
exp = 'ssl_oem_dwpc'
name_uda = 'dwpc_off'
