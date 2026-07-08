# ---------------------------------------------------------------
# DWPC ablation row A0 (UDA) — DAPCN-SSL/UDA baseline (DWPC off)
# Auto-generated from the A0-A7 matrix (DWPC_PROPOSAL.md §7).
# Inherits the full UDA OEM leaf; overrides only DWPC toggles.
# ---------------------------------------------------------------

_base_ = ['./uda_oem_dapcn_dynanchor_proto_mitb5.py']

uda = dict(dwpc={'enabled': False})

name = 'uda_oem_dwpc_off_mitb5'
exp = 'uda_oem_dwpc'
name_uda = 'dwpc_off'
