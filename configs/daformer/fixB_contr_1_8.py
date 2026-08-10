_base_ = ['./_dapcn18_base.py']
uda = dict(
    proto_correction=False,
    boundary_lambda=0.0,
    proto_lambda=0.0,
    contrastive_lambda=0.1,
    dynamic_anchor=None,
    dapg_loss=None,
)
work_dir = 'work_dirs/T_fixB_contr_1_8'
name = 'fixB_contr_1_8'
