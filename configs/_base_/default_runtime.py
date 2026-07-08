# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
# Email notification on training completion / error / eval milestones.
# Self-disabling when SMTP_* / NOTIFY_RECIPIENT env vars are absent.
custom_hooks = [
    dict(
        type='EmailNotificationHook',
        subject_prefix='[daformer]',
        milestone_every=1,
        log_tail_lines=50,
        nan_guard=True),
]
