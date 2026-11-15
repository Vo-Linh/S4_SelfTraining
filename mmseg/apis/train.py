# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Add ddp_wrapper from mmgen

import random
import warnings

import mmcv
import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import HOOKS, build_optimizer, build_runner
from mmcv.utils import build_from_cfg

from mmseg.core import DistEvalHook, EvalHook
from mmseg.core.ddp_wrapper import DistributedDataParallelWrapper
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import get_root_logger


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_subset_optimizer(model, optimizer_cfg):
    """Build an optimizer over a subset of the model's parameters.

    Same ``paramwise_cfg`` rules as mmcv's DefaultOptimizerConstructor,
    then the parameter groups are filtered by name (names as in
    ``model.module.named_parameters()``, e.g. ``model.backbone.*``,
    ``dynamic_anchor.query_proj.weight``):

        param_prefixes    keep only names starting with one of these
                          (None: keep all)
        exclude_prefixes  then drop names starting with one of these
    """
    from mmcv.runner.optimizer import DefaultOptimizerConstructor
    from mmcv.runner.optimizer.builder import OPTIMIZERS

    cfg = dict(optimizer_cfg)
    include = cfg.pop('param_prefixes', None)
    exclude = tuple(cfg.pop('exclude_prefixes', None) or ())
    paramwise_cfg = cfg.pop('paramwise_cfg', None) or dict()
    module = model.module if hasattr(model, 'module') else model

    constructor = DefaultOptimizerConstructor(cfg, paramwise_cfg)
    groups = []
    constructor.add_params(groups, module)
    names = {id(p): n for n, p in module.named_parameters()}
    keep = []
    for g in groups:
        p = g['params'][0]
        n = names[id(p)]
        if not p.requires_grad:
            continue
        if include is not None and not n.startswith(tuple(include)):
            continue
        if exclude and n.startswith(exclude):
            continue
        keep.append(g)
    assert keep, 'optimizer selects no parameter: %r' % (optimizer_cfg, )
    cfg['params'] = keep
    return build_from_cfg(cfg, OPTIMIZERS)


def train_segmentor(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):
    """Launch segmentor training."""
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=True) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        use_ddp_wrapper = cfg.get('use_ddp_wrapper', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        if use_ddp_wrapper:
            mmcv.print_log('Use DDP Wrapper.', 'mmseg')
            model = DistributedDataParallelWrapper(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    # A second optimizer (``optimizer2``) turns the runner's optimizer into
    # dict(main=..., aux=...); each one holds only the parameters its
    # ``param_prefixes`` / ``exclude_prefixes`` select.  Without
    # ``optimizer2`` nothing changes.
    if cfg.get('optimizer2') is not None:
        optimizer = dict(
            main=build_subset_optimizer(model, cfg.optimizer),
            aux=build_subset_optimizer(model, cfg.optimizer2))
        for _k, _o in optimizer.items():
            mmcv.print_log(
                'optimizer %s: %d params in %d groups' % (
                    _k, sum(p.numel() for g in _o.param_groups
                            for p in g['params']), len(_o.param_groups)),
                'mmseg')
    else:
        optimizer = build_optimizer(model, cfg.optimizer)

    if cfg.get('runner') is None:
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # register hooks
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    # register user-defined hooks (e.g. ProtoCorrectionDebugHook)
    for hook_cfg in cfg.get('custom_hooks', None) or []:
        assert isinstance(hook_cfg, dict), \
            f'Each item in custom_hooks must be a dict, but got {hook_cfg}'
        hook_cfg = hook_cfg.copy()
        priority = hook_cfg.pop('priority', 'NORMAL')
        runner.register_hook(build_from_cfg(hook_cfg, HOOKS), priority=priority)

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        # priority='LOW' puts EvalHook *after* IterTimerHook. mmcv >=1.4
        # makes EvalHook flush the logger and then `log_buffer.clear()`
        # before evaluating; if IterTimerHook runs afterwards it re-adds
        # 'time' (but not 'data_time', pushed back in before_train_iter)
        # to the emptied buffer, so TextLoggerHook takes the next line for
        # a training log and dies on KeyError: 'data_time'.
        # mmseg 0.16 targets mmcv 1.3.7, which lacks that flush — hence
        # the stock ordering breaks against the installed mmcv 1.4.0.
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
