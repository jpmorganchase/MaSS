# SPDX-License-Identifier: Apache-2.0
# Copyright : JP Morgan Chase & Co

# adapt from https://github.com/facebookresearch/deit/blob/main/utils.py

"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time
from collections import defaultdict, deque
import datetime
import tempfile
import logging
import argparse
import itertools
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
from fvcore.common.checkpoint import Checkpointer
from fvcore.nn.precise_bn import get_bn_modules, _PopulationVarianceEstimator


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total],
                         dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def load_checkpoint(model, state_dict, mode=None):

    # reuse Checkpointer in fvcore to support flexible loading
    ckpt = Checkpointer(model, save_to_disk=False)
    logging.basicConfig()
    ckpt.logger.setLevel(logging.DEBUG)
    # since Checkpointer requires the weight to be put under `model` field, we need to save it to disk
    tmp_path = tempfile.NamedTemporaryFile('w+b')
    torch.save({'model': state_dict}, tmp_path.name)
    ckpt.load(tmp_path.name)


# create a keyvalue class
class KeyValue(argparse.Action):
    # Constructor calling
    def __call__(self, parser, namespace,
                 values, option_string=None):
        setattr(namespace, self.dest, dict())

        for value in values:
            # split it into key and value
            key, value = value.split('=')
            # assign into dictionary
            try:
                value = float(value)
            except:
                value = value
            getattr(namespace, self.dest)[key] = value


@torch.no_grad()
def update_bn_stats(
    model: nn.Module,
    data_loader: Iterable[Any],
    num_iters: int = 200,
    progress: Optional[str] = None,
) -> None:
    """
    Recompute and update the batch norm stats to make them more precise. During
    training both BN stats and the weight are changing after every iteration, so
    the running average can not precisely reflect the actual stats of the
    current model.
    In this function, the BN stats are recomputed with fixed weights, to make
    the running average more precise. Specifically, it computes the true average
    of per-batch mean/variance instead of the running average.
    See Sec. 3 of the paper "Rethinking Batch in BatchNorm" for details.

    Args:
        model (nn.Module): the model whose bn stats will be recomputed.

            Note that:

            1. This function will not alter the training mode of the given model.
               Users are responsible for setting the layers that needs
               precise-BN to training mode, prior to calling this function.

            2. Be careful if your models contain other stateful layers in
               addition to BN, i.e. layers whose state can change in forward
               iterations.  This function will alter their state. If you wish
               them unchanged, you need to either pass in a submodule without
               those layers, or backup the states.
        data_loader (iterator): an iterator. Produce data as inputs to the model.
        num_iters (int): number of iterations to compute the stats.
        progress: None or "tqdm". If set, use tqdm to report the progress.
    """
    bn_layers = get_bn_modules(model)

    if len(bn_layers) == 0:
        return
    print(
        f"Computing precise BN statistics for {len(bn_layers)} BN layers ...")

    if len(data_loader) < num_iters:
        print(f"Number of iterations is larger than the length of data loader. Change to num_iters to the length of dataloader")
        num_iters = len(data_loader)

    # In order to make the running stats only reflect the current batch, the
    # momentum is disabled.
    # bn.running_mean = (1 - momentum) * bn.running_mean + momentum * batch_mean
    # Setting the momentum to 1.0 to compute the stats without momentum.
    momentum_actual = [bn.momentum for bn in bn_layers]
    for bn in bn_layers:
        bn.momentum = 1.0

    batch_size_per_bn_layer: Dict[nn.Module, int] = {}

    def get_bn_batch_size_hook(
        module: nn.Module, input: Tuple[torch.Tensor]
    ) -> Tuple[torch.Tensor]:
        assert (
            module not in batch_size_per_bn_layer
        ), "Some BN layers are reused. This is not supported and probably not desired."
        x = input[0]
        assert isinstance(
            x, torch.Tensor
        ), f"BN layer should take tensor as input. Got {input}"
        # consider spatial dimensions as batch as well
        batch_size = x.numel() // x.shape[1]
        batch_size_per_bn_layer[module] = batch_size
        return (x,)

    hooks_to_remove = [
        bn.register_forward_pre_hook(get_bn_batch_size_hook) for bn in bn_layers
    ]

    estimators = [
        _PopulationVarianceEstimator(bn.running_mean, bn.running_var)
        for bn in bn_layers
    ]

    ind = -1
    for inputs, _ in tqdm.tqdm(
        itertools.islice(data_loader, num_iters),
        total=num_iters,
        disable=progress != "tqdm",
    ):
        ind += 1
        batch_size_per_bn_layer.clear()
        model(inputs)

        for i, bn in enumerate(bn_layers):
            # Accumulates the bn stats.
            batch_size = batch_size_per_bn_layer.get(bn, None)
            if batch_size is None:
                continue  # the layer was unused in this forward
            estimators[i].update(bn.running_mean, bn.running_var, batch_size)
    assert ind == num_iters - 1, (
        "update_bn_stats is meant to run for {} iterations, "
        "but the dataloader stops at {} iterations.".format(num_iters, ind)
    )

    for i, bn in enumerate(bn_layers):
        # Sets the precise bn stats.
        bn.running_mean = estimators[i].pop_mean
        bn.running_var = estimators[i].pop_var
        bn.momentum = momentum_actual[i]
    for hook in hooks_to_remove:
        hook.remove()
