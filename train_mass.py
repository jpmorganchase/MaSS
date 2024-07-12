# SPDX-License-Identifier: Apache-2.0
# Copyright : JP Morgan Chase & Co
 
import argparse
import datetime
import time
from pathlib import Path
import json
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma


import models
from utils.datasets import build_dataset
from engine.mass_engine import train_one_epoch, evaluate, get_basic_stats, final_eval_one_epoch
import utils.misc as utils
from parser import get_args_parser

def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2"
        cudnn.deterministic = True
    
    cudnn.benchmark = True

    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    print(f"Creating model: {args.model}")
    if args.dataset == 'audiomnist':
        in_channels = 768
    elif args.dataset == 'motionsense':
        in_channels = 2
    
    # create GAN trainer
    # 0: generator, using MLP now
    generator = create_model(
        args.gen_model,
        pretrained=False,
        in_chans=in_channels,
        with_residual=args.gen_with_residual
        )

    # 1: discriminators
    attributes = list(args.loss_n.keys()) + list(args.loss_n.keys()) + list(args.loss_m.keys()) + \
        args.eval_attributes \
        + ['infonce']
    discriminators = nn.ModuleDict()
    for idx, attr in enumerate(attributes):
        if attr == 'infonce':
            model_name = args.disc_infonce_model
            num_classes = 128 # for both audiomnist and motionsense
        else:
            model_name = args.disc_model
            num_classes = dataset_train.num_classes[attr]
        tmp = create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes,
            in_chans=in_channels
        )
        # load the pretrained weights
        if args.attribute_models_weights is not None:
            if attr in args.attribute_models_weights:
                state_dict = torch.load(
                    args.attribute_models_weights[attr], map_location='cpu')['model']
                if attr == 'infonce':
                    state_dict = {k.replace('module.', '')
                                            : v for k, v in state_dict.items()}
                # print(tmp.load_state_dict(state_dict))
                utils.load_checkpoint(tmp, state_dict)
                print(f"loading {attr} model from {args.attribute_models_weights[attr]}")
        discriminators[attr] = tmp

    # 2: build the trainer
    model = create_model(
        args.model,
        pretrained=False,
        generator=generator,
        discriminators=discriminators,
        supp_attr=list(args.loss_m.keys()),
        # pres_attr=list(args.loss_n.keys()),
        pres_attr=list(args.loss_n.keys()),
        eval_attr=args.eval_attributes,
        in_chans=in_channels
    )

    print(model)
    model.to(device)

    model_ema = None
    if args.model_ema: 
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * \
        utils.get_world_size() / 256  # 512.0
    args.lr = linear_scaled_lr

    optimizer = create_optimizer(args, model.module.get_optimizer_params())
    loss_scaler = NativeScaler()
    optimizer_dis = create_optimizer(
        args, model.module.discriminators_xp.parameters())
    loss_scaler_dis = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)
    lr_scheduler_dis, _ = create_scheduler(args, optimizer_dis)

    criterion = {}
    for attr in args.eval_attributes:
        criterion[attr] = 'eval'
    for attr in args.loss_m.keys():
        criterion[attr] = 'sup'
    for attr in args.loss_n.keys():
        criterion[attr] = 'pre'
    criterion['infonce'] = 'infonce'

    output_dir = Path(args.output_dir)

    mutual_information_attr_x_dict, prior_attr_dict = get_basic_stats(
        data_loader_train, model, device, args)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if args.eval:
        args.resume = args.output_dir + 'checkpoint.pth'
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        utils.load_checkpoint(model_without_ddp, checkpoint['model'])
        # model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            optimizer_dis.load_state_dict(checkpoint['optimizer_dis'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            lr_scheduler_dis.load_state_dict(checkpoint['lr_scheduler_dis'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(
                    model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
                loss_scaler_dis.load_state_dict(checkpoint['scaler_dis'])
        lr_scheduler.step(args.start_epoch)
        lr_scheduler_dis.step(args.start_epoch)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device,
                              args, criterion, 0, mutual_information_attr_x_dict, prior_attr_dict)
        final_eval(model, attributes, criterion, in_channels, device, output_dir, dataset_train,
                   data_loader_train, data_loader_val, args)
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, optimizer_dis, device, epoch, loss_scaler, loss_scaler_dis,
            args.clip_grad, model_ema, mixup_fn,
            mutual_information_attr_x_dict=mutual_information_attr_x_dict, prior_attr_dict=prior_attr_dict,
            args=args
        )

        lr_scheduler.step(epoch)
        lr_scheduler_dis.step(epoch)

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'optimizer_dis': optimizer_dis.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'lr_scheduler_dis': lr_scheduler_dis.state_dict(),
                    'epoch': epoch,
                    'scaler': loss_scaler.state_dict(),
                    'scaler_dis': loss_scaler_dis.state_dict(),
                    'args': args,
                }, checkpoint_path)

        test_stats = evaluate(data_loader_val, model, device,
                              args, criterion, epoch, mutual_information_attr_x_dict, prior_attr_dict)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    mass_results = {}
    for attr in args.loss_m:
        for k, v in test_stats.items():
            if attr in k:
                mass_results[f'test_{k}'] = v

    attributes = list(set(list(args.loss_n.keys()) + args.eval_attributes))
    criterion = {}
    for attr in args.loss_n.keys():
        criterion[attr] = 'pre'
    for attr in args.eval_attributes:
        criterion[attr] = 'eval'
    test_stats = final_eval(model, attributes, criterion, in_channels, device, output_dir, dataset_train,
               data_loader_train, data_loader_val, args)
    
    for attr in attributes:
        for k, v in test_stats.items():
            if attr in k:
                mass_results[f'test_{k}'] = v

    with (output_dir / "mass_log.txt").open("a") as f:
        f.write(json.dumps(mass_results) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def final_eval(model, attributes, criterion, in_channels, device, output_dir, dataset_train, data_loader_train, data_loader_val, args):
    discriminators_new = nn.ModuleDict()
    for idx, attr in enumerate(attributes):
        model_name = args.disc_model
        num_classes = dataset_train.num_classes[attr]
        tmp = create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes,
            # drop_rate=args.drop,
            # drop_path_rate=args.drop_path,
            # drop_block_rate=None,
            in_chans=in_channels
        )
        if args.new_dis_attribute_models_weights is not None:
            if args.new_dis_attribute_models_weights.get(attr, None):
                state_dict = torch.load(
                    args.new_dis_attribute_models_weights[attr], map_location='cpu')['model']
                utils.load_checkpoint(tmp, state_dict)
        discriminators_new[attr] = tmp
    discriminators_new.to(device)

    n_parameters = sum(p.numel()
                       for p in discriminators_new.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # args.lr = 0.0001
    # args.weight_decay=0.05
    optimizer_dis = create_optimizer(
        args, discriminators_new.parameters())
    loss_scaler_dis = NativeScaler()
    lr_scheduler_dis, _ = create_scheduler(args, optimizer_dis)

    test_stats = final_eval_one_epoch(model, discriminators_new, criterion,
                                      data_loader_val, None, None,
                                      device, -1, None, None, max_norm=args.clip_grad, mode='test', args=args)

    for epoch in range(0, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = final_eval_one_epoch(model, discriminators_new, criterion,
                                           data_loader_train, None, optimizer_dis,
                                           device, epoch, None, loss_scaler_dis, max_norm=args.clip_grad, args=args)

        lr_scheduler_dis.step(epoch)

        test_stats = final_eval_one_epoch(model, discriminators_new, criterion,
                                          data_loader_val, None, None,
                                          device, epoch, None, None, max_norm=args.clip_grad, mode='test', args=args)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    return test_stats

if __name__ == '__main__':
    # select(0.8)
    parser = argparse.ArgumentParser(
        'MaSS trainer', parents=[get_args_parser()])
    args = parser.parse_args()
    args.output_dir = f'{args.output_dir}/{args.gen_model}-m-{args.loss_m}-n-{args.loss_n}-eval-{args.eval_attributes}-lr{args.lr}-seed{args.seed}/'
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # compose sensitive attribute based on loss m
    args.s_attributes = [k for k in args.loss_m.keys()]
    main(args)


