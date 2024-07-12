# SPDX-License-Identifier: Apache-2.0
# Copyright : JP Morgan Chase & Co
 
import math
import sys
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from timm.loss import LabelSmoothingCrossEntropy

from utils.losses import InfoNCELoss
import utils.misc as utils


def get_basic_stats(data_loader, model, device, args):

    # statistics are done in a single GPU

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Basic Stats:'
    # switch to evaluation mode
    model.module.eval_and_freeze()

    attributes = model.module.pres_attr + \
        model.module.supp_attr + model.module.eval_attr

    # create holders for statistics
    attr_counter = {}
    for attr in attributes:
        attr_counter[attr] = torch.zeros(
            data_loader.dataset.num_classes[attr]).to(device, non_blocking=True)

    cross_attr_counter = {}
    for attr in model.module.supp_attr:
        cross_attr_counter[attr] = {}
        for attr2 in model.module.pres_attr:
            cross_attr_counter[attr][attr2] = torch.zeros((
                data_loader.dataset.num_classes[attr], data_loader.dataset.num_classes[attr2])).to(device, non_blocking=True)

    cond_ent_attr_x_dict = {}
    for attr in attributes:
        cond_ent_attr_x_dict[attr] = 0

    total_num = 0

    for images, targets in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        targets = {attr: target.to(device, non_blocking=True)
                   for attr, target in targets.items() if attr != 'file_name'}

        # compute statistics for each batch
        with torch.cuda.amp.autocast():
            with torch.no_grad():

                total_num += images.shape[0]

                # H(U_j|X)
                references = model.module.run_discriminators(images)
                for attr in attributes:
                    cond_ent_attr_x = Categorical(
                        logits=references[1][attr]).entropy().sum()
                    cond_ent_attr_x_dict[attr] += cond_ent_attr_x

                # P(U) and P(S)
                for attr in attributes:
                    target = targets[attr]
                    for t in target:
                        attr_counter[attr][t] += 1

                # P(U|S)
                for attr in model.module.supp_attr:
                    for attr2 in model.module.pres_attr:
                        target = targets[attr]
                        target2 = targets[attr2]
                        for t_index in range(target.shape[0]):
                            t = target[t_index]
                            t2 = target2[t_index]
                            cross_attr_counter[attr][attr2][t, t2] += 1

    # aggregate the statistics
    cond_ent_attr_x_dict = {k: v/total_num for k,
                            v in cond_ent_attr_x_dict.items()}
    prior_attr_dict = {k: v/total_num for k, v in attr_counter.items()}
    ent_attr_dict = {k: Categorical(probs=v).entropy()
                     for k, v in prior_attr_dict.items()}
    # mutual_information_attr_x_dict = {
    #     k: v - cond_ent_attr_x_dict[k] for k, v in ent_attr_dict.items()}
    mutual_information_attr_x_dict = ent_attr_dict

    joint_prob_u_s = {}
    for attr in model.module.supp_attr:
        joint_prob_u_s[attr] = {}
        for attr2 in model.module.pres_attr:
            joint_prob_u_s[attr][attr2] = cross_attr_counter[attr][attr2] / total_num
            tmp = joint_prob_u_s[attr][attr2].sum(dim=1)
            
    cond_prob_u_s = {}
    for attr in model.module.supp_attr:
        cond_prob_u_s[attr] = {}
        for attr2 in model.module.pres_attr:
            mask = (prior_attr_dict[attr] != 0)
            cond_prob_u_s[attr][attr2] = joint_prob_u_s[attr][attr2][mask] / \
                prior_attr_dict[attr][:, None][mask]
            
    cond_ent_u_s = {}
    for attr in model.module.supp_attr:
        cond_ent_u_s[attr] = {}
        for attr2 in model.module.pres_attr:
            mask = (prior_attr_dict[attr] != 0)
            cond_ent_u_s[attr][attr2] = torch.sum(Categorical(
                probs=cond_prob_u_s[attr][attr2]).entropy() * prior_attr_dict[attr][mask])

    # reporting

    # statistics
    msg = f'Basic Statistics: \ncond_ent_attr_x_dict: {cond_ent_attr_x_dict}\nprior_attr_dict: {prior_attr_dict}\nent_attr_dict: {ent_attr_dict}\nmutual_information_attr_x_dict: {mutual_information_attr_x_dict}\njoint_prob_u_s: {joint_prob_u_s}\ncond_prob_u_s: {cond_prob_u_s}\ncond_ent_u_s: {cond_ent_u_s}\n'
    print(msg)
    print(prior_attr_dict['id'].max())

    # solution check
    for attr in model.module.supp_attr:
        if args.loss_m[attr] < 0:
            print(f'loss_m for {attr} >= 0 not satisified. Exit')
            exit(0)
    loss_n_all = {**args.loss_n}
    for attr in model.module.pres_attr:
        if loss_n_all[attr] > ent_attr_dict[attr]:
            print(f'loss_n for {attr} <= its entropy not satisified. Exit')
            exit(0)
    for attr in model.module.supp_attr:
        for attr2 in model.module.pres_attr:
            if loss_n_all[attr2] > args.loss_m[attr] + cond_ent_u_s[attr][attr2]:
                print(f'loss_n <= loss_m + H(u|s) not satisfied. Exit')
    print("possible solution exists.")
    # returning useful statistics only
    return mutual_information_attr_x_dict, prior_attr_dict



def train_one_epoch(model: torch.nn.Module, criterion: nn.ModuleDict,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, optimizer_dis: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, loss_scaler_dis, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, mutual_information_attr_x_dict=None, prior_attr_dict=None, mode='train', args=None):

    if mode == 'train':
        model.module.train_and_unfreeze()
    else:
        model.module.eval_and_freeze()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if mode == 'train':
        metric_logger.add_meter('lr', utils.SmoothedValue(
            window_size=1, fmt='{value:.6f}'))
    header = mode + ' Epoch: [{}]'.format(epoch)
    print_freq = 10

    nce_loss = InfoNCELoss(temperature=0.07)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = {attr: target.to(device, non_blocking=True)
                   for attr, target in targets.items() if attr != 'file_name'}

        batch_size = samples.size(0)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=(args.deterministic == False and mode == 'train')):
            anonymized_x, outs, _, _, _, outsp, _ = model(samples, False)

            loss = 0.0
            losses = {}
            for attr, criter in criterion.items():
                # -------- unannotated attributes, F --------
                if criter == 'infonce': # Ours
                    f1, f2 = F.normalize(
                        outs[attr], dim=-1), F.normalize(outsp[attr], dim=-1)
                    features = torch.cat(
                        [f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    losses[attr] = args.loss_lambda_infonce * nce_loss(features)
                # -------- Suppressed attributes, S --------
                elif criter == 'sup': # this is the one used
                    kld = nn.CrossEntropyLoss(reduction="mean")(input=outsp[attr], target=targets[attr])
                    mi = mutual_information_attr_x_dict[attr]
                    m = args.loss_m[attr]
                    violation = min(kld - mi + m, torch.tensor(0., device=device))
                    losses[attr] = args.loss_lambda_s_2 * \
                        torch.square(violation) + \
                        args.loss_lambda_s_1 * torch.abs(violation)
                # -------- Annotated useful attributes, U --------
                elif criter == 'pre':
                    kld = nn.CrossEntropyLoss(reduction="mean")(
                            input=outsp[attr], target=targets[attr])
                    mi = mutual_information_attr_x_dict[attr]
                    n = args.loss_n[attr]
                    violation = max(
                        kld - mi + n, torch.tensor(0, device=device))
                    losses[attr] = args.loss_lambda_u_2 * \
                        torch.square(violation) + \
                        args.loss_lambda_u_1 * torch.abs(violation)
                elif criter == 'eval':
                    losses[attr] = torch.tensor(0)

                loss = loss + losses[attr]

        loss_value = loss.item()
        # finetune the discrimators
        if mode == 'train':
            optimizer.zero_grad()
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.module.get_optimizer_params())

        with torch.cuda.amp.autocast(enabled=(args.deterministic == False and mode == 'train')):
            anonymized_x = anonymized_x.detach()
            outsp, _ = model(anonymized_x, True)

            loss_dis = 0.0
            losses_dis = {}
            for attr, criter in criterion.items():

                if criter != 'infonce':
                    kld = nn.CrossEntropyLoss(reduction="mean")(
                        input=outsp[attr], target=targets[attr])

                    losses_dis[attr] = args.loss_lambda_dis * kld
                    loss_dis = loss_dis + losses_dis[attr]
                    acc1, acc5 = accuracy(
                        outsp[attr], targets[attr], topk=(1, 5))
                    losses_dis[attr + '_acc1'] = acc1
                    losses_dis[attr + '_acc5'] = acc5

        loss_value_dis = loss_dis.item()

        if mode == 'train':
            optimizer_dis.zero_grad()
            loss_scaler_dis(loss_dis, optimizer_dis, clip_grad=max_norm,
                            parameters=model.module.discriminators_xp.parameters())

        # del outsp, acc1, acc5

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_dis=loss_value_dis)

        for attr, value in losses.items():
            metric_logger.meters[attr].update(value.item(), n=batch_size)
        for attr, value in losses_dis.items():
            metric_logger.meters[f'{attr}_dis'].update(
                value.item(), n=batch_size)
        if mode == 'train':
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args, criterion, epoch, mutual_information_attr_x_dict, prior_attr_dict):
    return train_one_epoch(model, criterion, data_loader, None, None, device, epoch, None, None, mutual_information_attr_x_dict=mutual_information_attr_x_dict, prior_attr_dict=prior_attr_dict, mode='test', args=args)


def final_eval_one_epoch(model: torch.nn.Module, discriminators_new, criterion: nn.ModuleDict,
                         data_loader: Iterable, optimizer: torch.optim.Optimizer, optimizer_dis: torch.optim.Optimizer,
                         device: torch.device, epoch: int, loss_scaler, loss_scaler_dis, max_norm: float = 0,
                         model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, mutual_information_attr_x_dict=None, prior_attr_dict=None, mode='train', args=None):
    # final epoch test
    # results would be put into metric logger
    model.module.eval_and_freeze()
    if mode == 'train':
        discriminators_new.train()
    else:
        discriminators_new.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    header = mode + ' Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = {attr: target.to(device, non_blocking=True)
                   for attr, target in targets.items() if attr != 'file_name'}

        batch_size = samples.size(0)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        # with torch.no_grad():

        with torch.cuda.amp.autocast(enabled=(args.deterministic == False and mode == 'train')):

            # 0: anomyized_x, outs: {id: logits, age: logits, gender: logits}, outs_feat: {id: features, age: features, gender: features}
            anonymized_x = model.module.anonymize(samples)
            anonymized_x = anonymized_x.detach()
            outsp = {}
            for attr, net in discriminators_new.items():
                outsp[attr], _ = net(anonymized_x, feature=True)

            loss_dis = 0.0
            losses_dis = {}
            for attr, criter in criterion.items():

                if criter != 'infonce':
                    kld = nn.CrossEntropyLoss(reduction="mean")(
                        input=outsp[attr], target=targets[attr])

                    losses_dis[attr] = args.loss_lambda_dis * kld
                    loss_dis = loss_dis + losses_dis[attr]
                    acc1, acc5 = accuracy(
                        outsp[attr], targets[attr], topk=(1, 5))
                    losses_dis[attr + '_acc1'] = acc1
                    losses_dis[attr + '_acc5'] = acc5        

        loss_value_dis = loss_dis.item()

        if mode == 'train':
            optimizer_dis.zero_grad()
            loss_scaler_dis(loss_dis, optimizer_dis, clip_grad=max_norm,
                            parameters=discriminators_new.parameters())

        # del outsp, acc1, acc5

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss_dis=loss_value_dis)

        for attr, value in losses_dis.items():
            metric_logger.meters[f'{attr}_dis'].update(
                value.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

