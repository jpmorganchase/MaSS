import os
import sys
import argparse
import time
import math
import pickle

import torch.nn.functional as F
from torchvision import transforms, datasets
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch
from timm.models import create_model
import numpy as np

from utils.datasets import build_dataset
from utils.losses import InfoNCELoss
from utils.misc import MetricLogger, SmoothedValue
import models


def parse_option(): # TODO: cleanup options
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=20,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    parser.add_argument('--resume', type=str, default='',
                        help='resume checkpoint')

    # model dataset
    parser.add_argument('--model', type=str, default='audiomnist_infonce_mlp')
    parser.add_argument('--dataset', type=str, default='path', help='dataset')
    parser.add_argument('--data_dir', type=str,
                        default=None, help='path to custom dataset')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    # other setting
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument('--output_dir', type=str, default='./',
                        help='output folder for log and model')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='Enable determinstic for cudnn and disable fp16')


    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_dir is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_dir is None:
        opt.data_dir = './datasets/'
    # opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'InfoNCE_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    # if opt.batch_size > 256:
    #     opt.warm = True
    # if opt.warm:
    #     opt.model_name = '{}_warm'.format(opt.model_name)
    #     opt.warmup_from = 0.01
    #     opt.warm_epochs = 10
    #     eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
    #     opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
    #         1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
    
    # opt.save_folder = os.path.join(opt.output_dir, opt.model_name)
    opt.save_folder = opt.output_dir
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.log_path = os.path.join(opt.save_folder, 'log.txt')

    return opt


def set_loader(args):
    
    args.data_dir = args.data_dir
    if args.dataset == 'adience':
        args.input_size = 80
    train_dataset = build_dataset(is_train=True, args=args)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)
    return train_loader


def set_model(args):

    if args.dataset == 'adience': # TODO: maybe integrate here?
        in_channels = 3
        num_classes = 1111 # TODO
    elif args.dataset == 'audiomnist':
        in_channels = 768
        num_classes = 128
    elif args.dataset == 'motionsense':
        in_channels = 2
        num_classes = 128

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=num_classes,
        in_chans=in_channels
    )
    criterion = InfoNCELoss(temperature=args.temp)
    # if opt.syncBN:
    #     model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
        if args.deterministic:
            os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2"
            cudnn.deterministic = True
    
    return model, criterion

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    eta_min = lr * (args.lr_decay_rate ** 3)
    lr = eta_min + (lr - eta_min) * (
            1 + math.cos(math.pi * epoch / args.epochs)) / 2
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


# def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
#     if args.warm and epoch <= args.warm_epochs:
#         p = (batch_id + (epoch - 1) * total_batches) / \
#             (args.warm_epochs * total_batches)
#         lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr
#         return lr
#     else:
#         return None

def train(train_loader, model, criterion, optimizer, epoch, lr, opt):
    """one epoch training"""
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('Batch time', SmoothedValue(window_size=1, fmt='{value:.4f}'))
    # metric_logger.add_meter('Loss', SmoothedValue(window_size=1, fmt='{value:.4f}'))
    end = time.time()

    idx = 0
    for images, _ in metric_logger.log_every(train_loader, opt.print_freq, ""):

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
        bsz = images.shape[0]  # // 2
        
        # warm_lr = warmup_learning_rate(
        #     opt, epoch, idx, len(train_loader), optimizer)
        # lr = warm_lr if warm_lr is not None else lr

        # compute loss
        features = model(images)
        features = F.normalize(features, dim=1)
        f1 = features
        f2 = f1.clone()
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)    
        loss = criterion(features)
        
        # update metric
        # losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        metric_logger.update(batch_time=time.time() - end)
        metric_logger.update(loss=loss.item())
        # batch_time.update()
        # end = time.time()

        # print info
        # if (idx + 1) % opt.print_freq == 0:
        #     print('Train: [{0}][{1}/{2}]\t'
        #           'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'MT {model_time.val:.3f} ({model_time.avg:.3f})\t'
        #           'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'LR {lr:.3f}\t'
        #           'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
        #               epoch, idx + 1, len(train_loader), batch_time=batch_time,
        #               data_time=data_time, model_time=model_time, loss=losses, lr=lr))
        #     sys.stdout.flush()
        idx += 1
    # metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    # print('Train: [{0}][{1}/{2}]\t'
    #       'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
    #       'MT {model_time.val:.3f} ({model_time.avg:.3f})\t'
    #       'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
    #       'LR {lr:.3f}\t'
    #       'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
    #           epoch, idx + 1, len(train_loader), batch_time=batch_time,
    #           data_time=data_time, model_time=model_time, loss=losses, lr=lr))
    return metric_logger.meters['loss'].global_avg


def main():
    opt = parse_option()

    seed = opt.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    
    if opt.resume:
        ckpt = torch.load(opt.resume, map_location='cpu')
        print(model.load_state_dict(ckpt['model']))

    # training routine
    for epoch in range(1, opt.epochs + 1):
        lr = adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion,
                     optimizer, epoch, lr, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f} seconds'.format(epoch, time2 - time1))
        with open(opt.log_path, 'a') as f:
            print('epoch {}, loss {:.2f}, total time {:.2f} seconds'.format(
                epoch, loss, time2 - time1), file=f)
        
        state_dict = {
            'opt': opt,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
        save_file = os.path.join(opt.save_folder, 'checkpoint.pth')
        torch.save(state_dict, save_file)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'checkpoint_{epoch}.pth'.format(epoch=epoch))
            torch.save(state_dict, save_file)

    # save the last model
    save_file = os.path.join(opt.save_folder, 'last.pth')
    state_dict = {
            'opt': opt,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
    torch.save(state_dict, save_file)


if __name__ == '__main__':
    main()