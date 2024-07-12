# SPDX-License-Identifier: Apache-2.0
# Copyright : JP Morgan Chase & Co
 
 import argparse
import utils.misc as utils

def get_args_parser():
    parser = argparse.ArgumentParser(
        'Pretraining VAE with multiple constraints', add_help=False)

    # General

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.001,
                        help='weight decay (default: 0.05)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.0,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Misc.
    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--eval_sensitive', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true',
                        default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument(
        '--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=False)
    parser.add_argument('--model-ema-decay', type=float,
                        default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu',
                        action='store_true', default=False, help='')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # Dataset parameters
    parser.add_argument('--data_dir', default='./data/audiomnist/', type=str, metavar='DATADIR',
                        help='Path to dataset')
    parser.add_argument('--dataset', default='audiomnist',
                        type=str, metavar='STR', help='which dataset')
    parser.add_argument('--data_file', default=None, type=str, metavar='DATAFILE', help='Path to csv')

    # Model parameters
    parser.add_argument('--model', default='gan_trainer', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--gen_model', default='gen_mlp_4', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--gen_with_residual', action='store_true')
    parser.add_argument('--disc_model', default='disc_mlp', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--disc_infonce_model', default='disc_mlp', type=str, metavar='MODEL',
                        help='Name of model to train')


    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=50, type=int)


    parser.add_argument('--attribute_models', default=None,
                        nargs='+', type=str, help='model name of attribute models')
    parser.add_argument('--attribute_models_weights', default=None, nargs='*',
                        action=utils.KeyValue, help='path of pretrained models for attributes')
    parser.add_argument('--new_dis_attribute_models_weights', default=None, nargs='*',
                        action=utils.KeyValue, help='path of pretrained models for attributes')
    parser.add_argument('--eval-attributes', default=[],
                        nargs='*', type=str, help='evaluation only attributes')

    parser.add_argument('--attributes', default='Response', type=str, help='type of attributes')

    # n, m in the paper, using loss n tune instead of n.
    parser.add_argument('--loss-n', default={}, nargs='*', action=utils.KeyValue,
                        help='n in loss')
    parser.add_argument('--loss-m', default={}, nargs='*', action=utils.KeyValue,
                        help='m in loss')

    # can be removed as it will be enabled always
    parser.add_argument('--loss-lambda-u-2', type=float, default=1, metavar='M',
                        help='lambda U for quadratic penalty in the loss function')
    parser.add_argument('--loss-lambda-s-2', type=float, default=1, metavar='M',
                        help='lambda S for quadratic penaltyin the loss function')
    parser.add_argument('--loss-lambda-u-1', type=float, default=1, metavar='M',
                        help='lambda U for linear penaltyin the loss function')
    parser.add_argument('--loss-lambda-s-1', type=float, default=1, metavar='M',
                        help='lambda S for linear penaltyin the loss function')
    parser.add_argument('--loss-lambda-dis', type=float, default=1, metavar='M',
                        help='lambda discriminator in the loss function')
    parser.add_argument('--loss-lambda-infonce', type=float, default=1, metavar='M',
                        help='lambda infonce in the loss function')
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='Enable determinstic for cudnn and disable fp16')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')

    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    
    return parser
