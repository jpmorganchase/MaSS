# SPDX-License-Identifier: Apache-2.0
# Copyright : JP Morgan Chase & Co

import os

from .audiomnist import AudioMNIST
from .motionsense import MotionSense

def build_dataset(is_train, args):

    if args.dataset == 'audiomnist':
        data_pickle = os.path.join(args.data_dir, f'audiomnist_{"train" if is_train else "val"}_fp16.pkl')
        metadata = os.path.join(args.data_dir, f'df_{"train" if is_train else "val"}_audio_mnist_data_to_split_digit_balanced.csv')
        dataset = AudioMNIST(data_pickle, metadata)
    elif args.dataset == 'motionsense':
        data_path = os.path.join(args.data_dir, 'motionsense_train.pkl' if is_train else 'motionsense_val.pkl')
        dataset = MotionSense(data_path)
    else:
        print(f"Unknown dataset, {args.dataset}")
        exit(0)
    
    return dataset
