# SPDX-License-Identifier: Apache-2.0
# Copyright : JP Morgan Chase & Co


import torch
from torch.utils.data import Dataset


class MotionSense(Dataset):
    def __init__(self, data_path):

        self.num_classes = {'gender': 2, 'id': 24, 'act': 4}
        self.data = torch.load(data_path, map_location='cpu')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
    ds = MotionSense('./data/motionsense/motionsense_train.pkl')
    print(len(ds))
    for d in ds:
        print(d[0].shape, d[1])
        exit(0)
