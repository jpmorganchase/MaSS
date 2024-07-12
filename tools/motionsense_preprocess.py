# SPDX-License-Identifier: Apache-2.0
# Copyright : JP Morgan Chase & Co

import pickle
import glob
import os
import warnings
import itertools
import argparse

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


def get_args_parser():
    parser = argparse.ArgumentParser('Preprocess Motion Sense data for MaSS experiment (following the preprocess in the original paper)')
    parser.add_argument('--data_path', type=str, metavar='PATH', help='path point to the root of git repo of motion sense')
    parser.add_argument('--output_path', type=str, metavar='PATH', help='path to store the preprocessed results')
    return parser

def get_ds_infos(data_path):
    """
    Read the file includes data subject information.

    Data Columns:
    0: code [1-24]
    1: weight [kg]
    2: height [cm]
    3: age [years]
    4: gender [0:Female, 1:Male]

    Returns:
        A pandas DataFrame that contains inforamtion about data subjects' attributes 
    """

    dss = pd.read_csv(data_path + "/data/data_subjects_info.csv")
    print("[INFO] -- Data subjects' information is imported.")

    return dss


def set_data_types(data_types=["userAcceleration"]):
    """
    Select the sensors and the mode to shape the final dataset.

    Args:
        data_types: A list of sensor data type from this list: [attitude, gravity, rotationRate, userAcceleration] 

    Returns:
        It returns a list of columns to use for creating time-series from files.
    """
    dt_list = []
    for t in data_types:
        if t != "attitude":
            dt_list.append([t+".x", t+".y", t+".z"])
        else:
            dt_list.append([t+".roll", t+".pitch", t+".yaw"])

    return dt_list


def creat_time_series(data_path, dt_list, act_labels, trial_codes, mode="mag", labeled=True):
    """
    Args:
        dt_list: A list of columns that shows the type of data we want.
        act_labels: list of activites
        trial_codes: list of trials
        mode: It can be "raw" which means you want raw data
        for every dimention of each data type,
        [attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)].
        or it can be "mag" which means you only want the magnitude for each data type: (x^2+y^2+z^2)^(1/2)
        labeled: True, if we want a labeld dataset. False, if we only want sensor values.

    Returns:
        It returns a time-series of sensor data.

    """
    num_data_cols = len(dt_list) if mode == "mag" else len(dt_list*3)

    if labeled:
        # "7" --> [act, code, weight, height, age, gender, trial]
        dataset = np.zeros((0, num_data_cols+7))
    else:
        dataset = np.zeros((0, num_data_cols))

    ds_list = get_ds_infos(args.data_path)

    print("[INFO] -- Creating Time-Series")
    for sub_id in ds_list["code"]:
        for act_id, act in enumerate(act_labels):
            for trial in trial_codes[act_id]:
                fname = data_path + '/data/A_DeviceMotion_data/'+act+'_' + \
                    str(trial)+'/sub_'+str(int(sub_id))+'.csv'
                raw_data = pd.read_csv(fname)
                raw_data = raw_data.drop(['Unnamed: 0'], axis=1)
                vals = np.zeros((len(raw_data), num_data_cols))
                for x_id, axes in enumerate(dt_list):
                    if mode == "mag":
                        vals[:, x_id] = (raw_data[axes]**2).sum(axis=1)**0.5
                    else:
                        vals[:, x_id*3:(x_id+1)*3] = raw_data[axes].values
                    vals = vals[:, :num_data_cols]
                if labeled:
                    lbls = np.array([[act_id,
                                      sub_id-1,
                                      ds_list["weight"][sub_id-1],
                                      ds_list["height"][sub_id-1],
                                      ds_list["age"][sub_id-1],
                                      ds_list["gender"][sub_id-1],
                                      trial
                                      ]]*len(raw_data))
                    vals = np.concatenate((vals, lbls), axis=1)
                dataset = np.append(dataset, vals, axis=0)
    cols = []
    for axes in dt_list:
        if mode == "raw":
            cols += axes
        else:
            cols += [str(axes[0][:-2])]

    if labeled:
        cols += ["act", "id", "weight", "height", "age", "gender", "trial"]

    dataset = pd.DataFrame(data=dataset, columns=cols)
    return dataset
# ________________________________


class DataFrameSplitter:
    def __init__(self, method="ratio"):
        self.method = method

    def train_test_split(self, dataset, labels, verbose=0, **options):
        if self.method == "trials":
            train_trials = options.get('train_trials', None)
            trial_col = options.get('trial_col', None)
        elif self.method == "ratio":
            train_ratio = options.get('train_ratio', None)
        else:
            raise ValueError(
                "You must define the method of splitting: 'trials' or 'ratio'")

        columns = dataset.columns
        train_data = pd.DataFrame(columns=columns)
        test_data = pd.DataFrame(columns=columns)

        label_values = list()
        for label in labels:
            unique_vals = sorted(dataset[label].unique())
            label_values.append(unique_vals)
        combs_of_label_values = list(itertools.product(*label_values))

        for i, comb in enumerate(combs_of_label_values):
            seg_data = dataset.copy()
            for j, label in enumerate(labels):
                seg_data = seg_data[seg_data[label] == comb[j]]
            seg_data.reset_index(drop=True, inplace=True)

            if seg_data.shape[0] > 0:
                if self.method == "trials":
                    if seg_data[trial_col][0] in train_trials:
                        train_data = train_data.append(seg_data)
                    else:
                        test_data = test_data.append(seg_data)
                elif self.method == "ratio":
                    split_index = int(seg_data.shape[0] * train_ratio)
                    train_data = train_data.append(seg_data[:split_index])
                    test_data = test_data.append(seg_data[split_index:])

            if verbose > 2:
                print("Seg_Shape:{} | TrainData:{} | TestData:{} | {}:{} | progress:{}%.".format(
                    seg_data.shape, train_data.shape, test_data.shape, labels, comb,
                    round((i / len(combs_of_label_values)) * 100)))
            elif verbose > 1:
                print("Seg_Shape:{} | TrainData:{} | TestData:{} | {}:{} | progress:{}%.".format(
                    seg_data.shape, train_data.shape, test_data.shape, labels, comb,
                    round((i / len(combs_of_label_values)) * 100)), end="\r")
            elif verbose > 0:
                print("progress:{}%.".format(
                    round((i / len(combs_of_label_values)) * 100)), end="\r")

        assert dataset.shape[0] == train_data.shape[0] + test_data.shape[0]
        assert dataset.shape[1] == train_data.shape[1] == test_data.shape[1]

        return train_data, test_data


def time_series_to_section(dataset, sliding_window_size, step_size_of_sliding_window, standardize=False, **options):
    data = dataset[:, 0:2]
    act_labels = dataset[:, 2]
    id_labels = dataset[:, 3]
    gen_labels = dataset[:, 7]
    mean = 0
    std = 1

    if standardize:
        # Standardize each sensor’s data to have a zero mean and unity standard deviation.
        # As usual, we normalize test dataset by training dataset's parameters
        if options:
            mean = options.get("mean")
            std = options.get("std")
            print("----> Test Data has been standardized")
        else:
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            print("----> Training Data has been standardized:\n the mean is = ",
                  str(mean.mean()), " ; and the std is = ", str(std.mean()))

        data -= mean
        data /= std
    else:
        print("----> Without Standardization.....")

    # We want the Rows of matrices show each Feature and the Columns show time points.
    data = data.T

    size_features = data.shape[0]
    size_data = data.shape[1]
    number_of_secs = round(
        ((size_data - sliding_window_size)/step_size_of_sliding_window))

    # Create a 3D matrix for Storing Snapshots
    secs_data = np.zeros((number_of_secs, size_features, sliding_window_size))
    id_secs_labels = np.zeros(number_of_secs)
    act_secs_labels = np.zeros(number_of_secs)
    gen_secs_labels = np.zeros(number_of_secs)

    k = 0
    for i in range(0, (size_data)-sliding_window_size, step_size_of_sliding_window):
        j = i // step_size_of_sliding_window
        if (j >= number_of_secs):
            break
        if (gen_labels[i] != gen_labels[i+sliding_window_size-1]):
            continue
        if (not (act_labels[i] == act_labels[i+sliding_window_size-1]).all()):
            continue
        if (id_labels[i] != id_labels[i+sliding_window_size-1]):
            continue
        secs_data[k] = data[0:size_features, i:i+sliding_window_size]
        act_secs_labels[k] = act_labels[i].astype(int)
        gen_secs_labels[k] = gen_labels[i].astype(int)
        id_secs_labels[k] = id_labels[i].astype(int)
        k = k+1
    secs_data = secs_data[0:k]
    act_secs_labels = act_secs_labels[0:k]
    gen_secs_labels = gen_secs_labels[0:k]
    id_secs_labels = id_secs_labels[0:k]

    # convert to our format here
    res = []
    for i in range(k):
        sample = secs_data[i]
        sample = torch.tensor(sample).float()
        res.append((sample, {
                   'gender': int(gen_secs_labels[i]), 'id': int(id_secs_labels[i]), 'act': int(act_secs_labels[i])}))

    return res, mean, std
    # ________________________________________________________________


if __name__ == '__main__':
    args = get_args_parser().parse_args()

    ACT_LABELS = ["dws", "ups", "wlk", "jog", "std", "sit"]
    TRIAL_CODES = {
        ACT_LABELS[0]: [1, 2, 11],
        ACT_LABELS[1]: [3, 4, 12],
        ACT_LABELS[2]: [7, 8, 15],
        ACT_LABELS[3]: [9, 16],
        ACT_LABELS[4]: [6, 14],
        ACT_LABELS[5]: [5, 13]
    }
    # Here we set parameter to build labeld time-series from dataset of "(A)DeviceMotion_data"
    # attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)
    sdt = ["rotationRate", "userAcceleration"]
    print("[INFO] -- Selected sensor data types: "+str(sdt))
    act_labels = ACT_LABELS[0:4]
    print("[INFO] -- Selected activites: "+str(act_labels))
    trial_codes = [TRIAL_CODES[act] for act in act_labels]
    dt_list = set_data_types(sdt)
    dataset = creat_time_series(args.data_path, 
        dt_list, act_labels, trial_codes, mode="mag", labeled=True)
    print("[INFO] -- Shape of time-Series dataset:"+str(dataset.shape))
    print(dataset.head())
    print(sorted(dataset['trial'].unique()))

    dfs = DataFrameSplitter(method="trials")

    train_data, test_data = dfs.train_test_split(dataset=dataset,
                                                 labels=("id", "trial"),
                                                 trial_col='trial',
                                                 train_trials=[
                                                     1., 2., 3., 4., 5., 6., 7., 8., 9.],
                                                 verbose=2)
    print(train_data.shape, test_data.shape)
    print(train_data.head(), test_data.head())

    # This Variable Defines the Size of Sliding Window
    # ( e.g. 100 means in each snapshot we just consider 100 consecutive observations of each sensor)
    # 50 Equals to 1 second for MotionSense Dataset (it is on 50Hz samplig rate)
    sliding_window_size = 128
    # Here We Choose Step Size for Building Diffrent Snapshots from Time-Series Data
    # ( smaller step size will increase the amount of the instances and higher computational cost may be incurred )
    step_size_of_sliding_window = 10
    num_features = 2
    print("--> Sectioning Training and Test datasets: shape of each section will be: (",
          num_features, "x", sliding_window_size, ")")

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    train_data_sec, train_mean, train_std = time_series_to_section(train_data.to_numpy(),
                                                                   sliding_window_size,
                                                                   step_size_of_sliding_window,
                                                                   standardize=True)
    torch.save(train_data_sec, args.output_path + '/motionsense_train.pkl')
    test_data_sec, test_mean, test_std = time_series_to_section(test_data.to_numpy(),
                                                                sliding_window_size,
                                                                step_size_of_sliding_window,
                                                                standardize=True,
                                                                mean=train_mean,
                                                                std=train_std)
    torch.save(test_data_sec, args.output_path + '/motionsense_val.pkl')

    print("--> Shape of Training Sections:", len(train_data_sec),
          train_data_sec[0][0].shape, train_data_sec[0][1])
    print("--> Shape of Test Sections:", len(test_data_sec),
          test_data_sec[0][0].shape, test_data_sec[0][1])
