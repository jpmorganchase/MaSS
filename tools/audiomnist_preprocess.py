# SPDX-License-Identifier: Apache-2.0
# Copyright : JP Morgan Chase & Co

import glob
import os
import argparse
import pandas as pd

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F


def get_args_parser():
    parser = argparse.ArgumentParser('Preprocess AudioMNIST from raw audio with HuBERT model')
    parser.add_argument('--audio_path', type=str, metavar='PATH', help='path point to the root of AudioMNIST git repo')
    # parser.add_argument('--metadata_location', default='./data/audiomnist/', type=str, metavar='PATH', help='path point to metadata of AudioMNIST')
    parser.add_argument('--max_len', default=30000, type=int, metavar='PATH', help='max length for an audio signal')
    parser.add_argument('--device', default='cuda', type=str, choices=['cuda', 'cpu'], help='use cpu or cuda')
    
    return parser


@torch.no_grad()
def main(args):
    # audio_path, metadata_location, output_name='audiomnist_train.pkl', length=30000

    device = torch.device(args.device)
    tmp = glob.glob(args.audio_path + 'data/**/*.wav', recursive=True)
    audio_paths = {os.path.basename(x):x for x in tmp}

    bundle = torchaudio.pipelines.HUBERT_BASE
    hubert = bundle.get_model()
    av_pool = nn.AdaptiveAvgPool2d((1, 768))
    hubert.eval()
    hubert.to(device)

    for split in ['train', 'val']:
        output_file = f'./data/audiomnist/audiomnist_{split}_fp16.pkl'
        metadata = f'./data/audiomnist/df_{split}_audio_mnist_data_to_split_digit_balanced.csv'
        df = pd.read_csv(metadata)
        cache = {}

        for idx in range(len(df)):
            df_data = df.iloc[[idx]]
            f_name = df_data["file_name"].values[0]
        
            waveform, _ = torchaudio.load(audio_paths[f_name])
            waveform = waveform[0]
            if waveform.shape[0] > args.max_len:
                waveform=waveform[:args.max_len]
            waveform = torch.nn.functional.pad(waveform, pad=(0, args.max_len - waveform.numel()), mode='constant', value=0).unsqueeze_(0).to(device)
            with torch.inference_mode():# model to eval mode
                samples = hubert.extract_features(waveform)[0][-1]
                with torch.cuda.amp.autocast():
                    samples = av_pool(samples)
                    samples = F.normalize(samples, p=2, dim=-1)
                    samples = samples.view(-1)
            cache[idx] = samples.float().cpu()
            cache[f_name] = samples.float().cpu()
        torch.save(cache, output_file)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
