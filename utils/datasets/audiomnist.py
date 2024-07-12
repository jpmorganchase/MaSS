# SPDX-License-Identifier: Apache-2.0
# Copyright : JP Morgan Chase & Co


from torch.utils.data import Dataset
import torch
import pandas as pd

class AudioMNIST(Dataset):
    def __init__(self, data_path, metadata_location):
        self.df = pd.read_csv(metadata_location)

        self.num_classes = {
            k: len(pd.unique(self.df[k])) for k in self.df.keys()
        }
        self.num_classes['id'] = 60
        self.data = torch.load(data_path, map_location='cpu')

    def __len__(self):
        return len(self.df.index)
    
    def __getitem__(self, idx):

        df_data = self.df.iloc[[idx]]
        f_name = df_data["file_name"].values[0]
        f_id = str(f_name[2:4])
        digit = int(f_name.split("_")[0])
        waveform = self.data.get(f_name, None) # was the above one
        
        out = {
            "file_name": f_name,
            "id": int(f_id) - 1,
            "age": df_data["age"].values[0],
            "gender": df_data["gender"].values[0],
            "digit": digit,
            "accent": df_data["accent"].values[0],
        }

        return waveform, out


if __name__ == "__main__":

    ds = AudioMNIST('./data/audiomnist/audiomnist_train_fp16.pkl', metadata_location='./data/audiomnist/df_train_audio_mnist_data_to_split_digit_balanced.csv')

    print(len(ds))
    for d in ds:
        print(d[0].shape, d[1])
        exit(0)
