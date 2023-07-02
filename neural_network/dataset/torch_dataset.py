import random
import pickle
import os

import torch
from torch.utils.data import Dataset, Sampler

DATA_PATH_TRAIN = "./training_dataset/training_data.p"
DATA_PATH_VAL = "./training_dataset/val_data.p"
device = "cuda" if torch.cuda.is_available() else "cpu"


class CustomSampler(Sampler):
    def __init__(self, custom_indices):
        self.custom_indices = custom_indices

    def __iter__(self):
        random.shuffle(self.custom_indices)
        return iter(self.custom_indices)

    def __len__(self):
        return len(self.custom_indices)


class CompetitionDataset(Dataset):

    def __init__(self, mode='train'):

        if mode == "train":
            self.input_data_x, self.input_data_y = pickle.load(open(DATA_PATH_TRAIN, "rb"))
        elif mode == "val":
            self.input_data_x, self.input_data_y = pickle.load(open(DATA_PATH_VAL, "rb"))
        else:
            raise ValueError(f"mode {mode} is not valid.")

        self.filenames = dict()
        self.filenames['x'] = self.input_data_x
        self.filenames['y'] = self.input_data_y

    def __getitem__(self, index):
        batch = dict()

        print("SHAPE 1", self.input_data_x.shape)

        features_columns = [col for col in self.filenames['x'].columns if col not in ["date", "id"]]
        subset_x = self.filenames['x'][self.filenames['x']["date"] == index][features_columns]
        subset_y = self.filenames['y'][self.filenames['y']["date"] == index]

        print("SHAPE", subset_x.shape, subset_y.shape)

        batch['x'] = torch.from_numpy(subset_x.values)
        batch['y'] = torch.from_numpy(subset_y["y"].values)

        return batch

    def __len__(self):
        return len(self.filenames['x'])
