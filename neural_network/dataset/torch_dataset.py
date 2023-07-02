import pickle
import os

import torch
from torch.utils.data import Dataset

DATA_PATH_TRAIN = "./training_dataset/training_data.p"
DATA_PATH_VAL = "./training_dataset/val_data.p"
device = "cuda" if torch.cuda.is_available() else "cpu"


class CompetitionDataset(Dataset):

    def __init__(self, mode='train'):

        if mode == "train":
            self.input_data_x, self.input_data_y = pickle.load(open(DATA_PATH_TRAIN, "rb"))
        elif mode == "val":
            self.input_data_x, self.input_data_y = pickle.load(open(DATA_PATH_VAL, "rb"))
        else:
            raise ValueError(f"mode {mode} is not valid.")

        print("DATASET SHAPE", self.input_data_x.shape)

        self.filenames = dict()
        self.filenames['x'] = self.input_data_x
        self.filenames['y'] = self.input_data_y

    def __getitem__(self, index):
        batch = dict()
        batch['x'] = torch.from_numpy(self.filenames['x'][index])
        batch['y'] = torch.from_numpy(self.filenames['y'][index])

        return batch

    def __len__(self):
        return len(self.filenames['x'])