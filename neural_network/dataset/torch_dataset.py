import random
import torch
from torch.utils.data import Dataset, Sampler

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

    def __init__(self, data_x, data_y):
        self.input_data_x = data_x
        self.input_data_y = data_y

        self.filenames = dict()
        self.filenames['x'] = self.input_data_x
        self.filenames['y'] = self.input_data_y

    def __getitem__(self, index):
        batch = dict()

        subset_x = self.filenames['x'][self.filenames['x']["date"] == index]
        subset_y = self.filenames['y'][self.filenames['y']["date"] == index]

        subset_x = subset_x.drop(columns=['id', 'date'], inplace=False)
        subset_y = subset_y.drop(columns=['id', 'date'], inplace=False)

        batch['x'] = torch.from_numpy(subset_x.values)
        batch['y'] = torch.from_numpy(subset_y["y"].values)

        return batch

    def __len__(self):
        return len(self.filenames['x'])
