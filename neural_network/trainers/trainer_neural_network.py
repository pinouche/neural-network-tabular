import numpy as np

import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from neural_network.dataset.torch_dataset import CompetitionDataset, CustomSampler
from neural_network.models.model import FeedForward
from neural_network.trainers.trainer import Trainer
from neural_network.metrics.reconstruction import ReconstructionMetric


# def collate_fn(batch):
# #     print(batch, len(batch))
# #     #batch = batch.groupby("date").apply(lambda x: x)
# #
# #     print("BATCH IN COLLATE FUNCTION", batch)
# #     return batch


class NNTrainer(Trainer):
    def create_data(self):

        self.train_dataset = CompetitionDataset(mode='train')
        self.val_dataset = CompetitionDataset(mode='val')

        custom_indices_train = np.unique(self.train_dataset.input_data_x["date"])
        custom_indices_val = np.unique(self.val_dataset.input_data_x["date"])
        sampler_train = CustomSampler(custom_indices_train)
        sampler_val = CustomSampler(custom_indices_val)

        print("WE ARE HERE")
        print(self.val_dataset.input_data_x.shape)
        self.train_dataloader = DataLoader(self.train_dataset, sampler=sampler_train, batch_size=1)
        self.val_dataloader = DataLoader(self.val_dataset, sampler=sampler_val, batch_size=1)

    def create_model(self):
        self.model = FeedForward(self.train_dataset.input_data_x.shape[1], self.config.hidden_size, 1)

    def create_loss(self):
        self.loss_fn = nn.MSELoss()

    def create_optimiser(self):
        parameters_with_grad = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimiser = Adam(parameters_with_grad, self.config.learning_rate, weight_decay=self.config.weight_decay)

    def create_metrics(self):
        self.train_metrics = ReconstructionMetric('train')
        self.val_metrics = ReconstructionMetric('val')

    def forward_model(self, batch):
        return self.model(batch['x'])

    def forward_loss(self, batch, output):
        return self.loss_fn(output, batch['y'])
