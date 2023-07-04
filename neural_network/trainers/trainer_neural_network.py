import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from neural_network.dataset.torch_dataset import CompetitionDataset, CustomSampler
from neural_network.models.feed_forward_model import FeedForward
from neural_network.models.transformer_model import TransformerEncoder
from neural_network.trainers.trainer import Trainer
from neural_network.metrics.spearman_metric import SpearmanCorrCoefMetric, differentiable_spearman
from torchmetrics import PearsonCorrCoef, SpearmanCorrCoef

device = "cuda" if torch.cuda.is_available() else "cpu"


class NNTrainer(Trainer):
    def create_data(self):

        self.train_dataset = CompetitionDataset(mode='train')
        self.val_dataset = CompetitionDataset(mode='val')

        custom_indices_train = np.unique(self.train_dataset.input_data_x["date"])
        custom_indices_val = np.unique(self.val_dataset.input_data_x["date"])
        sampler_train = CustomSampler(custom_indices_train)
        sampler_val = CustomSampler(custom_indices_val)

        self.train_dataloader = DataLoader(self.train_dataset, sampler=sampler_train, batch_size=1)
        self.val_dataloader = DataLoader(self.val_dataset, sampler=sampler_val, batch_size=1)

    def create_model(self):
        if self.config.model_str == "mlp":
            self.model = FeedForward(self.train_dataset.input_data_x.shape[1] - 2, self.config.hidden_size, 1)
        elif self.config.model_str == "transformer":

            self.model = TransformerEncoder(self.train_dataset.input_data_x.shape[1] - 2, 1, self.config.num_layers,
                                            self.config.hidden_size,
                                            self.config.num_heads,
                                            self.config.dropout)
        else:
            raise ValueError(f"model type {self.config.model_str} has no associated model")

    def create_loss(self):
        # self.loss_fn = nn.MSELoss()
        self.loss_fn = PearsonCorrCoef().to(device)
        # self.loss_fn = differentiable_spearman

    def create_optimiser(self):
        parameters_with_grad = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimiser = Adam(parameters_with_grad,
                              self.config.learning_rate,
                              weight_decay=self.config.weight_decay)

    def create_metrics(self):
        self.metric = SpearmanCorrCoefMetric()

    def forward_model(self, batch):
        return self.model(batch['x'])

    def forward_loss(self, batch, output):
        loss = self.loss_fn(torch.squeeze(output), torch.squeeze(batch['y']))
        return loss
