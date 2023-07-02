import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from neural_network.dataset.torch_dataset import CompetitionDataset
from neural_network.models.model import FeedForward
from neural_network.trainers.trainer import Trainer
from neural_network.metrics.reconstruction import ReconstructionMetric


class NNTrainer(Trainer):
    def create_data(self):
        def variable_size_collate_fn(batch):
            return batch

        self.train_dataset = CompetitionDataset(mode='train')
        self.val_dataset = CompetitionDataset(mode='val')

        self.train_dataloader = DataLoader(self.train_dataset, self.config.batch_size, shuffle=True,
                                           collate_fn=variable_size_collate_fn)
        self.val_dataloader = DataLoader(self.val_dataset, self.config.batch_size, shuffle=False,
                                         collate_fn=variable_size_collate_fn)

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
