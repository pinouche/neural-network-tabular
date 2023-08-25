import torch.nn as nn
import torch


class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, drop_out_rate):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        # non-parametrized function
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_out_rate)

    def forward(self, x, feature_mask):
        x = x * feature_mask
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.fc3(x)

        return x
