import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(self, input_size, output_size, num_layers, hidden_size, num_heads, dropout):
        super(TransformerEncoder, self).__init__()

        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dim_feedforward=hidden_size, dropout=dropout),
            num_layers=num_layers
        )

        self.fc1 = nn.Linear(input_size*2, input_size)
        self.fc2 = nn.Linear(input_size, output_size, bias=False)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        output = self.transformer(src)
        # output = torch.unsqueeze(output, dim=0).repeat(src.shape[0], 1)
        output = torch.cat((output, src), dim=1)
        output = self.dropout(output)
        output = self.fc1(output)
        output = self.dropout(output)
        output = self.relu(output)
        output = self.fc2(output)

        return output  # Squeeze the output to a single number
