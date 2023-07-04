import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(self, input_size, output_size, num_layers, hidden_size, num_heads, dropout):
        super(TransformerEncoder, self).__init__()

        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dim_feedforward=hidden_size, dropout=dropout),
            num_layers=num_layers
        )

        self.relu = nn.ReLU()
        self.fc = nn.Linear(input_size*2, input_size)
        self.fc2 = nn.Linear(input_size, output_size)

    def forward(self, src):
        output = self.transformer(src)
        output = torch.cat((output, src), dim=1)
        output = self.fc(output)
        output = self.relu(output)
        output = self.fc2(output)
        return output  # Squeeze the output to a single number
