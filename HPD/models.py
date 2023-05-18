'''ml models defination'''

import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, d_model=21, nhead=3, num_layers=2):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model*2, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(d_model*2, d_model)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.transformer_encoder(x)
        x = self.linear(x)
        return x