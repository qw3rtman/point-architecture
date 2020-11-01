import torch
import torch.nn as nn
import numpy as np

from .const import HEIGHT, WIDTH
from .util import SpatialSoftmax

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class AttentivePolicy(nn.Module):
    def __init__(self, steps=5, temperature=1.0, hidden_size=512, nhead=8, num_layers=2, **kwargs):
        super().__init__()

        self.positional_fc = nn.Sequential(
            nn.Linear(2, (hidden_size-7)//2),
            nn.ReLU(True),
            nn.Linear((hidden_size-7)//2, hidden_size-7)
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.deconv = nn.Sequential(
            nn.BatchNorm2d(512), # 2048 for resnet50
            nn.ConvTranspose2d(512, 256, 3, 2, 1, 1), # 2048 for resnet50
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.ReLU(True))

        self.extract = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(64),
                nn.Conv2d(64,steps,1,1,0),
                SpatialSoftmax(temperature)
            ) for i in range(4)
        ])

    def forward(self, M):
        """
        M : map, (N x 9)
        """
        M = torch.cat([self.positional_fc(M[..., :2]), M[..., 2:]], dim=-1)
        return self.transformer(M).mean(dim=1)
