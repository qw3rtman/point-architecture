import math
from itertools import chain

import torch
import torch.nn as nn
import numpy as np

from .const import HEIGHT, WIDTH, C
from .util import SpatialSoftmax

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class AttentivePolicy(nn.Module):
    def __init__(self, steps=5, temperature=1.0, hidden_size=512, nhead=8, num_layers=2, **kwargs):
        super().__init__()

        h = hidden_size//2

        self.class_embedding = nn.Parameter(torch.empty(C, h))
        stdv = 1./math.sqrt(h)
        self.class_embedding.data.uniform_(-stdv, stdv)

        self.positional_encoding = nn.Sequential(
            nn.Linear(2, h//2),
            nn.ReLU(True),
            nn.Linear(h//2, h)
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.deconv = nn.Sequential(*chain(*[
            (
                nn.BatchNorm2d(dim),
                nn.ConvTranspose2d(dim, dim//2, 3, 2, 1, 1),
                nn.GELU() # NOTE
            ) for dim in hidden_size//(2**np.arange(3))
        ]))

        self.extract = nn.Sequential(
            nn.BatchNorm2d(hidden_size//8),
            nn.Conv2d(hidden_size//8, steps, 1, 1, 0),
            SpatialSoftmax(temperature)
        )

        """ TODO: action conditioned
        self.extract = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(hidden_size//8),
                nn.Conv2d(hidden_size//8, steps, 1, 1, 0),
                SpatialSoftmax(temperature)
            ) for i in range(3)
        ])
        """

    def forward(self, M):
        """
        M : map, (N x 3)
        """
        return self.extract(self.deconv(self.transformer(torch.cat([
            self.positional_encoding(M[..., :2]),
            self.class_embedding[M[..., 2].long()]
        ], dim=-1).permute(1,0,2))[:1].permute(1,2,0).unsqueeze(-1)))
