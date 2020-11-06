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
    def __init__(self, steps=5, hidden_size=512, nhead=8, num_layers=2, **kwargs):
        super().__init__()

        self.steps = steps
        h = hidden_size//2

        self.class_embedding = nn.Parameter(torch.empty(C+1, h))
        stdv = 1./math.sqrt(h)
        self.class_embedding.data.uniform_(-stdv, stdv)

        self.positional_encoding = nn.Sequential(
            nn.Linear(2, h//2),
            nn.GELU(),
            nn.Linear(h//2, h)
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # TODO: action conditioned
        self.extract = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.GELU(),
            nn.Linear(hidden_size//2, steps*2)
        )

    def forward(self, M_pad, M_mask):
        """
        M : map, (N x 3)
        """

        pos = self.positional_encoding(M_pad[..., :2])
        c = self.class_embedding[M_pad[..., 2].long()]

        x = torch.cat([pos, c], dim=-1)
        attn = self.transformer(x.permute(1,0,2), src_key_padding_mask=M_mask)[0]

        waypoints = self.extract(attn).reshape(-1, self.steps, 2)

        return waypoints


if __name__ == '__main__':
    net = AttentivePolicy()
    M_pad = torch.rand((4,250,3))
    M_mask = torch.ones((4,250), dtype=torch.float)

    net(M_pad, M_mask)
