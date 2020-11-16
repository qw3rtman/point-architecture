import math
from itertools import chain

import torch
import torch.nn as nn
import numpy as np

from .const import COLORS
from .util import _get_clones


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model//nhead

        stdv = 1./math.sqrt(d_model)

        # query, projection for each head, distinct to each layer
        self.W_q = nn.Parameter(torch.empty(self.nhead, self.d_model, self.d_head))
        self.W_q.data.uniform_(-stdv, stdv)

    def forward(self, ego, mask, keys, value):
        """
        ego:     (b, 1, d)
        keys:    (h, b, N-1, h_d)
        value:   (b, N-1, h_d)

        out:     (b, 1, d)
        """

        out = torch.empty(ego.shape[0], 1, self.d_model).to(self.W_q.device)
        for head in range(self.nhead):
            query = torch.einsum('bnd,dh->bnh', ego, self.W_q[head])

            x = torch.bmm(query, keys[head].permute(0,2,1))[:,0]
            x /= math.sqrt(self.d_head)
            x[~mask[:,1:]] = float('-inf') # batch-wise pad mask
            x = nn.functional.softmax(x, dim=-1).unsqueeze(dim=1)

            out[...,(head*self.d_head):((head+1)*self.d_head)] = torch.bmm(x, value)

        return out


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=512, nhead=8):
        super().__init__()

        self.d_model = d_model

        self.attention = MultiheadAttention(d_model, nhead)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2048),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(2048, d_model))

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, ego, mask, keys, value):
        """
        ego:     (b, 1, d)
        keys:    (h, b, N-1, h_d)
        value:   (b, N-1, h_d)

        out:     (b, 1, d)
        """

        x1 = self.attention(ego, mask, keys, value)
        ego = ego + self.dropout1(x1)

        ego = self.norm1(ego)
        x1 = self.mlp(ego)
        ego = ego + self.dropout2(x1)

        return self.norm2(ego)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers=8):
        super().__init__()

        self.d_model = encoder_layer.d_model
        self.nhead = encoder_layer.attention.nhead
        self.d_head = self.d_model//self.nhead
        stdv = 1./math.sqrt(self.d_model)

        self.layers = _get_clones(encoder_layer, num_layers)

        # key, projection for each head, shared among all layers
        self.W_k = nn.Parameter(torch.empty(self.nhead, self.d_model, self.d_head))
        self.W_k.data.uniform_(-stdv, stdv)

        # value, shared value for all heads, shared among all layers
        self.W_v = nn.Parameter(torch.empty(self.d_model, self.d_head))
        self.W_v.data.uniform_(-stdv, stdv)

    def forward(self, ego, other, mask):
        """
        ego:     (b, 1, d)
        other:   (b, N-1, d)
        mask:    (b, N-1)

        out:     (b, 1, d)
        """

        # shared among all layers
        keys = [torch.matmul(other, self.W_k[head]) for head in range(self.nhead)]
        value = torch.matmul(other, self.W_v)

        x = ego
        for attn_layer in self.layers:
            x = attn_layer(x, mask, keys, value)

        return x


class AttentivePolicy(nn.Module):
    def __init__(self, steps=5, hidden_size=512, nhead=8, num_layers=2, **kwargs):
        super().__init__()

        self.hidden_size = hidden_size
        self.steps = steps

        self.class_embedding = nn.Parameter(torch.empty(len(COLORS), hidden_size))
        stdv = 1./math.sqrt(hidden_size)
        self.class_embedding.data.uniform_(-stdv, stdv)

        self.positional_encoding = nn.Sequential(
            nn.Linear(5, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, hidden_size)
        )

        encoder_layer = TransformerEncoderLayer(hidden_size, nhead)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)

        # action conditioned
        self.extract = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size//2),
                nn.ReLU(),
                nn.Linear(hidden_size//2, steps*2)
            ) for _ in range(6)
        ])

    def forward(self, M_pad, M_mask, action):
        """
        M : map, (N x 3)
        """

        # construct inputs. NOTE: cat or sum?
        pos = self.positional_encoding(M_pad[..., :5])
        c = self.class_embedding[M_pad[..., -1].long()]
        x = pos + c #torch.cat([pos, c], dim=-1)
        ego, other = x[:,:1], x[:,1:] # TODO: bug!!!!

        attn = self.transformer(ego, other, M_mask)[:,0]

        # velocity late-fusion, like in LbC
        #fusion = torch.cat([attn, velocity.repeat(1, self.hidden_size//2)], dim=1)

        # extract waypoints
        out = torch.empty((action.shape[0], self.steps, 2)).to(self.class_embedding.device)
        for a in action.long().unique():
            out[action==a] = self.extract[a](attn[action==a]).reshape(-1, self.steps, 2)

        return out


if __name__ == '__main__':
    net = AttentivePolicy().cuda()
    M_pad = torch.rand((5,250,5)).cuda()
    M_mask = torch.ones((5,250), dtype=torch.bool).cuda()
    velocity = torch.Tensor([1.5, 2.5, 3.5, 4.5]).unsqueeze(dim=1).cuda()
    M_mask[0,150:] = 0
    M_mask[1,200:] = 0
    M_mask[2,225:] = 0

    net(M_pad, M_mask, torch.Tensor([0,0,0,0]))#, velocity)
