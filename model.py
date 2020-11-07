import math
from itertools import chain
import copy

import torch
import torch.nn as nn
import numpy as np

from .const import HEIGHT, WIDTH, C
from .util import SpatialSoftmax


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


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

    def forward(self, ego, keys, value):
        """
        ego:     (b, 1, d)
        keys:    (h, b, N-1, h_d)
        value:   (b, N-1, h_d)

        out:     (b, 1, d)
        """

        out = torch.empty(ego.shape[0], 1, self.d_model)
        for head in range(self.nhead):
            query = torch.einsum('bnd,dh->bnh', ego, self.W_q[head])

            x = torch.bmm(query, keys[head].permute(0,2,1))
            x /= math.sqrt(self.d_head)
            x = nn.functional.softmax(x, dim=0)

            out[...,(head*self.d_head):((head+1)*self.d_head)] = torch.bmm(x, value)

        return out


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=512, nhead=8):
        super().__init__()

        self.d_model = d_model

        self.attention = MultiheadAttention(d_model, nhead)

    def forward(self, ego, keys, value):
        """
        ego:     (b, 1, d)
        keys:    (h, b, N-1, h_d)
        value:   (b, N-1, h_d)

        out:     (b, 1, d)
        """

        return self.attention(ego, keys, value)
        # TODO: MLP
        # TODO: residual
        # TODO: dropout?


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

    def forward(self, ego, other):
        """
        ego:     (b, 1, d)
        other:   (b, N-1, d)

        out:     (b, 1, d)
        """

        # shared among all layers
        keys = [torch.mm(other, self.W_k[head]) for head in range(self.nhead)]
        value = torch.mm(other, self.W_v)

        x = ego
        for attn_layer in self.layers:
            x = attn_layer(x, keys, value)

        return x


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
            nn.ReLU(),
            nn.Linear(h//2, h)
        )

        encoder_layer = TransformerEncoderLayer(hidden_size, nhead)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)

        # action conditioned
        self.extract = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size//2),
                nn.ReLU(),
                nn.Linear(hidden_size//2, steps*2)
            ) for _ in range(3)
        ])

    def forward(self, M_pad, M_mask, action):
        """
        M : map, (N x 3)
        """

        # construct inputs. NOTE: cat or sum?
        pos = self.positional_encoding(M_pad[..., :2])
        c = self.class_embedding[M_pad[..., 2].long()]
        x = torch.cat([pos, c], dim=-1)

        ego, other = x[:,:1], x[:,1:]

        attn = self.transformer(ego, other)
        import pdb; pdb.set_trace()
        #src_key_padding_mask=M_mask

        # extract waypoints
        out = torch.empty((action.shape[0], self.steps, 2))#.cuda()
        for a in action.long().unique():
            out[action==a] = self.extract[a](attn[action==a]).reshape(-1, self.steps, 2)

        return out


if __name__ == '__main__':
    net = AttentivePolicy()
    M_pad = torch.rand((4,250,3))
    M_mask = torch.ones((4,250), dtype=torch.float)

    net(M_pad, M_mask, torch.Tensor([0]))
