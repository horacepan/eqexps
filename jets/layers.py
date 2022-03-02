import numpy as np
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, nin, dim):
        '''
        Attention(Q, K, V) = softmax(\frac{QK^\top}{d_k})V
        '''
        super(SelfAttention, self).__init__()
        self.q = nn.Linear(nin, dim)
        self.k = nn.Linear(nin, dim)
        self.v = nn.Linear(nin, dim)
        self.scale = dim**(-0.5)

    def forward(self, x, mask=None):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        if mask is not None:
            dots = dots.masked_fill(mask, -np.inf)
        weights = torch.softmax(dots, dim=-1)
        return torch.einsum('bij,bjd->bid', weights, v)

class SABlock(nn.Module):
    def __init__(self, dim):
        self.att = SelfAttention(dim, dim)

    def forward(self, x, mask=None):
        '''
        Assert x dim is equal to output dim of attention layer
        '''
        xa = self.att(x, mask) # B x N x dim
        return x + xa

class MultiHeadAttention(nn.Module):
    def __init__(self, nin, dim, nhead):
        super(MultiHeadAttention, self).__init__()
        self.qs = nn.Linear(nin, dim * nhead)
        self.vs = nn.Linear(nin, dim * nhead)
        self.ks = nn.Linear(nin, dim * nhead)
        self.scale = dim**(-0.5)
        self.dim = dim
        self.nhead = nhead

    def forward(self, x, mask=None):
        B, N, nin = x.shape
        qs = self.qs(x).view(self.nhead, B, N, self.dim)
        ks = self.ks(x).view(self.nhead, B, N, self.dim)
        vs = self.vs(x).view(self.nhead, B, N, self.dim)
        att = torch.einsum('hbid,hbjd->hbij',qs, ks) * self.scale

        if mask is not None:
            att.masked_fill(mask, -np.inf)

        output = torch.einsum('hbij,hbjd->hbid', att, vs)
        output = output.permute(1, 2, 3, 0)
        output = output.reshape(B, N, -1)
        return  output
