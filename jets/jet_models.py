import sys
sys.path.append('../')
from eq_models import Eq2to2, MiniEq2to2
import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from layers import SelfAttention, SABlock, MultiHeadAttention

class SmallSetNet(nn.Module):
    def __init__(self, nin, nhid, nout):
        super(SmallSetNet, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(nin, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nhid)
        )
        self.dec = nn.Sequential(
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nout)
        )

    def forward(self, batch):
        x = batch.x
        x = self.enc(x)
        x = torch_geometric.nn.global_mean_pool(x, batch.batch)
        x = self.dec(x)
        return x

    def forward_xb(self, x, batch_idx):
        x = self.enc(x)
        x = torch_geometric.nn.global_mean_pool(x, batch_idx)
        x = self.dec(x)
        return x

class SmallAttNet(nn.Module):
    def __init__(self, nin, nhid, nout):
        super(SmallAttNet, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(nin, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nhid)
        )
        self.att1 = SABlock(nhid, nhid)
        self.att2 = SABlock(nhid, nhid)

    def forward(self):
        pass

class SmallEq2Net(nn.Module):
    def __init__(self, nin, nhid, neqhid, ndechid, nout, **kwargs):
        super(SmallEq2Net, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(nin, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nhid)
        )
        self.eq_fc1 = nn.Linear(nhid, neqhid)
        self.eq1 = Eq2to2(neqhid, nhid)

        self.dec = nn.Sequential(
            nn.Linear(nhid, ndechid),
            nn.ReLU(),
            nn.Linear(ndechid, ndechid),
            nn.ReLU(),
            nn.Linear(ndechid, nout)
        )

    def forward(self, x, device='cpu'):
        x = self.enc(x)
        x = torch.einsum('bid,bjd->bijd', x, x)
        x = self.eq_fc1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.eq1(x)
        x = F.relu(x)
        x = x.mean(dim=(-1, -2))
        x = self.dec(x)
        return x

class MiniEq2Net(nn.Module):
    def __init__(self, nin, nhid, ndechid, nout):
        super(MiniEq2Net, self).__init__()
        self.eq1 = MiniEq2to2(2*nin, nhid) # 8 * 10 * nhid
        self.eq2 = MiniEq2to2(nhid, nhid) # nhid * 10 * nhid
        self.dec = nn.Sequential(
            nn.Linear(nhid, ndechid),
            nn.ReLU(),
            nn.Linear(ndechid, ndechid),
            nn.ReLU(),
            nn.Linear(ndechid, nout)
        )

    def forward(self, x):
        x1 = torch.diag_embed(x.permute(0, 2, 1)) # B x d x N x N
        x2 = torch.einsum('bid,bjd->bdij', x, x)
        x = torch.cat([x1, x2], dim=1) # B x 2d x N x N
        x = self.eq1(x) # B x D x N x N
        x = F.relu(x)
        x = self.eq2(x) # B x D x N x N
        x = F.relu(x)
        x = x.sum(dim=(-1, -2)) # B x D
        x = F.relu(x)
        x = self.dec(x)
        return x

class MediumEq2Net(nn.Module):
    def __init__(self, nin, nenchid, nhid, ndechid, nout):
        super(MediumEq2Net, self).__init__()
        self.enc = nn.Linear(2*nin, nenchid)
        self.eq1 = Eq2to2(nenchid, nhid)
        self.dec = nn.Sequential(
            nn.Linear(nhid, ndechid),
            nn.ReLU(),
            nn.Linear(ndechid, nout)
        )

    def forward(self, x):
        x1 = torch.diag_embed(x.permute(0, 2, 1)) # B x d x N x N
        x2 = torch.einsum('bid,bjd->bdij', x, x)
        x = torch.cat([x1, x2], dim=1) # B x 2d x N x N
        x = self.enc(x.permute(0, 2, 3, 1))
        x = F.relu(x)
        x = self.eq1(x.permute(0, 3, 1, 2)) # B x D x N x N
        x = F.relu(x)
        x = x.sum(dim=(-1, -2)) # B x D
        x = self.dec(x)
        return x

class SmallEq2NetMini(nn.Module):
    def __init__(self, nin, nhid, neqhid, ndechid, nout, **kwargs):
        super(SmallEq2NetMini, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(nin, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nhid)
        )
        self.eq_fc1 = nn.Linear(nhid, neqhid)
        self.eq1 = MiniEq2to2(neqhid, nhid)
        self.dec = nn.Sequential(
            nn.Linear(nhid, ndechid),
            nn.ReLU(),
            nn.Linear(ndechid, ndechid),
            nn.ReLU(),
            nn.Linear(ndechid, nout)
        )

    def forward(self, x, device='cpu'):
        x = self.enc(x)
        x = torch.einsum('bid,bjd->bijd', x, x)

        # b n n d
        x = self.eq_fc1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.eq1(x)
        x = F.relu(x)
        x = x.mean(dim=(-1, -2))
        x = self.dec(x)
        return x

class TensorNet(nn.Module):
    def __init__(self, nin, nhid, ndechid, nout):
        super(TensorNet, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(nin, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nhid)
        )
        self.dec = nn.Sequential(
            nn.Linear(nhid, ndechid),
            nn.ReLU(),
            nn.Linear(ndechid, ndechid),
            nn.ReLU(),
            nn.Linear(ndechid, nout)
        )

    def forward(self, x):
        x = self.enc(x)
        x = F.relu(x)
        x = torch.einsum('bid,bjd->bijd', x, x)
        x = torch.mean(x, dim=(1, 2))
        x = F.relu(x)
        x = self.dec(x)
        return x

class TensorMLPNet(nn.Module):
    def __init__(self, nin, nhid, ndechid, nout):
        super(TensorMLPNet, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(nin, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nhid)
        )
        self.tens_dec = nn.Sequential(
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nhid)
        )
        self.dec = nn.Sequential(
            nn.Linear(nhid, ndechid),
            nn.ReLU(),
            nn.Linear(ndechid, ndechid),
            nn.ReLU(),
            nn.Linear(ndechid, nout)
        )

    def forward(self, x):
        x = self.enc(x)
        x = F.relu(x)
        x = torch.einsum('bid,bjd->bijd', x, x)
        x = self.tens_dec(x)
        x = torch.mean(x, dim=(1, 2))
        x = F.relu(x)
        x = self.dec(x)
        return x
# python smallnet.py --model SmallEq2Net --nhid 32 --neqhid 8 --batch_size 512 --lr 0.001
