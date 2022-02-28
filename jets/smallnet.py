import pdb
import time
import os
import logging
from tqdm import tqdm

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import DataLoader

from particlenet import ParticleDataset, AwkwardDataset, validate_model, check_memory

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

PROJECT_DIR = './'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
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

class Eq2Net(nn.Module):
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

def main(args):
    train_fn = './data/convertedbig/train_file_0.awkd'
    val_fn = './data/convertedbig/val_file_0.awkd'
    batch_size = 1024
    num_workers = 0
    nhid = 32
    lr = 0.001

    logging.info("Loading big data!")
    train_awkward = AwkwardDataset(train_fn, data_format='channel_last')
    val_awkward = AwkwardDataset(val_fn, data_format='channel_last')
    train_dataset = ParticleDataset(train_awkward, os.path.join(PROJECT_DIR, 'data', 'graph', 'trainbig'))
    val_dataset = ParticleDataset(val_awkward, os.path.join(PROJECT_DIR, 'data', 'graph', 'valbig'))
    logging.info("Done loading big data!")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = SmallSetNet(nin=4, nhid=nhid, nout=2)
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    logging.info(f"Running with num workers: {num_workers}, batch size: {batch_size}")

    for e in range(10):
        for batch in tqdm(train_dataloader):
            opt.zero_grad()
            batch = batch.to(DEVICE)
            ypred = model.forward(batch)
            loss = criterion(ypred, torch.LongTensor(batch.y).to(DEVICE))

            loss.backward()
            opt.step()

        # do the validation
        val_corr, val_total = validate_model(model, val_dataloader)
        logging.info("Epoch: {:2d} | Val acc: {:.3f}".format(e, val_corr/val_total))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--nhid', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    main(args)
