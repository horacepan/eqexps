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
from torch_geometric.utils import to_dense_batch

from particlenet import ParticleDataset, AwkwardDataset, validate_model, check_memory, nparams
import sys
sys.path.append('../')
from eq_models import Eq2to2
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
        batch = batch.to(DEVICE)
        x = batch.x
        x = self.enc(x)
        x = torch_geometric.nn.global_mean_pool(x, batch.batch)
        x = self.dec(x)
        return x

class SmallEq2Net(nn.Module):
    def __init__(self, nin, nhid, nout):
        super(SmallEq2Net, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(nin, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nhid)
        )

        self.eq = nn.Sequential(
            Eq2to2(nhid, nhid),
            nn.ReLU(),
            Eq2to2(nhid, nhid),
            nn.ReLU()
        )
        self.dec = nn.Linear(nhid, nout)

    def forward(self, batch):
        x, mask = to_dense_batch(batch.x, batch.batch)
        x = x.to(DEVICE)
        x = self.enc(x)
        x = torch.einsum('bid,bjd->bdij', x, x)
        x =  self.eq(x)
        x = x.mean(dim=(-1, -2))
        x = self.dec(x)
        return x

def main(args):
    train_fn = './data/convertedbig/train_file_0.awkd'
    val_fn = './data/convertedbig/val_file_0.awkd'
    test_fn = './data/convertedbig/test_file_0.awkd'
    torch.manual_seed(args.seed)

    logging.info("Starting to load big data ...")
    train_awkward = AwkwardDataset(train_fn, data_format='channel_last')
    val_awkward = AwkwardDataset(val_fn, data_format='channel_last')
    test_awkward = AwkwardDataset(test_fn, data_format='channel_last')
    logging.info("Done loading awk datasets! | Memory used: {:.2f}mb".format(check_memory(False)))

    train_dataset = ParticleDataset(train_awkward, os.path.join(PROJECT_DIR, 'data', 'graph', 'trainbig'))
    val_dataset = ParticleDataset(val_awkward, os.path.join(PROJECT_DIR, 'data', 'graph', 'valbig'))
    test_dataset = ParticleDataset(test_awkward, os.path.join(PROJECT_DIR, 'data', 'graph', 'testbig'))
    logging.info("Done loading particle datasets! | Memory used: {:.2f}mb".format(check_memory(False)))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    logging.info("Done making dataloaders! | Memory used: {:.2f}mb".format(check_memory(False)))

    if args.model == 'SmallSetNet':
        model = SmallSetNet(nin=4, nhid=args.nhid, nout=2)
    elif args.model == 'SmallEq2Net':
        model = SmallEq2Net(nin=4, nhid=args.nhid, nout=2)

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    logging.info(f"Running with num workers: {args.num_workers}, batch size: {args.batch_size}")
    logging.info("Memory used: {:.2f}mb | Num params: {}".format(check_memory(False), nparams(model)))
    logging.info("Model name: {}".format(model.__class__))

    for e in range(100):
        for batch in tqdm(train_dataloader):
            opt.zero_grad()
            ypred = model.forward(batch)
            loss = criterion(ypred, torch.LongTensor(batch.y).to(DEVICE))

            loss.backward()
            opt.step()

        # do the validation
        val_corr, val_total = validate_model(model, val_dataloader)
        logging.info("Epoch: {:2d} | Val acc: {:.3f}".format(e, val_corr/val_total))

        if e % 5 == 0 and e > 0 and not(test_dataloader is None):
            test_corr, test_total = validate_model(model, test_dataloader)
            logging.info("Epoch: {:2d} | Test acc: {:.3f}".format(e, test_corr/test_total))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--model', type=str, default='SmallSetNet')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--nhid', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0003)
    args = parser.parse_args()
    main(args)
