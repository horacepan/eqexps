import sys
sys.path.append('../')

import pdb
import time
import os
import logging
from tqdm import tqdm

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_batch

from jet_utils import check_memory, tensor_validate_model
from particlenet import ParticleDataset, AwkwardDataset, validate_model, check_memory, nparams
from eq_models import Eq2to2, Eq2to2Fixed
from jet_models import *
from layers import *
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

PROJECT_DIR = './'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
def load_tensor_data(xfn, yfn, loader_kwargs):
    xs = torch.load(xfn)
    ys = torch.load(yfn)
    ds = torch.utils.data.TensorDataset(xs, ys)
    dataloader = torch.utils.data.DataLoader(ds, **loader_kwargs)
    return dataloader
def load_small_tensor_loaders(loader_kwargs):
    pass

def load_tensor_loaders(loader_kwargs):
    xs_fmt = './data/convertedbig/{}_stack.pt'
    ys_fmt = './data/convertedbig/{}y.pt'
    fs = ['train', 'val', 'test']
    xs_fns = [xs_fmt.format(t) for t in fs]
    ys_fns = [ys_fmt.format(t) for t in fs]
    return [load_tensor_data(x, y, loader_kwargs) for x, y in zip(xs_fns, ys_fns)]

def load_graph_dataset(fn, particle_fn, loader_kwargs):
    awk_data = AwkwardDataset(fn, data_format='channel_last')
    dataset = ParticleDataset(awk_data, particle_fn)
    loader = DataLoader(dataset, **loader_kwargs)
    return loader

def load_graph_loaders(loader_kwargs):
    train_fn = './data/convertedbig/train_file_0.awkd'
    val_fn = './data/convertedbig/val_file_0.awkd'
    test_fn = './data/convertedbig/test_file_0.awkd'

    train_pn = './data/graph/trainbig'
    val_pn = './data/graph/valbig'
    test_pn = './data/graph/testbig'
    fns = [train_fn, val_fn, test_fn]
    pns = [train_pn, val_pn, test_pn]
    return [load_graph_dataset(f, p, loader_kwargs) for f, p in zip(fns, pns)]

def main(args):
    train_fn = './data/convertedbig/train_file_0.awkd'
    val_fn = './data/convertedbig/val_file_0.awkd'
    test_fn = './data/convertedbig/test_file_0.awkd'
    torch.manual_seed(args.seed)

    logging.info("Starting to load dataloaders")
    loader_kwargs = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': args.num_workers}
    train_dataloader, val_dataloader, test_dataloader = load_tensor_loaders(loader_kwargs)
    logging.info("Done loading dataloaders")
    logging.info("Memory usage: {:.2f}mb".format(check_memory(False)))
    #logging.info("Starting to load big data ...")
    #train_awkward = AwkwardDataset(train_fn, data_format='channel_last')
    #val_awkward = AwkwardDataset(val_fn, data_format='channel_last')
    #test_awkward = AwkwardDataset(test_fn, data_format='channel_last')
    #logging.info("Done loading awk datasets! | Memory used: {:.2f}mb".format(check_memory(False)))

    #train_dataset = ParticleDataset(train_awkward, os.path.join(PROJECT_DIR, 'data', 'graph', 'trainbig'))
    #val_dataset = ParticleDataset(val_awkward, os.path.join(PROJECT_DIR, 'data', 'graph', 'valbig'))
    #test_dataset = ParticleDataset(test_awkward, os.path.join(PROJECT_DIR, 'data', 'graph', 'testbig'))
    #logging.info("Done loading particle datasets! | Memory used: {:.2f}mb".format(check_memory(False)))

    #train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    #val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    #test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    #logging.info("Done making dataloaders! | Memory used: {:.2f}mb".format(check_memory(False)))

    if args.model == 'SmallSetNet':
        model = SmallSetNet(nin=4, nhid=args.nhid, nout=2)
    elif args.model == 'SmallEq2Net':
        model = SmallEq2Net(nin=4, nhid=args.nhid, neqhid=args.neqhid, ndechid=args.ndechid, nout=2)
    elif args.model == 'MiniEq2Net':
        model = MiniEq2Net(nin=4, nhid=args.nhid, ndechid=args.ndechid, nout=2)

    #scaler = torch.cuda.amp.GradScaler()
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    logging.info(f"Running with num workers: {args.num_workers}, batch size: {args.batch_size}")
    logging.info("Memory used: {:.2f}mb | Num params: {}".format(check_memory(False), nparams(model)))
    logging.info("Model name: {}".format(model.__class__))

    for e in range(100):
        for batch in tqdm(train_dataloader):
            opt.zero_grad()
            #if args.model == 'SmallSetNet':
            #    ypred = model.forward(batch)
            #    loss = criterion(ypred, torch.LongTensor(batch.y).to(DEVICE))
            #else:
            bx, by = batch
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            ypred = model.forward(bx)
            loss = criterion(ypred, by)

            loss.backward()
            opt.step()
            #scaler.scale(loss).backward()
            #scaler.step(opt)
            #scaler.update()

        # do the validation
        if e % args.val_check == 0:
            val_corr, val_total = tensor_validate_model(model, val_dataloader)
            logging.info("Epoch: {:2d} | Val acc: {:.3f}".format(e, val_corr/val_total))

        if e % 5 == 0 and e > 0 and not(test_dataloader is None):
            test_corr, test_total = tensor_validate_model(model, test_dataloader)
            logging.info("Epoch: {:2d} | Test acc: {:.3f}".format(e, test_corr/test_total))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--model', type=str, default='SmallSetNet')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--nhid', type=int, default=32)
    parser.add_argument('--neqhid', type=int, default=8)
    parser.add_argument('--ndechid', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--val_check', type=int, default=2)
    args = parser.parse_args()
    main(args)
