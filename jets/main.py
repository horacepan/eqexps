import sys
sys.path.append('../')

import pdb
import time
import os
#from tqdm import tqdm

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_batch

from jet_utils import check_memory, tensor_validate_model, validate_model, nparams
from particlenet import ParticleDataset, AwkwardDataset, check_memory
from jet_models import *
from layers import *
from utils import get_logger, setup_experiment_log, save_checkpoint, load_checkpoint

PROJECT_DIR = './'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
def load_tensor_data(xfn, yfn, loader_kwargs):
    xs = torch.load(xfn)
    ys = torch.load(yfn)
    ds = torch.utils.data.TensorDataset(xs, ys)
    dataloader = torch.utils.data.DataLoader(ds, **loader_kwargs)
    return dataloader

def load_small_tensor_loaders(loader_kwargs):
    loc_fmt = './data/converted100/{}.pt'
    fs = ['train', 'val', 'test']
    pt_fns = [loc_fmt.format(t) for t in fs]
    tensor_loaders = []

    for f in pt_fns:
        d = torch.load(f)
        x = d['x']
        y = d['y']
        ds = torch.utils.data.TensorDataset(x, y)
        dl = torch.utils.data.DataLoader(ds, **loader_kwargs)
        tensor_loaders.append(dl)

    return tensor_loaders

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

def load_graph_loaders(loader_kwargs, test=False):
    if test:
        _dir = 'converted100'
        pn_suffix = '100'
    else:
        _dir = 'convertedbig'
        pn_suffix = 'big'

    train_fn = f'./data/{_dir}/train_file_0.awkd'
    val_fn = f'./data/{_dir}/val_file_0.awkd'
    test_fn = f'./data/{_dir}/test_file_0.awkd'

    train_pn = './data/graph/train{pn_suffix}'
    val_pn = './data/graph/val{pn_suffix}'
    test_pn = './data/graph/test{pn_suffix}'
    fns = [train_fn, val_fn, test_fn]
    pns = [train_pn, val_pn, test_pn]
    return [load_graph_dataset(f, p, loader_kwargs) for f, p in zip(fns, pns)]

def main(args):
    torch.manual_seed(args.seed)
    log_fn, swr = setup_experiment_log(args, args.savedir, args.exp_name, save=args.save)
    log = get_logger(log_fn)
    log.info('Starting experiment! Saving in: {}'.format(log_fn))
    log.info('Command line:')
    log.info('python ' + ' '.join(sys.argv))

    log.info("Starting to load dataloaders")
    loader_kwargs = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': args.num_workers}
    if args.model == 'SmallSetNet':
        train_dataloader, val_dataloader, test_dataloader = load_graph_loaders(loader_kwargs, test=args.test)
    else:
        if not args.test:
            train_dataloader, val_dataloader, test_dataloader = load_tensor_loaders(loader_kwargs)
        else:
            log.info("Using the smaller data")
            train_dataloader, val_dataloader, test_dataloader = load_small_tensor_loaders(loader_kwargs)

    log.info("Done loading dataloaders")
    log.info("Memory usage: {:.2f}mb".format(check_memory(False)))
    log.info("Args: {}".format(args))

    if args.model == 'SmallSetNet':
        model = SmallSetNet(nin=4, nhid=args.nhid, nout=2)
    elif args.model == 'SmallEq2Net':
        model = SmallEq2Net(nin=4, nhid=args.nhid, neqhid=args.neqhid, ndechid=args.ndechid, nout=2)
    elif args.model == 'MiniEq2Net':
        model = MiniEq2Net(nin=4, nhid=args.nhid, ndechid=args.ndechid, nout=2)
    elif args.model == 'MediumEq2Net':
        model = MediumEq2Net(nin=4, nenchid=args.nenchid, nhid=args.nhid, ndechid=args.ndechid, nout=2)
    elif args.model == 'SmallEq2NetMini':
        model = SmallEq2NetMini(nin=4, nhid=args.nhid, neqhid=args.neqhid, ndechid=args.ndechid, nout=2)
    elif args.model == 'TensorNet':
        model = TensorNet(nin=4, nhid=args.nhid, ndechid=args.ndechid, nout=2)
    elif args.model == 'TensorMLPNet':
        model = TensorMLPNet(nin=4, nhid=args.nhid, ndechid=args.ndechid, nout=2)

    #scaler = torch.cuda.amp.GradScaler()
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    savedir = os.path.join(args.savedir, args.exp_name)
    checkpoint_fn = os.path.join(savedir, 'checkpoint.pth')
    if args.lr_decay:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.8, patience=args.patience)
    else:
        scheduler = None

    model, opt, start_epoch, _ = load_checkpoint(model, opt, log, checkpoint_fn, scheduler)
    log.info(f"Running with num workers: {args.num_workers}, batch size: {args.batch_size}")
    log.info("Memory used: {:.2f}mb | Num params: {}".format(check_memory(False), nparams(model)))
    log.info("Model name: {}".format(model.__class__))

    for e in range(start_epoch, start_epoch + args.epochs + 1):
        for batch in (train_dataloader):
            if args.model == 'SmallSetNet':
                batch = batch.to(DEVICE)
                ypred = model.forward(batch)
                loss = criterion(ypred, torch.LongTensor(batch.y).to(DEVICE))
            else:
                bx, by = batch
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                ypred = model.forward(bx)
                loss = criterion(ypred, by)

            loss.backward()
            opt.step()
            opt.zero_grad()
            #scaler.scale(loss).backward()
            #scaler.step(opt)
            #scaler.update()

        if e % args.val_check == 0:
            if args.model == 'SmallSetNet':
                val_corr, val_total = validate_model(model, val_dataloader, DEVICE)
            else:
                val_corr, val_total = tensor_validate_model(model, val_dataloader, DEVICE)

            val_acc = val_corr/val_total
            log.info("Epoch: {:2d} |  Val acc: {:.4f}".format(e, val_corr/val_total))

            if args.save:
                save_checkpoint(e, model, opt, checkpoint_fn)
            if scheduler:
                log.info("Stepping scheduler")
                scheduler.step(val_acc)

        if e % 5 == 0 and e > 0 and not(test_dataloader is None):
            if args.model == 'SmallSetNet':
                test_corr, test_total = validate_model(model, test_dataloader, DEVICE)
            else:
                test_corr, test_total = tensor_validate_model(model, test_dataloader, DEVICE)
            log.info("Epoch: {:2d} | Test acc: {:.4f}".format(e, test_corr/test_total))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--model', type=str, default='SmallSetNet')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--nhid', type=int, default=32)
    parser.add_argument('--neqhid', type=int, default=8)
    parser.add_argument('--ndechid', type=int, default=128)
    parser.add_argument('--nenchid', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--val_check', type=int, default=1)

    parser.add_argument('--savedir', type=str, default='./results/jets')
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--lr_decay', action='store_true', default=False)
    parser.add_argument('--patience', type=int, default=3)

    args = parser.parse_args()
    main(args)
