import pdb
import os
import psutil
from tqdm import tqdm
import numpy as np
import torch

def check_memory(verbose=True):
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    if verbose:
        print("Consumed {:.2f}mb memory".format(mem))
    return mem

def nparams(model):
    tot = 0
    for p in model.parameters():
        tot += p.numel()
    return tot

def tensor_validate_model(acc, auroc, model, loader, device='cpu'):
    with torch.no_grad():
        for batch in (loader):
            bx, by = batch[0].to(device), batch[1].to(device)
            ypred = model.forward(bx)
            acc.update(ypred, by)
            auroc.update(ypred.softmax(dim=1), by)

    return acc, auroc

def validate_model(acc, auroc, model, loader, device='cpu'):
    ncorr = 0
    total = 0

    with torch.no_grad():
        for batch in (loader):
            batch = batch.to(device)
            ypred = model.forward(batch)
            # get acc of the model
            y = torch.LongTensor(batch.y).to(device)
            acc.update(ypred, y)
            auroc.update(ypred.softmax(dim=1), y)

    return acc, auroc

def init_weights(net):
    for p in net.parameters():
        if len(p.shape) == 1:
            p.data.zero_()
        else:
            nin = p.shape[0]
            nout = p.shape[1]
            std = 2 / (nin + nout)
            p.data.normal_(0, np.sqrt(2 / (nin + nout)))
