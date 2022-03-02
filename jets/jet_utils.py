import os
import psutil
import torch
from tqdm import tqdm

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

def tensor_validate_model(model, loader, device='cpu'):
    ncorr = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader):
            bx, by = batch[0].to(device), batch[1].to(device)
            ypred = model.forward(bx)
            ncorr += (ypred.max(dim=1)[1] == by).sum()
            total += len(batch[1])

    return ncorr, total


