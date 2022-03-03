import pdb
import os
import sys
import time
import logging
import pickle
import json
from tqdm import tqdm

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from eq_models import *

def ncorrect(output, tgt):
    _, predicted = torch.max(output.data, 1)
    correct = (predicted == tgt).sum().item()
    return correct

def nparams(model):
    tot = 0
    for p in model.parameters():
        tot += p.numel()
    return tot

def get_logger(fname=None, stdout=True):
    '''
    fname: file location to store the log file
    Returns: logger object

    Use the logger object anywhere where you might use a print statement. The logger
    object will print log messages to stdout in addition to writing it to a log file.
    '''
    handlers = []
    if stdout:
        stdout_handler = logging.StreamHandler(sys.stdout)
        handlers.append(stdout_handler)
    if fname:
        file_handler = logging.FileHandler(filename=fname)
        handlers.append(file_handler)

    str_fmt = '[%(asctime)s.%(msecs)03d] %(message)s'
    date_fmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(
        level=logging.INFO,
        format=str_fmt,
        datefmt=date_fmt,
        handlers=handlers
    )

    logger = logging.getLogger(__name__)
    return logger

def setup_experiment_log(args, savedir='./results/', exp_name='test', save=False):
    '''
    savedir: str location to save contents in
    exp_name: Name of experiment
    save: boolean
    Returns: tuple of str (log file) and SummaryWriter
        SummaryWriter will write to the location specified by: {savedir}/{exp_name}/summary
        The returned logfile string will be: {savedir}/{exp_name}/output.log
        If the existing logfile exists (suppose you want to rerun an experiment with
        the model reloaded from a save checkpoint to continue training), the output log
        file will instead be output1.log/output2.log/etc.
    '''
    if not save:
        return None, None

    if os.path.exists(savedir):
        exp_dir = os.path.join(savedir, exp_name)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        sumdir = os.path.join(exp_dir, 'summary')
        swr = SummaryWriter(sumdir)
        json.dump(args.__dict__, open(os.path.join(exp_dir, 'args.json'), 'w'))
        logfile = os.path.join(exp_dir, 'output.log')
        cnt = 1
        while os.path.exists(logfile):
            logfile = os.path.join(exp_dir, f'output{cnt}.log')
            cnt += 1

    else:
        # make the save dir, retry
        os.makedirs(savedir)
        return setup_experiment_log(args, savedir, exp_name, save)

    return logfile, swr

def save_checkpoint(epoch, model, optimizer, fname, scheduler=None):
    '''
    epoch: int, epoch number
    model: nn.Module, the model to save
    optimizer: torch.optim optimizer object
    fname: string, location to save the checkpoint file to

    Saves a state dict of the model and optimizer so that it can be reloaded.
    '''
    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    if scheduler:
        state['scheduler'] = scheduler.state_dict()
    torch.save(state, fname)

def load_checkpoint(model, optimizer, log, fname, scheduler=None):
    '''
    model: nn.Module, the model to reload
    optimizer: torch.optim optimizer, the optizer to reload
    log: logger object
    fname: string, file name to load checkpoint from

    Returns: tuple of the reloaded model, optimizer, epoch of the checkpoint, and a boolean
        indicating whether or not the checkpoint was succesffully loaded.
    Loads the state dict for the given model, optimizer
    '''
    start_epoch = 0
    if os.path.isfile(fname):
        log.info("=> loading checkpoint '{}'".format(fname))
        checkpoint = torch.load(fname)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        log.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(fname, checkpoint['epoch']))

        if scheduler and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        success = True
    else:
        log.info("=> no checkpoint found at '{}'".format(fname))
        success = False

    return model, optimizer, start_epoch, success

def _validate_model(dataloader, model, device):
    '''
    Returns: float, accuracy of model on the data in the given dataloader
    '''
    tot_correct = 0
    tot = 0
    with torch.no_grad():
        for batch in (dataloader):
            d1 = batch[0].to(device)
            d2 = batch[1].to(device)
            target = batch[2].to(device)
            ypred = model.forward(d1, d2)
            tot_correct += ncorrect(ypred, target)
            tot += len(batch[2])

    acc = tot_correct / tot
    return acc, tot_correct

def load_csl_data(fn):
    graphs = pickle.load(open(fn, 'rb'))
    return graphs

def adj_tensor(adj, feats):
    return torch.einsum('bij,bkd->bijkd', adj, feats)

def make_dense_graphs(sp_graphs):
    return [s.todense() for s in sp_graphs]

def onehot_features(graphs, onehot_dim=10):
    feats = []

    for g in graphs:
        mat = np.zeros((g.shape[0], onehot_dim + g.shape[0]))
        for i in range(g.shape[0]):
            j = i % onehot_dim
            mat[i][j] = 1
        mat[:, onehot_dim:] = torch.eye(g.shape[0])
        feats.append(mat)

    return feats

def coo_edge_idx(coo):
    edge_idx = torch.LongTensor([coo.row, coo.col])
    return edge_idx

class CSLData():
    def __init__(self, fn, yfn, graph_fmt='sp'):
        self.sp_graphs = load_csl_data(fn)
        self.feats = onehot_features(self.sp_graphs)
        self.labels = torch.load(yfn)
        if graph_fmt == 'sp':
            self.graphs = [coo_edge_idx(g) for g in self.sp_graphs]
        else:
            self.graphs = [g.todense() for g in self.sp_graphs]

    def __len__(self):
        return len(self.sp_graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.feats[idx], self.labels[idx]

#class BasicGCN(nn.Module):
#    def __init__(self, nin, nhid, nout, nlayers):
#        self.enc = nn.Linear(nin, nhid)
#        self.dec = nn.Linear(nhid, nout)
#        self.layers = nn.ModuleList([GCNConv(nhid, nhid) for _ in nlayers])
#
#    def forward_batch(self, batch):
#        x, edge_index, bidx = batch.x, batch.edge_index, batch.batch
#        return self.forward(x, edge_index, bidx)
#
#    def forward(self, x, edge_idx, bidx):
#        x = self.enc(x)
#
#        for gconv in self.gcn_layers:
#            x = gconv(x)
#            x = F.relu(x)
#
#        x = self.dec(x)
#        return x

class Eq2Net(nn.Module):
    def __init__(self, nin, nhid, nout, nlayers):
        super(Eq2Net, self).__init__()
        self.enc = nn.Linear(nin, nhid)
        self.dec = nn.Linear(nhid, nout)
        self.layers = nn.ModuleList([
            Eq2to2(nhid, nhid) for _ in range(nlayers)
        ])

    def forward(self, adj, feats):
        feats = self.enc(feats)
        x = torch.einsum('bid,bjd->bijd', feats, feats)
        x = x.permute(0, 3, 1, 2)

        # how is the adj used
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
        x = x.permute(0, 2, 3, 1)
        x = x.sum(dim=(1, 2))
        x = self.dec(x)
        return x

class Eq3Net(nn.Module):
    def __init__(self, nin, nhid, nout, gen_third_order_tensor, nlayers=2):
        super(Eq3Net, self).__init__()
        self.enc = nn.Linear(nin, nhid)
        self.dec = nn.Linear(nhid, nout)
        self.layers = nn.ModuleList([
            SetEq3to3(nhid, nhid) for _ in range(nlayers)
        ])
        self.gen_third_order_tensor = gen_third_order_tensor

    def forward(self, adj, feats):
        feats = self.enc(feats)
        x = torch.einsum('bij,bkd->bijkd', adj, feats)
        x = x.permute(0, 4, 1, 2, 3) # move the feat dim to the 2nd index

        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
        # b d n n n
        x = x.permute(0, 2, 3, 4, 1) # move the feat dim to the 2nd index
        x = torch.sum(x, dim=(1, 2, 3))
        x = self.dec(x)
        return x

class CSL(Dataset):
    def __init__(self, graphs, targets):
        self.graphs = graphs
        self.feats = torch.FloatTensor(onehot_features(graphs))
        self.targets = targets

    def edge_feats(self, i):
        # i, j encodes the pair feature
        graph = self.graphs[i]
        feat = self.feats[i]
        ef = torch.zeros(graph.shape)
        for i in range(n):
            for j in range(n):
                if graph[i][j] != 0:
                    ef[i][j] = 1
        return ef

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.graphs[idx], self.feats[idx], self.targets[idx]

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fn = './data/graphs_Kary_Deterministic_Graphs.pkl'
    yfn = os.path.join('./data/', 'y_Kary_Deterministic_Graphs.pt')
    seed = 0
    lr = 0.001
    batch_size = 16
    epochs = 20
    nin = 10
    nhid = 16
    nclasses = 10

    graphs = load_csl_data(fn)
    dense_graphs = torch.stack([torch.FloatTensor(g.todense()) for g in graphs])
    targets = torch.load(yfn)
    dataset = CSL(dense_graphs, targets)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                    (120, 30),
                                    torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

    #net = Eq3Net(nin, nhid, nclasses, adj_tensor).to(device)
    net = Eq2Net(nin + 41, nhid, nclasses, 2).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    print("Starting training")

    for e in (range(epochs)):
        for batch in tqdm(train_loader):
            opt.zero_grad()
            adj, feats, target = batch
            adj, feats, target = adj.to(device), feats.to(device), target.to(device)

            ypred = net.forward(adj, feats)
            loss = criterion(ypred, target)
            loss.backward()
            opt.step()
        val_acc = _validate_model(test_loader, net, device)
        print("Epoch {:3d} | Test acc: {:.3f}".format(e, val_acc))

if __name__ == '__main__':
    main()
