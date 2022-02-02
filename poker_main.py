import pdb
import os
import time
import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from poker_loader import PokerDataset
from utils import nparams, ncorrect, get_logger, _validate_model, setup_experiment_log, save_checkpoint, load_checkpoint
from eq_models import Eq2to2, SetEq3to3

class PokerNet(nn.Module):
    def __init__(self, embed_dim, hid_dim, out_dim):
        super(PokerNet, self).__init__()
        self.suit_enc = nn.Embedding(4, embed_dim)
        self.num_enc = nn.Embedding(13, embed_dim)
        self.mix = nn.Linear(2*embed_dim, hid_dim)
        self.enc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
        )

        self.dec = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, suits, nums):
        s = self.suit_enc(suits)
        n = self.num_enc(nums)
        x = torch.cat([s, n], dim=2)
        x = self.mix(x)
        x = x.sum(dim=1)
        x = self.dec(x)
        return x

class Poker2Net(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(Poker2Net, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(2, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
        )

        self.dec = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
        )
        #self.dec = nn.Sequential(
        #    nn.ReLU(),
        #    nn.Linear(hid_dim, out_dim)
        #)

    def forward(self, suits, nums):
        x = torch.stack([suits, nums], dim=-1).float()
        x = self.enc(x)
        x = x.sum(dim=1)
        x = self.dec(x)
        return x

class PokerEq2Net(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(PokerEq2Net, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(2, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
        )

        self.eq = nn.Sequential(
            Eq2to2(hid_dim, hid_dim),
            nn.ReLU(),
            Eq2to2(hid_dim, hid_dim),
            nn.ReLU()
        )

        self.dec = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
        )
        #self.dec = nn.Sequential(
        #    nn.ReLU(),
        #    nn.Linear(hid_dim, out_dim)
        #)

    def forward(self, suits, nums):
        x = torch.stack([suits, nums], dim=-1).float()
        x = self.enc(x)
        x = torch.einsum('bid,bjd->bijd', x, x)
        x = x.permute(0, 3, 1, 2)
        x = self.eq(x)
        x = x.permute(0, 2, 3, 1)
        x = x.sum(dim=(1, 2))
        x = self.dec(x)
        return x

class PokerEq3Net(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(PokerEq3Net, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(2, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
        )

        self.eq = nn.Sequential(
            SetEq3to3(hid_dim, hid_dim),
            nn.ReLU(),
            SetEq3to3(hid_dim, hid_dim),
            nn.ReLU()
        )

        self.dec = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, suits, nums):
        x = torch.stack([suits, nums], dim=-1).float()
        x = self.enc(x)
        x = torch.einsum('bid,bjd,bkd->bijkd', x, x, x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.eq(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = x.sum(dim=(1, 2, 3))
        x = self.dec(x)
        return x

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    log_fn, swr = setup_experiment_log(args, args.savedir, args.exp_name, save=args.save)
    log = get_logger(log_fn)
    log.info('Loading data')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.all_test:
        nsample=None # None to use all of it
    else:
        nsample = 50000 # 2x train data size
    train_data = PokerDataset(args.train_data_fn, mode=args.mode)
    test_data = PokerDataset(args.test_data_fn, nsample=nsample, mode=args.mode)
    log.info('Done loading data | Train size: {} | Test size: {}'.format(len(train_data), len(test_data)))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

    if args.model == 'PokerNet':
        model = PokerNet(args.embed_dim, args.hid_dim, train_data.num_classes).to(device)
    elif args.model == 'PokerEq2Net':
        model = PokerEq2Net(args.hid_dim, train_data.num_classes).to(device)
    elif args.model == 'PokerEq3Net':
        model = PokerEq3Net(args.hid_dim, train_data.num_classes).to(device)
    else:
        model = Poker2Net(args.hid_dim, train_data.num_classes).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    savedir = os.path.join(args.savedir, args.exp_name)
    checkpoint_fn = os.path.join(savedir, 'checkpoint.pth')
    model, opt, start_epoch, _ = load_checkpoint(model, opt, log, checkpoint_fn)
    log.info('Starting')
    log.info('Model: {} | params: {}'.format(model.__class__, nparams(model)))

    for e in range(start_epoch, start_epoch + args.epochs + 1):
        losses = []
        correct = 0
        for batch in train_loader:
            suits, nums, target = batch
            suits = suits.to(device)
            nums = nums.to(device)
            target = target.to(device)

            opt.zero_grad()
            ypred = model(suits, nums)
            loss = criterion(ypred, target)
            loss.backward()
            opt.step()

            losses.append(loss.item())
            correct += ncorrect(ypred, target)

        train_acc = correct / len(train_data)
        if e % args.test_iter == 0:
            test_acc, tcorrect = _validate_model(test_loader, model, device)
            log.info('Epoch {:4d} | Train acc: {:.2f} | Test acc: {:.2f}'.format(e, train_acc, test_acc))
            if args.save:
                save_checkpoint(e, model, opt, checkpoint_fn)
        else:
            log.info('Epoch {:4d} | Train acc: {:.2f}'.format(e, train_acc, correct,  test_acc, tcorrect))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', type=str, default= 'PokerNet')
    parser.add_argument('--savedir', type=str, default='./results/poker/')
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--train_data_fn', type=str, default= './data/poker/poker-hand-training-true.data')
    parser.add_argument('--test_data_fn', type=str, default= './data/poker/poker-hand-testing.data')
    parser.add_argument('--test_iter', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--hid_dim', type=int, default=32)
    parser.add_argument('--nlayers', type=int, default=2)
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--mode', type=str, default='binary')
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--all_test', action='store_true', default=False)
    parser.add_argument('--save', action='store_true', default=False)

    args = parser.parse_args()
    main(args)
