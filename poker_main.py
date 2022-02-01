import pdb
import time
import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from poker_loader import PokerDataset
from utils import nparams, ncorrect, get_logger, _validate_model

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

def main(args):
    log = get_logger()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = PokerDataset(args.data_fn)
    train_len = int(len(dataset) * args.train_pct)
    train_data, test_data = torch.utils.data.random_split(dataset,
                                                          (train_len, len(dataset) - train_len),
                                                          torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_data, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)
    model = PokerNet(args.embed_dim, args.hid_dim, dataset.num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    log.info('Starting')

    for e in range(args.epochs):
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
        test_acc = _validate_model(test_loader, model, device)
        log.info('Epoch {:4d} | Train loss: {:.2f} | Acc: {:.2f} | Test loss: {:.2f}'.format(e, np.mean(losses), train_acc,  test_acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_fn', type=str, default= './data/poker/poker-hand-training-true.data')
    parser.add_argument('--train_pct', type=float, default=0.8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--hid_dim', type=int, default=32)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--cuda', action='store_true', default=False)

    args = parser.parse_args()
    main(args)
