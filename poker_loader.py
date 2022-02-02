import pdb
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

class PokerDataset(Dataset):
    def __init__(self, fn, nsample=None, mode='binary'):
        self.mode = mode
        self.data = pd.read_csv(fn, header=None)
        if nsample:
            self.data = self.data.sample(n=nsample)
        self._suit_feats = torch.eye(4)
        self._num_feats = torch.eye(13)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        suits = torch.LongTensor([row[i] for i in range(0, 10, 2)]) - 1
        nums = torch.LongTensor([row[i] for i in range(1, 10, 2)]) - 1
        tgt = int(row[10] == 0) if self.mode == 'binary' else row[10]
        #tgt = row[10]
        return suits, nums, tgt

    def __len__(self):
        return len(self.data)

    @property
    def num_classes(self):
        return 2 if self.mode == 'binary' else 10

if __name__ == '__main__':
    fn = './data/poker/poker-hand-training-true.data'
    data = PokerDataset(fn)
    loader = DataLoader(data, batch_size=32)
    batch = next(iter(loader))
    pdb.set_trace()
