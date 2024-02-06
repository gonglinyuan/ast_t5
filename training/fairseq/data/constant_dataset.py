import numpy as np
import torch

from . import FairseqDataset


class ConstantDataset(FairseqDataset):
    def __init__(self, constant, length):
        super().__init__()
        self.constant = torch.tensor(constant, dtype=torch.long)
        self.length = length
        self.sizes = np.full(length, len(constant), dtype=np.long)

    def __getitem__(self, idx):
        return self.constant

    def __len__(self):
        return self.length

    def num_tokens(self, idx):
        return self.sizes[idx]

    def size(self, idx):
        return self.sizes[idx]

    @property
    def supports_prefetch(self):
        return False
