import numpy as np
from torch.utils.data.dataloader import default_collate

from . import FairseqDataset


class TensorListDataset(FairseqDataset):
    def __init__(self, seqs):
        self.seqs = seqs
        self._sizes = np.array([len(seq) for seq in seqs], dtype=np.long)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx]

    @property
    def sizes(self):
        return self._sizes

    def num_tokens(self, index):
        return self._sizes[index]

    def size(self, index):
        return self._sizes[index]

    def collater(self, samples):
        return default_collate(samples)

    @property
    def supports_prefetch(self):
        return False
