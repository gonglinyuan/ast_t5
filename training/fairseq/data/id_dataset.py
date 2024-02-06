# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from . import FairseqDataset


class IdDataset(FairseqDataset):
    def __init__(self, offset=0):
        super(IdDataset, self).__init__()
        self.offset = 0

    def __getitem__(self, index):
        return index + self.offset

    def __len__(self):
        return 0

    def collater(self, samples):
        return torch.tensor(samples)
