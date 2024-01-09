# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.data import data_utils

from . import BaseWrapperDataset


class PadDataset(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx, left_pad):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad

    def collater(self, samples):
        return data_utils.collate_tokens(samples, self.pad_idx, left_pad=self.left_pad, pad_to_multiple=8)


class LeftPadDataset(PadDataset):
    def __init__(self, dataset, pad_idx):
        super().__init__(dataset, pad_idx, left_pad=True)


class RightPadDataset(PadDataset):
    def __init__(self, dataset, pad_idx):
        super().__init__(dataset, pad_idx, left_pad=False)


class PadShiftDataset(BaseWrapperDataset):

    def __init__(self, dataset, pad_idx, start_idx):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.start_idx = start_idx

    def collater(self, samples):
        return data_utils.collate_tokens(
            samples,
            self.pad_idx,
            eos_idx=self.start_idx,
            left_pad=False,
            move_eos_to_beginning=True,
            pad_to_multiple=8
        )
