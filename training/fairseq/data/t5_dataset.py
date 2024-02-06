from functools import lru_cache

import numpy as np
import torch

from fairseq.data import data_utils, Dictionary
from . import BaseWrapperDataset, LRUCacheDataset


def generate_masks(item, mask_prob, empty_prob, mask_multiple_length):
    sz = len(item)

    # decide elements to mask
    mask = np.full(sz, False)
    num_mask = int(
        # probabilistic rounding
        mask_prob * (1.0 - empty_prob) * sz / ((1.0 + mask_multiple_length) * 0.5) + np.random.rand()
    )
    mask_idx = np.random.choice(sz, num_mask, replace=False)
    mask[mask_idx] = True

    for i in range(0, mask_multiple_length - 1):
        num_mask = int(
            num_mask * (1.0 - (1.0 / (mask_multiple_length - i))) + np.random.rand()
        )
        mask_idx = np.random.choice(mask_idx, num_mask, replace=False)
        mask_idx = (mask_idx + 1) % sz
        mask[mask_idx] = True

    if empty_prob > 0:
        num_empty = max(0, int(
            mask_prob * sz - np.count_nonzero(mask) + np.random.rand()
        ))
        num_empty = min(num_empty, int(mask_prob * empty_prob * sz + 1))
        cannot_empty = np.concatenate([
            mask[1:2] | mask[:1],
            mask[2:] | mask[1:-1] | mask[:-2],
            mask[-2:-1] | mask[-1:]
        ])
        empty_idx = np.nonzero(~cannot_empty)[0]
        empty_idx = np.random.choice(empty_idx, min(num_empty, len(empty_idx)), replace=False)
    else:
        empty_idx = None

    return mask, empty_idx


class T5Dataset(BaseWrapperDataset):

    @classmethod
    def apply_mask(cls, dataset: torch.utils.data.Dataset, *args, **kwargs):
        dataset = LRUCacheDataset(dataset)
        return (
            LRUCacheDataset(cls(dataset, *args, **kwargs, return_target=False)),
            LRUCacheDataset(cls(dataset, *args, **kwargs, return_target=True)),
        )

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        vocab: Dictionary,
        pad_idx: int,
        sentinel_idx_begin: int,
        return_target: bool = False,
        seed: int = 1,
        mask_prob: float = 0.15,
        empty_prob: float = 0.1,
        mask_multiple_length: int = 1,
        max_positions: int = 1e6,
    ):
        assert 0.0 < mask_prob < 1.0

        self.dataset = dataset
        self.vocab = vocab
        self.pad_idx = pad_idx
        self.sentinel_idx_begin = sentinel_idx_begin
        self.return_target = return_target
        self.seed = seed
        self.mask_prob = mask_prob
        self.empty_prob = empty_prob
        self.mask_multiple_length = mask_multiple_length
        self.max_positions = max_positions
        self.epoch = 0

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.seed, self.epoch, index)

    @lru_cache(maxsize=8)
    def __getitem_cached__(self, seed: int, epoch: int, index: int):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            item = self.dataset[index]

            assert (item < self.sentinel_idx_begin).all(), \
                f'Dataset contains index larger than sentinel_idx_begin {self.sentinel_idx_begin}'

            mask, empty_idx = generate_masks(item, self.mask_prob, self.empty_prob, self.mask_multiple_length)

            mask_init_idx = np.concatenate([mask[:1], mask[1:] & ~mask[:-1]])
            if empty_idx is not None:
                mask_init_idx[empty_idx] = True
            mask_init_idx = mask_init_idx.nonzero()[0]
            new_item = np.copy(item)
            item_with_sentinel = np.insert(
                new_item, mask_init_idx,
                np.arange(len(mask_init_idx)) + self.sentinel_idx_begin + 1
            )

            if self.return_target:
                mask_with_sentinel_1 = np.insert(mask, mask_init_idx, True)
                result = torch.from_numpy(item_with_sentinel[mask_with_sentinel_1])
                if result.size(0) > self.max_positions - 1:
                    result = result[:self.max_positions]
                return torch.cat([result, item.new([self.sentinel_idx_begin])])
            else:
                mask_with_sentinel_0 = np.insert(mask, mask_init_idx, False)
                result = torch.from_numpy(item_with_sentinel[~mask_with_sentinel_0])
                if result.size(0) > self.max_positions:
                    result = result[:self.max_positions]
                return result
