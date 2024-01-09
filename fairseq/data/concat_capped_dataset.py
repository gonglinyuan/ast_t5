import bisect
from functools import lru_cache

import numpy as np
from torch.utils.data.dataloader import default_collate

from . import FairseqDataset, data_utils


class ConcatCappedDataset(FairseqDataset):
    @staticmethod
    def cumsum(sequence, caps):
        r, s = [], 0
        for e, cap in zip(sequence, caps):
            curr_len = min(len(e), cap)
            r.append(curr_len + s)
            s += curr_len
        return r

    def __init__(self, datasets, dataset_caps, seed, shuffle):
        super(ConcatCappedDataset, self).__init__()
        assert len(datasets) > 0, "datasets should not be an empty iterable"
        self.datasets = list(datasets)
        self.dataset_caps = dataset_caps
        self.cumulative_sizes = self.cumsum(self.datasets, dataset_caps)
        self.real_sizes = [len(d) for d in self.datasets]
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

        self.perm = [None for _ in self.datasets]
        with data_utils.numpy_seed(seed):
            for i, (d, cap) in enumerate(zip(self.datasets, dataset_caps)):
                if len(d) > cap:
                    self.perm[i] = np.random.permutation(len(d))

        self.sizes = None
        self.src_sizes = None
        self.tgt_sizes = None

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self._get_dataset_and_sample_index(self.seed, self.epoch, idx)
        return self.datasets[dataset_idx][sample_idx]

    @lru_cache(maxsize=8)
    def _get_dataset_and_sample_index(self, seed: int, epoch: int, idx: int):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        if self.real_sizes[dataset_idx] > self.dataset_caps[dataset_idx]:
            sample_idx = self.epoch * self.dataset_caps[dataset_idx] + sample_idx
            sample_idx = self.perm[dataset_idx][sample_idx % self.real_sizes[dataset_idx]]
        return dataset_idx, sample_idx

    def collater(self, samples, **extra_args):
        # For now only supports datasets with same underlying collater implementations
        if hasattr(self.datasets[0], "collater"):
            return self.datasets[0].collater(samples, **extra_args)
        else:
            return default_collate(samples, **extra_args)

    def size(self, idx: int):
        return (
            self.src_sizes[idx],
            self.tgt_sizes[idx] if self.tgt_sizes is not None else 0,
        )

    def num_tokens(self, idx: int):
        dataset_idx, sample_idx = self._get_dataset_and_sample_index(self.seed, self.epoch, idx)
        return self.datasets[dataset_idx].num_tokens(sample_idx)

    def num_tokens_vec(self, indices):
        sizes = self.src_sizes[indices]
        if self.tgt_sizes is not None:
            sizes = np.maximum(sizes, self.tgt_sizes[indices])
        return sizes

    def attr(self, attr: str, index: int):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, index)
        return getattr(self.datasets[dataset_idx], attr, None)

    @property
    def supports_prefetch(self):
        return all(d.supports_prefetch for d in self.datasets)

    def ordered_indices(self):
        """
        Returns indices sorted by length. So less padding is needed.
        """
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)

        # sort by target length, then source length
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind="mergesort")]
        return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]

    def prefetch(self, indices):
        to_prefetch = {i: [] for i in range(len(self.datasets))}
        for idx in indices:
            i, j = self._get_dataset_and_sample_index(self.seed, self.epoch, idx)
            to_prefetch[i].append(j)
        for i, j_s in to_prefetch.items():
            if j_s:
                assert getattr(self.datasets[i], "supports_prefetch", False)
                self.datasets[i].prefetch(j_s)

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False  # because the sizes are dynamic across epochs

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self.epoch = epoch

        sizes, src_sizes, tgt_sizes = [], [], []
        for i, (d, cap) in enumerate(zip(self.datasets, self.dataset_caps)):
            if len(d) > cap:
                indices = np.arange(epoch * cap, (epoch + 1) * cap)
                indices = self.perm[i][indices % len(d)]
                sizes.append(d.sizes[indices, :])
                src_sizes.append(d.src_sizes[indices])
                if d.tgt_sizes is None:
                    tgt_sizes.append(np.zeros_like(d.src_sizes[indices]))
                else:
                    tgt_sizes.append(d.tgt_sizes[indices])
            else:
                sizes.append(d.sizes)
                src_sizes.append(d.src_sizes)
                if d.tgt_sizes is None:
                    tgt_sizes.append(np.zeros_like(d.src_sizes))
                else:
                    tgt_sizes.append(d.tgt_sizes)
        self.sizes = np.concatenate(sizes)
        self.src_sizes = np.concatenate(src_sizes)
        self.tgt_sizes = np.concatenate(tgt_sizes)
        assert self.sizes.shape[0] == self.src_sizes.shape[0] == self.tgt_sizes.shape[0] == len(self)

        for ds in self.datasets:
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)

    def filter_indices_by_size(self, indices, max_sizes):
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.
        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)
        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        return data_utils.filter_paired_dataset_indices_by_size(
            self.src_sizes,
            self.tgt_sizes,
            indices,
            max_sizes,
        )