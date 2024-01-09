import bisect
from typing import List
from functools import lru_cache

import numpy as np
from torch.utils.data.dataloader import default_collate

from . import FairseqDataset, data_utils


class ConcatWeightedDataset(FairseqDataset):
    """
    Explanation of the algorithm:

    **Input:**

    - Number of datapoints in each subset: $c_1, c_2, c_3, \dots, c_n$

    - The expected weight of each subset: $w_1, w_2, w_3, \dots, w_n$

    **Output:**

    - The number of datapoints sampled for each subset $c'_1, c'_2,c'_3,\dots,c'_n$

    **Objective:**

    Maximize $\sum_j c_j'$

    **Constraint:**

    For each $i$, $c_i' = w_i \cdot \sum_j c_j'$ and $c_i' \le c_i$

    **Solution**

    Let $Z = \sum_j c_j'$ then $c_i' = w_i Z$

    $w_i Z \le c_i \implies Z \le c_i / w_i$

    To maximize $Z$, $Z = \min_j\{c_j / w_j\}$

    and $c_i' = w_i Z$
    """

    @staticmethod
    def cumsum(arr):
        r, s = [], 0
        for e in arr:
            r.append(e + s)
            s += e
        return r

    def __init__(
        self,
        datasets: List[FairseqDataset],
        dataset_weights: List[float],
        dataset_sizes: List[np.ndarray],
        seed: int,
        shuffle: bool,
        min_size: int = 0
    ):
        super(ConcatWeightedDataset, self).__init__()
        assert len(datasets) > 0, "datasets should not be an empty iterable"
        self.datasets = list(datasets)
        assert abs(sum(dataset_weights) - 1.0) < 1e-8
        self.dataset_weights = dataset_weights
        z = min(len(d) / w for d, w in zip(self.datasets, dataset_weights))
        self.sampled_n = [int(z * w) for w in self.dataset_weights]
        assert all(sn <= len(d) for sn, d in zip(self.sampled_n, self.datasets))
        self.cumulative_sizes = self.cumsum(self.sampled_n)
        assert self.cumulative_sizes[-1] <= z
        self.dataset_sizes = dataset_sizes
        self.real_sizes = [len(d) for d in self.datasets]
        self.seed = seed
        self.shuffle = shuffle
        self.min_size = min_size
        self.epoch = 0

        self.perm: List[np.ndarray] = [None for _ in self.datasets]
        with data_utils.numpy_seed(seed):
            for i, (d, sn) in enumerate(zip(self.datasets, self.sampled_n)):
                if len(d) > sn:
                    self.perm[i] = np.random.permutation(len(d))

        self.sizes_for_filter = None

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
        if self.real_sizes[dataset_idx] > self.sampled_n[dataset_idx]:
            sample_idx = self.epoch * self.sampled_n[dataset_idx] + sample_idx
            sample_idx = self.perm[dataset_idx][sample_idx % self.real_sizes[dataset_idx]]
        return dataset_idx, sample_idx

    def collater(self, samples, **extra_args):
        # For now only supports datasets with same underlying collater implementations
        if hasattr(self.datasets[0], "collater"):
            return self.datasets[0].collater(samples, **extra_args)
        else:
            return default_collate(samples, **extra_args)

    def size(self, idx: int):
        dataset_idx, sample_idx = self._get_dataset_and_sample_index(self.seed, self.epoch, idx)
        return self.datasets[dataset_idx].size(sample_idx)

    def num_tokens(self, idx: int):
        dataset_idx, sample_idx = self._get_dataset_and_sample_index(self.seed, self.epoch, idx)
        return self.datasets[dataset_idx].num_tokens(sample_idx)

    def num_tokens_vec(self, indices):
        return self.sizes_for_filter[indices]

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
        return indices[np.argsort(self.sizes_for_filter[indices], kind="mergesort")]

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

        sizes = []
        for i, (d, dsizes, sn) in enumerate(zip(self.datasets, self.dataset_sizes, self.sampled_n)):
            if len(d) > sn:
                indices = np.arange(epoch * sn, (epoch + 1) * sn)
                indices = self.perm[i][indices % len(d)]
                sizes.append(dsizes[indices])
            else:
                sizes.append(dsizes)
        self.sizes_for_filter = np.concatenate(sizes)
        assert self.sizes_for_filter.shape[0] == len(self)

        for ds in self.datasets:
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)

    def filter_indices_by_size(self, indices, max_sizes):
        assert isinstance(max_sizes, tuple) and len(max_sizes) == 2 and max_sizes[0] == max_sizes[1]
        indices_mask = (
            (self.sizes_for_filter[indices] <= max_sizes[0])
            & (self.sizes_for_filter[indices] >= self.min_size)
        )
        ignored = indices[~indices_mask].tolist()
        indices = indices[indices_mask]
        return indices, ignored
