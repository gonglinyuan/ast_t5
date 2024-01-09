from functools import lru_cache

import numpy as np
import torch

import fairseq.ast_t5_dataset_fast
from fairseq.data import Dictionary, data_utils
from . import BaseWrapperDataset, LRUCacheDataset
from .t5_dataset import generate_masks as t5_generate_masks


@lru_cache(maxsize=8)
def generate_masks(
    item, binarized_ndtypes,
    mask_prob, flat_multiple_len, threshold_lb, threshold_ub, obf_prob, obf_ratio, t2c_prob, t2c_ratio, seed
):
    mask, mask_init_idx, sentinels, span_ndtypes, source, target = fairseq.ast_t5_dataset_fast.generate_masks(
        item.long().numpy(),
        binarized_ndtypes.long().numpy(),
        mask_prob,
        flat_multiple_len,
        threshold_lb,
        threshold_ub,
        obf_prob,
        obf_ratio,
        t2c_prob,
        t2c_ratio,
        seed
    )
    return (
        mask.astype(np.bool),
        np.array(mask_init_idx, dtype=np.int64),
        np.array(sentinels, dtype=np.int64),
        np.array(span_ndtypes, dtype=np.int64),
        source,
        target
    )


class AstT5Dataset(BaseWrapperDataset):

    @classmethod
    def apply_mask(cls, dataset: torch.utils.data.Dataset, ndtype_dataset: torch.utils.data.Dataset, *args, **kwargs):
        dataset = LRUCacheDataset(dataset)
        ndtype_dataset = LRUCacheDataset(ndtype_dataset)
        return (
            LRUCacheDataset(cls(dataset, ndtype_dataset, *args, **kwargs, return_target=False)),
            LRUCacheDataset(cls(dataset, ndtype_dataset, *args, **kwargs, return_target=True)),
        )

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        ndtype_dataset: torch.utils.data.Dataset,
        ndtype_vocab: Dictionary,
        sentinel_idx_begin: int,
        return_target: bool = False,
        seed: int = 1,
        mask_prob: float = 0.15,
        mask_multiple_length: int = 1,
        threshold_lb: int = 5,
        threshold_ub: int = 50,
        obf_prob: float = 0.0,
        obf_ratio: float = 0.0,
        t2c_prob: float = 0.0,
        t2c_ratio: float = 0.0,
        max_positions: int = 1e6,
    ):
        assert 0.0 < mask_prob < 1.0

        self.dataset = dataset
        self.ndtype_dataset = ndtype_dataset
        self.ndtype_vocab = ndtype_vocab
        self.sentinel_idx_begin = sentinel_idx_begin
        self.return_target = return_target
        self.seed = seed
        self.mask_prob = mask_prob
        self.flat_multiple_len = mask_multiple_length
        self.threshold_lb = threshold_lb
        self.threshold_ub = threshold_ub
        self.obf_prob = obf_prob
        self.obf_ratio = obf_ratio
        self.t2c_prob = t2c_prob
        self.t2c_ratio = t2c_ratio
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
        if not fairseq.ast_t5_dataset_fast.configured:
            fairseq.ast_t5_dataset_fast.configure(
                self.ndtype_vocab.index("("),
                self.ndtype_vocab.index(")"),
                self.ndtype_vocab.index("p000000"),
                self.ndtype_vocab.index("_python_function_definition"),
                self.ndtype_vocab.index("_python_block"),
                self.ndtype_vocab.index("_python_expression_statement"),
                self.ndtype_vocab.index("_python_string"),
                self.ndtype_vocab.index("_python_function_body"),
                [self.ndtype_vocab.index(f"_{lang}_identifier") for lang in ["python", "java", "cpp", "csharp"]],
                self.sentinel_idx_begin,
                self.max_positions,
                self.ndtype_vocab.index(f"p{self.max_positions - 1:06d}") + 1
            )
        with data_utils.numpy_seed(seed, epoch, index):
            item = self.dataset[index]
            binarized_ndtypes = self.ndtype_dataset[index]

            assert (item < self.sentinel_idx_begin).all(), \
                f'Dataset contains index larger than sentinel_idx_begin {self.sentinel_idx_begin}'

            if len(binarized_ndtypes) <= 6000:
                mask, mask_init_idx, sentinels, span_ndtypes, source, target = generate_masks(
                    item,
                    binarized_ndtypes,
                    self.mask_prob,
                    self.flat_multiple_len,
                    self.threshold_lb,
                    self.threshold_ub,
                    self.obf_prob,
                    self.obf_ratio,
                    self.t2c_prob,
                    self.t2c_ratio,
                    seed=np.random.randint(np.iinfo(np.int64).max)
                )
                if self.return_target:
                    result = torch.from_numpy(target)
                else:
                    result = torch.from_numpy(source)
            else:
                mask, _ = t5_generate_masks(
                    item,
                    self.mask_prob,
                    0.0,
                    self.flat_multiple_len
                )
                mask_init_idx = np.concatenate([mask[:1], mask[1:] & ~mask[:-1]])
                mask_init_idx = mask_init_idx.nonzero()[0]
                sentinels = np.arange(len(mask_init_idx))
                new_item = np.copy(item)
                item_with_sentinel = np.insert(
                    new_item, mask_init_idx,
                    sentinels + self.sentinel_idx_begin + 1
                )
                if self.return_target:
                    mask_with_sentinel_1 = np.insert(mask, mask_init_idx, True)
                    result = torch.from_numpy(item_with_sentinel[mask_with_sentinel_1])
                else:
                    mask_with_sentinel_0 = np.insert(mask, mask_init_idx, False)
                    result = torch.from_numpy(item_with_sentinel[~mask_with_sentinel_0])

            if self.return_target:
                if result.size(0) > self.max_positions - 1:
                    result = result[:self.max_positions]
                return torch.cat([result, item.new([self.sentinel_idx_begin])])
            else:
                if result.size(0) > self.max_positions:
                    result = result[:self.max_positions]
                return result
