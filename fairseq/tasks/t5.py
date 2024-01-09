# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass, field

import numpy as np
from omegaconf import MISSING, II

from fairseq import utils
from fairseq.data import (
    Dictionary,
    IdDataset,
    T5Dataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PrependTokenDataset,
    RightPadDataset,
    PadShiftDataset,
    SortDataset,
    TokenBlockDataset,
    data_utils,
)
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
from .language_modeling import SAMPLE_BREAK_MODE_CHOICES, SHORTEN_METHOD_CHOICES

logger = logging.getLogger(__name__)


@dataclass
class T5Config(FairseqDataclass):
    data: str = field(
        default=MISSING,
        metadata={
            "help": "colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner"
        },
    )
    sample_break_mode: SAMPLE_BREAK_MODE_CHOICES = field(
        default="none",
        metadata={
            "help": 'If omitted or "none", fills each sample with tokens-per-sample '
                    'tokens. If set to "complete", splits samples only at the end '
                    "of sentence, but may include multiple sentences per sample. "
                    '"complete_doc" is similar but respects doc boundaries. '
                    'If set to "eos", includes only one sentence per sample.'
        },
    )
    tokens_per_sample: int = field(
        default=1024,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )
    mask_prob: float = field(
        default=0.15,
        metadata={"help": "probability of replacing a token with mask"},
    )
    empty_prob: float = field(
        default=0.1,
        metadata={"help": "probability of replacing adding a sentinel for an empty span"}
    )
    mask_multiple_length: int = field(
        default=1,
        metadata={"help": "repeat the mask indices multiple times"},
    )
    shorten_method: SHORTEN_METHOD_CHOICES = field(
        default="none",
        metadata={
            "help": "if not none, shorten sequences that exceed --tokens-per-sample"
        },
    )
    shorten_data_split_list: str = field(
        default="",
        metadata={
            "help": "comma-separated list of dataset splits to apply shortening to, "
                    'e.g., "train,valid" (default: all dataset splits)'
        },
    )
    seed: int = II("common.seed")


@register_task("t5", dataclass=T5Config)
class T5Task(FairseqTask):
    cfg: T5Config

    def __init__(self, cfg: T5Config, dictionary):
        super().__init__(cfg)
        self.dictionary = dictionary

        # add mask token
        self.sentinel_idx_begin = dictionary.add_symbol('<sen000>')
        for i in range(1, cfg.tokens_per_sample):
            dictionary.add_symbol(f'<sen{i:03d}>')

    @classmethod
    def setup_task(cls, cfg: T5Config, **kwargs):
        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(cfg, dictionary)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        dataset = data_utils.load_indexed_dataset(
            split_path,
            self.source_dictionary,
            combine=combine,
        )
        if dataset is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path)
            )

        dataset = maybe_shorten_dataset(
            dataset,
            split,
            self.cfg.shorten_data_split_list,
            self.cfg.shorten_method,
            self.cfg.tokens_per_sample - 1,
            self.cfg.seed,
        )

        # create continuous blocks of tokens
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.cfg.tokens_per_sample - 1,  # one less for <s>
            pad=self.source_dictionary.pad(),
            eos=self.source_dictionary.eos(),
            break_mode=self.cfg.sample_break_mode,
        )
        logger.info("loaded {} blocks from: {}".format(len(dataset), split_path))

        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())

        # create masked input and targets
        src_dataset, tgt_dataset = T5Dataset.apply_mask(
            dataset,
            self.source_dictionary,
            pad_idx=self.source_dictionary.pad(),
            sentinel_idx_begin=self.sentinel_idx_begin,
            seed=self.cfg.seed,
            mask_prob=self.cfg.mask_prob,
            empty_prob=self.cfg.empty_prob,
            mask_multiple_length=self.cfg.mask_multiple_length,
            max_positions=self.cfg.tokens_per_sample
        )

        with data_utils.numpy_seed(self.cfg.seed):
            shuffle = np.random.permutation(len(src_dataset))

        target_dataset = RightPadDataset(
            tgt_dataset,
            pad_idx=self.source_dictionary.pad(),
        )

        input_dict = {
            "src_tokens": RightPadDataset(
                src_dataset,
                pad_idx=self.source_dictionary.pad(),
            ),
            "src_lengths": NumelDataset(src_dataset, reduce=False),
            "prev_output_tokens": PadShiftDataset(
                tgt_dataset,
                pad_idx=self.source_dictionary.pad(),
                start_idx=self.sentinel_idx_begin
            ),
        }

        self.datasets[split] = SortDataset(
            NestedDictionaryDataset(
                {
                    "id": IdDataset(),
                    "net_input": input_dict,
                    "target": target_dataset,
                    "nsentences": NumSamplesDataset(),
                    "ntokens": NumelDataset(dataset, reduce=True),
                    "sample_size": NumelDataset(tgt_dataset, reduce=True),
                },
                sizes=[dataset.sizes],
            ),
            sort_order=[
                shuffle,
                dataset.sizes,
            ],
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, sort=True):
        raise NotImplementedError()
        # src_dataset = RightPadDataset(
        #     TokenBlockDataset(
        #         src_tokens,
        #         src_lengths,
        #         self.cfg.tokens_per_sample - 1,  # one less for <s>
        #         pad=self.source_dictionary.pad(),
        #         eos=self.source_dictionary.eos(),
        #         break_mode="eos",
        #     ),
        #     pad_idx=self.source_dictionary.pad(),
        # )
        # src_dataset = PrependTokenDataset(src_dataset, self.source_dictionary.bos())
        # src_dataset = NestedDictionaryDataset(
        #     {
        #         "id": IdDataset(),
        #         "net_input": {
        #             "src_tokens": src_dataset,
        #             "src_lengths": NumelDataset(src_dataset, reduce=False),
        #         },
        #     },
        #     sizes=src_lengths,
        # )
        # if sort:
        #     src_dataset = SortDataset(src_dataset, sort_order=[src_lengths])
        # return src_dataset

    def max_positions(self):
        return self.cfg.tokens_per_sample

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
        prefix_allowed_tokens_fn=None,
    ):
        generator = super().build_generator(models, args, seq_gen_cls, extra_gen_cls_kwargs, prefix_allowed_tokens_fn)
        generator.eos = self.sentinel_idx_begin
        return generator
