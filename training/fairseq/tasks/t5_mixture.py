import json
import logging
import os
from dataclasses import dataclass, field

import numpy as np
from omegaconf import MISSING, II

from fairseq import utils
from fairseq.data import (
    ConcatWeightedDataset,
    Dictionary,
    IdDataset,
    T5Dataset,
    AstT5Dataset,
    FFRecordDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
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
class T5MixtureConfig(FairseqDataclass):
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
    min_tokens_per_sample: int = field(
        default=0,
        metadata={"help": "min number of tokens per sample for LM dataset"},
    )
    subtree_mask: bool = field(
        default=False,
        metadata={"help": "use subtree masking"},
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
    threshold_lb: int = field(
        default=5,
        metadata={"help": "lower bound of the threshold for subtree masking greedy masking"}
    )
    threshold_ub: int = field(
        default=50,
        metadata={"help": "upper bound of the threshold for subtree masking greedy masking"}
    )
    obf_prob: float = field(
        default=0.0,
        metadata={"help": "probability to apply obfuscation"}
    )
    obf_ratio: float = field(
        default=0.0,
        metadata={"help": "the proportion of obfuscated tokens in all masked tokens"}
    )
    t2c_prob: float = field(
        default=0.0,
        metadata={"help": "probability to do text2code generation"}
    )
    t2c_ratio: float = field(
        default=0.0,
        metadata={"help": "the max proportion of code masked in all masked tokens"}
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


@register_task("t5_mixture", dataclass=T5MixtureConfig)
class T5MixtureTask(FairseqTask):
    cfg: T5MixtureConfig

    def __init__(self, cfg: T5MixtureConfig, dictionary, ndtype_dictionary):
        super().__init__(cfg)
        self.dictionary = dictionary
        self.ndtype_dictionary = ndtype_dictionary

        # add mask token
        self.sentinel_idx_begin = dictionary.add_symbol('<sen000>')
        for i in range(1, cfg.tokens_per_sample):
            dictionary.add_symbol(f'<sen{i:03d}>')

        # add function body
        ndtype_dictionary.add_symbol('_python_function_body')

    @classmethod
    def setup_task(cls, cfg: T5MixtureConfig, **kwargs):
        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))
        ndtype_dictionary = Dictionary.load(os.path.join(paths[0], "dict_ndtype.txt"))
        logger.info("dictionary: {} types".format(len(dictionary)))
        logger.info("ndtype_dictionary: {} types".format(len(ndtype_dictionary)))
        return cls(cfg, dictionary, ndtype_dictionary)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        if split == "train":
            with open(os.path.join(self.cfg.data, "data_weights.json"), "r", encoding="utf-8") as f:
                data_weights = json.load(f)
            subsets, subset_weights, subset_sizes, cur_offset = [], [], [], 0
            for lang, w in data_weights.items():
                subset, subset_size = self._load_single_dataset(
                    os.path.join(self.cfg.data, lang, "train"), split=split, offset=cur_offset, sort_mode="post",
                    ndtype_data_path=os.path.join(self.cfg.data, lang, "ndtypes", "train") if lang != "text" else None
                )
                subsets.append(subset)
                subset_weights.append(w)
                subset_sizes.append(subset_size)
                cur_offset += len(subset)

            self.datasets[split] = ConcatWeightedDataset(
                subsets, subset_weights, subset_sizes,
                seed=self.cfg.seed, shuffle=True, min_size=self.cfg.min_tokens_per_sample
            )
        elif split.startswith("valid_"):
            lang = split[len("valid_"):]
            self.datasets[split] = self._load_single_dataset(
                os.path.join(self.cfg.data, lang, "valid"), split=split, offset=0, sort_mode="pre",
                ndtype_data_path=os.path.join(self.cfg.data, lang, "ndtypes", "valid") if lang != "text" else None
            )
        else:
            raise ValueError(split)

    def _load_single_dataset(self, data_path, split, offset, sort_mode, ndtype_data_path=None):
        dataset = data_utils.load_indexed_dataset(
            data_path, self.source_dictionary, combine=False, )
        if dataset is None:
            raise FileNotFoundError(
                "Dataset not found: {}".format(data_path)
            )

        dataset = maybe_shorten_dataset(
            dataset,
            split,
            self.cfg.shorten_data_split_list,
            self.cfg.shorten_method,
            self.cfg.tokens_per_sample,
            self.cfg.seed,
        )
        logger.info("loaded {} sequences from: {}".format(len(dataset), data_path))

        # create continuous blocks of tokens
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.cfg.tokens_per_sample,
            pad=self.source_dictionary.pad(),
            eos=self.source_dictionary.eos(),
            break_mode=self.cfg.sample_break_mode,
        )
        logger.info("loaded {} blocks from: {}".format(len(dataset), data_path))

        # create masked input and targets
        if self.cfg.subtree_mask and ndtype_data_path is not None:
            if FFRecordDataset is not None:
                ndtype_dataset = FFRecordDataset(ndtype_data_path)
            else:
                ndtype_dataset = data_utils.load_indexed_dataset(
                    ndtype_data_path, self.source_dictionary, combine=False, )
            if ndtype_dataset is None:
                raise FileNotFoundError(
                    "Ndtype Dataset not found: {}".format(ndtype_data_path)
                )
            logger.info("loaded {} ndtype sequences from: {}".format(len(ndtype_dataset), ndtype_data_path))
            assert len(ndtype_dataset) == len(ndtype_dataset)

            src_dataset, tgt_dataset = AstT5Dataset.apply_mask(
                dataset,
                ndtype_dataset,
                ndtype_vocab=self.ndtype_dictionary,
                sentinel_idx_begin=self.sentinel_idx_begin,
                seed=self.cfg.seed,
                mask_prob=self.cfg.mask_prob,
                mask_multiple_length=self.cfg.mask_multiple_length,
                threshold_lb=self.cfg.threshold_lb,
                threshold_ub=self.cfg.threshold_ub,
                obf_prob=self.cfg.obf_prob,
                obf_ratio=self.cfg.obf_ratio,
                t2c_prob=self.cfg.t2c_prob,
                t2c_ratio=self.cfg.t2c_ratio,
                max_positions=self.cfg.tokens_per_sample
            )
        else:
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

        nested_dictionary_dataset = NestedDictionaryDataset(
            {
                "id": IdDataset(offset),
                "net_input": input_dict,
                "target": target_dataset,
                "nsentences": NumSamplesDataset(),
                "ntokens": NumelDataset(dataset, reduce=True),
                "sample_size": NumelDataset(tgt_dataset, reduce=True),
            },
            sizes=[dataset.sizes],
        )
        if sort_mode == "pre":
            return SortDataset(
                nested_dictionary_dataset, sort_order=[shuffle, dataset.sizes, ], )
        elif sort_mode == "post":
            return nested_dictionary_dataset, dataset.sizes
        else:
            raise ValueError(sort_mode)

    def build_dataset_for_inference(self, src_tokens, src_lengths, sort=True):
        raise NotImplementedError()

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
