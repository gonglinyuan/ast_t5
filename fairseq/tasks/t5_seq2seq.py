import itertools
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from omegaconf import II

from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    TruncateDataset,
    data_utils,
    indexed_dataset,
)
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import FairseqTask, register_task

logger = logging.getLogger(__name__)


def load_langpair_dataset(
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    combine,
    dataset_impl,
    upsample_primary,
    max_source_positions,
    max_target_positions,
    truncate_source=False,
    truncate_target=False,
    append_source_id=False,
    shuffle=True,
    pad_to_multiple=1
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError("Dataset not found: {} ({})".format(split, data_path))

        src_dataset = data_utils.load_indexed_dataset(prefix + src, src_dict, dataset_impl)
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())

        if truncate_source:
            src_dataset = TruncateDataset(src_dataset, max_source_positions, src_dict.eos())
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(prefix + tgt, tgt_dict, dataset_impl)
        if tgt_dataset is not None:
            if truncate_target:
                tgt_dataset = TruncateDataset(tgt_dataset, max_target_positions, tgt_dict.eos())
            tgt_datasets.append(tgt_dataset)

        logger.info("{} {} {}-{} {} examples".format(data_path, split_k, src, tgt, len(src_datasets[-1])))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(src_dataset, src_dict.index("[{}]".format(src)))
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(tgt_dataset, tgt_dict.index("[{}]".format(tgt)))
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return LanguagePairDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=False,
        left_pad_target=False,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=0,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple
    )


@dataclass
class T5Seq2seqConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None, metadata={
            "help": "colon separated path to data directories list, will be iterated upon during epochs "
                    "in round-robin manner; however, valid and test data are always in the first directory "
                    "to avoid the need for repeating them in all directories"
        }
    )
    truncate_source: bool = field(default=False, metadata={"help": "truncate source to max-source-positions"})
    truncate_target: bool = field(default=False, metadata={"help": "truncate source to max-target-positions"})
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II("dataset.dataset_impl")
    tokens_per_sample: int = field(default=1024, metadata={"help": "max number of tokens per sample for LM dataset"})
    seed: int = II("common.seed")


@register_task("t5_seq2seq", dataclass=T5Seq2seqConfig)
class T5Seq2seqTask(FairseqTask):
    cfg: T5Seq2seqConfig

    def __init__(self, cfg: T5Seq2seqConfig, dictionary):
        super().__init__(cfg)
        self.dictionary = dictionary

    @classmethod
    def setup_task(cls, cfg: T5Seq2seqConfig, **kwargs):
        # load dictionaries
        dictionary = cls.load_dictionary(os.path.join(cfg.data, "dict.txt"))
        # add mask token
        dictionary.add_symbol('<sen000>')
        for i in range(1, cfg.tokens_per_sample):
            dictionary.add_symbol(f'<sen{i:03d}>')
        logger.info("[{}] dictionary: {} types".format("src", len(dictionary)))

        return cls(cfg, dictionary)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        self.datasets[split] = load_langpair_dataset(
            self.cfg.data,
            split,
            "src",
            self.dictionary,
            "tgt",
            self.dictionary,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=-1,
            max_source_positions=self.cfg.tokens_per_sample,
            max_target_positions=self.cfg.tokens_per_sample,
            truncate_source=self.cfg.truncate_source,
            truncate_target=self.cfg.truncate_target,
            shuffle=(split == "train"),
            pad_to_multiple=8
        )

        return self.datasets[split]

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        raise NotImplementedError()

    def max_positions(self):
        return self.cfg.tokens_per_sample

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary
