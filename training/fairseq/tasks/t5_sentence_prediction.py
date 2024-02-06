import json
import logging
import os
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional, List

import editdistance
import numpy as np
from omegaconf import MISSING, II

from fairseq import glue_utils
from fairseq.data import (
    ConcatSentencesDataset,
    ConstantDataset,
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    OffsetTokensDataset,
    PrependTokenDataset,
    RightPadDataset,
    PadShiftDataset,
    SortDataset,
    StripTokenDataset,
    TableLookupDataset,
    TensorListDataset,
    data_utils, RawLabelDataset,
)
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.dataclass import ChoiceEnum
from fairseq.tasks import FairseqDataclass, FairseqTask, register_task

logger = logging.getLogger(__name__)
SHORTEN_METHOD_CHOICES = ChoiceEnum(["none", "truncate", "random_crop"])


@dataclass
class T5SentencePredictionConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})

    no_shuffle: bool = field(
        default=False,
    )
    shorten_method: SHORTEN_METHOD_CHOICES = field(
        default="none",
        metadata={
            "help": "if not none, shorten sequences that exceed tokens_per_sample"
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

    tokens_per_sample: int = field(
        default=1024,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )

    # prompt
    init_token: Optional[int] = field(
        default=None,
        metadata={"help": "add token at the beginning of each batch item"},
    )
    separator_token: Optional[int] = field(
        default=None,
        metadata={"help": "add separator token between inputs"},
    )
    prefix_0: Optional[str] = field(
        default=None,
        metadata={"help": "prefix to prepend before input0"}
    )
    prefix_1: Optional[str] = field(
        default=None,
        metadata={"help": "prefix to prepend before input1"}
    )

    # classification
    label_lookup_table: List[str] = field(
        default_factory=list,
        metadata={"help": "a list that maps label ids to strings (prompt)"}
    )
    label_lookup_table_submit: List[str] = field(
        default_factory=list,
        metadata={"help": "a list that maps label ids to strings (submission)"}
    )
    binary_classification: bool = field(
        default=False,
        metadata={
            "help": "compute metrics for binary classification"
        }
    )

    # regression
    regression_target: bool = field(
        default=False,
        metadata={
            "help": "use regression target"
        }
    )
    num_classes: int = field(
        default=-1,
        metadata={"help": "number of classes or regression targets"},
    )
    regression_quant_mul: Optional[float] = field(
        default=None,
        metadata={"help": "quantization multiplier for regression targets"}
    )
    regression_quant_frac: Optional[int] = field(
        default=None,
        metadata={"help": "quantization fraction digits for regression targets"}
    )
    regression_fallback: Optional[float] = field(
        default=float("nan"),
        metadata={"help": "Fallback value when regression prediction is nan"}
    )

    # eval
    eval_generate: bool = field(
        default=False,
        metadata={
            "help": "evaluation generated sequence with metrics (acc/f1/mcc/corr)"
        }
    )
    eval_generate_args: str = field(
        default="{}",
        metadata={
            "help": "args for generation as a JSON string"
        }
    )
    eval_generate_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )


@register_task("t5_sentence_prediction", dataclass=T5SentencePredictionConfig)
class T5SentencePredictionTask(FairseqTask):
    cfg: T5SentencePredictionConfig
    dictionary: Dictionary

    def __init__(self, cfg, data_dictionary, label_dictionary):
        super().__init__(cfg)
        self.dictionary = data_dictionary
        self._label_dictionary = label_dictionary
        self.tokenizer = None
        self.bpe = None

    @classmethod
    def setup_task(cls, cfg: T5SentencePredictionConfig, **kwargs):
        assert cfg.num_classes > 0, "Must set task.num_classes"

        # load data dictionary
        data_dict = cls.load_dictionary(
            os.path.join(cfg.data, "input0", "dict.txt"),
        )
        # add mask token
        data_dict.add_symbol('<sen000>')
        for i in range(1, cfg.tokens_per_sample):
            data_dict.add_symbol(f'<sen{i:03d}>')
        logger.info("[input] dictionary: {} types".format(len(data_dict)))

        # load label dictionary
        if not cfg.regression_target:
            label_dict = cls.load_dictionary(
                os.path.join(cfg.data, "label", "dict.txt"),
            )
            logger.info("[label] dictionary: {} types".format(len(label_dict)))
        else:
            label_dict = data_dict
        return cls(cfg, data_dict, label_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        def get_path(key, split):
            return os.path.join(self.cfg.data, key, split)

        def make_dataset(key, dictionary):
            split_path = get_path(key, split)

            try:
                dataset = data_utils.load_indexed_dataset(
                    split_path,
                    dictionary,
                    combine=combine,
                )
            except Exception as e:
                if "StorageException: [404] Path not found" in str(e):
                    logger.warning(f"dataset {e} not found")
                    dataset = None
                else:
                    raise e
            return dataset

        input0 = make_dataset("input0", self.source_dictionary)
        assert input0 is not None, "could not find dataset: {}".format(
            get_path("input0", split)
        )
        input1 = make_dataset("input1", self.source_dictionary)

        if self.cfg.prefix_0 is not None:
            prefix_0 = self.bpe.encode(self.cfg.prefix_0)
            prefix_0 = self.source_dictionary.encode_line(prefix_0, add_if_not_exist=False, append_eos=False)
            input0 = ConcatSentencesDataset(ConstantDataset(prefix_0.tolist(), len(input0)), input0)
        if self.cfg.prefix_1 is not None:
            assert input1 is not None
            prefix_1 = self.bpe.encode(self.cfg.prefix_1)
            prefix_1 = self.source_dictionary.encode_line(prefix_1, add_if_not_exist=False, append_eos=False)
            input1 = ConcatSentencesDataset(ConstantDataset(prefix_1.tolist(), len(input1)), input1)

        if self.cfg.init_token is not None:
            input0 = PrependTokenDataset(input0, self.cfg.init_token)

        if input1 is None:
            src_tokens = input0
        else:
            if self.cfg.separator_token is not None:
                input1 = PrependTokenDataset(input1, self.cfg.separator_token)

            src_tokens = ConcatSentencesDataset(input0, input1)

        if split.startswith("train"):
            with data_utils.numpy_seed(self.cfg.seed):
                shuffle = np.random.permutation(len(src_tokens))

        src_tokens = maybe_shorten_dataset(
            src_tokens,
            split,
            self.cfg.shorten_data_split_list,
            self.cfg.shorten_method,
            self.cfg.tokens_per_sample,
            self.cfg.seed,
        )

        label_dataset, tgt_dataset = None, None
        if not self.cfg.regression_target:
            label_dataset = make_dataset("label", self.label_dictionary)
            if label_dataset is not None:
                label_dataset = OffsetTokensDataset(
                    StripTokenDataset(
                        label_dataset,
                        id_to_strip=self.label_dictionary.eos(),
                    ),
                    offset=-self.label_dictionary.nspecial,
                )
                label_token_ids_lookup_table = [
                    self.source_dictionary.encode_line(
                        self.bpe.encode(s), add_if_not_exist=False, append_eos=True
                    ).tolist()
                    for s in self.cfg.label_lookup_table
                ]
                logger.info("Label token ids: " + repr(label_token_ids_lookup_table))
                tgt_dataset = TableLookupDataset(
                    label_dataset, label_token_ids_lookup_table
                )
                tgt_dataset = RightPadDataset(
                    tgt_dataset,
                    pad_idx=self.source_dictionary.pad(),
                )
        else:
            label_path = "{0}.label".format(get_path("label", split))
            if os.path.exists(label_path):
                def parse_regression_target(i, line):
                    values = line.split()
                    assert (
                        len(values) == self.cfg.num_classes
                    ), f'expected num_classes={self.cfg.num_classes} regression target values on line {i}, found: "{line}"'
                    return [float(x) for x in values]

                def label_to_str(parsed):
                    return " ".join([("{:." + str(self.cfg.regression_quant_frac) + "f}").format(
                        round(float(x) * self.cfg.regression_quant_mul) / self.cfg.regression_quant_mul
                    ) for x in parsed])

                with open(label_path) as h:
                    parsed_labels = [
                        parse_regression_target(i, line.strip())
                        for i, line in enumerate(h.readlines())
                    ]
                tgt_dataset = TensorListDataset([
                    self.source_dictionary.encode_line(
                        self.bpe.encode(label_to_str(label)),
                        add_if_not_exist=False, append_eos=True
                    ).long()
                    for label in parsed_labels
                ])
                label_dataset = RawLabelDataset(parsed_labels)
                assert len(tgt_dataset) == len(label_dataset) == len(src_tokens)
                tgt_dataset = RightPadDataset(
                    tgt_dataset,
                    pad_idx=self.source_dictionary.pad(),
                )

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": RightPadDataset(
                    src_tokens,
                    pad_idx=self.source_dictionary.pad(),
                ),
                "src_lengths": NumelDataset(src_tokens, reduce=False),
            },
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens, reduce=True)
        }
        if tgt_dataset is not None:
            dataset["target"] = tgt_dataset
            dataset["net_input"]["prev_output_tokens"] = PadShiftDataset(
                tgt_dataset,
                pad_idx=self.source_dictionary.pad(),
                start_idx=self.source_dictionary.eos()
            )
            dataset["sample_size"] = NumelDataset(tgt_dataset, reduce=True)
        if label_dataset is not None:
            dataset["label"] = label_dataset

        dataset = NestedDictionaryDataset(
            dataset,
            sizes=[src_tokens.sizes]
        )

        if split.startswith("train") and not self.cfg.no_shuffle:
            dataset = SortDataset(
                dataset,
                sort_order=[shuffle],
            )

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))
        self.datasets[split] = dataset
        return self.datasets[split]

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": RightPadDataset(
                    src_tokens,
                    pad_idx=self.source_dictionary.pad(),
                ),
                "src_lengths": NumelDataset(src_tokens, reduce=False),
            },
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens, reduce=True)
        }
        return NestedDictionaryDataset(
            dataset,
            sizes=[src_tokens.sizes]
        )

    def build_model(self, cfg: FairseqDataclass, from_checkpoint=False):
        model = super().build_model(cfg, from_checkpoint)
        if self.cfg.eval_generate:
            gen_args = json.loads(self.cfg.eval_generate_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.cfg.eval_generate:
            preds = self.inference(self.sequence_generator, sample, model)
            preds = sample["label"].new_tensor(preds)
            glue_utils.compute_eval_metrics(
                logging_output, preds, sample["label"],
                self.cfg.regression_target, self.cfg.binary_classification
            )
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.cfg.eval_generate:
            glue_utils.reduce_metrics(logging_outputs)

    def logging_outputs_can_be_summed(self, criterion) -> bool:
        return not self.cfg.regression_target

    def max_positions(self):
        return self.cfg.tokens_per_sample

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    @property
    def label_dictionary(self):
        return self._label_dictionary

    def inference(self, generator, sample, model, for_submit=False):
        def decode(toks):
            s = self.dictionary.string(toks.int().cpu())
            if self.bpe:
                s = self.bpe.decode(s)
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        pred_strs, pred_labels = [], []
        for i in range(len(gen_out)):
            pred_str = decode(gen_out[i][0]["tokens"])
            pred_strs.append(pred_str)
            if not self.cfg.regression_target:
                min_ed, pred_label = 1e6, None
                for j in range(len(self.cfg.label_lookup_table)):
                    ed = editdistance.eval(pred_str, self.cfg.label_lookup_table[j])
                    if ed < min_ed:
                        min_ed = ed
                        pred_label = j
                pred_labels.append(pred_label)
            else:
                try:
                    pred_label = float(pred_str)
                except ValueError:
                    try:
                        pred_str_split = pred_str.strip().split()
                        if len(pred_str_split) == 0:
                            pred_label = self.cfg.regression_fallback
                        else:
                            pred_label = float(pred_str_split[0])
                    except ValueError:
                        pred_label = self.cfg.regression_fallback
                pred_labels.append(pred_label)
        if self.cfg.eval_generate_print_samples:
            logger.info("example prediction string: " + pred_strs[0])
            logger.info("example prediction label: " + repr(pred_labels[0]))
            if "label" in sample:
                logger.info("example groundtruth label: " + repr(sample["label"][0]))
        if for_submit and not self.cfg.regression_target:
            pred_labels = [self.cfg.label_lookup_table_submit[l] for l in pred_labels]
        return pred_labels
