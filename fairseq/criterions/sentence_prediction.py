# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from fairseq import glue_utils, metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


@dataclass
class SentencePredictionConfig(FairseqDataclass):
    classification_head_name: str = field(
        default="sentence_classification_head",
        metadata={"help": "name of the classification head to use"},
    )
    regression_target: bool = field(
        default=False,
    )
    binary_classification: bool = field(
        default=False
    )


@register_criterion("sentence_prediction", dataclass=SentencePredictionConfig)
class SentencePredictionCriterion(FairseqCriterion):
    def __init__(self, cfg: SentencePredictionConfig, task):
        super().__init__(task)
        self.classification_head_name = cfg.classification_head_name
        self.regression_target = cfg.regression_target
        self.binary_classification = cfg.binary_classification

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert (
            hasattr(model, "classification_heads")
            and self.classification_head_name in model.classification_heads
        ), "model must provide sentence classification head for --criterion=sentence_prediction"

        logits, _ = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.classification_head_name,
        )
        targets = model.get_targets(sample, [logits]).view(-1)
        sample_size = targets.numel()

        if not self.regression_target:
            lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            task_loss = F.nll_loss(lprobs, targets, reduction="sum")
        else:
            logits = logits.view(-1).float()
            targets = targets.float()
            task_loss = F.mse_loss(logits, targets, reduction="sum")

        loss = task_loss
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample_size,
            "sample_size": sample_size,
        }

        with torch.no_grad():
            preds = (
                logits.argmax(dim=1)
                if not self.regression_target
                else logits
            )
        glue_utils.compute_eval_metrics(
            logging_output, preds, targets,
            self.regression_target, self.binary_classification
        )

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        glue_utils.reduce_metrics(logging_outputs)

    def logging_outputs_can_be_summed(self) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return not self.regression_target
