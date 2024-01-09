import logging

import numpy as np
import scipy.stats as stats
import torch

from fairseq import metrics, utils
from fairseq.logging.meters import Meter

logger = logging.getLogger(__name__)


class ArrayListMeter(Meter):
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.array_list = []

    def update(self, val):
        self.array_list.append(val)

    def state_dict(self):
        return {
            "array_list": self.array_list
        }

    def load_state_dict(self, state_dict):
        self.array_list = state_dict["array_list"]

    @property
    def smoothed_value(self) -> float:
        raise RuntimeError(
            "smoothed_value method of a ArrayListMeter should never be called"
            "please prepend an underscore to the key of this meter"
        )


def compute_eval_metrics(logging_output, preds, labels, regression_target=False, binary_classification=False):
    with torch.no_grad():
        preds = preds.view(-1)
        labels = labels.view(-1)
        assert preds.size(0) == labels.size(0)
        if not regression_target:
            logging_output["n_correct"] = (preds == labels).sum()
            if binary_classification:
                logging_output["tp"] = ((preds == 1) & (labels == 1)).long().sum()
                logging_output["fp"] = ((preds == 1) & (labels == 0)).long().sum()
                logging_output["fn"] = ((preds == 0) & (labels == 1)).long().sum()
                logging_output["tn"] = ((preds == 0) & (labels == 0)).long().sum()
                assert (
                           logging_output["tp"] + logging_output["fp"] + logging_output["fn"] + logging_output["tn"]
                       ).item() == labels.size(0), "invalid size"
        else:
            logging_output["x"] = preds.detach().cpu()
            logging_output["y"] = labels.detach().cpu()


def _compute_f1(meters):
    tp, fp, fn, tn = [utils.item(meters[key].sum) for key in ["tp", "fp", "fn", "tn"]]
    tmp = 2 * tp + fp + fn
    return (2 * tp) / tmp if tmp else 0


def _compute_mcc(meters):
    tp, fp, fn, tn = [utils.item(meters[key].sum) for key in ["tp", "fp", "fn", "tn"]]
    tmp = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return (tp * tn - fp * fn) / (tmp ** 0.5) if tmp else 0


def _compute_acc_f1(meters):
    tp, fp, fn, tn = [utils.item(meters[key].sum) for key in ["tp", "fp", "fn", "tn"]]
    n = tp + fp + fn + tn
    acc = (tp + tn) / (tp + fp + fn + tn) if n else 0
    f1 = _compute_f1(meters)
    return (acc + f1) / 2


def _compute_corr(meters, method):
    x = np.concatenate(meters["_x"].array_list)
    y = np.concatenate(meters["_y"].array_list)
    x_mean = np.nanmean(x)
    if np.isnan(x_mean):
        return 0.0
    x = np.nan_to_num(x, nan=x_mean)
    if (x == x[0]).all():
        return 0.0
    if method == 'pearson':
        return stats.pearsonr(x, y)[0]
    elif method == 'spearman':
        return stats.spearmanr(x, y)[0]
    elif method == 'pearson_spearman':
        return (stats.pearsonr(x, y)[0] + stats.spearmanr(x, y)[0]) / 2
    else:
        raise NotImplementedError()


def reduce_metrics(logging_outputs, verbose=False):
    if len(logging_outputs) > 0:
        total = sum(log.get("nsentences", 0) for log in logging_outputs)
        if "n_correct" in logging_outputs[0]:  # classification
            metrics.log_scalar_sum("total", total)
            n_correct = sum(log.get("n_correct", 0) for log in logging_outputs)
            metrics.log_scalar_sum("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    utils.item(meters["n_correct"].sum) * 100.0 / utils.item(meters["total"].sum), 3
                )
                if meters["total"].sum > 0
                else 0,
            )
            if "tp" in logging_outputs[0]:  # binary classification
                tp = sum(log.get("tp", 0) for log in logging_outputs)
                fp = sum(log.get("fp", 0) for log in logging_outputs)
                fn = sum(log.get("fn", 0) for log in logging_outputs)
                tn = sum(log.get("tn", 0) for log in logging_outputs)
                assert tp + fp + fn + tn == total, "invalid size when aggregating"
                if verbose:
                    logger.info(f"tp = {tp}, fp = {fp}, fn = {fn}, tn = {tn}")
                metrics.log_scalar_sum("tp", tp)
                metrics.log_scalar_sum("fp", fp)
                metrics.log_scalar_sum("fn", fn)
                metrics.log_scalar_sum("tn", tn)
                metrics.log_derived("f1", lambda m: round(_compute_f1(m) * 100.0, 3))
                metrics.log_derived("mcc", lambda m: round(_compute_mcc(m) * 100.0, 3))
                metrics.log_derived("acc_f1", lambda m: round(_compute_acc_f1(m) * 100.0, 3))
        elif "x" in logging_outputs[0]:  # regression
            x = torch.cat([log["x"] for log in logging_outputs if "x" in log]).numpy()
            y = torch.cat([log["y"] for log in logging_outputs if "y" in log]).numpy()
            metrics.log_custom(ArrayListMeter, "_x", x)
            metrics.log_custom(ArrayListMeter, "_y", y)
            metrics.log_derived("pearson", lambda m: round(_compute_corr(m, "pearson") * 100.0, 3))
            metrics.log_derived("spearman", lambda m: round(_compute_corr(m, "spearman") * 100.0, 3))
            metrics.log_derived("pearson_spearman", lambda m: round(_compute_corr(m, "pearson_spearman") * 100.0, 3))
