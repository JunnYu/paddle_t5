import sklearn
from scipy.stats import pearsonr,spearmanr
import collections
import numpy as np
from sklearn.metrics import matthews_corrcoef,accuracy_score,f1_score


def accuracy(targets, predictions):
    return {"accuracy": 100 * accuracy_score(targets, predictions)}


def sklearn_metrics_wrapper(
    metric_str, metric_dict_str=None, metric_post_process_fn=None, **metric_fn_kwargs
):
    def fn(targets, predictions):
        if metric_str == "matthews_corrcoef":
            metric_fn = matthews_corrcoef
        else:
            metric_fn = getattr(sklearn.metrics, metric_str)
        metric_val = metric_fn(targets, predictions, **metric_fn_kwargs)
        if metric_post_process_fn is not None:
            metric_val = metric_post_process_fn(metric_val)
        return {metric_dict_str or metric_str: metric_val}

    return fn


def f1_score_with_invalid(targets, predictions):
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    invalid_idx_mask = np.logical_and(predictions != 0, predictions != 1)
    predictions[invalid_idx_mask] = 1 - targets[invalid_idx_mask]
    return {"f1": 100 * f1_score(targets, predictions)}


def pearson_corrcoef(targets, predictions):
    return {"pearson_corrcoef": 100 * pearsonr(targets, predictions)[0]}


def spearman_corrcoef(targets, predictions):
    return {"spearman_corrcoef": 100 * spearmanr(targets, predictions)[0]}


GLUE_METRICS = collections.OrderedDict(
    [
        (
            "cola",
            [
                sklearn_metrics_wrapper(
                    "matthews_corrcoef", metric_post_process_fn=lambda x: 100 * x
                )
            ],
        ),
        ("sst-2", [accuracy]),
        ("mrpc", [f1_score_with_invalid, accuracy]),
        ("sts-b", [pearson_corrcoef, spearman_corrcoef]),
        ("qqp", [f1_score_with_invalid, accuracy]),
        ("mnli", [accuracy]),
        ("qnli", [accuracy]),
        ("rte", [accuracy]),
        ("wnli", [accuracy]),
        ("ax", []),  # Only test set available.
    ]
)
