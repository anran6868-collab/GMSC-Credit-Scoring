"""二分类区分能力、阈值与概率校准评价。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


EPSILON = 1e-12


def _validated(y_true, probability) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p):
        raise ValueError("标签和预测概率必须是一维且长度一致。")
    if not np.isin(y, [0, 1]).all() or np.unique(y).size != 2:
        raise ValueError("标签必须同时包含0和1。")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("预测概率必须为0到1之间的有限数值。")
    return y, p


def ks_statistic(y_true, probability) -> tuple[float, float]:
    y, p = _validated(y_true, probability)
    fpr, tpr, thresholds = roc_curve(y, p)
    differences = tpr - fpr
    index = int(np.argmax(differences))
    return float(differences[index]), float(thresholds[index])


def select_f1_threshold(y_true, probability) -> float:
    """最大化F1；并列时先取Recall更高、再取更低阈值。"""
    y, p = _validated(y_true, probability)
    precision, recall, thresholds = precision_recall_curve(y, p)
    if thresholds.size == 0:
        raise ValueError("无法从预测概率确定分类阈值。")
    denom = precision[:-1] + recall[:-1]
    scores = np.divide(
        2 * precision[:-1] * recall[:-1],
        denom,
        out=np.zeros_like(denom),
        where=denom > 0,
    )
    best = np.flatnonzero(np.isclose(scores, scores.max(), rtol=0, atol=1e-15))
    best_recall = recall[best].max()
    best = best[np.isclose(recall[best], best_recall, rtol=0, atol=1e-15)]
    return float(thresholds[best].min())


def calibration_statistics(y_true, probability) -> dict[str, float]:
    y, p = _validated(y_true, probability)
    clipped = np.clip(p, EPSILON, 1 - EPSILON)
    logit = np.log(clipped / (1 - clipped))
    design = sm.add_constant(logit, has_constant="add")
    result = sm.GLM(y, design, family=sm.families.Binomial()).fit()
    return {
        "calibration_intercept": float(result.params[0]),
        "calibration_slope": float(result.params[1]),
    }


def binary_metrics(y_true, probability, threshold: float) -> dict[str, float]:
    y, p = _validated(y_true, probability)
    predicted = (p >= threshold).astype(int)
    ks, ks_threshold = ks_statistic(y, p)
    result = {
        "auc": float(roc_auc_score(y, p)),
        "ks": ks,
        "ks_threshold": ks_threshold,
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "threshold": float(threshold),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
    }
    result.update(calibration_statistics(y, p))
    return result


def calibration_table(y_true, probability, n_bins: int = 10) -> pd.DataFrame:
    y, p = _validated(y_true, probability)
    ranks = pd.Series(p).rank(method="first")
    bins = pd.qcut(ranks, q=n_bins, labels=False)
    frame = pd.DataFrame({"bin": bins + 1, "target": y, "pd": p})
    return (
        frame.groupby("bin", observed=True)
        .agg(count=("target", "size"), observed_bad_rate=("target", "mean"), mean_pd=("pd", "mean"))
        .reset_index()
    )
