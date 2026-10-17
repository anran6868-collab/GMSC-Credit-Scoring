"""第五阶段统一模型比较、预测稳定性和复杂度统计。"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd

from .metrics import binary_metrics


def probability_psi(train_probability, other_probability, n_bins: int = 10) -> tuple[float, pd.DataFrame]:
    train = np.asarray(train_probability, dtype=float)
    other = np.asarray(other_probability, dtype=float)
    if train.ndim != 1 or other.ndim != 1 or not len(train) or not len(other):
        raise ValueError("预测概率 PSI 要求两个非空的一维数组。")
    if not np.isfinite(train).all() or not np.isfinite(other).all():
        raise ValueError("预测概率 PSI 不接受非有限数值。")
    interior = np.unique(np.quantile(train, np.linspace(0, 1, n_bins + 1)[1:-1]))
    edges = np.concatenate(([-np.inf], interior, [np.inf]))
    train_bins = pd.cut(train, edges, include_lowest=True)
    other_bins = pd.cut(other, edges, include_lowest=True)
    categories = train_bins.categories
    left = pd.Series(train_bins).value_counts(sort=False).reindex(categories, fill_value=0).to_numpy(float)
    right = pd.Series(other_bins).value_counts(sort=False).reindex(categories, fill_value=0).to_numpy(float)
    left = np.maximum(left / left.sum(), 1e-6)
    right = np.maximum(right / right.sum(), 1e-6)
    left /= left.sum()
    right /= right.sum()
    contributions = (right - left) * np.log(right / left)
    detail = pd.DataFrame({
        "bin": [str(value) for value in categories],
        "train_share": left,
        "comparison_share": right,
        "psi_contribution": contributions,
    })
    return float(contributions.sum()), detail


def calibration_pass(metrics: dict[str, float]) -> bool:
    return bool(
        0.8 <= metrics["calibration_slope"] <= 1.2
        and abs(metrics["calibration_intercept"]) <= 0.10
    )


def compare_models(
    y_true: pd.Series,
    probabilities: dict[str, np.ndarray],
    thresholds: dict[str, float],
    train_probabilities: dict[str, np.ndarray] | None = None,
    include_ranks: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    psi_tables = []
    for name, probability in probabilities.items():
        metrics = binary_metrics(y_true, probability, thresholds[name])
        psi = np.nan
        if train_probabilities is not None:
            psi, detail = probability_psi(train_probabilities[name], probability)
            detail.insert(0, "model", name)
            psi_tables.append(detail)
        rows.append({"model": name, **metrics, "probability_psi": psi, "calibration_pass": calibration_pass(metrics)})
    comparison = pd.DataFrame(rows)
    if include_ranks:
        directions = {"auc": False, "ks": False, "pr_auc": False, "f1": False, "brier": True, "log_loss": True}
        rank_columns = []
        for metric, ascending in directions.items():
            column = f"rank_{metric}"
            comparison[column] = comparison[metric].rank(method="average", ascending=ascending)
            rank_columns.append(column)
        comparison["rank_score"] = comparison[rank_columns].mean(axis=1)
        comparison = comparison.sort_values(["rank_score", "brier", "model"]).reset_index(drop=True)
    details = pd.concat(psi_tables, ignore_index=True) if psi_tables else pd.DataFrame()
    return comparison, details


def model_complexity(model_name: str, estimator, model_path: Path, predict: Callable[[], np.ndarray]) -> dict[str, object]:
    started = perf_counter()
    probability = predict()
    elapsed = perf_counter() - started
    rows = len(probability)
    result: dict[str, object] = {
        "model": model_name,
        "model_bytes": model_path.stat().st_size if model_path.is_file() else np.nan,
        "inference_rows": rows,
        "inference_ms_per_1000": elapsed * 1000 * 1000 / rows if rows else np.nan,
    }
    if hasattr(estimator, "estimators_"):
        trees = list(estimator.estimators_)
        result.update({
            "tree_count": len(trees),
            "total_nodes": int(sum(tree.tree_.node_count for tree in trees)),
            "total_leaves": int(sum(tree.tree_.n_leaves for tree in trees)),
            "max_depth": int(max(tree.tree_.max_depth for tree in trees)),
        })
    elif hasattr(estimator, "get_booster"):
        booster = estimator.get_booster()
        frame = booster.trees_to_dataframe()
        leaf_mask = frame["Feature"].eq("Leaf")
        result.update({
            "tree_count": int(frame["Tree"].nunique()),
            "total_nodes": int(len(frame)),
            "total_leaves": int(leaf_mask.sum()),
            "max_depth": np.nan,
        })
    return result
