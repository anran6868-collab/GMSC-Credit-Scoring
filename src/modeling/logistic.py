"""第四阶段 Logistic 候选模型、交叉验证和验证集选择。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.evaluation.metrics import binary_metrics, select_f1_threshold


@dataclass(frozen=True)
class TrainingConfig:
    c_grid: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
    folds: int = 5
    random_state: int = 42
    max_iter: int = 2000
    calibration_slope_min: float = 0.8
    calibration_slope_max: float = 1.2
    calibration_intercept_abs_max: float = 0.10


@dataclass
class LogisticCandidate:
    name: str
    class_weight: str | None
    c_value: float
    features: tuple[str, ...]
    estimator: LogisticRegression
    adjusted_intercept: float
    threshold: float

    @property
    def coefficients(self) -> np.ndarray:
        return self.estimator.coef_[0].astype(float)

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        missing = sorted(set(self.features).difference(X.columns))
        if missing:
            raise ValueError(f"Logistic输入缺少字段：{missing}")
        values = X.loc[:, list(self.features)].to_numpy(dtype=float)
        return values @ self.coefficients + self.adjusted_intercept

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        return expit(self.decision_function(X))


def balanced_prior_adjustment(y) -> float:
    values = np.asarray(y, dtype=int)
    bad = float(np.sum(values == 1))
    good = float(np.sum(values == 0))
    if bad == 0 or good == 0:
        raise ValueError("类别权重先验修正要求标签同时包含0和1。")
    return float(np.log(good / bad))


def _fit_estimator(X, y, c_value: float, class_weight: str | None, max_iter: int) -> tuple[LogisticRegression, float]:
    model = LogisticRegression(
        C=c_value,
        l1_ratio=0.0,
        solver="lbfgs",
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=42,
    )
    model.fit(X, y)
    adjustment = 0.0
    if class_weight == "balanced":
        adjustment = balanced_prior_adjustment(y)
    adjusted_intercept = float(model.intercept_[0] - adjustment)
    return model, adjusted_intercept


def _candidate_oof(
    X: pd.DataFrame,
    y: pd.Series,
    c_value: float,
    class_weight: str | None,
    config: TrainingConfig,
) -> np.ndarray:
    predictions = np.empty(len(X), dtype=float)
    splitter = StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=config.random_state)
    for train_index, holdout_index in splitter.split(X, y):
        model, intercept = _fit_estimator(
            X.iloc[train_index], y.iloc[train_index], c_value, class_weight, config.max_iter
        )
        logits = X.iloc[holdout_index].to_numpy(dtype=float) @ model.coef_[0] + intercept
        predictions[holdout_index] = expit(logits)
    return predictions


def _rank_grid(results: pd.DataFrame) -> pd.DataFrame:
    ranked = results.copy()
    directions = {
        "auc": False,
        "ks": False,
        "pr_auc": False,
        "f1": False,
        "brier": True,
        "log_loss": True,
    }
    rank_columns = []
    for metric, ascending in directions.items():
        column = f"rank_{metric}"
        ranked[column] = ranked[metric].rank(method="average", ascending=ascending)
        rank_columns.append(column)
    ranked["rank_score"] = ranked[rank_columns].mean(axis=1)
    return ranked.sort_values(["rank_score", "brier", "C"], ascending=[True, True, True]).reset_index(drop=True)


def train_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: TrainingConfig | None = None,
) -> tuple[dict[str, LogisticCandidate], pd.DataFrame]:
    cfg = config or TrainingConfig()
    candidates: dict[str, LogisticCandidate] = {}
    all_results = []
    for name, class_weight in (("原始比例", None), ("类别权重", "balanced")):
        scheme_results = []
        thresholds: dict[float, float] = {}
        for c_value in cfg.c_grid:
            oof = _candidate_oof(X_train, y_train, c_value, class_weight, cfg)
            threshold = select_f1_threshold(y_train, oof)
            thresholds[c_value] = threshold
            row = binary_metrics(y_train, oof, threshold)
            row.update({"scheme": name, "class_weight": class_weight or "none", "C": c_value})
            scheme_results.append(row)
        ranked = _rank_grid(pd.DataFrame(scheme_results))
        all_results.append(ranked)
        best = ranked.iloc[0]
        c_value = float(best["C"])
        estimator, intercept = _fit_estimator(X_train, y_train, c_value, class_weight, cfg.max_iter)
        candidates[name] = LogisticCandidate(
            name=name,
            class_weight=class_weight,
            c_value=c_value,
            features=tuple(X_train.columns),
            estimator=estimator,
            adjusted_intercept=intercept,
            threshold=thresholds[c_value],
        )
    return candidates, pd.concat(all_results, ignore_index=True)


def _passes_calibration(row: pd.Series, config: TrainingConfig) -> bool:
    return bool(
        config.calibration_slope_min <= float(row["calibration_slope"]) <= config.calibration_slope_max
        and abs(float(row["calibration_intercept"])) <= config.calibration_intercept_abs_max
    )


def choose_validation_model(
    candidates: dict[str, LogisticCandidate],
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    config: TrainingConfig | None = None,
) -> tuple[LogisticCandidate, pd.DataFrame, dict[str, np.ndarray]]:
    cfg = config or TrainingConfig()
    rows = []
    predictions = {}
    for name, candidate in candidates.items():
        probability = candidate.predict_pd(X_validation)
        predictions[name] = probability
        row = binary_metrics(y_validation, probability, candidate.threshold)
        row.update({"scheme": name, "C": candidate.c_value, "class_weight": candidate.class_weight or "none"})
        rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison["calibration_pass"] = comparison.apply(_passes_calibration, axis=1, config=cfg)
    eligible = comparison.loc[comparison["calibration_pass"]].copy()
    if eligible.empty:
        raise RuntimeError("原始比例与类别权重方案均未通过验证集校准门槛，停止生成正式评分卡。")
    metrics = (("auc", False), ("ks", False), ("pr_auc", False), ("f1", False), ("brier", True), ("log_loss", True))
    wins = {name: 0 for name in eligible["scheme"]}
    for metric, lower_better in metrics:
        best_value = eligible[metric].min() if lower_better else eligible[metric].max()
        for name in eligible.loc[np.isclose(eligible[metric], best_value, rtol=0, atol=1e-12), "scheme"]:
            wins[name] += 1
    eligible["metric_wins"] = eligible["scheme"].map(wins)
    eligible["raw_preference"] = eligible["scheme"].eq("原始比例").astype(int)
    winner = eligible.sort_values(
        ["metric_wins", "brier", "raw_preference"], ascending=[False, True, False]
    ).iloc[0]["scheme"]
    comparison["metric_wins"] = comparison["scheme"].map(wins).fillna(0).astype(int)
    comparison["selected"] = comparison["scheme"].eq(winner)
    return candidates[str(winner)], comparison, predictions
