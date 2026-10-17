"""第五阶段树模型训练、调参和可重载端到端推理管线。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterSampler, StratifiedKFold

from src.data_processing.cleaning import CleaningRules
from src.data_processing.schema import FEATURES
from src.evaluation.metrics import binary_metrics, select_f1_threshold
from src.feature_engineering.features import CANDIDATE_FEATURES, FeatureEngineer


@dataclass(frozen=True)
class TreeTrainingConfig:
    folds: int = 5
    candidates_per_model: int = 16
    random_state: int = 42
    xgb_early_stopping_rounds: int = 50
    xgb_max_estimators: int = 1500


@dataclass
class TreeModelBundle:
    """从原始十字段到 PD 的完整树模型推理对象。"""

    model_name: str
    estimator: Any
    imputer: SimpleImputer
    cleaning_rules: CleaningRules
    feature_engineer: FeatureEngineer
    feature_names: tuple[str, ...]
    threshold: float
    training_metadata: dict[str, Any] = field(default_factory=dict)

    def _validate_raw(self, raw: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(set(FEATURES).difference(raw.columns))
        if missing:
            raise ValueError(f"客户输入缺少原始字段：{missing}")
        selected = raw.loc[:, list(FEATURES)].copy()
        for name in FEATURES:
            if not pd.api.types.is_numeric_dtype(selected[name]):
                raise ValueError(f"客户字段 {name} 必须为数值型。")
        values = selected.to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError("客户输入存在正无穷或负无穷，无法评分。")
        return selected

    def raw_to_features(self, raw: pd.DataFrame) -> pd.DataFrame:
        selected = self._validate_raw(raw)
        clean = self.cleaning_rules.transform(selected)
        return self.feature_engineer.transform(clean).loc[:, list(self.feature_names)]

    def transform_features(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(set(self.feature_names).difference(features.columns))
        if missing:
            raise ValueError(f"树模型输入缺少字段：{missing}")
        selected = features.loc[:, list(self.feature_names)].copy()
        values = selected.to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError("树模型输入存在正无穷或负无穷。")
        transformed = self.imputer.transform(selected)
        if not np.isfinite(transformed).all():
            raise ValueError("树模型填补后仍存在非有限数值。")
        return pd.DataFrame(transformed, columns=self.feature_names, index=selected.index)

    def predict_pd_features(self, features: pd.DataFrame) -> np.ndarray:
        transformed = self.transform_features(features)
        probability = np.asarray(self.estimator.predict_proba(transformed)[:, 1], dtype=float)
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise RuntimeError("树模型输出了无效的严重逾期概率。")
        return probability

    def predict_pd(self, raw: pd.DataFrame) -> np.ndarray:
        return self.predict_pd_features(self.raw_to_features(raw))

    def predict(self, raw: pd.DataFrame) -> np.ndarray:
        return (self.predict_pd(raw) >= self.threshold).astype(int)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "TreeModelBundle":
        if not path.is_file():
            raise ValueError(f"未找到树模型文件：{path}")
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError("序列化文件不是 TreeModelBundle。")
        return loaded

    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "feature_names": list(self.feature_names),
            "threshold": self.threshold,
            "imputer_statistics": dict(zip(self.feature_names, self.imputer.statistics_.tolist())),
            **self.training_metadata,
        }


RF_SPACE: dict[str, list[Any]] = {
    "n_estimators": [200, 350, 500],
    "max_depth": [6, 10, 14, None],
    "min_samples_leaf": [10, 20, 50, 100],
    "max_features": ["sqrt", 0.5, 0.8],
    "bootstrap": [True],
}

XGB_SPACE: dict[str, list[Any]] = {
    "learning_rate": [0.02, 0.05, 0.10],
    "max_depth": [3, 4, 5, 6],
    "min_child_weight": [1, 5, 10],
    "subsample": [0.70, 0.85, 1.0],
    "colsample_bytree": [0.70, 0.85, 1.0],
    "reg_lambda": [1.0, 5.0, 10.0],
    "reg_alpha": [0.0, 0.1, 1.0],
    "gamma": [0.0, 0.1, 0.5],
}


def _rank_results(results: pd.DataFrame) -> pd.DataFrame:
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
    return ranked.sort_values(["rank_score", "brier", "candidate_id"]).reset_index(drop=True)


def _fit_imputer(X: pd.DataFrame) -> SimpleImputer:
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    imputer.fit(X)
    if not np.isfinite(imputer.statistics_).all():
        raise RuntimeError("训练集无法为全部树模型特征拟合中位数。")
    return imputer


def _rf_oof(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict[str, Any],
    config: TreeTrainingConfig,
) -> tuple[np.ndarray, float]:
    predictions = np.empty(len(X), dtype=float)
    splitter = StratifiedKFold(config.folds, shuffle=True, random_state=config.random_state)
    started = perf_counter()
    for fold_train, fold_holdout in splitter.split(X, y):
        imputer = _fit_imputer(X.iloc[fold_train])
        train_values = imputer.transform(X.iloc[fold_train])
        holdout_values = imputer.transform(X.iloc[fold_holdout])
        estimator = RandomForestClassifier(
            **params,
            class_weight=None,
            random_state=config.random_state,
            n_jobs=-1,
        )
        estimator.fit(train_values, y.iloc[fold_train])
        predictions[fold_holdout] = estimator.predict_proba(holdout_values)[:, 1]
    return predictions, perf_counter() - started


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cleaning_rules: CleaningRules,
    config: TreeTrainingConfig | None = None,
) -> tuple[TreeModelBundle, pd.DataFrame, np.ndarray]:
    cfg = config or TreeTrainingConfig()
    sampled = list(ParameterSampler(RF_SPACE, n_iter=cfg.candidates_per_model, random_state=cfg.random_state))
    rows: list[dict[str, Any]] = []
    oof_by_id: dict[int, np.ndarray] = {}
    for candidate_id, params in enumerate(sampled, start=1):
        oof, elapsed = _rf_oof(X_train, y_train, params, cfg)
        threshold = select_f1_threshold(y_train, oof)
        metrics = binary_metrics(y_train, oof, threshold)
        rows.append({
            "candidate_id": candidate_id,
            "parameters": json.dumps(params, ensure_ascii=False, sort_keys=True),
            "fit_seconds": elapsed,
            **metrics,
        })
        oof_by_id[candidate_id] = oof
    results = _rank_results(pd.DataFrame(rows))
    best = results.iloc[0]
    best_id = int(best["candidate_id"])
    params = sampled[best_id - 1]
    imputer = _fit_imputer(X_train)
    estimator = RandomForestClassifier(
        **params,
        class_weight=None,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    started = perf_counter()
    estimator.fit(pd.DataFrame(imputer.transform(X_train), columns=X_train.columns), y_train)
    final_seconds = perf_counter() - started
    bundle = TreeModelBundle(
        model_name="Random Forest",
        estimator=estimator,
        imputer=imputer,
        cleaning_rules=cleaning_rules,
        feature_engineer=FeatureEngineer(),
        feature_names=tuple(X_train.columns),
        threshold=float(best["threshold"]),
        training_metadata={
            "parameters": params,
            "training_config": asdict(cfg),
            "final_fit_seconds": final_seconds,
            "selection_source": "train_oof_only",
        },
    )
    return bundle, results, oof_by_id[best_id]


def _xgb_classifier(params: dict[str, Any], config: TreeTrainingConfig, *, early_stopping: bool):
    from xgboost import XGBClassifier

    kwargs = {
        **params,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": "cpu",
        "n_jobs": -1,
        "random_state": config.random_state,
        "n_estimators": config.xgb_max_estimators,
    }
    if early_stopping:
        kwargs["early_stopping_rounds"] = config.xgb_early_stopping_rounds
    return XGBClassifier(**kwargs)


def _xgb_oof(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict[str, Any],
    config: TreeTrainingConfig,
) -> tuple[np.ndarray, list[int], float]:
    predictions = np.empty(len(X), dtype=float)
    iterations: list[int] = []
    splitter = StratifiedKFold(config.folds, shuffle=True, random_state=config.random_state)
    started = perf_counter()
    for fold_train, fold_holdout in splitter.split(X, y):
        imputer = _fit_imputer(X.iloc[fold_train])
        train_values = pd.DataFrame(imputer.transform(X.iloc[fold_train]), columns=X.columns)
        holdout_values = pd.DataFrame(imputer.transform(X.iloc[fold_holdout]), columns=X.columns)
        estimator = _xgb_classifier(params, config, early_stopping=True)
        estimator.fit(
            train_values,
            y.iloc[fold_train],
            eval_set=[(holdout_values, y.iloc[fold_holdout])],
            verbose=False,
        )
        predictions[fold_holdout] = estimator.predict_proba(holdout_values)[:, 1]
        best_iteration = getattr(estimator, "best_iteration", config.xgb_max_estimators - 1)
        iterations.append(int(best_iteration) + 1)
    return predictions, iterations, perf_counter() - started


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cleaning_rules: CleaningRules,
    config: TreeTrainingConfig | None = None,
) -> tuple[TreeModelBundle, pd.DataFrame, np.ndarray]:
    cfg = config or TreeTrainingConfig()
    sampled = list(ParameterSampler(XGB_SPACE, n_iter=cfg.candidates_per_model, random_state=cfg.random_state))
    rows: list[dict[str, Any]] = []
    oof_by_id: dict[int, np.ndarray] = {}
    iterations_by_id: dict[int, list[int]] = {}
    for candidate_id, params in enumerate(sampled, start=1):
        oof, iterations, elapsed = _xgb_oof(X_train, y_train, params, cfg)
        threshold = select_f1_threshold(y_train, oof)
        metrics = binary_metrics(y_train, oof, threshold)
        rows.append({
            "candidate_id": candidate_id,
            "parameters": json.dumps(params, ensure_ascii=False, sort_keys=True),
            "best_iterations": json.dumps(iterations),
            "median_best_iteration": int(np.median(iterations)),
            "fit_seconds": elapsed,
            **metrics,
        })
        oof_by_id[candidate_id] = oof
        iterations_by_id[candidate_id] = iterations
    results = _rank_results(pd.DataFrame(rows))
    best = results.iloc[0]
    best_id = int(best["candidate_id"])
    params = sampled[best_id - 1]
    final_estimators = max(1, int(np.median(iterations_by_id[best_id])))
    imputer = _fit_imputer(X_train)
    estimator = _xgb_classifier(params, cfg, early_stopping=False)
    estimator.set_params(n_estimators=final_estimators)
    started = perf_counter()
    estimator.fit(pd.DataFrame(imputer.transform(X_train), columns=X_train.columns), y_train, verbose=False)
    final_seconds = perf_counter() - started
    bundle = TreeModelBundle(
        model_name="XGBoost",
        estimator=estimator,
        imputer=imputer,
        cleaning_rules=cleaning_rules,
        feature_engineer=FeatureEngineer(),
        feature_names=tuple(X_train.columns),
        threshold=float(best["threshold"]),
        training_metadata={
            "parameters": {**params, "n_estimators": final_estimators},
            "fold_best_iterations": iterations_by_id[best_id],
            "training_config": asdict(cfg),
            "final_fit_seconds": final_seconds,
            "selection_source": "train_oof_only",
        },
    )
    return bundle, results, oof_by_id[best_id]
