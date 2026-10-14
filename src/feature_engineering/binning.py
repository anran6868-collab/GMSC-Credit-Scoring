"""只在训练集拟合的 OptimalBinning 封装。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd

_MPL_CACHE = Path(tempfile.gettempdir()) / "gmsc-matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

from optbinning import OptimalBinning

from .features import CANDIDATE_FEATURES


CATEGORICAL_FEATURES = {
    "severe_delinquency_flag",
    "qa_MonthlyIncome_missing",
    "qa_NumberOfDependents_missing",
    "qa_late_special",
}

ASCENDING_RISK_FEATURES = {
    "RevolvingUtilizationOfUnsecuredLines",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "total_delinquency_count",
    "delinquency_severity_score",
}

DESCENDING_RISK_FEATURES = {
    "age",
    "MonthlyIncome",
    "monthly_income_per_capita",
    "residual_income_per_capita",
}


@dataclass(frozen=True)
class BinningConfig:
    max_n_prebins: int = 20
    min_n_bins: int = 2
    max_n_bins: int = 6
    min_bin_size: float = 0.05
    min_bin_n_event: int = 30
    min_bin_n_nonevent: int = 30
    max_pvalue: float = 0.05
    time_limit: int = 60


class WOEBinningPipeline:
    """逐变量拟合并保存分箱器，避免任何验证/测试标签进入拟合。"""

    def __init__(self, config: BinningConfig | None = None) -> None:
        self.config = config or BinningConfig()
        self.binners: dict[str, OptimalBinning] = {}
        self.fitted_rows: int | None = None

    @staticmethod
    def _trend(feature: str) -> str | None:
        if feature in ASCENDING_RISK_FEATURES:
            return "ascending"
        if feature in DESCENDING_RISK_FEATURES:
            return "descending"
        if feature in CATEGORICAL_FEATURES:
            return None
        return "auto"

    def _make_binner(self, feature: str) -> OptimalBinning:
        cfg = self.config
        params: dict[str, object] = {
            "name": feature,
            "dtype": "categorical" if feature in CATEGORICAL_FEATURES else "numerical",
            "solver": "cp",
            "prebinning_method": "cart",
            "max_n_prebins": cfg.max_n_prebins,
            "min_n_bins": cfg.min_n_bins,
            "max_n_bins": cfg.max_n_bins,
            "min_bin_size": cfg.min_bin_size,
            "min_bin_n_event": cfg.min_bin_n_event,
            "min_bin_n_nonevent": cfg.min_bin_n_nonevent,
            "max_pvalue": cfg.max_pvalue,
            "max_pvalue_policy": "consecutive",
            "time_limit": cfg.time_limit,
        }
        trend = self._trend(feature)
        if trend is not None:
            params["monotonic_trend"] = trend
        return OptimalBinning(**params)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "WOEBinningPipeline":
        missing = sorted(set(CANDIDATE_FEATURES).difference(X_train.columns))
        if missing:
            raise ValueError(f"分箱训练数据缺少字段：{missing}")
        if len(X_train) != len(y_train):
            raise ValueError("分箱训练特征与目标变量行数不一致。")
        y = pd.Series(y_train).reset_index(drop=True)
        if y.isna().any() or not y.isin([0, 1]).all() or y.nunique() != 2:
            raise ValueError("训练目标必须同时包含0和1，且不得缺失。")

        fitted: dict[str, OptimalBinning] = {}
        for feature in CANDIDATE_FEATURES:
            binner = self._make_binner(feature)
            binner.fit(X_train[feature].to_numpy(), y.to_numpy())
            if binner.status != "OPTIMAL":
                raise RuntimeError(
                    f"变量 {feature} 的最优分箱状态为 {binner.status}，"
                    "未达到 OPTIMAL；请人工检查数据与约束。"
                )
            fitted[feature] = binner
        self.binners = fitted
        self.fitted_rows = len(X_train)
        return self

    def _check_fitted(self, X: pd.DataFrame) -> None:
        if tuple(self.binners) != tuple(CANDIDATE_FEATURES):
            raise RuntimeError("分箱器尚未完整拟合。")
        missing = sorted(set(CANDIDATE_FEATURES).difference(X.columns))
        if missing:
            raise ValueError(f"待转换数据缺少字段：{missing}")

    def transform(self, X: pd.DataFrame, metric: str = "woe") -> pd.DataFrame:
        self._check_fitted(X)
        if metric not in {"woe", "bins", "indices"}:
            raise ValueError("转换指标仅支持 woe、bins 或 indices。")
        transformed: dict[str, np.ndarray] = {}
        for feature, binner in self.binners.items():
            kwargs: dict[str, object] = {"metric": metric}
            if metric == "woe":
                kwargs.update(metric_missing="empirical", metric_special="empirical")
            elif metric == "indices":
                kwargs.update(metric_missing=-1, metric_special=-2)
            transformed[feature] = binner.transform(X[feature].to_numpy(), **kwargs)
        result = pd.DataFrame(transformed, index=X.index)
        if metric == "woe" and not np.isfinite(result.to_numpy(dtype=float)).all():
            raise ValueError("WOE 转换结果存在 NaN 或无穷值。")
        return result

    def binning_tables(self) -> pd.DataFrame:
        if not self.binners:
            raise RuntimeError("分箱器尚未拟合。")
        tables = []
        for feature, binner in self.binners.items():
            table = binner.binning_table.build().copy()
            table.insert(0, "table_row", table.index.astype(str))
            table.insert(0, "feature", feature)
            tables.append(table.reset_index(drop=True))
        return pd.concat(tables, ignore_index=True)

    def summary(self) -> pd.DataFrame:
        tables = self.binning_tables()
        rows = []
        for feature, binner in self.binners.items():
            feature_table = tables.loc[tables["feature"].eq(feature)]
            total = feature_table.loc[feature_table["table_row"].eq("Totals")]
            iv = float(total.iloc[0]["IV"]) if len(total) else float(feature_table["IV"].sum())
            regular = feature_table.loc[
                ~feature_table["table_row"].eq("Totals")
                & ~feature_table["Bin"].astype(str).isin(["Special", "Missing"])
            ]
            rows.append({
                "feature": feature,
                "dtype": "categorical" if feature in CATEGORICAL_FEATURES else "numerical",
                "configured_trend": self._trend(feature) or "not_applicable",
                "status": binner.status,
                "n_bins": int(regular["Count"].gt(0).sum()),
                "iv": iv,
                "splits": repr(binner.splits),
            })
        return pd.DataFrame(rows)

    def configuration_table(self) -> pd.DataFrame:
        base = asdict(self.config)
        return pd.DataFrame([
            {
                "feature": feature,
                "dtype": "categorical" if feature in CATEGORICAL_FEATURES else "numerical",
                "monotonic_trend": self._trend(feature) or "not_applicable",
                **base,
            }
            for feature in CANDIDATE_FEATURES
        ])

    def rule_manifest(self) -> dict[str, object]:
        if not self.binners:
            raise RuntimeError("分箱器尚未拟合。")
        return {
            "fitted_rows": self.fitted_rows,
            "config": asdict(self.config),
            "features": {
                feature: {
                    "status": binner.status,
                    "dtype": "categorical" if feature in CATEGORICAL_FEATURES else "numerical",
                    "trend": self._trend(feature),
                    "splits": np.asarray(binner.splits, dtype=object).tolist(),
                }
                for feature, binner in self.binners.items()
            },
        }

    def save(self, path: Path) -> None:
        if not self.binners:
            raise RuntimeError("分箱器尚未拟合，无法保存。")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "WOEBinningPipeline":
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError("序列化文件不是 WOEBinningPipeline。")
        return loaded
