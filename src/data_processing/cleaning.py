"""第二阶段清洗规则：只在训练集拟合，并保留特殊风险信息。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .schema import FEATURES, LATE_FEATURES


SPECIAL_LATE_VALUES = (96, 98)


@dataclass(frozen=True)
class CleaningRules:
    medians: dict[str, float]
    special_late_values: tuple[int, int] = SPECIAL_LATE_VALUES

    @classmethod
    def fit(cls, train: pd.DataFrame) -> "CleaningRules":
        """所有中位数仅从训练集计算。"""
        source = train.loc[:, list(FEATURES)].copy()
        source.loc[source["age"].le(0), "age"] = float("nan")
        for feature in LATE_FEATURES:
            source.loc[source[feature].isin(SPECIAL_LATE_VALUES), feature] = float("nan")
        medians: dict[str, float] = {}
        for feature in FEATURES:
            median = source[feature].median()
            if pd.isna(median):
                raise ValueError(f"训练集字段 {feature} 全部缺失，无法拟合填补规则。")
            medians[feature] = float(median)
        return cls(medians=medians)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """统一处理三组数据；质量标记仅供审计与后续阶段评估。"""
        result = data.copy()
        result["qa_age_invalid"] = result["age"].le(0).astype("int8")
        result.loc[result["age"].le(0), "age"] = float("nan")
        special = pd.Series(False, index=result.index)
        for feature in LATE_FEATURES:
            mask = result[feature].isin(self.special_late_values)
            special |= mask
            result[f"qa_{feature}_special_code"] = result[feature].where(mask, 0).fillna(0).astype("int16")
            result.loc[mask, feature] = float("nan")
        result["qa_late_special"] = special.astype("int8")
        for feature in FEATURES:
            result[f"qa_{feature}_missing"] = result[feature].isna().astype("int8")
            result[feature] = result[feature].fillna(self.medians[feature])
        if result.loc[:, list(FEATURES)].isna().any().any():
            raise ValueError("清洗后仍存在未处理的数值缺失。")
        return result

    def as_dict(self) -> dict[str, object]:
        return {
            "fit_dataset": "train_only",
            "medians": self.medians,
            "special_late_values": list(self.special_late_values),
            "policy": {
                "age_le_zero": "标记后用训练集中位数填补",
                "late_96_98": "标记后用各字段训练集中位数填补",
                "other_missing": "标记后用各字段训练集中位数填补",
                "extreme_ratios": "保留原值，进入第三阶段专业分箱时再评估",
            },
        }
