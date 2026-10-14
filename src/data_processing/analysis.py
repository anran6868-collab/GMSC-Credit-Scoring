"""只针对训练集计算第二阶段 EDA 统计，不进行正式特征筛选。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from .schema import FEATURES, LATE_FEATURES, TARGET


@dataclass(frozen=True)
class EDAResults:
    overview: pd.DataFrame
    quality: pd.DataFrame
    anomalies: pd.DataFrame
    group_risks: dict[str, pd.DataFrame]
    correlation: pd.DataFrame
    interactions: dict[str, pd.DataFrame]
    risk_segments: pd.DataFrame
    discrimination: pd.DataFrame


def _labels(values: pd.Series, feature: str) -> pd.Series:
    """EDA 描述性分组；这些边界不会作为第三阶段 WOE 分箱规则。"""
    s = values.astype("float64")
    if feature == "age":
        bins = [-np.inf, 0, 25, 35, 45, 55, 65, np.inf]
        labels = ["异常年龄", "≤25岁", "26–35岁", "36–45岁", "46–55岁", "56–65岁", ">65岁"]
    elif feature == "MonthlyIncome":
        try:
            groups = pd.qcut(s, q=5, duplicates="drop")
            return groups.cat.add_categories(["缺失"]).fillna("缺失")
        except ValueError:
            return s.astype("string").fillna("缺失")
    elif feature == "DebtRatio":
        bins = [-np.inf, 0, 0.25, 0.5, 1, 5, np.inf]
        labels = ["0及以下", "(0,0.25]", "(0.25,0.5]", "(0.5,1]", "(1,5]", ">5"]
    elif feature == "RevolvingUtilizationOfUnsecuredLines":
        bins = [-np.inf, 0, 0.3, 0.7, 1, 5, np.inf]
        labels = ["0及以下", "(0,0.3]", "(0.3,0.7]", "(0.7,1]", "(1,5]", ">5"]
    elif feature in LATE_FEATURES:
        normal = s.mask(s.isin([96, 98]))
        groups = pd.cut(normal, [-np.inf, 0, 1, 2, np.inf], labels=["0次", "1次", "2次", "≥3次"])
        result = groups.cat.add_categories(["96/98特殊值", "缺失"]).fillna("缺失")
        result.loc[s.isin([96, 98])] = "96/98特殊值"
        return result
    elif feature == "NumberOfOpenCreditLinesAndLoans":
        bins = [-np.inf, 0, 3, 7, 15, np.inf]
        labels = ["0条", "1–3条", "4–7条", "8–15条", ">15条"]
    elif feature == "NumberRealEstateLoansOrLines":
        bins = [-np.inf, 0, 1, 2, np.inf]
        labels = ["0条", "1条", "2条", "≥3条"]
    elif feature == "NumberOfDependents":
        bins = [-np.inf, 0, 1, 2, np.inf]
        labels = ["0人", "1人", "2人", "≥3人"]
    else:
        raise ValueError(f"未知字段：{feature}")
    return pd.cut(s, bins=bins, labels=labels).cat.add_categories(["缺失"]).fillna("缺失")


def group_risk(data: pd.DataFrame, feature: str) -> pd.DataFrame:
    """输出每组样本、坏客户及坏率，并标记不稳定小组。"""
    frame = pd.DataFrame({"group": _labels(data[feature], feature), TARGET: data[TARGET]})
    summary = frame.groupby("group", dropna=False, observed=True)[TARGET].agg(sample_count="size", bad_count="sum")
    summary = summary.reset_index()
    summary["group"] = summary["group"].astype(str)
    summary["sample_share"] = summary["sample_count"] / len(data)
    summary["bad_rate"] = summary["bad_count"] / summary["sample_count"]
    summary["risk_lift"] = summary["bad_rate"] / data[TARGET].mean()
    summary["unstable"] = (summary["sample_count"] < 100) | (summary["bad_count"] < 20)
    summary.insert(0, "feature", feature)
    return summary


def _quality(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in FEATURES:
        s = data[feature]
        rows.append({
            "feature": feature,
            "dtype": str(s.dtype),
            "missing_count": int(s.isna().sum()),
            "missing_rate": float(s.isna().mean()),
            "zero_count": int(s.eq(0).sum()),
            "min": float(s.min()),
            "p01": float(s.quantile(0.01)),
            "median": float(s.median()),
            "p99": float(s.quantile(0.99)),
            "max": float(s.max()),
        })
    return pd.DataFrame(rows)


def _anomalies(data: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("年龄为零或负值", data["age"].le(0), "无效年龄，单独标记并用训练集中位数填补"),
        ("逾期字段出现96/98", data.loc[:, list(LATE_FEATURES)].isin([96, 98]).any(axis=1), "特殊编码，保留标记后填补"),
        ("额度使用率大于1", data["RevolvingUtilizationOfUnsecuredLines"].gt(1), "可能有业务含义，第二阶段保留原值"),
        ("债务比率大于1", data["DebtRatio"].gt(1), "结合收入和字段定义解读，第二阶段保留原值"),
        ("月收入为零", data["MonthlyIncome"].eq(0), "与缺失区分，第二阶段保留原值"),
    ]
    return pd.DataFrame([
        {
            "issue": name,
            "row_count": int(mask.sum()),
            "row_rate": float(mask.mean()),
            "bad_rate": float(data.loc[mask, TARGET].mean()) if mask.any() else np.nan,
            "handling": handling,
        }
        for name, mask, handling in checks
    ])


def _discrimination(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in FEATURES:
        score = data[feature].copy()
        if feature == "age":
            score = score.mask(score.le(0))
        if feature in LATE_FEATURES:
            score = score.mask(score.isin([96, 98]))
        valid = score.notna()
        y = data.loc[valid, TARGET]
        if score.loc[valid].nunique() < 2 or y.nunique() < 2:
            auc, ks, direction = np.nan, np.nan, "无法计算"
        else:
            auc_raw = roc_auc_score(y, score.loc[valid])
            direction = "值越大风险越高" if auc_raw >= 0.5 else "值越小风险越高"
            oriented = score.loc[valid] if auc_raw >= 0.5 else -score.loc[valid]
            fpr, tpr, _ = roc_curve(y, oriented)
            auc, ks = max(auc_raw, 1 - auc_raw), float(np.max(tpr - fpr))
        rows.append({
            "feature": feature,
            "valid_count": int(valid.sum()),
            "excluded_count": int((~valid).sum()),
            "single_auc": auc,
            "single_ks": ks,
            "direction": direction,
            "limitation": "仅训练集原始单变量探索；缺失/特殊值单独排除；不等同于入模效果",
        })
    return pd.DataFrame(rows).sort_values("single_auc", ascending=False)


def _interaction(data: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    frame = pd.DataFrame({
        "group_a": _labels(data[left], left),
        "group_b": _labels(data[right], right),
        TARGET: data[TARGET],
    })
    summary = frame.groupby(["group_a", "group_b"], dropna=False, observed=True)[TARGET].agg(
        sample_count="size", bad_count="sum"
    ).reset_index()
    summary["group_a"] = summary["group_a"].astype(str)
    summary["group_b"] = summary["group_b"].astype(str)
    summary["bad_rate"] = summary["bad_count"] / summary["sample_count"]
    summary["unstable"] = (summary["sample_count"] < 100) | (summary["bad_count"] < 20)
    return summary


def _risk_segments(data: pd.DataFrame) -> pd.DataFrame:
    late = data.loc[:, list(LATE_FEATURES)]
    special = late.isin([96, 98]).any(axis=1)
    severe = late[[LATE_FEATURES[1], LATE_FEATURES[2]]].gt(0).any(axis=1)
    mild = late[LATE_FEATURES[0]].gt(0)
    segment = np.select(
        [special, severe, mild],
        ["逾期特殊值", "有60–89或90天逾期", "仅有30–59天逾期"],
        default="三类逾期均为0",
    )
    frame = pd.DataFrame({"segment": segment, TARGET: data[TARGET]})
    result = frame.groupby("segment")[TARGET].agg(sample_count="size", bad_count="sum").reset_index()
    result["sample_share"] = result["sample_count"] / len(data)
    result["bad_rate"] = result["bad_count"] / result["sample_count"]
    result["risk_lift"] = result["bad_rate"] / data[TARGET].mean()
    return result


def analyze_training(train: pd.DataFrame) -> EDAResults:
    """分析仅接受分层后的训练集；调用方不得传入验证/测试集。"""
    overview = pd.DataFrame([{
        "sample_count": len(train),
        "feature_count": len(FEATURES),
        "good_count": int(train[TARGET].eq(0).sum()),
        "bad_count": int(train[TARGET].eq(1).sum()),
        "bad_rate": float(train[TARGET].mean()),
        "duplicate_rows": int(train.drop(columns=["source_row_id"]).duplicated().sum()),
    }])
    correlation_source = train.loc[:, list(FEATURES)].copy()
    correlation_source.loc[correlation_source["age"].le(0), "age"] = np.nan
    for feature in LATE_FEATURES:
        correlation_source.loc[correlation_source[feature].isin([96, 98]), feature] = np.nan
    pairs = {
        "年龄×90天逾期": ("age", "NumberOfTimes90DaysLate"),
        "收入×债务比率": ("MonthlyIncome", "DebtRatio"),
        "额度使用率×30–59天逾期": (
            "RevolvingUtilizationOfUnsecuredLines", "NumberOfTime30-59DaysPastDueNotWorse"
        ),
    }
    return EDAResults(
        overview=overview,
        quality=_quality(train),
        anomalies=_anomalies(train),
        group_risks={feature: group_risk(train, feature) for feature in FEATURES},
        correlation=correlation_source.corr(method="spearman"),
        interactions={name: _interaction(train, *pair) for name, pair in pairs.items()},
        risk_segments=_risk_segments(train),
        discrimination=_discrimination(train),
    )
