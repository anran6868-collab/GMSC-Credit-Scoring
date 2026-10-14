"""第三阶段的无标签 PSI、相关性诊断与单变量初筛。"""

from __future__ import annotations

import numpy as np
import pandas as pd


PSI_EPSILON = 1e-6


def calculate_psi(
    train_bins: pd.DataFrame,
    validation_bins: pd.DataFrame,
    epsilon: float = PSI_EPSILON,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按训练分箱索引计算验证集 PSI，不需要也不接受标签。"""
    if list(train_bins.columns) != list(validation_bins.columns):
        raise ValueError("训练集与验证集的分箱字段不一致。")
    details: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for feature in train_bins.columns:
        train_counts = train_bins[feature].value_counts(dropna=False)
        validation_counts = validation_bins[feature].value_counts(dropna=False)
        bins = train_counts.index.union(validation_counts.index)
        train_pct = train_counts.reindex(bins, fill_value=0).astype(float).to_numpy()
        validation_pct = validation_counts.reindex(bins, fill_value=0).astype(float).to_numpy()
        train_pct = train_pct / train_pct.sum()
        validation_pct = validation_pct / validation_pct.sum()
        train_safe = np.maximum(train_pct, epsilon)
        validation_safe = np.maximum(validation_pct, epsilon)
        train_safe = train_safe / train_safe.sum()
        validation_safe = validation_safe / validation_safe.sum()
        contributions = (validation_safe - train_safe) * np.log(validation_safe / train_safe)
        psi = float(contributions.sum())
        if psi < 0.10:
            status = "稳定"
        elif psi < 0.25:
            status = "关注"
        else:
            status = "不稳定"
        summaries.append({"feature": feature, "psi": psi, "psi_status": status})
        for bin_value, left, right, contribution in zip(
            bins, train_safe, validation_safe, contributions
        ):
            details.append({
                "feature": feature,
                "bin_index": bin_value,
                "train_share": float(left),
                "validation_share": float(right),
                "psi_contribution": float(contribution),
            })
    return pd.DataFrame(summaries), pd.DataFrame(details)


def high_correlation_pairs(woe_train: pd.DataFrame, threshold: float = 0.70) -> tuple[pd.DataFrame, pd.DataFrame]:
    correlation = woe_train.corr(method="spearman")
    pairs: list[dict[str, object]] = []
    columns = list(correlation.columns)
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1:]:
            value = correlation.loc[left, right]
            if pd.notna(value) and abs(value) >= threshold:
                pairs.append({
                    "feature_a": left,
                    "feature_b": right,
                    "spearman": float(value),
                    "abs_spearman": float(abs(value)),
                    "note": "仅标记冗余候选，第三阶段不删除变量",
                })
    pair_table = pd.DataFrame(pairs)
    if not pair_table.empty:
        pair_table = pair_table.sort_values("abs_spearman", ascending=False).reset_index(drop=True)
    return correlation, pair_table


def screen_candidates(iv_summary: pd.DataFrame, psi_summary: pd.DataFrame) -> pd.DataFrame:
    result = iv_summary.merge(psi_summary, on="feature", validate="one_to_one")

    def classify(row: pd.Series) -> tuple[str, str]:
        iv = float(row["iv"])
        psi = float(row["psi"])
        if iv >= 0.50:
            return "高IV待复核", "检查数据泄露、特殊编码和变量重复后再决定"
        if psi >= 0.25:
            return "稳定性待复核", "验证集PSI不稳定，不默认进入阶段四"
        if iv < 0.02:
            return "弱变量", "单变量IV低，不推荐直接进入阶段四"
        return "阶段四候选", "保留至阶段四进行VIF、系数、显著性和验证表现综合筛选"

    classifications = result.apply(classify, axis=1, result_type="expand")
    classifications.columns = ["screening_status", "screening_note"]
    return pd.concat([result, classifications], axis=1).sort_values("iv", ascending=False).reset_index(drop=True)
