"""第二阶段 EDA 图表；只呈现训练集的真实汇总统计。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .analysis import EDAResults
from .narratives import FEATURE_NAMES
from .schema import FEATURES


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Microsoft YaHei")


def _save(fig: plt.Figure, output: Path) -> None:
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def create_eda_charts(train: pd.DataFrame, results: EDAResults, figure_dir: Path) -> tuple[dict[str, Path], int]:
    """为每张图返回稳定 ID，供 Notebook 与自动说明对应。"""
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    quality = results.quality.loc[results.quality["missing_count"].gt(0)].sort_values(
        "missing_rate", ascending=True
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh([FEATURE_NAMES[f] for f in quality["feature"]], quality["missing_rate"] * 100)
    ax.set(xlabel="缺失率（%）", title="训练集有缺失的变量（其他变量均为0）")
    for i, row in enumerate(quality.itertuples()):
        ax.text(row.missing_rate * 100 + 0.15, i, f"{row.missing_count:,} 条", va="center", fontsize=8)
    paths["missing_rate"] = figure_dir / "missing_rate.png"
    _save(fig, paths["missing_rate"])

    sample = train.sample(n=min(500, len(train)), random_state=42)
    missing_matrix = sample.loc[:, list(FEATURES)].isna().astype(int)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.imshow(missing_matrix.to_numpy(), aspect="auto", interpolation="nearest", cmap="Purples")
    ax.set_xticks(range(len(FEATURES)), [FEATURE_NAMES[f] for f in FEATURES], rotation=65, ha="right")
    ax.set(ylabel="训练集抽样记录", title=f"缺失模式（固定随机抽样 {len(sample)} 行；深色表示缺失）")
    paths["missing_pattern"] = figure_dir / "missing_pattern.png"
    _save(fig, paths["missing_pattern"])

    dist_features = ["age", "MonthlyIncome", "DebtRatio", "RevolvingUtilizationOfUnsecuredLines"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, feature in zip(axes.flat, dist_features):
        s = train[feature].dropna()
        cap = s.quantile(0.99)
        visible = s.loc[s.le(cap)]
        if feature == "DebtRatio":
            visible = np.log1p(visible)
            xlabel = "log(1+债务比率)，显示至原值99分位"
        else:
            xlabel = "原始取值（显示至99分位）"
        bins = np.arange(20, 111, 5) if feature == "age" else 40
        sns.histplot(visible, bins=bins, ax=ax, color="#2d6a8a")
        ax.set(title=FEATURE_NAMES[feature], xlabel=xlabel, ylabel="样本数")
        ax.text(0.98, 0.95, f">99分位：{int(s.gt(cap).sum()):,}条", transform=ax.transAxes,
                va="top", ha="right", fontsize=8)
    paths["feature_distribution"] = figure_dir / "feature_distribution.png"
    _save(fig, paths["feature_distribution"])

    anomaly = results.anomalies
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(anomaly["issue"], anomaly["row_count"], color="#ad6538")
    ax.set(xlabel="涉及记录数", title="训练集重点异常与特殊取值")
    ax.invert_yaxis()
    paths["anomaly_counts"] = figure_dir / "anomaly_counts.png"
    _save(fig, paths["anomaly_counts"])

    for feature in FEATURES:
        table = results.group_risks[feature]
        labels = table["group"].astype(str).tolist()
        x = np.arange(len(table))
        fig, (top, bottom) = plt.subplots(2, 1, figsize=(max(9, len(labels) * 1.1), 6), sharex=True)
        top.bar(x, table["sample_count"], color="#669db4")
        top.set(ylabel="样本数", title=f"{FEATURE_NAMES[feature]}：分组规模与坏客户率")
        bottom.plot(x, table["bad_rate"] * 100, marker="o", color="#b8513e")
        bottom.axhline(results.overview.iloc[0]["bad_rate"] * 100,
                       linestyle="--", color="#777777", label="训练集整体坏率")
        bottom.set(ylabel="坏客户率（%）", xlabel="描述性分组")
        bottom.legend(loc="best", fontsize=8)
        bottom.set_xticks(x, [label + ("*" if unstable else "")
                              for label, unstable in zip(labels, table["unstable"])],
                          rotation=35, ha="right")
        bottom.text(0, -0.5, "* 样本数 <100 或坏客户数 <20，解读需谨慎", transform=bottom.transAxes,
                    fontsize=8)
        key = f"risk_{feature}"
        paths[key] = figure_dir / f"{key}.png"
        _save(fig, paths[key])

    corr = results.correlation.rename(index=FEATURE_NAMES, columns=FEATURE_NAMES)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap="coolwarm", annot=True, fmt=".2f",
                linewidths=0.3, ax=ax, cbar_kws={"label": "Spearman相关系数"})
    ax.set(title="训练集原始变量相关性")
    paths["correlation"] = figure_dir / "correlation.png"
    _save(fig, paths["correlation"])

    for index, (name, table) in enumerate(results.interactions.items(), start=1):
        display = table.copy()
        display.loc[display["unstable"], "bad_rate"] = np.nan
        matrix = display.pivot(index="group_a", columns="group_b", values="bad_rate")
        matrix = matrix.reindex(
            index=pd.unique(table["group_a"]), columns=pd.unique(table["group_b"])
        )
        fig, ax = plt.subplots(figsize=(max(9, len(matrix.columns) * 1.3), max(5, len(matrix.index) * 0.7)))
        sns.heatmap(matrix * 100, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                    cbar_kws={"label": "坏客户率（%）"})
        left_name, right_name = name.split("×")
        ax.set(title=f"{name}：二维群体坏客户率（稀疏单元格留空）",
               xlabel=right_name, ylabel=left_name)
        key = f"interaction_{index}"
        paths[key] = figure_dir / f"{key}.png"
        _save(fig, paths[key])

    segments = results.risk_segments.sort_values("bad_rate", ascending=False)
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))
    left.barh(segments["segment"], segments["bad_rate"] * 100, color="#b8513e")
    left.set(xlabel="坏客户率（%）", title="历史逾期群体的观察风险")
    left.invert_yaxis()
    right.barh(segments["segment"], segments["sample_share"] * 100, color="#669db4")
    right.set(xlabel="样本占比（%）", title="各群体规模")
    right.invert_yaxis()
    paths["risk_segments"] = figure_dir / "risk_segments.png"
    _save(fig, paths["risk_segments"])

    discrimination = results.discrimination.sort_values("single_auc")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([FEATURE_NAMES[f] for f in discrimination["feature"]],
            discrimination["single_auc"], color="#3c8d75")
    ax.axvline(0.5, color="#777777", linestyle="--", label="随机排序 0.5")
    ax.set(xlabel="训练集单变量方向校正 AUC", title="原始变量初步区分能力")
    ax.legend(loc="lower right")
    paths["single_feature_discrimination"] = figure_dir / "single_feature_discrimination.png"
    _save(fig, paths["single_feature_discrimination"])
    return paths, len(sample)
