"""第四阶段 Logistic 与评分卡的 Excel、图表和中文说明。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from src.evaluation.metrics import calibration_table
from src.feature_engineering.reporting import FEATURE_NAMES


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def grade_summary(explanations: pd.DataFrame, target: pd.Series, split: str) -> pd.DataFrame:
    frame = explanations.loc[:, ["risk_grade", "pd", "score_raw"]].copy()
    frame["target"] = np.asarray(target, dtype=int)
    result = (
        frame.groupby("risk_grade", observed=True)
        .agg(count=("target", "size"), mean_score=("score_raw", "mean"), mean_pd=("pd", "mean"), bad_rate=("target", "mean"))
        .reindex(list("ABCDE"))
        .reset_index()
    )
    result.insert(0, "split", split)
    return result


def write_stage4_excel(path: Path, tables: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet, table in tables.items():
            table.to_excel(writer, sheet_name=sheet[:31], index=False)


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def create_stage4_charts(
    figure_dir: Path,
    y_validation: pd.Series,
    predictions: dict[str, np.ndarray],
    comparison: pd.DataFrame,
    selected_name: str,
    train_explanations: pd.DataFrame,
    validation_explanations: pd.DataFrame,
    grade_statistics: pd.DataFrame,
    scorecard_rules: pd.DataFrame,
) -> dict[str, Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    metrics = ["auc", "ks", "pr_auc", "f1"]
    labels = ["AUC", "KS", "PR-AUC", "F1"]
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    for index, (_, row) in enumerate(comparison.iterrows()):
        ax.bar(x + (index - 0.5) * width, [row[m] for m in metrics], width, label=row["scheme"])
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_title("验证集 Logistic 方案综合比较")
    ax.legend()
    paths["model_comparison"] = _save(fig, figure_dir / "model_comparison.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, probability in predictions.items():
        fpr, tpr, _ = roc_curve(y_validation, probability)
        auc = comparison.set_index("scheme").loc[name, "auc"]
        ax.plot(fpr, tpr, label=f"{name} AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="#777777")
    ax.set(xlabel="假阳性率", ylabel="真正率", title="验证集 ROC 曲线")
    ax.legend()
    paths["roc"] = _save(fig, figure_dir / "validation_roc.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, probability in predictions.items():
        precision, recall, _ = precision_recall_curve(y_validation, probability)
        pr_auc = comparison.set_index("scheme").loc[name, "pr_auc"]
        ax.plot(recall, precision, label=f"{name} PR-AUC={pr_auc:.4f}")
    ax.set(xlabel="召回率", ylabel="精确率", title="验证集 PR 曲线")
    ax.legend()
    paths["pr"] = _save(fig, figure_dir / "validation_pr.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, probability in predictions.items():
        fpr, tpr, thresholds = roc_curve(y_validation, probability)
        order = np.argsort(thresholds)
        ax.plot(thresholds[order], (tpr - fpr)[order], label=name)
    ax.set(xlabel="PD阈值", ylabel="TPR-FPR", title="验证集 KS 曲线")
    ax.set_xlim(0, 1)
    ax.legend()
    paths["ks"] = _save(fig, figure_dir / "validation_ks.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, probability in predictions.items():
        table = calibration_table(y_validation, probability)
        ax.plot(table["mean_pd"], table["observed_bad_rate"], marker="o", label=name)
    ax.plot([0, 1], [0, 1], "--", color="#777777")
    ax.set(xlabel="平均预测PD", ylabel="实际坏率", title="验证集十分位校准")
    ax.legend()
    paths["calibration"] = _save(fig, figure_dir / "validation_calibration.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(train_explanations["score_raw"], bins=50, alpha=0.55, density=True, label="训练集")
    ax.hist(validation_explanations["score_raw"], bins=50, alpha=0.55, density=True, label="验证集")
    ax.set(xlabel="信用分", ylabel="密度", title="训练集与验证集评分分布")
    ax.legend()
    paths["score_distribution"] = _save(fig, figure_dir / "score_distribution.png")

    validation_grades = grade_statistics.loc[grade_statistics["split"].eq("validation")]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(validation_grades["risk_grade"], validation_grades["bad_rate"] * 100, color="#4f8f73")
    ax.set(xlabel="风险等级", ylabel="实际坏率（%）", title="验证集 A–E 等级实际坏率")
    paths["risk_grades"] = _save(fig, figure_dir / "risk_grades.png")

    average_points = scorecard_rules.assign(abs_points=scorecard_rules["points"].abs()).groupby("feature")["abs_points"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([FEATURE_NAMES.get(name, name) for name in average_points.index], average_points.values, color="#5b7fa3")
    ax.set(xlabel="分箱绝对积分均值", title=f"最终模型（{selected_name}）变量积分幅度")
    paths["contributions"] = _save(fig, figure_dir / "variable_contributions.png")
    return paths


def write_stage4_explanations(
    path: Path,
    selected_name: str,
    selected_features: list[str],
    comparison: pd.DataFrame,
    grade_statistics: pd.DataFrame,
    chart_paths: dict[str, Path],
) -> None:
    selected = comparison.set_index("scheme").loc[selected_name]
    validation_grades = grade_statistics.loc[grade_statistics["split"].eq("validation")]
    lines = [
        "# GMSC 第四阶段 Logistic 与信用评分卡说明",
        "",
        "变量筛选、交叉验证和参数搜索只使用训练集；验证集只用于候选方案选择。测试集标签未读取。",
        "本项目预测未来两年严重逾期概率，仅用于教学研究，不代表真实授信审批结论。",
        "",
        "## 最终方案",
        "",
        f"最终选择“{selected_name}”方案，入模变量 {len(selected_features)} 个："
        + "、".join(FEATURE_NAMES.get(name, name) for name in selected_features)
        + "。",
        "",
        f"验证集 AUC={selected['auc']:.6f}，KS={selected['ks']:.6f}，PR-AUC={selected['pr_auc']:.6f}，"
        f"F1={selected['f1']:.6f}，Brier={selected['brier']:.6f}，Log Loss={selected['log_loss']:.6f}。",
        "",
        f"校准截距={selected['calibration_intercept']:.6f}，校准斜率={selected['calibration_slope']:.6f}；"
        f"校准门槛判定={'通过' if selected['calibration_pass'] else '未通过'}。",
        "",
    ]
    sections = [
        ("方案综合比较", "model_comparison"), ("ROC", "roc"), ("PR", "pr"), ("KS", "ks"),
        ("概率校准", "calibration"), ("评分分布", "score_distribution"),
        ("风险等级", "risk_grades"), ("变量积分", "contributions"),
    ]
    for title, key in sections:
        lines.extend([f"## {title}", "", f"![{title}](figures/stage4/{chart_paths[key].name})", ""])
    lines.extend(["## A–E 等级验证结果", ""])
    for _, row in validation_grades.iterrows():
        lines.append(
            f"- {row['risk_grade']}级：{int(row['count']):,} 人，平均分 {row['mean_score']:.2f}，"
            f"平均PD {row['mean_pd']:.2%}，实际坏率 {row['bad_rate']:.2%}。"
        )
    lines.extend(["", "等级边界来自训练集等频分位数，验证集结果未用于调整边界。"])
    path.write_text("\n".join(lines), encoding="utf-8")
