"""第三阶段真实数据报表、图表与中文解释。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_processing.narratives import FEATURE_NAMES as RAW_FEATURE_NAMES

from .features import FEATURE_DEFINITIONS


FEATURE_NAMES = {
    **RAW_FEATURE_NAMES,
    "total_delinquency_count": "历史逾期总次数",
    "severe_delinquency_flag": "历史严重逾期标记",
    "delinquency_severity_score": "历史逾期严重度",
    "real_estate_loan_share": "房地产贷款账户占比",
    "monthly_income_per_capita": "人均月收入",
    "estimated_monthly_debt": "估算月债务",
    "residual_income_per_capita": "人均剩余收入",
    "qa_MonthlyIncome_missing": "月收入原始缺失标记",
    "qa_NumberOfDependents_missing": "家属人数原始缺失标记",
    "qa_late_special": "逾期96/98特殊编码标记",
}

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Microsoft YaHei")


def feature_definition_table() -> pd.DataFrame:
    rows = []
    for feature, name in FEATURE_NAMES.items():
        if feature in FEATURE_DEFINITIONS:
            kind = "业务衍生变量"
            formula = FEATURE_DEFINITIONS[feature]
        elif feature.startswith("qa_"):
            kind = "质量标记"
            formula = "第二阶段保留的0/1数据质量标记"
        else:
            kind = "原始变量"
            formula = "GMSC原始字段；原始缺失或特殊编码按质量标记恢复为Missing箱"
        rows.append({"feature": feature, "中文名称": name, "类型": kind, "定义或处理": formula})
    return pd.DataFrame(rows)


def write_analysis_excel(
    path: Path,
    *,
    configuration: pd.DataFrame,
    binning_tables: pd.DataFrame,
    screening: pd.DataFrame,
    psi_detail: pd.DataFrame,
    correlation: pd.DataFrame,
    correlation_pairs: pd.DataFrame,
    run_manifest: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        feature_definition_table().to_excel(writer, sheet_name="特征定义", index=False)
        configuration.to_excel(writer, sheet_name="分箱参数", index=False)
        binning_tables.to_excel(writer, sheet_name="训练分箱明细", index=False)
        screening.to_excel(writer, sheet_name="IV_PSI与初筛", index=False)
        psi_detail.to_excel(writer, sheet_name="验证PSI明细", index=False)
        correlation.to_excel(writer, sheet_name="WOE相关矩阵")
        correlation_pairs.to_excel(writer, sheet_name="高相关变量对", index=False)
        run_manifest.to_excel(writer, sheet_name="运行清单", index=False)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def create_stage3_charts(
    figure_dir: Path,
    screening: pd.DataFrame,
    binning_tables: pd.DataFrame,
) -> dict[str, Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    ordered = screening.sort_values("iv")
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh([FEATURE_NAMES[name] for name in ordered["feature"]], ordered["iv"], color="#3c8d75")
    ax.axvline(0.02, color="#777777", linestyle="--", label="IV=0.02")
    ax.axvline(0.50, color="#b8513e", linestyle="--", label="IV=0.50")
    ax.set(xlabel="训练集 IV", title="训练集候选变量 IV 排名")
    for index, value in enumerate(ordered["iv"]):
        ax.text(value + 0.01, index, f"{value:.3f}", va="center", fontsize=8)
    ax.legend(loc="lower right")
    paths["iv_ranking"] = figure_dir / "iv_ranking.png"
    _save(fig, paths["iv_ranking"])

    ordered = screening.sort_values("psi")
    colors = np.where(ordered["psi"].ge(0.25), "#b8513e", np.where(ordered["psi"].ge(0.10), "#d39a3d", "#4f8f73"))
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh([FEATURE_NAMES[name] for name in ordered["feature"]], ordered["psi"], color=colors)
    ax.axvline(0.10, color="#777777", linestyle="--", label="PSI=0.10")
    ax.axvline(0.25, color="#b8513e", linestyle="--", label="PSI=0.25")
    ax.set(xlabel="训练集与验证集 PSI（不使用验证标签）", title="分箱分布稳定性")
    ax.set_xlim(0, max(0.26, float(ordered["psi"].max()) * 1.2))
    for index, value in enumerate(ordered["psi"]):
        ax.text(value + 0.001, index, f"{value:.4f}", va="center", fontsize=8)
    ax.legend(loc="lower right")
    paths["validation_psi"] = figure_dir / "validation_psi.png"
    _save(fig, paths["validation_psi"])

    for feature in screening["feature"]:
        table = binning_tables.loc[binning_tables["feature"].eq(feature)].copy()
        table = table.loc[~table["table_row"].eq("Totals") & table["Count"].gt(0)]
        labels = table["Bin"].astype(str).tolist()
        x = np.arange(len(table))
        fig, (top, bottom) = plt.subplots(2, 1, figsize=(max(9, len(table) * 1.3), 7), sharex=True)
        top.bar(x, table["Count"], color="#669db4")
        top.set(ylabel="训练样本数", title=f"{FEATURE_NAMES[feature]}：最优分箱规模、坏率与WOE")
        bad_axis = bottom
        woe_axis = bottom.twinx()
        bad_axis.plot(x, table["Event rate"] * 100, marker="o", color="#b8513e", label="坏客户率")
        woe_axis.plot(x, table["WoE"], marker="s", color="#3c6e8f", label="WOE")
        bad_axis.set(ylabel="坏客户率（%）", xlabel="训练集分箱")
        woe_axis.set_ylabel("WOE")
        bad_axis.set_xticks(x, labels, rotation=35, ha="right")
        lines = bad_axis.lines + woe_axis.lines
        bad_axis.legend(lines, [line.get_label() for line in lines], loc="best", fontsize=8)
        key = f"binning_{feature}"
        paths[key] = figure_dir / f"{key}.png"
        _save(fig, paths[key])
    return paths


def write_explanations(
    path: Path,
    screening: pd.DataFrame,
    binning_tables: pd.DataFrame,
    correlation_pairs: pd.DataFrame,
    chart_paths: dict[str, Path],
) -> None:
    strongest = screening.sort_values("iv", ascending=False).iloc[0]
    max_psi = screening.sort_values("psi", ascending=False).iloc[0]
    lines = [
        "# GMSC 第三阶段分箱、WOE 与 IV 业务解释",
        "",
        "所有切点、WOE 和 IV 仅使用训练集拟合。验证集只用于无标签 PSI；测试集不参与本报告结论。",
        "IV 与 PSI 仅作单变量初筛，最终变量选择留到第四阶段。",
        "",
        "## 训练集 IV 排名",
        "",
        f"![训练集 IV 排名](figures/stage3/{chart_paths['iv_ranking'].name})",
        "",
        f"训练集最高 IV 变量为“{FEATURE_NAMES[strongest['feature']]}”，IV={strongest['iv']:.6f}，"
        f"初筛状态为“{strongest['screening_status']}”。高 IV 不直接等同于最终入模资格。",
        "",
        "## 验证集无标签 PSI",
        "",
        f"![验证集 PSI](figures/stage3/{chart_paths['validation_psi'].name})",
        "",
        f"PSI 最高变量为“{FEATURE_NAMES[max_psi['feature']]}”，PSI={max_psi['psi']:.6f}，"
        f"稳定性判定为“{max_psi['psi_status']}”。PSI 只比较分箱占比，不使用验证集目标变量。",
        "",
        "## WOE 高相关诊断",
        "",
        f"训练集共发现 {len(correlation_pairs)} 对 |Spearman|≥0.70 的 WOE 候选变量；"
        "本阶段只标记，不删除变量。",
        "",
    ]
    indexed = screening.set_index("feature")
    for feature in screening["feature"]:
        row = indexed.loc[feature]
        table = binning_tables.loc[binning_tables["feature"].eq(feature)]
        nonempty = table.loc[~table["table_row"].eq("Totals") & table["Count"].gt(0)]
        min_rate = float(nonempty["Event rate"].min())
        max_rate = float(nonempty["Event rate"].max())
        image = chart_paths[f"binning_{feature}"].name
        lines.extend([
            f"## {FEATURE_NAMES[feature]}",
            "",
            f"![{FEATURE_NAMES[feature]}分箱](figures/stage3/{image})",
            "",
            f"训练集有效箱共 {len(nonempty)} 个，箱坏率范围为 {min_rate:.2%}–{max_rate:.2%}；"
            f"IV={row['iv']:.6f}，验证 PSI={row['psi']:.6f}。",
            "",
            f"初筛结论：{row['screening_status']}。{row['screening_note']}。"
            "分箱关系是统计关联，不代表因果或真实授信结论。",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")
