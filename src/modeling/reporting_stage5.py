"""第五阶段模型比较图表、结构化表和中文说明。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve, roc_curve

from src.feature_engineering.reporting import FEATURE_NAMES

from .tree_models import TreeModelBundle


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

MODEL_COLORS = {"Logistic 评分卡": "#4f779f", "Random Forest": "#c77c3b", "XGBoost": "#4f8f73"}


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def create_comparison_charts(
    output_dir: Path,
    y_validation: pd.Series,
    probabilities: dict[str, np.ndarray],
    comparison: pd.DataFrame,
    complexity: pd.DataFrame,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    indexed = comparison.set_index("model")

    metrics = ["auc", "ks", "pr_auc", "f1"]
    labels = ["AUC", "KS", "PR-AUC", "F1"]
    x = np.arange(len(metrics))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for idx, name in enumerate(probabilities):
        ax.bar(x + (idx - 1) * width, indexed.loc[name, metrics], width, label=name, color=MODEL_COLORS[name])
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_title("验证集三模型综合指标")
    ax.legend()
    paths["metrics"] = _save(fig, output_dir / "validation_model_comparison.png")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for name, probability in probabilities.items():
        fpr, tpr, _ = roc_curve(y_validation, probability)
        ax.plot(fpr, tpr, color=MODEL_COLORS[name], label=f"{name} AUC={indexed.loc[name, 'auc']:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="#777777")
    ax.set(xlabel="假阳性率", ylabel="真正率", title="验证集 ROC 曲线")
    ax.legend()
    paths["roc"] = _save(fig, output_dir / "validation_roc.png")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for name, probability in probabilities.items():
        precision, recall, _ = precision_recall_curve(y_validation, probability)
        ax.plot(recall, precision, color=MODEL_COLORS[name], label=f"{name} PR-AUC={indexed.loc[name, 'pr_auc']:.4f}")
    ax.set(xlabel="召回率", ylabel="精确率", title="验证集 PR 曲线")
    ax.legend()
    paths["pr"] = _save(fig, output_dir / "validation_pr.png")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for name, probability in probabilities.items():
        fpr, tpr, thresholds = roc_curve(y_validation, probability)
        finite = np.isfinite(thresholds)
        order = np.argsort(thresholds[finite])
        ax.plot(thresholds[finite][order], (tpr - fpr)[finite][order], color=MODEL_COLORS[name], label=name)
    ax.set(xlabel="PD 阈值", ylabel="TPR-FPR", title="验证集 KS 曲线", xlim=(0, 1))
    ax.legend()
    paths["ks"] = _save(fig, output_dir / "validation_ks.png")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for name, probability in probabilities.items():
        observed, predicted = calibration_curve(y_validation, probability, n_bins=10, strategy="quantile")
        ax.plot(predicted, observed, marker="o", color=MODEL_COLORS[name], label=name)
    ax.plot([0, 1], [0, 1], "--", color="#777777")
    ax.set(xlabel="平均预测 PD", ylabel="实际坏率", title="验证集十分位校准")
    ax.legend()
    paths["calibration"] = _save(fig, output_dir / "validation_calibration.png")

    display = complexity.set_index("model").reindex(list(probabilities))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].bar(display.index, display["model_bytes"] / 1024 / 1024, color=[MODEL_COLORS[name] for name in display.index])
    axes[0].set(ylabel="模型大小（MB）", title="序列化模型大小")
    axes[0].tick_params(axis="x", rotation=15)
    axes[1].bar(display.index, display["inference_ms_per_1000"], color=[MODEL_COLORS[name] for name in display.index])
    axes[1].set(ylabel="毫秒/千行", title="同机验证集推理时间")
    axes[1].tick_params(axis="x", rotation=15)
    paths["complexity"] = _save(fig, output_dir / "model_complexity.png")
    return paths


def tree_feature_importance(
    bundle: TreeModelBundle,
    validation_features: pd.DataFrame,
    y_validation: pd.Series,
    output_dir: Path,
) -> tuple[pd.DataFrame, Path]:
    transformed = bundle.transform_features(validation_features)
    native = np.asarray(bundle.estimator.feature_importances_, dtype=float)
    permutation = permutation_importance(
        bundle.estimator,
        transformed,
        y_validation,
        scoring="average_precision",
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )
    table = pd.DataFrame({
        "model": bundle.model_name,
        "feature": bundle.feature_names,
        "feature_cn": [FEATURE_NAMES.get(name, name) for name in bundle.feature_names],
        "native_importance": native,
        "validation_permutation_pr_auc_mean": permutation.importances_mean,
        "validation_permutation_pr_auc_std": permutation.importances_std,
    }).sort_values("validation_permutation_pr_auc_mean", ascending=False).reset_index(drop=True)
    plot = table.sort_values("validation_permutation_pr_auc_mean").tail(15)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(plot["feature_cn"], plot["validation_permutation_pr_auc_mean"], color="#597d9f")
    ax.set(xlabel="验证集 PR-AUC 置换重要性", title=f"{bundle.model_name} 特征重要性")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = _save(fig, output_dir / ("rf_feature_importance.png" if bundle.model_name == "Random Forest" else "xgboost_feature_importance.png"))
    return table, path


def write_tables(output_dir: Path, tables: dict[str, pd.DataFrame]) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}
    paths: dict[str, Path] = {}
    for name, table in tables.items():
        path = output_dir / f"{name}.csv"
        table.to_csv(path, index=False, encoding="utf-8-sig")
        manifest[name] = path.name
        paths[name] = path
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    workbook_payload = []
    for name, table in tables.items():
        serializable = table.astype(object).where(pd.notna(table), None)
        workbook_payload.append({
            "sheet": name[:31],
            "columns": list(serializable.columns),
            "rows": serializable.values.tolist(),
        })
    (output_dir / "workbook_tables.json").write_text(
        json.dumps(
            workbook_payload,
            ensure_ascii=False,
            default=lambda value: value.item() if hasattr(value, "item") else str(value),
        ),
        encoding="utf-8",
    )
    return paths


def write_explanations(
    path: Path,
    comparison: pd.DataFrame,
    test_audit: pd.DataFrame,
    best_challenger: str,
    chart_paths: dict[str, Path],
    shap_paths: dict[str, dict[str, Path]],
) -> None:
    validation = comparison.set_index("model")
    lines = [
        "# GMSC 第五阶段 Random Forest、XGBoost 与 SHAP 说明",
        "",
        "Logistic 评分卡保持主模型地位；Random Forest 与 XGBoost 是对照和挑战者。所有调参、缺失值规则、阈值和早停轮数仅来自训练集。",
        "验证集用于冻结比较结论；测试集只在冻结后进行一次审计，未参与选模或返工调参。",
        "本项目只预测未来两年严重逾期概率，不代表真实授信审批结论。",
        "",
        "## 验证集结论",
        "",
        f"最佳树模型挑战者：{best_challenger}。Logistic 评分卡仍是项目主路线。",
        "",
    ]
    for model in ["Logistic 评分卡", "Random Forest", "XGBoost"]:
        row = validation.loc[model]
        lines.append(
            f"- {model}：AUC={row['auc']:.6f}，KS={row['ks']:.6f}，PR-AUC={row['pr_auc']:.6f}，"
            f"F1={row['f1']:.6f}，Brier={row['brier']:.6f}，Log Loss={row['log_loss']:.6f}，"
            f"校准门槛={'通过' if row['calibration_pass'] else '未通过'}，预测 PSI={row['probability_psi']:.6f}。"
        )
    lines.extend(["", "## 测试集冻结后审计", ""])
    for _, row in test_audit.iterrows():
        lines.append(
            f"- {row['model']}：AUC={row['auc']:.6f}，KS={row['ks']:.6f}，PR-AUC={row['pr_auc']:.6f}，"
            f"F1={row['f1']:.6f}，Brier={row['brier']:.6f}，Log Loss={row['log_loss']:.6f}。"
        )
    lines.extend(["", "测试集结果仅用于报告泛化表现，不改变任何模型、阈值或验证集结论。", ""])
    titles = {
        "metrics": "验证集综合指标", "roc": "验证集 ROC", "pr": "验证集 PR", "ks": "验证集 KS",
        "calibration": "验证集概率校准", "complexity": "模型复杂度",
        "rf_importance": "Random Forest 特征重要性", "xgb_importance": "XGBoost 特征重要性",
    }
    for key, title in titles.items():
        if key in chart_paths:
            lines.extend([f"## {title}", "", f"![{title}](figures/stage5/{chart_paths[key].name})", ""])
    for model, paths in shap_paths.items():
        lines.extend([f"## {model} SHAP", ""])
        for title, key in (("全局重要性", "global_bar"), ("影响方向", "beeswarm")):
            lines.extend([f"### {title}", "", f"![{model}{title}](figures/stage5/{paths[key].name})", ""])
    lines.extend([
        "## 解释口径",
        "",
        "SHAP 使用 interventional 特征扰动并解释严重逾期概率；基准概率与各特征 SHAP 贡献之和等于模型 PD。",
        "个案来自验证集低风险、中位风险、高风险以及两树模型分歧最大的固定客户，选择过程不使用真实标签。",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")
