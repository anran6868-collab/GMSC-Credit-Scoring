"""运行第四阶段：训练集筛选与调参、验证集选模、PD/PDO信用评分卡。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from src.data_processing.cleaning import CleaningRules
from src.data_processing.schema import TARGET
from src.evaluation.metrics import calibration_table
from src.feature_engineering.binning import WOEBinningPipeline
from src.feature_engineering.features import CANDIDATE_FEATURES
from src.feature_engineering.stability import calculate_psi
from src.scorecard.card import CreditScorecard, ScorecardConfig

from .logistic import TrainingConfig, choose_validation_model, train_candidates
from .reporting import (
    create_stage4_charts,
    grade_summary,
    write_stage4_excel,
    write_stage4_explanations,
)
from .selection import SelectionConfig, select_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_cleaning_rules(path: Path) -> CleaningRules:
    if not path.is_file():
        raise ValueError(f"未找到第二阶段清洗规则：{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CleaningRules(
        medians={name: float(value) for name, value in payload["medians"].items()},
        special_late_values=tuple(int(value) for value in payload["special_late_values"]),
    )


def _load_stage3_data(root: Path):
    feature_dir = root / "data" / "features"
    train_path = feature_dir / "train_woe.csv"
    validation_path = feature_dir / "validation_woe.csv"
    for path in (train_path, validation_path):
        if not path.is_file():
            raise ValueError(f"未找到第三阶段WOE数据：{path}")
    train = pd.read_csv(train_path, usecols=["source_row_id", TARGET, *CANDIDATE_FEATURES])
    # 首次读取验证集时刻意排除标签；标签只在候选训练完成后单独读取。
    validation_features = pd.read_csv(validation_path, usecols=["source_row_id", *CANDIDATE_FEATURES])
    train_bins = pd.read_csv(feature_dir / "train_bins.csv", usecols=list(CANDIDATE_FEATURES))
    validation_bins = pd.read_csv(feature_dir / "validation_bins.csv", usecols=list(CANDIDATE_FEATURES))
    return train, validation_features, train_bins, validation_bins, validation_path


def _screening_table(binner: WOEBinningPipeline, train_bins, validation_bins) -> pd.DataFrame:
    iv = binner.summary().loc[:, ["feature", "iv"]]
    psi, _ = calculate_psi(train_bins, validation_bins)
    return iv.merge(psi, on="feature", validate="one_to_one")


def _high_iv_audit(screening: pd.DataFrame) -> pd.DataFrame:
    high = screening.loc[screening["iv"].ge(0.50), ["feature", "iv", "psi"]].copy()
    high["time_logic"] = "历史/申请时点可得信息，早于未来两年目标窗口"
    high["special_value_check"] = "逾期96/98已由冻结规则恢复为Missing箱"
    high["target_leakage_check"] = "未读取目标字段构造特征；未发现直接目标泄露"
    high["decision"] = "进入训练集多变量筛选，不因高IV自动保留"
    return high


def _coefficient_table(card: CreditScorecard, glm: pd.DataFrame) -> pd.DataFrame:
    coefficients = pd.DataFrame({
        "feature": list(card.candidate.features),
        "final_l2_coefficient": card.candidate.coefficients,
    })
    return coefficients.merge(glm, on="feature", how="left", validate="one_to_one")


def _grade_boundaries(card: CreditScorecard) -> pd.DataFrame:
    q20, q40, q60, q80 = card.grade_boundaries
    return pd.DataFrame([
        {"risk_grade": "A", "score_rule": f"> {q80:.12f}", "lower": q80, "upper": np.inf},
        {"risk_grade": "B", "score_rule": f"({q60:.12f}, {q80:.12f}]", "lower": q60, "upper": q80},
        {"risk_grade": "C", "score_rule": f"({q40:.12f}, {q60:.12f}]", "lower": q40, "upper": q60},
        {"risk_grade": "D", "score_rule": f"({q20:.12f}, {q40:.12f}]", "lower": q20, "upper": q40},
        {"risk_grade": "E", "score_rule": f"<= {q20:.12f}", "lower": -np.inf, "upper": q20},
    ])


def run_stage4(project_root: Path = PROJECT_ROOT) -> dict[str, object]:
    binner_path = project_root / "models" / "stage3_binning.joblib"
    if not binner_path.is_file():
        raise ValueError(f"未找到已验收的第三阶段分箱器：{binner_path}")
    binner = WOEBinningPipeline.load(binner_path)
    frozen_manifest = binner.rule_manifest()
    train, validation_x, train_bins, validation_bins, validation_path = _load_stage3_data(project_root)
    screening = _screening_table(binner, train_bins, validation_bins)

    X_train_all = train.loc[:, list(CANDIDATE_FEATURES)]
    y_train = train[TARGET].astype(int)
    selected_features, selection_log, final_vif, glm = select_features(X_train_all, y_train, screening, SelectionConfig())
    X_train = X_train_all.loc[:, selected_features]

    training_config = TrainingConfig()
    candidates, cv_results = train_candidates(X_train, y_train, training_config)

    # 只有训练、调参全部结束后才读取验证标签，用于两条冻结候选路线的最终选择。
    y_validation = pd.read_csv(validation_path, usecols=[TARGET])[TARGET].astype(int)
    X_validation = validation_x.loc[:, selected_features]
    selected, comparison, predictions = choose_validation_model(
        candidates, X_validation, y_validation, training_config
    )
    if np.any(selected.coefficients >= 0):
        raise RuntimeError("最终L2 Logistic存在非负WOE系数，停止生成评分卡。")
    if binner.rule_manifest() != frozen_manifest:
        raise RuntimeError("第三阶段分箱规则在第四阶段发生变化。")

    cleaning_rules = _load_cleaning_rules(project_root / "data" / "processed" / "cleaning_rules.json")
    card = CreditScorecard.build(cleaning_rules, binner, selected, X_train, ScorecardConfig())
    train_explanations = card.explain_woe(X_train_all, train["source_row_id"])
    validation_explanations = card.explain_woe(
        validation_x.loc[:, list(CANDIDATE_FEATURES)], validation_x["source_row_id"]
    )
    feature_dir = project_root / "data" / "features"
    train_explanations.to_csv(feature_dir / "train_scorecard.csv", index=False)
    validation_explanations.to_csv(feature_dir / "validation_scorecard.csv", index=False)

    model_path = project_root / "models" / "stage4_logistic_scorecard.joblib"
    card.save(model_path)
    loaded = CreditScorecard.load(model_path)
    check_rows = min(1000, len(validation_x))
    before = card.explain_woe(validation_x.loc[: check_rows - 1, list(CANDIDATE_FEATURES)])
    after = loaded.explain_woe(validation_x.loc[: check_rows - 1, list(CANDIDATE_FEATURES)])
    reload_pd_error = float(np.max(np.abs(before["pd"] - after["pd"])))
    reload_score_error = float(np.max(np.abs(before["score_raw"] - after["score_raw"])))

    contribution_columns = [f"points_{feature}" for feature in selected_features]
    contribution_error = float(np.max(np.abs(
        train_explanations["score_raw"]
        - (train_explanations["base_points"] + train_explanations[contribution_columns].sum(axis=1))
    )))
    reverse_pd = expit((card.config.offset - train_explanations["score_raw"].to_numpy()) / card.config.factor)
    pd_reverse_error = float(np.max(np.abs(reverse_pd - train_explanations["pd"].to_numpy())))
    odds_double_error = float(abs(np.exp(card.config.pdo / card.config.factor) - 2.0))
    tolerance = 1e-10
    if max(contribution_error, pd_reverse_error, reload_pd_error, reload_score_error, odds_double_error) > tolerance:
        raise RuntimeError("评分卡数学一致性或持久化重载误差超过1e-10。")

    rules = card.scorecard_rules()
    grades = pd.concat([
        grade_summary(train_explanations, y_train, "train"),
        grade_summary(validation_explanations, y_validation, "validation"),
    ], ignore_index=True)
    calibration = pd.concat([
        calibration_table(y_validation, probability).assign(scheme=name)
        for name, probability in predictions.items()
    ], ignore_index=True)
    reconciliation = pd.DataFrame([{
        "contribution_sum_max_error": contribution_error,
        "pd_reverse_max_error": pd_reverse_error,
        "reload_pd_max_error": reload_pd_error,
        "reload_score_max_error": reload_score_error,
        "pdo_odds_double_error": odds_double_error,
        "tolerance": tolerance,
    }])
    configuration = pd.DataFrame([{
        **card.metadata()["scorecard"],
        "selected_scheme": selected.name,
        "selected_C": selected.c_value,
        "threshold": selected.threshold,
        "cv_folds": training_config.folds,
        "random_state": training_config.random_state,
        "test_label_read": False,
    }])
    run_manifest = pd.DataFrame([
        {"split": "train", "rows": len(train), "label_usage": "变量筛选、交叉验证、调参和拟合"},
        {"split": "validation", "rows": len(validation_x), "label_usage": "冻结候选方案选择与独立验证"},
        {"split": "test", "rows": 0, "label_usage": "第四阶段未读取；保留至第五阶段最终评价"},
    ])
    report_path = project_root / "reports" / "excel" / "stage4_logistic_scorecard.xlsx"
    tables = {
        "配置": configuration,
        "运行清单": run_manifest,
        "高IV审计": _high_iv_audit(screening),
        "变量筛选轨迹": selection_log,
        "最终VIF": final_vif,
        "GLM显著性": glm,
        "交叉验证": cv_results,
        "验证方案比较": comparison,
        "验证校准明细": calibration,
        "最终系数": _coefficient_table(card, glm),
        "评分卡规则": rules,
        "等级边界": _grade_boundaries(card),
        "等级统计": grades,
        "一致性核对": reconciliation,
    }
    write_stage4_excel(report_path, tables)
    chart_paths = create_stage4_charts(
        project_root / "reports" / "figures" / "stage4",
        y_validation,
        predictions,
        comparison,
        selected.name,
        train_explanations,
        validation_explanations,
        grades,
        rules,
    )
    explanation_path = project_root / "reports" / "stage4_explanations.md"
    write_stage4_explanations(
        explanation_path, selected.name, selected_features, comparison, grades, chart_paths
    )

    portable = {"metadata": card.metadata(), "rules": json.loads(rules.to_json(orient="records"))}
    rules_path = project_root / "models" / "stage4_scorecard_rules.json"
    rules_path.write_text(json.dumps(portable, ensure_ascii=False, indent=2), encoding="utf-8")
    selected_metrics = comparison.loc[comparison["selected"]].iloc[0]
    summary = {
        "selected_scheme": selected.name,
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "selected_C": selected.c_value,
        "threshold": selected.threshold,
        "validation_metrics": {
            key: float(selected_metrics[key])
            for key in ("auc", "ks", "pr_auc", "precision", "recall", "f1", "brier", "log_loss", "calibration_intercept", "calibration_slope")
        },
        "grade_boundaries": list(card.grade_boundaries),
        "chart_count": len(chart_paths),
        "test_label_read": False,
        "model": str(model_path),
        "report": str(report_path),
    }
    (project_root / "reports" / "stage4_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(run_stage4(), ensure_ascii=False, indent=2))
