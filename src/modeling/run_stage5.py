"""运行第五阶段：RF、XGBoost、三模型比较、SHAP 与冻结后测试审计。"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

_MPL_CACHE = Path(tempfile.gettempdir()) / "gmsc-matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import numpy as np
import pandas as pd
import sklearn
import shap
import xgboost

from src.data_processing.cleaning import CleaningRules
from src.data_processing.schema import TARGET
from src.evaluation.comparison import compare_models, model_complexity
from src.evaluation.metrics import calibration_table
from src.feature_engineering.features import CANDIDATE_FEATURES
from src.scorecard.card import CreditScorecard

from .explainability import create_shap_charts, explain_tree_model, select_validation_cases
from .reporting_stage5 import (
    create_comparison_charts,
    tree_feature_importance,
    write_explanations,
    write_tables,
)
from .tree_models import TreeModelBundle, TreeTrainingConfig, train_random_forest, train_xgboost


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAMES = ("Logistic 评分卡", "Random Forest", "XGBoost")


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"无法序列化对象：{type(value).__name__}")


def _load_cleaning_rules(path: Path) -> CleaningRules:
    if not path.is_file():
        raise ValueError(f"未找到第二阶段清洗规则：{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CleaningRules(
        medians={name: float(value) for name, value in payload["medians"].items()},
        special_late_values=tuple(int(value) for value in payload["special_late_values"]),
    )


def _load_feature_split(path: Path, *, include_label: bool) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"未找到第三阶段特征数据：{path}")
    columns = ["source_row_id", *CANDIDATE_FEATURES]
    if include_label:
        columns.insert(1, TARGET)
    return pd.read_csv(path, usecols=columns)


def _verify_split_ids(root: Path, splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    manifest = pd.read_csv(root / "data" / "processed" / "split_manifest.csv")
    rows = []
    for split, frame in splits.items():
        expected = manifest.loc[manifest["split"].eq(split), "source_row_id"].to_numpy()
        actual = frame["source_row_id"].to_numpy()
        matches = bool(np.array_equal(expected, actual))
        rows.append({"split": split, "expected_rows": len(expected), "actual_rows": len(actual), "source_row_id_match": matches})
        if not matches:
            raise RuntimeError(f"{split} 特征数据与既有划分 source_row_id 不一致。")
    return pd.DataFrame(rows)


def _prediction_frame(
    ids: pd.Series,
    probabilities: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> pd.DataFrame:
    result = pd.DataFrame({"source_row_id": ids.to_numpy()})
    for model, probability in probabilities.items():
        key = {"Logistic 评分卡": "logistic", "Random Forest": "random_forest", "XGBoost": "xgboost"}[model]
        result[f"pd_{key}"] = probability
        result[f"predicted_bad_{key}"] = (probability >= thresholds[model]).astype(int)
    return result


def _freeze_manifest(
    path: Path,
    config: TreeTrainingConfig,
    thresholds: dict[str, float],
    validation_comparison: pd.DataFrame,
    best_challenger: str,
    model_paths: dict[str, Path],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "frozen_before_test_label_access",
        "primary_model": "Logistic 评分卡",
        "best_tree_challenger": best_challenger,
        "training_config": asdict(config),
        "thresholds": thresholds,
        "validation_comparison": validation_comparison.to_dict(orient="records"),
        "model_paths": {name: str(value) for name, value in model_paths.items()},
        "test_used_for_selection": False,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_json_default)
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return payload


def _evaluate_frozen_test(
    test_path: Path,
    frozen_manifest_path: Path,
    probabilities: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> tuple[pd.Series, pd.DataFrame]:
    if not frozen_manifest_path.is_file():
        raise RuntimeError("冻结清单尚未生成，禁止读取测试集标签。")
    frozen = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_test_label_access":
        raise RuntimeError("冻结清单状态无效，禁止读取测试集标签。")
    y_test = pd.read_csv(test_path, usecols=[TARGET])[TARGET].astype(int)
    audit, _ = compare_models(y_test, probabilities, thresholds, include_ranks=False)
    audit.insert(0, "evaluation_role", "冻结后一次性测试审计")
    return y_test, audit


def _select_challenger(comparison: pd.DataFrame) -> str:
    tree_rows = comparison.loc[comparison["model"].isin(["Random Forest", "XGBoost"])].copy()
    eligible = tree_rows.loc[tree_rows["calibration_pass"] & tree_rows["probability_psi"].lt(0.25)]
    if eligible.empty:
        return "无（树模型均未同时通过校准与稳定性门槛）"
    return str(eligible.sort_values(["rank_score", "brier", "model"]).iloc[0]["model"])


def run_stage5(
    project_root: Path = PROJECT_ROOT,
    training_config: TreeTrainingConfig | None = None,
) -> dict[str, object]:
    cfg = training_config or TreeTrainingConfig()
    feature_dir = project_root / "data" / "features"
    train_path = feature_dir / "train_features.csv"
    validation_path = feature_dir / "validation_features.csv"
    test_path = feature_dir / "test_features.csv"

    train = _load_feature_split(train_path, include_label=True)
    validation = _load_feature_split(validation_path, include_label=False)
    test = _load_feature_split(test_path, include_label=False)
    split_check = _verify_split_ids(project_root, {"train": train, "validation": validation, "test": test})
    X_train = train.loc[:, list(CANDIDATE_FEATURES)]
    y_train = train[TARGET].astype(int)
    X_validation = validation.loc[:, list(CANDIDATE_FEATURES)]
    X_test = test.loc[:, list(CANDIDATE_FEATURES)]
    cleaning_rules = _load_cleaning_rules(project_root / "data" / "processed" / "cleaning_rules.json")

    rf_bundle, rf_search, rf_oof = train_random_forest(X_train, y_train, cleaning_rules, cfg)
    xgb_bundle, xgb_search, xgb_oof = train_xgboost(X_train, y_train, cleaning_rules, cfg)
    model_dir = project_root / "models"
    rf_path = model_dir / "stage5_random_forest.joblib"
    xgb_path = model_dir / "stage5_xgboost_pipeline.joblib"
    xgb_json_path = model_dir / "stage5_xgboost.json"
    rf_bundle.save(rf_path)
    xgb_bundle.save(xgb_path)
    xgb_bundle.estimator.save_model(xgb_json_path)
    rf_loaded = TreeModelBundle.load(rf_path)
    xgb_loaded = TreeModelBundle.load(xgb_path)

    # 两个树模型及其阈值已完全由训练集冻结，此后才读取验证标签。
    y_validation = pd.read_csv(validation_path, usecols=[TARGET])[TARGET].astype(int)
    scorecard_path = model_dir / "stage4_logistic_scorecard.joblib"
    scorecard = CreditScorecard.load(scorecard_path)
    train_woe = pd.read_csv(feature_dir / "train_woe.csv", usecols=list(CANDIDATE_FEATURES))
    validation_woe = pd.read_csv(feature_dir / "validation_woe.csv", usecols=list(CANDIDATE_FEATURES))
    test_woe = pd.read_csv(feature_dir / "test_woe.csv", usecols=list(CANDIDATE_FEATURES))

    train_probabilities = {
        "Logistic 评分卡": scorecard.predict_pd_woe(train_woe),
        "Random Forest": rf_oof,
        "XGBoost": xgb_oof,
    }
    validation_probabilities = {
        "Logistic 评分卡": scorecard.predict_pd_woe(validation_woe),
        "Random Forest": rf_loaded.predict_pd_features(X_validation),
        "XGBoost": xgb_loaded.predict_pd_features(X_validation),
    }
    thresholds = {
        "Logistic 评分卡": float(scorecard.candidate.threshold),
        "Random Forest": float(rf_loaded.threshold),
        "XGBoost": float(xgb_loaded.threshold),
    }
    validation_comparison, probability_psi_detail = compare_models(
        y_validation, validation_probabilities, thresholds, train_probabilities
    )
    best_challenger = _select_challenger(validation_comparison)

    reload_checks = pd.DataFrame([
        {
            "model": "Random Forest",
            "max_reload_pd_error": float(np.max(np.abs(rf_bundle.predict_pd_features(X_validation) - validation_probabilities["Random Forest"]))),
        },
        {
            "model": "XGBoost",
            "max_reload_pd_error": float(np.max(np.abs(xgb_bundle.predict_pd_features(X_validation) - validation_probabilities["XGBoost"]))),
        },
    ])
    if reload_checks["max_reload_pd_error"].max() > 1e-12:
        raise RuntimeError("树模型保存重载后的预测误差超过1e-12。")

    model_paths = {"Logistic 评分卡": scorecard_path, "Random Forest": rf_path, "XGBoost": xgb_path}
    frozen_path = model_dir / "stage5_frozen_manifest.json"
    frozen = _freeze_manifest(frozen_path, cfg, thresholds, validation_comparison, best_challenger, model_paths)

    # 冻结完成后可以生成测试预测，但标签仍由独立审计函数读取一次。
    test_probabilities = {
        "Logistic 评分卡": scorecard.predict_pd_woe(test_woe),
        "Random Forest": rf_loaded.predict_pd_features(X_test),
        "XGBoost": xgb_loaded.predict_pd_features(X_test),
    }
    y_test, test_audit = _evaluate_frozen_test(test_path, frozen_path, test_probabilities, thresholds)

    prediction_paths = {}
    for split, ids, probabilities in (
        ("train", train["source_row_id"], train_probabilities),
        ("validation", validation["source_row_id"], validation_probabilities),
        ("test", test["source_row_id"], test_probabilities),
    ):
        output = feature_dir / f"{split}_model_predictions.csv"
        _prediction_frame(ids, probabilities, thresholds).to_csv(output, index=False)
        prediction_paths[split] = output

    complexity = pd.DataFrame([
        {
            **model_complexity("Logistic 评分卡", scorecard.candidate.estimator, scorecard_path, lambda: scorecard.predict_pd_woe(validation_woe)),
            "feature_count": len(scorecard.candidate.features),
            "tree_count": 0,
            "total_nodes": 0,
            "total_leaves": 0,
            "max_depth": 0,
        },
        {
            **model_complexity("Random Forest", rf_loaded.estimator, rf_path, lambda: rf_loaded.predict_pd_features(X_validation)),
            "feature_count": len(rf_loaded.feature_names),
        },
        {
            **model_complexity("XGBoost", xgb_loaded.estimator, xgb_path, lambda: xgb_loaded.predict_pd_features(X_validation)),
            "feature_count": len(xgb_loaded.feature_names),
        },
    ])

    figure_dir = project_root / "reports" / "figures" / "stage5"
    chart_paths = create_comparison_charts(figure_dir, y_validation, validation_probabilities, validation_comparison, complexity)
    rf_importance, chart_paths["rf_importance"] = tree_feature_importance(rf_loaded, X_validation, y_validation, figure_dir)
    xgb_importance, chart_paths["xgb_importance"] = tree_feature_importance(xgb_loaded, X_validation, y_validation, figure_dir)

    cases = select_validation_cases(
        validation["source_row_id"],
        validation_probabilities["Logistic 评分卡"],
        validation_probabilities["Random Forest"],
        validation_probabilities["XGBoost"],
    )
    rf_shap = explain_tree_model(rf_loaded, X_train, X_validation, validation["source_row_id"], cases)
    xgb_shap = explain_tree_model(xgb_loaded, X_train, X_validation, validation["source_row_id"], cases)
    shap_paths = {
        "Random Forest": create_shap_charts(rf_shap, cases, figure_dir),
        "XGBoost": create_shap_charts(xgb_shap, cases, figure_dir),
    }

    calibration = pd.concat([
        calibration_table(y_validation, probability).assign(model=name, split="validation")
        for name, probability in validation_probabilities.items()
    ] + [
        calibration_table(y_test, probability).assign(model=name, split="test")
        for name, probability in test_probabilities.items()
    ], ignore_index=True)
    configuration = pd.DataFrame([{
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "shap": shap.__version__,
        "platform": platform.platform(),
        **asdict(cfg),
        "tree_feature_count": len(CANDIDATE_FEATURES),
        "test_used_for_selection": False,
    }])
    run_manifest = pd.DataFrame([
        {"step": 1, "dataset": "train", "label_usage": "五折调参、OOF阈值、早停和最终拟合", "rows": len(train)},
        {"step": 2, "dataset": "validation", "label_usage": "冻结候选后的三模型比较与解释", "rows": len(validation)},
        {"step": 3, "dataset": "test", "label_usage": "冻结清单后一次性最终审计，不参与选模", "rows": len(test)},
        {"step": 0, "dataset": "test_metadata", "label_usage": "规划期曾暴露总体坏样本率；未产生模型指标且未用于任何决策", "rows": 0},
    ])
    reconciliation = pd.concat([
        reload_checks,
        pd.DataFrame([
            {"model": "Random Forest SHAP", "max_reload_pd_error": rf_shap.max_additivity_error},
            {"model": "XGBoost SHAP", "max_reload_pd_error": xgb_shap.max_additivity_error},
        ]),
    ], ignore_index=True)
    tables = {
        "总览": validation_comparison,
        "配置": configuration,
        "运行审计": run_manifest,
        "划分核对": split_check,
        "RF调参": rf_search,
        "XGB调参": xgb_search,
        "验证集比较": validation_comparison,
        "概率PSI明细": probability_psi_detail,
        "测试集审计": test_audit,
        "校准明细": calibration,
        "复杂度": complexity,
        "RF特征重要性": rf_importance,
        "XGB特征重要性": xgb_importance,
        "SHAP全局": pd.concat([rf_shap.global_table, xgb_shap.global_table], ignore_index=True),
        "SHAP个案": pd.concat([rf_shap.local_table, xgb_shap.local_table], ignore_index=True),
        "解释客户": cases,
        "一致性核对": reconciliation,
    }
    table_dir = project_root / "reports" / "stage5_tables"
    table_paths = write_tables(table_dir, tables)
    explanation_path = project_root / "reports" / "stage5_explanations.md"
    write_explanations(explanation_path, validation_comparison, test_audit, best_challenger, chart_paths, shap_paths)

    summary = {
        "primary_model": "Logistic 评分卡",
        "best_tree_challenger": best_challenger,
        "tree_features": list(CANDIDATE_FEATURES),
        "validation_metrics": validation_comparison.to_dict(orient="records"),
        "test_metrics_frozen_audit": test_audit.to_dict(orient="records"),
        "shap_additivity_max_error": {
            "Random Forest": rf_shap.max_additivity_error,
            "XGBoost": xgb_shap.max_additivity_error,
        },
        "frozen_manifest_sha256": frozen["sha256"],
        "test_used_for_selection": False,
        "table_manifest": {name: str(value) for name, value in table_paths.items()},
        "chart_count": len(chart_paths) + sum(len(value) for value in shap_paths.values()),
        "model_paths": {name: str(value) for name, value in model_paths.items()},
        "prediction_paths": {name: str(value) for name, value in prediction_paths.items()},
    }
    summary_path = project_root / "reports" / "stage5_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_stage5(), ensure_ascii=False, indent=2))
