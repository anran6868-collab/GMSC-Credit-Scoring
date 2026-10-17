"""只读地加载已验收产物，并为 Streamlit 提供可测试的数据接口。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from src.data_processing.schema import FEATURES, LATE_FEATURES, TARGET
from src.modeling.tree_models import TreeModelBundle
from src.scorecard.card import CreditScorecard


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_FILES = {
    "Logistic 评分卡": "stage4_logistic_scorecard.joblib",
    "Random Forest": "stage5_random_forest.joblib",
    "XGBoost": "stage5_xgboost_pipeline.joblib",
}
MODEL_COLUMNS = {
    "Logistic 评分卡": "pd_logistic",
    "Random Forest": "pd_random_forest",
    "XGBoost": "pd_xgboost",
}
REPORT_FILES = {
    "quality": "data_quality.xlsx",
    "binning": "stage3_woe_iv_analysis.xlsx",
    "scorecard": "stage4_logistic_scorecard.xlsx",
    "comparison": "stage5_model_comparison.xlsx",
}
COUNT_FIELDS = {
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
}
OPTIONAL_FIELDS = {"MonthlyIncome", "NumberOfDependents"}


def _required(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"缺少已验收产物：{path}。请先恢复本地项目文件，页面不会自动训练。")
    return path


def load_manifest(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    path = _required(root / "models" / "stage5_frozen_manifest.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("sha256", None)
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if recorded != actual or payload.get("status") != "frozen_before_test_label_access":
        raise ValueError("第五阶段冻结清单无效，请核对本地产物。")
    if payload.get("primary_model") != "Logistic 评分卡" or payload.get("test_used_for_selection") is not False:
        raise ValueError("冻结清单的模型路线或测试集状态不符合项目要求。")
    return payload


def load_models(root: Path = PROJECT_ROOT) -> dict[str, CreditScorecard | TreeModelBundle]:
    manifest = load_manifest(root)
    models: dict[str, CreditScorecard | TreeModelBundle] = {}
    for name, filename in MODEL_FILES.items():
        path = _required(root / "models" / filename)
        models[name] = CreditScorecard.load(path) if name == "Logistic 评分卡" else TreeModelBundle.load(path)
        expected = float(manifest["thresholds"][name])
        actual = models[name].candidate.threshold if name == "Logistic 评分卡" else models[name].threshold
        if not np.isclose(expected, actual, rtol=0, atol=1e-12):
            raise ValueError(f"{name} 的阈值与冻结清单不一致。")
    return models


def read_report(report: str, sheet: str, root: Path = PROJECT_ROOT) -> pd.DataFrame:
    if report not in REPORT_FILES:
        raise ValueError(f"未知报表：{report}")
    path = _required(root / "reports" / "excel" / REPORT_FILES[report])
    # 第五阶段工作簿有标题及说明行，表头在第五行。
    return pd.read_excel(path, sheet_name=sheet, header=4 if report == "comparison" else 0)


def read_summary(stage: int, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    path = _required(root / "reports" / f"stage{stage}_summary.json")
    return json.loads(path.read_text(encoding="utf-8"))


def representative_shap(root: Path = PROJECT_ROOT) -> pd.DataFrame:
    rows = read_report("comparison", "SHAP个案", root)
    needed = {"model", "case_type", "source_row_id", "feature_cn", "base_probability", "shap_probability_contribution", "predicted_pd"}
    if not needed.issubset(rows.columns) or rows.empty:
        raise ValueError("第五阶段代表客户 SHAP 明细不完整。")
    if not set(rows["model"]) == {"Random Forest", "XGBoost"}:
        raise ValueError("代表客户 SHAP 缺少树模型。")
    for (_, _, _), group in rows.groupby(["model", "case_type", "source_row_id"]):
        total = float(group["base_probability"].iloc[0] + group["shap_probability_contribution"].sum())
        if abs(total - float(group["predicted_pd"].iloc[0])) > 1e-6:
            raise ValueError("代表客户 SHAP 贡献与预测概率不一致。")
    return rows


def cleaning_anomalies(root: Path = PROJECT_ROOT) -> pd.DataFrame:
    before = read_report("quality", "异常与特殊值", root)
    columns = ["age", "RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "MonthlyIncome", *LATE_FEATURES]
    clean = pd.read_csv(_required(root / "data" / "processed" / "train_clean.csv"), usecols=columns)
    after = {
        "年龄为零或负值": int(clean["age"].le(0).sum()),
        "逾期字段出现96/98": int(clean[list(LATE_FEATURES)].isin([96, 98]).any(axis=1).sum()),
        "额度使用率大于1": int(clean["RevolvingUtilizationOfUnsecuredLines"].gt(1).sum()),
        "债务比率大于1": int(clean["DebtRatio"].gt(1).sum()),
        "月收入为零": int(clean["MonthlyIncome"].eq(0).sum()),
    }
    result = before.loc[:, ["issue", "row_count", "handling"]].copy()
    result["after_count"] = result["issue"].map(after)
    if result["after_count"].isna().any():
        raise ValueError("存在未定义清洗后统计口径的异常项。")
    return result


def validate_customer(values: dict[str, Any]) -> pd.DataFrame:
    missing = set(FEATURES).difference(values)
    if missing:
        raise ValueError(f"客户输入缺少字段：{', '.join(sorted(missing))}")
    normalized: dict[str, float] = {}
    for name in FEATURES:
        value = values[name]
        if value is None or value == "" or pd.isna(value):
            if name in OPTIONAL_FIELDS:
                normalized[name] = np.nan
                continue
            raise ValueError(f"请填写 {name}。")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} 必须是数字。") from exc
        if not np.isfinite(number):
            raise ValueError(f"{name} 必须是有限数值。")
        if number < 0:
            raise ValueError(f"{name} 不能为负数。")
        if name in COUNT_FIELDS and not number.is_integer():
            raise ValueError(f"{name} 必须为整数。")
        normalized[name] = number
    return pd.DataFrame([normalized], columns=FEATURES)


def score_customer(raw: pd.DataFrame, models: dict[str, CreditScorecard | TreeModelBundle]) -> dict[str, Any]:
    card = models["Logistic 评分卡"]
    if not isinstance(card, CreditScorecard):
        raise TypeError("主模型不是已验收的信用评分卡。")
    detail = card.explain(raw).iloc[0]
    points = pd.DataFrame([
        {"feature": name, "points": float(detail[f"points_{name}"])}
        for name in card.candidate.features
    ]).sort_values("points")
    probabilities = {"Logistic 评分卡": float(detail["pd"])}
    for name in ("Random Forest", "XGBoost"):
        probabilities[name] = float(models[name].predict_pd(raw)[0])
    if abs(float(detail["base_points"]) + points["points"].sum() - float(detail["score_raw"])) > 1e-8:
        raise RuntimeError("客户分值贡献与总分不一致，已停止展示。")
    return {
        "pd": float(detail["pd"]),
        "score": int(detail["score"]),
        "score_raw": float(detail["score_raw"]),
        "grade": str(detail["risk_grade"]),
        "base_points": float(detail["base_points"]),
        "points": points.reset_index(drop=True),
        "probabilities": probabilities,
    }


def validation_predictions(root: Path = PROJECT_ROOT) -> pd.DataFrame:
    labels = pd.read_csv(_required(root / "data" / "processed" / "validation_clean.csv"), usecols=["source_row_id", TARGET])
    predictions = pd.read_csv(_required(root / "data" / "features" / "validation_model_predictions.csv"), usecols=["source_row_id", *MODEL_COLUMNS.values()])
    if labels["source_row_id"].duplicated().any() or predictions["source_row_id"].duplicated().any():
        raise ValueError("验证集存在重复的原始行号。")
    merged = predictions.merge(labels, on="source_row_id", how="inner", validate="one_to_one")
    if len(merged) != len(labels) or len(merged) != len(predictions):
        raise ValueError("验证集预测与标签的原始行号不一致。")
    return merged


def validation_curves(data: pd.DataFrame) -> dict[str, dict[str, pd.DataFrame]]:
    y = data[TARGET].to_numpy(dtype=int)
    if not set(np.unique(y)) == {0, 1}:
        raise ValueError("验证集标签必须同时含有好客户和坏客户。")
    result: dict[str, dict[str, pd.DataFrame]] = {}
    for name, column in MODEL_COLUMNS.items():
        p = data[column].to_numpy(dtype=float)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError(f"{name} 验证集概率无效。")
        fpr, tpr, thresholds = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        bad_rate, mean_pd = calibration_curve(y, p, n_bins=10, strategy="quantile")
        finite = np.isfinite(thresholds)
        result[name] = {
            "roc": pd.DataFrame({"x": fpr, "y": tpr}),
            "pr": pd.DataFrame({"x": recall, "y": precision}),
            "ks": pd.DataFrame({"x": thresholds[finite], "y": (tpr - fpr)[finite]}).sort_values("x"),
            "calibration": pd.DataFrame({"x": mean_pd, "y": bad_rate}),
        }
    return result


def strategy_population(root: Path = PROJECT_ROOT) -> pd.DataFrame:
    scores = pd.read_csv(_required(root / "data" / "features" / "train_scorecard.csv"), usecols=["source_row_id", "score"])
    labels = pd.read_csv(_required(root / "data" / "processed" / "train_clean.csv"), usecols=["source_row_id", TARGET])
    if scores["source_row_id"].duplicated().any() or labels["source_row_id"].duplicated().any():
        raise ValueError("训练集评分或标签存在重复的原始行号。")
    merged = scores.merge(labels, on="source_row_id", how="inner", validate="one_to_one")
    if len(merged) != len(scores) or len(merged) != len(labels):
        raise ValueError("训练集评分与标签的原始行号不一致。")
    if merged.empty or not np.isfinite(merged["score"]).all() or not merged[TARGET].isin([0, 1]).all():
        raise ValueError("训练集评分或标签无效。")
    return merged


def strategy_at_cutoff(population: pd.DataFrame, cutoff: int) -> dict[str, float | int | None]:
    passed = population["score"].ge(cutoff)
    bad = population[TARGET].eq(1)
    n_passed = int(passed.sum())
    n_bad = int(bad.sum())
    if n_bad == 0:
        raise ValueError("训练集没有坏客户，无法计算捕获率。")
    return {
        "cutoff": int(cutoff),
        "passed": n_passed,
        "pass_rate": n_passed / len(population),
        "pass_bad_rate": float((passed & bad).sum() / n_passed) if n_passed else None,
        "rejected_bad_capture": float((~passed & bad).sum() / n_bad),
    }


def strategy_curve(population: pd.DataFrame, steps: int = 80) -> pd.DataFrame:
    low, high = int(population["score"].min()), int(population["score"].max()) + 1
    cutoffs = np.unique(np.rint(np.linspace(low, high, min(steps, high - low + 1))).astype(int))
    return pd.DataFrame([strategy_at_cutoff(population, int(value)) for value in cutoffs])
