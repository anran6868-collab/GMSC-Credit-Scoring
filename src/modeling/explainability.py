"""RF 与 XGBoost 的概率空间 SHAP 全局及局部解释。"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

_MPL_CACHE = Path(tempfile.gettempdir()) / "gmsc-matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.feature_engineering.reporting import FEATURE_NAMES

from .tree_models import TreeModelBundle


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class ShapResult:
    model_name: str
    evaluation_rows: pd.DataFrame
    transformed_rows: pd.DataFrame
    values: np.ndarray
    global_transformed_rows: pd.DataFrame
    global_values: np.ndarray
    base_value: float
    global_table: pd.DataFrame
    local_table: pd.DataFrame
    max_additivity_error: float


def select_validation_cases(
    source_row_ids: pd.Series,
    logistic_probability: np.ndarray,
    rf_probability: np.ndarray,
    xgb_probability: np.ndarray,
) -> pd.DataFrame:
    ids = np.asarray(source_row_ids)
    logistic = np.asarray(logistic_probability, dtype=float)
    labels = [("低风险", 0.10), ("中位风险", 0.50), ("高风险", 0.90)]
    selected: list[dict[str, object]] = []
    used: set[int] = set()
    for label, quantile in labels:
        target = float(np.quantile(logistic, quantile))
        order = np.argsort(np.abs(logistic - target), kind="stable")
        index = next(int(value) for value in order if int(value) not in used)
        used.add(index)
        selected.append({"case_type": label, "row_position": index, "source_row_id": ids[index]})
    disagreement_order = np.argsort(-np.abs(np.asarray(rf_probability) - np.asarray(xgb_probability)), kind="stable")
    index = next(int(value) for value in disagreement_order if int(value) not in used)
    selected.append({"case_type": "树模型分歧最大", "row_position": index, "source_row_id": ids[index]})
    return pd.DataFrame(selected)


def _positive_class(values, expected_value) -> tuple[np.ndarray, float]:
    array = np.asarray(values, dtype=float)
    base = np.asarray(expected_value, dtype=float)
    if array.ndim == 3:
        if array.shape[2] != 2:
            raise RuntimeError(f"二分类 SHAP 输出类别维度异常：{array.shape}")
        array = array[:, :, 1]
        base_value = float(base.reshape(-1)[1])
    elif array.ndim == 2:
        base_value = float(base.reshape(-1)[0])
    else:
        raise RuntimeError(f"SHAP 输出维度异常：{array.shape}")
    return array, base_value


def explain_tree_model(
    bundle: TreeModelBundle,
    train_features: pd.DataFrame,
    validation_features: pd.DataFrame,
    source_row_ids: pd.Series,
    cases: pd.DataFrame,
    *,
    random_state: int = 42,
    background_size: int = 256,
    global_size: int = 2000,
) -> ShapResult:
    rng = np.random.default_rng(random_state)
    background_positions = np.sort(rng.choice(len(train_features), size=min(background_size, len(train_features)), replace=False))
    evaluation_positions = np.sort(rng.choice(len(validation_features), size=min(global_size, len(validation_features)), replace=False))
    union_positions = np.unique(np.concatenate((evaluation_positions, cases["row_position"].to_numpy(dtype=int))))
    background = bundle.transform_features(train_features.iloc[background_positions])
    evaluation_raw = validation_features.iloc[union_positions].copy()
    transformed = bundle.transform_features(evaluation_raw)
    explainer = shap.TreeExplainer(
        bundle.estimator,
        data=background,
        feature_perturbation="interventional",
        model_output="probability",
    )
    explanation = explainer(transformed, check_additivity=False)
    values, base_value = _positive_class(explanation.values, explanation.base_values)
    probability = bundle.predict_pd_features(evaluation_raw)
    max_error = float(np.max(np.abs(base_value + values.sum(axis=1) - probability)))
    if max_error > 1e-6:
        raise RuntimeError(f"{bundle.model_name} SHAP 概率加总误差 {max_error:.3e} 超过1e-6。")

    position_to_local = {int(position): idx for idx, position in enumerate(union_positions)}
    global_local_positions = [position_to_local[int(position)] for position in evaluation_positions]
    global_values = values[global_local_positions]
    global_table = pd.DataFrame({
        "model": bundle.model_name,
        "feature": bundle.feature_names,
        "feature_cn": [FEATURE_NAMES.get(name, name) for name in bundle.feature_names],
        "mean_abs_shap_probability": np.mean(np.abs(global_values), axis=0),
        "mean_signed_shap_probability": np.mean(global_values, axis=0),
    }).sort_values("mean_abs_shap_probability", ascending=False).reset_index(drop=True)

    local_rows: list[dict[str, object]] = []
    ids = np.asarray(source_row_ids)
    for _, case in cases.iterrows():
        position = int(case["row_position"])
        local_index = position_to_local[position]
        for feature_index, feature in enumerate(bundle.feature_names):
            local_rows.append({
                "model": bundle.model_name,
                "case_type": case["case_type"],
                "source_row_id": ids[position],
                "feature": feature,
                "feature_cn": FEATURE_NAMES.get(feature, feature),
                "raw_feature_value": evaluation_raw.iloc[local_index][feature],
                "model_input_value": transformed.iloc[local_index][feature],
                "base_probability": base_value,
                "shap_probability_contribution": values[local_index, feature_index],
                "predicted_pd": probability[local_index],
            })
    local_table = pd.DataFrame(local_rows)
    evaluation_rows = pd.DataFrame({
        "row_position": union_positions,
        "source_row_id": ids[union_positions],
        "pd": probability,
    })
    return ShapResult(
        model_name=bundle.model_name,
        evaluation_rows=evaluation_rows,
        transformed_rows=transformed,
        values=values,
        global_transformed_rows=transformed.iloc[global_local_positions].reset_index(drop=True),
        global_values=global_values,
        base_value=base_value,
        global_table=global_table,
        local_table=local_table,
        max_additivity_error=max_error,
    )


def create_shap_charts(result: ShapResult, cases: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "rf" if result.model_name == "Random Forest" else "xgboost"
    paths: dict[str, Path] = {}
    ranked = result.global_table.sort_values("mean_abs_shap_probability").tail(15)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(ranked["feature_cn"], ranked["mean_abs_shap_probability"], color="#4f779f")
    ax.set(xlabel="平均绝对 SHAP 概率贡献", title=f"{result.model_name} 全局 SHAP 重要性")
    fig.tight_layout()
    path = output_dir / f"{safe_name}_shap_global_bar.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["global_bar"] = path

    shap.summary_plot(
        result.global_values,
        result.global_transformed_rows,
        feature_names=[FEATURE_NAMES.get(name, name) for name in result.global_transformed_rows.columns],
        show=False,
        max_display=15,
    )
    fig = plt.gcf()
    plt.gca().set_xlabel("SHAP 严重逾期概率贡献")
    fig.suptitle(f"{result.model_name} 全局 SHAP 方向", y=1.01)
    path = output_dir / f"{safe_name}_shap_beeswarm.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["beeswarm"] = path

    return paths
