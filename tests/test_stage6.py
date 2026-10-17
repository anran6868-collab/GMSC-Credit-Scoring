"""第六阶段展示服务的只读推理、既有解释与策略口径。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import calibration_curve

from src.dashboard.service import (
    PROJECT_ROOT,
    load_manifest,
    load_models,
    representative_shap,
    score_customer,
    strategy_at_cutoff,
    strategy_population,
    validate_customer,
    validation_curves,
    validation_predictions,
)
from src.data_processing.schema import FEATURES, RAW_ID


def test_missing_frozen_manifest_has_chinese_message(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="缺少已验收产物"):
        load_manifest(tmp_path)


def test_customer_input_preserves_optional_missing_and_special_codes() -> None:
    values = {name: 1 for name in FEATURES}
    values["MonthlyIncome"] = None
    values["NumberOfDependents"] = None
    values["NumberOfTimes90DaysLate"] = 96
    frame = validate_customer(values)
    assert pd.isna(frame.loc[0, "MonthlyIncome"])
    assert pd.isna(frame.loc[0, "NumberOfDependents"])
    assert frame.loc[0, "NumberOfTimes90DaysLate"] == 96
    with pytest.raises(ValueError, match="不能为负数"):
        validate_customer({**values, "age": -1})
    with pytest.raises(ValueError, match="必须为整数"):
        validate_customer({**values, "NumberOfOpenCreditLinesAndLoans": 1.5})


def test_strategy_cutoff_counts_and_empty_pass_group() -> None:
    population = pd.DataFrame({"score": [500, 600, 600, 700], "SeriousDlqin2yrs": [1, 0, 1, 0]})
    at_600 = strategy_at_cutoff(population, 600)
    assert at_600["pass_rate"] == 0.75
    assert at_600["pass_bad_rate"] == pytest.approx(1 / 3)
    assert at_600["rejected_bad_capture"] == 0.5
    at_701 = strategy_at_cutoff(population, 701)
    assert at_701["pass_rate"] == 0
    assert at_701["pass_bad_rate"] is None
    assert at_701["rejected_bad_capture"] == 1


def test_frozen_examples_and_validation_rows_are_consistent() -> None:
    shap = representative_shap()
    assert len(shap.groupby(["model", "case_type", "source_row_id"])) == 8
    validation = validation_predictions()
    assert len(validation) == 22500
    curves = validation_curves(validation)
    assert set(curves) == {"Logistic 评分卡", "Random Forest", "XGBoost"}
    assert all(not curves[name]["roc"].empty for name in curves)
    bad_rate, mean_pd = calibration_curve(
        validation["SeriousDlqin2yrs"], validation["pd_logistic"], n_bins=10, strategy="quantile"
    )
    assert np.allclose(curves["Logistic 评分卡"]["calibration"]["x"], mean_pd)
    assert np.allclose(curves["Logistic 评分卡"]["calibration"]["y"], bad_rate)


def test_saved_models_match_existing_validation_prediction() -> None:
    raw_file = PROJECT_ROOT / "data" / "raw" / "GiveMeSomeCredit-training.csv"
    if not raw_file.is_file():
        raw_file = PROJECT_ROOT / "data" / "raw" / "cs-training.csv"
    if not raw_file.is_file():
        pytest.skip("未提供原始数据，跳过真实客户推理一致性检查。")
    predictions = validation_predictions().iloc[0]
    source_id = int(predictions["source_row_id"])
    raw = pd.read_csv(raw_file, nrows=source_id, usecols=[RAW_ID, *FEATURES])
    row = raw.loc[raw[RAW_ID].eq(source_id), list(FEATURES)]
    assert len(row) == 1
    model_paths = [PROJECT_ROOT / "models" / filename for filename in (
        "stage4_logistic_scorecard.joblib", "stage5_random_forest.joblib", "stage5_xgboost_pipeline.joblib"
    )]
    before = [path.stat().st_mtime_ns for path in model_paths]
    result = score_customer(validate_customer(row.iloc[0].to_dict()), load_models())
    assert result["probabilities"]["Logistic 评分卡"] == pytest.approx(predictions["pd_logistic"], abs=1e-12)
    assert result["probabilities"]["Random Forest"] == pytest.approx(predictions["pd_random_forest"], abs=1e-12)
    assert result["probabilities"]["XGBoost"] == pytest.approx(predictions["pd_xgboost"], abs=1e-7)
    assert np.isfinite(result["score_raw"])
    assert [path.stat().st_mtime_ns for path in model_paths] == before


def test_training_strategy_population_uses_existing_score_and_label() -> None:
    data = strategy_population()
    assert len(data) == 105000
    assert set(data["SeriousDlqin2yrs"].unique()) == {0, 1}
