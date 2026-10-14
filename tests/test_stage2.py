"""第二阶段的数据边界、统计口径与训练集规则测试。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data_processing.analysis import analyze_training
from src.data_processing.cleaning import CleaningRules
from src.data_processing.schema import FEATURES, TARGET, load_labeled_training
from src.data_processing.split import stratified_split


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def real_splits():
    if not (ROOT / "data" / "raw" / "GiveMeSomeCredit-training.csv").exists():
        pytest.skip("未提供真实 GMSC 数据，跳过集成测试。")
    data, _ = load_labeled_training(ROOT / "data" / "raw")
    return data, stratified_split(data)


def test_real_data_schema_and_stratified_split(real_splits) -> None:
    data, splits = real_splits
    assert len(data) == len(splits.train) + len(splits.validation) + len(splits.test)
    assert [len(splits.train), len(splits.validation), len(splits.test)] == [105000, 22500, 22500]
    ids = [set(part["source_row_id"]) for part in (splits.train, splits.validation, splits.test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert set.union(*ids) == set(data["source_row_id"])
    overall = data[TARGET].mean()
    assert all(abs(part[TARGET].mean() - overall) < 0.001 for part in
               (splits.train, splits.validation, splits.test))
    repeated = stratified_split(data)
    assert splits.train["source_row_id"].equals(repeated.train["source_row_id"])


def test_train_only_cleaning_rules_and_special_markers() -> None:
    train = pd.DataFrame({feature: [2.0, 4.0, 6.0] for feature in FEATURES})
    train["age"] = [30.0, 40.0, 50.0]
    train["MonthlyIncome"] = [1000.0, 2000.0, None]
    train["NumberOfDependents"] = [0.0, 1.0, None]
    for feature in (
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
        "NumberOfTime60-89DaysPastDueNotWorse",
    ):
        train[feature] = [0.0, 1.0, 98.0]
    rules = CleaningRules.fit(train)
    validation = train.iloc[[0]].copy()
    validation["age"] = 0.0
    validation["MonthlyIncome"] = None
    validation["NumberOfTimes90DaysLate"] = 96.0
    cleaned = rules.transform(validation)
    assert rules.medians["MonthlyIncome"] == 1500.0
    assert cleaned.iloc[0]["MonthlyIncome"] == 1500.0
    assert cleaned.iloc[0]["age"] == 40.0
    assert cleaned.iloc[0]["NumberOfTimes90DaysLate"] == 0.5
    assert cleaned.iloc[0]["qa_age_invalid"] == 1
    assert cleaned.iloc[0]["qa_late_special"] == 1
    assert cleaned.iloc[0]["qa_NumberOfTimes90DaysLate_special_code"] == 96
    assert cleaned.iloc[0]["qa_MonthlyIncome_missing"] == 1
    assert CleaningRules.fit(train).medians == rules.medians


def test_eda_group_totals_use_training_only(real_splits) -> None:
    _, splits = real_splits
    results = analyze_training(splits.train)
    assert results.overview.iloc[0]["sample_count"] == len(splits.train)
    for table in results.group_risks.values():
        assert table["sample_count"].sum() == len(splits.train)
        assert table["bad_count"].sum() == splits.train[TARGET].sum()
    assert results.risk_segments["sample_count"].sum() == len(splits.train)
