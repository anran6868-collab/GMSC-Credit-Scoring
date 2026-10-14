"""第三阶段特征公式、泄露边界、分箱、WOE、IV 与 PSI 测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_processing.schema import FEATURES, LATE_FEATURES, TARGET
from src.feature_engineering.binning import (
    ASCENDING_RISK_FEATURES,
    DESCENDING_RISK_FEATURES,
    WOEBinningPipeline,
)
from src.feature_engineering.features import CANDIDATE_FEATURES, FeatureEngineer
from src.feature_engineering.stability import calculate_psi


ROOT = Path(__file__).resolve().parents[1]


def _input_frame() -> pd.DataFrame:
    frame = pd.DataFrame({feature: [1.0, 2.0, 3.0] for feature in FEATURES})
    frame["RevolvingUtilizationOfUnsecuredLines"] = [0.2, 0.5, 0.8]
    frame["age"] = [40.0, 50.0, 60.0]
    frame["NumberOfTime30-59DaysPastDueNotWorse"] = [1.0, 2.0, 0.0]
    frame["NumberOfTime60-89DaysPastDueNotWorse"] = [2.0, 1.0, 0.0]
    frame["NumberOfTimes90DaysLate"] = [3.0, 1.0, 0.0]
    frame["DebtRatio"] = [0.25, 0.5, 0.1]
    frame["MonthlyIncome"] = [8000.0, 9000.0, 7000.0]
    frame["NumberOfOpenCreditLinesAndLoans"] = [10.0, 0.0, 5.0]
    frame["NumberRealEstateLoansOrLines"] = [2.0, 0.0, 1.0]
    frame["NumberOfDependents"] = [1.0, 2.0, 0.0]
    frame["qa_age_invalid"] = [0, 0, 0]
    frame["qa_MonthlyIncome_missing"] = [0, 1, 0]
    frame["qa_NumberOfDependents_missing"] = [0, 0, 1]
    frame["qa_late_special"] = [0, 1, 0]
    for feature in LATE_FEATURES:
        frame[f"qa_{feature}_special_code"] = [0, 96, 0]
    return frame


def test_feature_engineering_formulas_and_missing_propagation() -> None:
    result = FeatureEngineer().transform(_input_frame())
    assert list(result.columns) == list(CANDIDATE_FEATURES)
    first = result.iloc[0]
    assert first["total_delinquency_count"] == 6
    assert first["severe_delinquency_flag"] == 1
    assert first["delinquency_severity_score"] == 14
    assert first["real_estate_loan_share"] == pytest.approx(0.2)
    assert first["monthly_income_per_capita"] == 4000
    assert first["estimated_monthly_debt"] == 2000
    assert first["residual_income_per_capita"] == 3000

    second = result.iloc[1]
    assert pd.isna(second["MonthlyIncome"])
    assert pd.isna(second["total_delinquency_count"])
    assert pd.isna(second["severe_delinquency_flag"])
    assert pd.isna(second["delinquency_severity_score"])
    assert pd.isna(second["real_estate_loan_share"])
    assert pd.isna(second["monthly_income_per_capita"])
    assert pd.isna(second["estimated_monthly_debt"])
    assert pd.isna(second["residual_income_per_capita"])

    third = result.iloc[2]
    assert pd.isna(third["NumberOfDependents"])
    assert pd.isna(third["monthly_income_per_capita"])
    assert third["estimated_monthly_debt"] == 700
    assert pd.isna(third["residual_income_per_capita"])


def test_feature_engineer_does_not_require_target() -> None:
    result = FeatureEngineer().transform(_input_frame())
    assert TARGET not in result.columns


def test_psi_identical_and_shifted_distributions() -> None:
    train = pd.DataFrame({"x": [0, 0, 1, 1], "y": [0, 1, 2, 3]})
    same_summary, _ = calculate_psi(train, train.copy())
    assert np.allclose(same_summary["psi"], 0)
    shifted = pd.DataFrame({"x": [1, 1, 1, 1], "y": [3, 3, 3, 3]})
    shifted_summary, detail = calculate_psi(train, shifted)
    assert shifted_summary["psi"].gt(0).all()
    assert detail.groupby("feature")["psi_contribution"].sum().gt(0).all()


@pytest.fixture(scope="module")
def stage3_artifacts():
    artifact = ROOT / "models" / "stage3_binning.joblib"
    if not artifact.exists():
        pytest.skip("尚未运行第三阶段，跳过真实产物集成测试。")
    pipeline = WOEBinningPipeline.load(artifact)
    train = pd.read_csv(ROOT / "data" / "features" / "train_features.csv")
    validation = pd.read_csv(ROOT / "data" / "features" / "validation_features.csv")
    test = pd.read_csv(ROOT / "data" / "features" / "test_features.csv")
    return pipeline, train, validation, test


def test_real_outputs_keep_split_identity_and_finite_woe(stage3_artifacts) -> None:
    pipeline, train, validation, test = stage3_artifacts
    assert [len(train), len(validation), len(test)] == [105000, 22500, 22500]
    ids = [set(frame["source_row_id"]) for frame in (train, validation, test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    for frame in (train, validation, test):
        transformed = pipeline.transform(frame.loc[:, list(CANDIDATE_FEATURES)])
        assert transformed.shape == (len(frame), len(CANDIDATE_FEATURES))
        assert np.isfinite(transformed.to_numpy()).all()


def test_validation_and_test_changes_cannot_change_fitted_rules(stage3_artifacts) -> None:
    pipeline, _, validation, test = stage3_artifacts
    before = pipeline.rule_manifest()
    for frame in (validation, test):
        changed = frame.loc[:, list(CANDIDATE_FEATURES)].copy()
        changed.iloc[:, :] = 0
        pipeline.transform(changed)
    assert pipeline.rule_manifest() == before


def test_serialized_pipeline_round_trip(stage3_artifacts, tmp_path: Path) -> None:
    pipeline, _, validation, _ = stage3_artifacts
    sample = validation.loc[:99, list(CANDIDATE_FEATURES)]
    expected = pipeline.transform(sample)
    path = tmp_path / "binning.joblib"
    pipeline.save(path)
    actual = WOEBinningPipeline.load(path).transform(sample)
    pd.testing.assert_frame_equal(expected, actual)


def test_real_binning_counts_constraints_and_trends(stage3_artifacts) -> None:
    pipeline, train, _, _ = stage3_artifacts
    tables = pipeline.binning_tables()
    min_count = len(train) * pipeline.config.min_bin_size
    for feature, table in tables.groupby("feature", sort=False):
        total = table.loc[table["table_row"].eq("Totals")].iloc[0]
        assert int(total["Count"]) == len(train)
        regular = table.loc[
            ~table["table_row"].eq("Totals")
            & ~table["Bin"].astype(str).isin(["Special", "Missing"])
            & table["Count"].gt(0)
        ]
        assert len(regular) <= pipeline.config.max_n_bins
        assert regular["Count"].ge(min_count - 1).all()
        assert regular["Event"].ge(pipeline.config.min_bin_n_event).all()
        assert regular["Non-event"].ge(pipeline.config.min_bin_n_nonevent).all()
        rates = regular["Event rate"].to_numpy(dtype=float)
        if feature in ASCENDING_RISK_FEATURES:
            assert np.all(np.diff(rates) >= -1e-12)
        if feature in DESCENDING_RISK_FEATURES:
            assert np.all(np.diff(rates) <= 1e-12)


def test_exported_training_woe_matches_empirical_formula(stage3_artifacts) -> None:
    pipeline, train, _, _ = stage3_artifacts
    X = train.loc[:, list(CANDIDATE_FEATURES)]
    y = train[TARGET]
    bins = pipeline.transform(X, metric="bins")
    woe = pipeline.transform(X, metric="woe")
    total_good = int(y.eq(0).sum())
    total_bad = int(y.eq(1).sum())
    for feature in CANDIDATE_FEATURES:
        check = pd.DataFrame({"bin": bins[feature], "woe": woe[feature], "target": y})
        grouped = check.groupby("bin", dropna=False, observed=True)
        for _, group in grouped:
            good = int(group["target"].eq(0).sum())
            bad = int(group["target"].eq(1).sum())
            assert good > 0 and bad > 0
            expected = np.log((good / total_good) / (bad / total_bad))
            assert group["woe"].nunique() == 1
            assert group["woe"].iloc[0] == pytest.approx(expected, abs=1e-10)
