"""第四阶段 Logistic、评价、PDO 评分与持久化测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_processing.schema import FEATURES
from src.evaluation.metrics import binary_metrics, ks_statistic, select_f1_threshold
from src.feature_engineering.binning import WOEBinningPipeline
from src.feature_engineering.features import CANDIDATE_FEATURES
from src.modeling.logistic import balanced_prior_adjustment
from src.modeling.selection import select_features, vif_table
from src.scorecard.card import CreditScorecard, ScorecardConfig


ROOT = Path(__file__).resolve().parents[1]


def test_ks_and_f1_threshold_are_deterministic() -> None:
    y = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.4, 0.35, 0.8])
    ks, threshold = ks_statistic(y, probability)
    assert ks == pytest.approx(0.5)
    assert threshold == pytest.approx(0.8)
    assert select_f1_threshold(y, probability) == pytest.approx(0.35)
    metrics = binary_metrics(y, probability, 0.35)
    assert metrics["auc"] == pytest.approx(0.75)
    assert metrics["f1"] == pytest.approx(0.8)


def test_pdo_parameters_and_inverse_formula() -> None:
    config = ScorecardConfig()
    base_pd = 1 / 21
    base_score = config.offset - config.factor * np.log(base_pd / (1 - base_pd))
    assert base_score == pytest.approx(600)
    assert np.exp(config.pdo / config.factor) == pytest.approx(2)
    scores = np.array([450.0, 600.0, 750.0])
    pd_values = 1 / (1 + np.exp((scores - config.offset) / config.factor))
    recovered = config.offset - config.factor * np.log(pd_values / (1 - pd_values))
    np.testing.assert_allclose(recovered, scores, atol=1e-10)


def test_balanced_prior_adjustment_uses_training_prevalence() -> None:
    y = np.array([0] * 80 + [1] * 20)
    assert balanced_prior_adjustment(y) == pytest.approx(np.log(4))


def test_training_selection_is_deterministic_for_iv_correlation_vif_direction_and_significance() -> None:
    rng = np.random.default_rng(42)
    rows = 5000
    age = rng.normal(size=rows)
    debt = rng.normal(size=rows)
    correlated = age + rng.normal(scale=0.001, size=rows)
    wrong_direction = rng.normal(size=rows)
    logits = -1.5 * age - debt + 0.8 * wrong_direction - 2.0
    y = pd.Series(rng.binomial(1, 1 / (1 + np.exp(-logits))))
    X = pd.DataFrame({
        "age": age,
        "MonthlyIncome": correlated,
        "DebtRatio": debt,
        "NumberOfDependents": wrong_direction,
        "qa_late_special": rng.integers(0, 2, size=rows),
    })
    screening = pd.DataFrame({
        "feature": X.columns,
        "iv": [0.20, 0.19, 0.10, 0.08, 0.0],
        "psi": [0.0] * X.shape[1],
    })
    selected, log, final_vif, glm = select_features(X, y, screening)
    assert selected == ["age", "DebtRatio"]
    reasons = log.loc[log["action"].eq("排除")].set_index("feature")["stage"]
    assert reasons["qa_late_special"] == "IV_PSI"
    assert reasons["MonthlyIncome"] == "相关性"
    assert reasons["NumberOfDependents"].startswith("方向迭代")
    assert final_vif["vif"].le(5).all()
    assert glm.loc[glm["feature"].ne("const"), "coefficient"].lt(0).all()


@pytest.fixture(scope="module")
def stage4_artifact() -> CreditScorecard:
    path = ROOT / "models" / "stage4_logistic_scorecard.joblib"
    if not path.exists():
        pytest.skip("尚未运行第四阶段，跳过真实产物集成测试。")
    return CreditScorecard.load(path)


def test_real_scorecard_has_expected_direction_and_grades(stage4_artifact: CreditScorecard) -> None:
    assert np.all(stage4_artifact.candidate.coefficients < 0)
    assert np.all(np.diff(stage4_artifact.grade_boundaries) > 0)
    train = pd.read_csv(ROOT / "data" / "features" / "train_woe.csv")
    explained = stage4_artifact.explain_woe(train.loc[:, list(CANDIDATE_FEATURES)])
    assert set(explained["risk_grade"]) == set("ABCDE")
    assert explained["pd"].between(0, 1).all()


def test_pd_score_contributions_and_reload_match(stage4_artifact: CreditScorecard, tmp_path: Path) -> None:
    validation = pd.read_csv(ROOT / "data" / "features" / "validation_woe.csv", nrows=500)
    X = validation.loc[:, list(CANDIDATE_FEATURES)]
    explained = stage4_artifact.explain_woe(X)
    point_columns = [f"points_{feature}" for feature in stage4_artifact.candidate.features]
    np.testing.assert_allclose(
        explained["score_raw"],
        explained["base_points"] + explained[point_columns].sum(axis=1),
        atol=1e-10,
    )
    recovered_pd = 1 / (
        1 + np.exp((explained["score_raw"] - stage4_artifact.config.offset) / stage4_artifact.config.factor)
    )
    np.testing.assert_allclose(recovered_pd, explained["pd"], atol=1e-10)
    path = tmp_path / "scorecard.joblib"
    stage4_artifact.save(path)
    reloaded = CreditScorecard.load(path)
    pd.testing.assert_frame_equal(explained, reloaded.explain_woe(X))


def test_end_to_end_raw_input_matches_frozen_woe(stage4_artifact: CreditScorecard) -> None:
    clean = pd.read_csv(ROOT / "data" / "processed" / "validation_clean.csv")
    usable = clean.loc[
        clean["qa_age_invalid"].eq(0)
        & clean["qa_MonthlyIncome_missing"].eq(0)
        & clean["qa_NumberOfDependents_missing"].eq(0)
        & clean["qa_late_special"].eq(0)
    ].head(100)
    woe = pd.read_csv(ROOT / "data" / "features" / "validation_woe.csv").set_index("source_row_id")
    direct = stage4_artifact.explain_woe(woe.loc[usable["source_row_id"], list(CANDIDATE_FEATURES)].reset_index(drop=True))
    end_to_end = stage4_artifact.explain(usable.loc[:, list(FEATURES)].reset_index(drop=True))
    np.testing.assert_allclose(direct["pd"], end_to_end["pd"], atol=1e-12)
    np.testing.assert_allclose(direct["score_raw"], end_to_end["score_raw"], atol=1e-10)


def test_stage3_rules_unchanged_and_test_labels_reserved(stage4_artifact: CreditScorecard) -> None:
    stage3 = WOEBinningPipeline.load(ROOT / "models" / "stage3_binning.joblib")
    assert stage4_artifact.binner.rule_manifest() == stage3.rule_manifest()
    summary = json.loads((ROOT / "reports" / "stage4_summary.json").read_text(encoding="utf-8"))
    assert summary["test_label_read"] is False
    assert not (ROOT / "data" / "features" / "test_scorecard.csv").exists()
