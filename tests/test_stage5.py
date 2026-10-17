"""第五阶段树模型、统一评价、SHAP 与测试隔离测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

from src.data_processing.cleaning import CleaningRules
from src.data_processing.schema import FEATURES, TARGET
from src.evaluation.comparison import probability_psi
from src.feature_engineering.features import CANDIDATE_FEATURES, FeatureEngineer
from src.modeling.explainability import select_validation_cases
from src.modeling.run_stage5 import _evaluate_frozen_test
from src.modeling import tree_models
from src.modeling.tree_models import TreeModelBundle, TreeTrainingConfig, train_random_forest


ROOT = Path(__file__).resolve().parents[1]


def test_probability_psi_is_zero_for_identical_distribution() -> None:
    probability = np.linspace(0.001, 0.9, 1000)
    psi, detail = probability_psi(probability, probability)
    assert psi == pytest.approx(0.0, abs=1e-15)
    assert detail["psi_contribution"].abs().max() == pytest.approx(0.0, abs=1e-15)


def test_validation_case_selection_is_deterministic_and_label_free() -> None:
    ids = pd.Series(np.arange(100, 200))
    logistic = np.linspace(0, 1, 100)
    rf = np.linspace(0.1, 0.9, 100)
    xgb = rf.copy()
    xgb[73] = 0.0
    first = select_validation_cases(ids, logistic, rf, xgb)
    second = select_validation_cases(ids, logistic, rf, xgb)
    pd.testing.assert_frame_equal(first, second)
    assert first["source_row_id"].is_unique
    assert set(first["case_type"]) == {"低风险", "中位风险", "高风险", "树模型分歧最大"}


def test_test_labels_cannot_be_read_before_freeze(tmp_path: Path) -> None:
    path = tmp_path / "test.csv"
    pd.DataFrame({TARGET: [0, 1, 0, 1]}).to_csv(path, index=False)
    probabilities = {name: np.array([0.1, 0.8, 0.2, 0.7]) for name in ("a", "b", "c")}
    thresholds = {name: 0.5 for name in probabilities}
    with pytest.raises(RuntimeError, match="冻结清单"):
        _evaluate_frozen_test(path, tmp_path / "missing.json", probabilities, thresholds)
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({"status": "frozen_before_test_label_access"}), encoding="utf-8")
    y, audit = _evaluate_frozen_test(path, frozen, probabilities, thresholds)
    assert y.tolist() == [0, 1, 0, 1]
    assert set(audit["model"]) == set(probabilities)
    assert not any(column.startswith("rank_") for column in audit.columns)
    assert "rank_score" not in audit.columns


def test_tree_bundle_accepts_column_reordering_and_rejects_invalid_input() -> None:
    rng = np.random.default_rng(42)
    raw = pd.DataFrame({feature: rng.uniform(1, 5, 80) for feature in FEATURES})
    raw["age"] = rng.integers(21, 75, 80)
    rules = CleaningRules.fit(raw)
    clean = rules.transform(raw)
    engineered = FeatureEngineer().transform(clean)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(engineered)
    y = pd.Series(rng.integers(0, 2, len(raw)))
    estimator = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42).fit(imputer.transform(engineered), y)
    bundle = TreeModelBundle(
        "Random Forest", estimator, imputer, rules, FeatureEngineer(), tuple(CANDIDATE_FEATURES), 0.5
    )
    baseline = bundle.predict_pd(raw)
    reordered = bundle.predict_pd(raw.loc[:, list(reversed(FEATURES))])
    np.testing.assert_allclose(baseline, reordered, atol=0)
    with pytest.raises(ValueError, match="缺少原始字段"):
        bundle.predict_pd(raw.drop(columns=[FEATURES[0]]))
    invalid = raw.copy()
    invalid.loc[0, FEATURES[0]] = np.inf
    with pytest.raises(ValueError, match="无穷"):
        bundle.predict_pd(invalid)
    with pytest.raises(ValueError, match="未找到树模型文件"):
        TreeModelBundle.load(ROOT / "models" / "不存在的第五阶段模型.joblib")


def test_rf_sampling_oof_selection_and_threshold_are_reproducible(monkeypatch) -> None:
    rng = np.random.default_rng(42)
    raw = pd.DataFrame({feature: rng.uniform(1, 5, 120) for feature in FEATURES})
    raw["age"] = rng.integers(21, 75, len(raw))
    rules = CleaningRules.fit(raw)
    X = FeatureEngineer().transform(rules.transform(raw)).loc[:, list(CANDIDATE_FEATURES)]
    y = pd.Series(np.tile([0, 1], len(raw) // 2))
    monkeypatch.setattr(
        tree_models,
        "RF_SPACE",
        {
            "n_estimators": [8, 10],
            "max_depth": [3],
            "min_samples_leaf": [2],
            "max_features": ["sqrt"],
            "bootstrap": [True],
        },
    )
    config = TreeTrainingConfig(folds=3, candidates_per_model=2, random_state=42)
    first_bundle, first_search, first_oof = train_random_forest(X, y, rules, config)
    second_bundle, second_search, second_oof = train_random_forest(X, y, rules, config)
    pd.testing.assert_frame_equal(
        first_search.drop(columns="fit_seconds"),
        second_search.drop(columns="fit_seconds"),
    )
    np.testing.assert_allclose(first_oof, second_oof, atol=1e-15, rtol=0)
    assert first_bundle.threshold == pytest.approx(second_bundle.threshold, abs=1e-15)
    assert first_bundle.training_metadata["parameters"] == second_bundle.training_metadata["parameters"]


@pytest.fixture(scope="module")
def stage5_artifacts() -> tuple[TreeModelBundle, TreeModelBundle]:
    rf_path = ROOT / "models" / "stage5_random_forest.joblib"
    xgb_path = ROOT / "models" / "stage5_xgboost_pipeline.joblib"
    if not rf_path.exists() or not xgb_path.exists():
        pytest.skip("尚未运行第五阶段，跳过真实产物集成测试。")
    return TreeModelBundle.load(rf_path), TreeModelBundle.load(xgb_path)


def test_real_tree_models_reload_and_predict(stage5_artifacts) -> None:
    rf, xgb = stage5_artifacts
    validation = pd.read_csv(ROOT / "data" / "features" / "validation_features.csv", nrows=300)
    X = validation.loc[:, list(CANDIDATE_FEATURES)]
    for bundle in (rf, xgb):
        probability = bundle.predict_pd_features(X)
        assert probability.shape == (len(X),)
        assert np.isfinite(probability).all()
        assert ((probability >= 0) & (probability <= 1)).all()


def test_xgboost_native_json_reload_matches_pipeline(stage5_artifacts) -> None:
    _, xgb = stage5_artifacts
    validation = pd.read_csv(ROOT / "data" / "features" / "validation_features.csv", nrows=300)
    X = validation.loc[:, list(CANDIDATE_FEATURES)]
    transformed = xgb.transform_features(X)
    native = XGBClassifier()
    native_path = ROOT / "models" / "stage5_xgboost.json"
    native.load_model(bytearray(native_path.read_bytes()))
    expected = xgb.predict_pd_features(X)
    actual = native.predict_proba(transformed)[:, 1]
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


def test_stage5_summary_records_frozen_test_policy(stage5_artifacts) -> None:
    summary = json.loads((ROOT / "reports" / "stage5_summary.json").read_text(encoding="utf-8"))
    frozen = json.loads((ROOT / "models" / "stage5_frozen_manifest.json").read_text(encoding="utf-8"))
    assert summary["primary_model"] == "Logistic 评分卡"
    assert summary["test_used_for_selection"] is False
    assert frozen["status"] == "frozen_before_test_label_access"
    assert len(frozen["sha256"]) == 64
    assert (ROOT / "models" / "stage5_xgboost.json").is_file()
