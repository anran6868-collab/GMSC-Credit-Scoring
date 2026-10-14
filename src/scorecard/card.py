"""PDO 信用评分卡、风险等级和端到端可重载推理管线。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_processing.cleaning import CleaningRules
from src.data_processing.schema import FEATURES
from src.feature_engineering.binning import WOEBinningPipeline
from src.feature_engineering.features import FeatureEngineer
from src.modeling.logistic import LogisticCandidate


@dataclass(frozen=True)
class ScorecardConfig:
    base_score: float = 600.0
    pdo: float = 20.0
    base_good_bad_odds: float = 20.0

    @property
    def factor(self) -> float:
        return float(self.pdo / np.log(2.0))

    @property
    def offset(self) -> float:
        return float(self.base_score - self.factor * np.log(self.base_good_bad_odds))


@dataclass
class CreditScorecard:
    cleaning_rules: CleaningRules
    feature_engineer: FeatureEngineer
    binner: WOEBinningPipeline
    candidate: LogisticCandidate
    grade_boundaries: tuple[float, float, float, float]
    config: ScorecardConfig = ScorecardConfig()

    @classmethod
    def build(
        cls,
        cleaning_rules: CleaningRules,
        binner: WOEBinningPipeline,
        candidate: LogisticCandidate,
        train_woe: pd.DataFrame,
        config: ScorecardConfig | None = None,
    ) -> "CreditScorecard":
        cfg = config or ScorecardConfig()
        temporary = cls(cleaning_rules, FeatureEngineer(), binner, candidate, (0, 0, 0, 0), cfg)
        scores = temporary.score_woe(train_woe)
        boundaries = tuple(float(value) for value in np.quantile(scores, [0.2, 0.4, 0.6, 0.8], method="linear"))
        if len(set(boundaries)) != 4:
            distinct = np.unique(scores)
            adjusted = []
            for value in boundaries:
                choices = distinct[distinct > (adjusted[-1] if adjusted else -np.inf)]
                if choices.size == 0:
                    raise RuntimeError("训练分数无法形成五个非空风险等级。")
                adjusted.append(float(choices[np.argmin(np.abs(choices - value))]))
            boundaries = tuple(adjusted)
        if not np.all(np.diff(boundaries) > 0):
            raise RuntimeError("训练分数的风险等级边界不严格递增。")
        card = cls(cleaning_rules, FeatureEngineer(), binner, candidate, boundaries, cfg)
        grades = card.assign_grade(scores)
        if set(grades.unique()) != set("ABCDE"):
            raise RuntimeError("训练分数无法形成五个非空风险等级。")
        return card

    @property
    def base_points(self) -> float:
        return float(self.config.offset - self.config.factor * self.candidate.adjusted_intercept)

    def _raw_to_woe(self, raw: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(set(FEATURES).difference(raw.columns))
        if missing:
            raise ValueError(f"客户输入缺少原始字段：{missing}")
        clean = self.cleaning_rules.transform(raw.loc[:, list(FEATURES)].copy())
        engineered = self.feature_engineer.transform(clean)
        return self.binner.transform(engineered, metric="woe")

    def predict_pd_woe(self, woe: pd.DataFrame) -> np.ndarray:
        return self.candidate.predict_pd(woe)

    def predict_pd(self, raw: pd.DataFrame) -> np.ndarray:
        return self.predict_pd_woe(self._raw_to_woe(raw))

    def score_woe(self, woe: pd.DataFrame) -> np.ndarray:
        logit = self.candidate.decision_function(woe)
        return self.config.offset - self.config.factor * logit

    def score(self, raw: pd.DataFrame) -> np.ndarray:
        return self.score_woe(self._raw_to_woe(raw))

    def assign_grade(self, scores) -> pd.Series:
        values = np.asarray(scores, dtype=float)
        q20, q40, q60, q80 = self.grade_boundaries
        grades = np.select(
            [values > q80, values > q60, values > q40, values > q20],
            ["A", "B", "C", "D"],
            default="E",
        )
        return pd.Series(grades)

    def explain_woe(self, woe: pd.DataFrame, source_row_id=None) -> pd.DataFrame:
        selected = woe.loc[:, list(self.candidate.features)].astype(float)
        coefficients = pd.Series(self.candidate.coefficients, index=self.candidate.features)
        contributions = selected.mul(-self.config.factor * coefficients, axis=1)
        contributions.columns = [f"points_{name}" for name in contributions.columns]
        score_raw = self.base_points + contributions.sum(axis=1).to_numpy()
        probability = self.predict_pd_woe(woe)
        result = pd.DataFrame(index=woe.index)
        if source_row_id is not None:
            result["source_row_id"] = np.asarray(source_row_id)
        result["pd"] = probability
        result["score_raw"] = score_raw
        result["score"] = np.rint(score_raw).astype(int)
        result["risk_grade"] = self.assign_grade(score_raw).to_numpy()
        result["predicted_bad"] = (probability >= self.candidate.threshold).astype(int)
        result["base_points"] = self.base_points
        for feature in self.candidate.features:
            result[f"woe_{feature}"] = selected[feature].to_numpy()
            result[f"points_{feature}"] = contributions[f"points_{feature}"].to_numpy()
        return result.reset_index(drop=True)

    def explain(self, raw: pd.DataFrame, source_row_id=None) -> pd.DataFrame:
        return self.explain_woe(self._raw_to_woe(raw), source_row_id=source_row_id)

    def scorecard_rules(self) -> pd.DataFrame:
        tables = self.binner.binning_tables()
        rows = []
        coefficients = dict(zip(self.candidate.features, self.candidate.coefficients))
        for feature in self.candidate.features:
            table = tables.loc[tables["feature"].eq(feature) & ~tables["table_row"].eq("Totals")]
            for _, row in table.iterrows():
                if int(row["Count"]) <= 0:
                    continue
                woe = float(row["WoE"])
                rows.append({
                    "feature": feature,
                    "bin": str(row["Bin"]),
                    "count": int(row["Count"]),
                    "bad_rate": float(row["Event rate"]),
                    "woe": woe,
                    "coefficient": float(coefficients[feature]),
                    "points": float(-self.config.factor * coefficients[feature] * woe),
                })
        return pd.DataFrame(rows)

    def metadata(self) -> dict[str, object]:
        return {
            "features": list(self.candidate.features),
            "scheme": self.candidate.name,
            "class_weight": self.candidate.class_weight or "none",
            "C": self.candidate.c_value,
            "threshold": self.candidate.threshold,
            "adjusted_intercept": self.candidate.adjusted_intercept,
            "coefficients": dict(zip(self.candidate.features, self.candidate.coefficients.tolist())),
            "grade_boundaries": list(self.grade_boundaries),
            "scorecard": {**asdict(self.config), "factor": self.config.factor, "offset": self.config.offset, "base_points": self.base_points},
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "CreditScorecard":
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError("序列化文件不是 CreditScorecard。")
        return loaded
