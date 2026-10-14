"""第三阶段业务特征工程；所有公式均不读取目标变量。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data_processing.schema import FEATURES, LATE_FEATURES


DERIVED_FEATURES = (
    "total_delinquency_count",
    "severe_delinquency_flag",
    "delinquency_severity_score",
    "real_estate_loan_share",
    "monthly_income_per_capita",
    "estimated_monthly_debt",
    "residual_income_per_capita",
)

QUALITY_FEATURES = (
    "qa_MonthlyIncome_missing",
    "qa_NumberOfDependents_missing",
    "qa_late_special",
)

CANDIDATE_FEATURES = (*FEATURES, *DERIVED_FEATURES, *QUALITY_FEATURES)

FEATURE_DEFINITIONS = {
    "total_delinquency_count": "30–59天、60–89天和90天及以上历史逾期次数之和",
    "severe_delinquency_flag": "60–89天或90天及以上历史逾期次数大于0记为1",
    "delinquency_severity_score": "30–59天次数 + 2×60–89天次数 + 3×90天及以上次数",
    "real_estate_loan_share": "房地产贷款或额度数 ÷ 开放信贷账户数",
    "monthly_income_per_capita": "月收入 ÷（家属人数 + 1）",
    "estimated_monthly_debt": "月收入 × 债务比率",
    "residual_income_per_capita": "月收入 ×（1−债务比率）÷（家属人数 + 1）",
}


@dataclass(frozen=True)
class FeatureEngineer:
    """恢复可审计缺失状态并计算业务衍生变量。"""

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        required = {*FEATURES, *QUALITY_FEATURES, "qa_age_invalid"}
        required.update(f"qa_{feature}_special_code" for feature in LATE_FEATURES)
        missing = sorted(required.difference(data.columns))
        if missing:
            raise ValueError(f"特征工程输入缺少字段：{missing}")

        result = data.loc[:, list(FEATURES)].copy()
        result.loc[data["qa_age_invalid"].eq(1), "age"] = np.nan
        result.loc[data["qa_MonthlyIncome_missing"].eq(1), "MonthlyIncome"] = np.nan
        result.loc[data["qa_NumberOfDependents_missing"].eq(1), "NumberOfDependents"] = np.nan
        for feature in LATE_FEATURES:
            marker = f"qa_{feature}_special_code"
            result.loc[data[marker].ne(0), feature] = np.nan

        late_valid = result.loc[:, list(LATE_FEATURES)].notna().all(axis=1)
        late_total = result.loc[:, list(LATE_FEATURES)].sum(axis=1, min_count=len(LATE_FEATURES))
        result["total_delinquency_count"] = late_total
        result["severe_delinquency_flag"] = (
            result["NumberOfTime60-89DaysPastDueNotWorse"].gt(0)
            | result["NumberOfTimes90DaysLate"].gt(0)
        ).astype(float)
        result.loc[~late_valid, "severe_delinquency_flag"] = np.nan
        result["delinquency_severity_score"] = (
            result["NumberOfTime30-59DaysPastDueNotWorse"]
            + 2 * result["NumberOfTime60-89DaysPastDueNotWorse"]
            + 3 * result["NumberOfTimes90DaysLate"]
        )

        open_credit = result["NumberOfOpenCreditLinesAndLoans"]
        result["real_estate_loan_share"] = (
            result["NumberRealEstateLoansOrLines"] / open_credit.where(open_credit.gt(0))
        )

        income = result["MonthlyIncome"].where(result["MonthlyIncome"].gt(0))
        household_size = result["NumberOfDependents"] + 1
        household_valid = household_size.gt(0)
        result["monthly_income_per_capita"] = income / household_size.where(household_valid)
        result["estimated_monthly_debt"] = income * result["DebtRatio"]
        result["residual_income_per_capita"] = (
            income * (1 - result["DebtRatio"]) / household_size.where(household_valid)
        )

        for feature in QUALITY_FEATURES:
            values = data[feature]
            if not values.isin([0, 1]).all():
                raise ValueError(f"质量标记 {feature} 必须仅包含0或1。")
            result[feature] = values.astype("int8")

        return result.loc[:, list(CANDIDATE_FEATURES)]
