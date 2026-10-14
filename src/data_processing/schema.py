"""GMSC 原始数据读取与字段校验。原始文件始终只读。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


TARGET = "SeriousDlqin2yrs"
RAW_ID = "Unnamed: 0"
FEATURES = (
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
)
LATE_FEATURES = (
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
)
TRAIN_NAMES = ("GiveMeSomeCredit-training.csv", "cs-training.csv")
KAGGLE_TEST_NAMES = ("GiveMeSomeCredit-testing.csv", "cs-test.csv")


def locate_raw_file(raw_dir: Path, candidates: tuple[str, ...]) -> Path:
    """按已知文件名寻找用户提供的数据，不下载或重命名。"""
    found = [raw_dir / name for name in candidates if (raw_dir / name).is_file()]
    if len(found) != 1:
        names = "、".join(candidates)
        raise ValueError(f"请在 {raw_dir} 中保留且仅保留一个匹配文件：{names}")
    return found[0]


def load_labeled_training(raw_dir: Path) -> tuple[pd.DataFrame, Path]:
    """读取并校验有标签的 Kaggle 训练文件。"""
    path = locate_raw_file(raw_dir, TRAIN_NAMES)
    data = pd.read_csv(path)
    expected = {RAW_ID, TARGET, *FEATURES}
    missing = expected.difference(data.columns)
    extra = set(data.columns).difference(expected)
    if missing or extra:
        raise ValueError(f"原始训练文件字段不符；缺少：{sorted(missing)}；多出：{sorted(extra)}")
    if data.empty:
        raise ValueError("原始训练文件为空。")
    if data[RAW_ID].isna().any() or not data[RAW_ID].is_unique:
        raise ValueError("原始行号存在缺失或重复，无法可靠地划分数据。")
    if data[TARGET].isna().any() or not data[TARGET].isin([0, 1]).all():
        raise ValueError("目标变量 SeriousDlqin2yrs 必须全部为 0 或 1。")
    if data[TARGET].nunique() != 2:
        raise ValueError("训练文件必须同时包含好客户和坏客户。")
    for feature in FEATURES:
        if not pd.api.types.is_numeric_dtype(data[feature]):
            raise ValueError(f"字段 {feature} 必须为数值型。")
    data = data.rename(columns={RAW_ID: "source_row_id"})
    return data, path


def inspect_unlabeled_file(raw_dir: Path) -> dict[str, object]:
    """仅核对 Kaggle 无标签文件的结构，不将其作为项目测试集。"""
    path = locate_raw_file(raw_dir, KAGGLE_TEST_NAMES)
    data = pd.read_csv(path, usecols=[RAW_ID, TARGET])
    if not data[TARGET].isna().all():
        raise ValueError("Kaggle 测试文件的目标列并非全空，请核对文件来源。")
    return {"file": path.name, "rows": len(data), "label_status": "全部缺失，未用于项目测试集"}
