"""仅对有标签的 Kaggle 训练文件进行分层划分。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from .schema import TARGET


SEED = 42


@dataclass(frozen=True)
class DataSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def stratified_split(data: pd.DataFrame, seed: int = SEED) -> DataSplits:
    """按 70% / 15% / 15% 分层；测试集只留待最终评价。"""
    train, remainder = train_test_split(
        data, test_size=0.30, random_state=seed, stratify=data[TARGET]
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.50,
        random_state=seed,
        stratify=remainder[TARGET],
    )
    return DataSplits(
        train=train.sort_values("source_row_id").reset_index(drop=True),
        validation=validation.sort_values("source_row_id").reset_index(drop=True),
        test=test.sort_values("source_row_id").reset_index(drop=True),
    )
