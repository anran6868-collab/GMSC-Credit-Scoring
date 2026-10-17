"""模型训练与持久化模块。"""

from .logistic import LogisticCandidate, TrainingConfig, choose_validation_model, train_candidates
from .selection import SelectionConfig, select_features

__all__ = [
    "LogisticCandidate",
    "TrainingConfig",
    "SelectionConfig",
    "choose_validation_model",
    "select_features",
    "train_candidates",
]
"""Logistic 与树模型训练、解释和持久化。"""

from .tree_models import TreeModelBundle, TreeTrainingConfig

__all__ = ["TreeModelBundle", "TreeTrainingConfig"]
