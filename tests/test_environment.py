"""第一阶段开发环境与项目结构验收测试。"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRECTORIES = (
    "data/raw",
    "data/processed",
    "data/features",
    "src/data_processing",
    "src/feature_engineering",
    "src/modeling",
    "src/scorecard",
    "src/evaluation",
    "notebooks",
    "models",
    "app",
    "reports",
    "tests",
    "docs",
    "logs",
)

REQUIRED_FILES = (
    ".gitignore",
    "requirements.txt",
    "README.md",
    "AGENTS.md",
    "REQUIREMENTS.md",
    "PROJECT_PROGRESS.md",
    "DECISIONS.md",
    "CHANGELOG.md",
    "data/raw/README.md",
    "启动GMSC系统.bat",
)

DEPENDENCY_IMPORTS = (
    "numpy",
    "pandas",
    "sklearn",
    "jupyter_core",
    "matplotlib",
    "seaborn",
    "openpyxl",
    "plotly",
    "streamlit",
    "joblib",
    "pytest",
)


def test_python_version_meets_project_standard() -> None:
    """项目必须在已选定的 Python 3.11 环境中运行。"""
    assert sys.version_info >= (3, 10), "Python 版本必须不低于 3.10"
    assert sys.version_info[:2] == (3, 11), "第一阶段标准环境必须使用 Python 3.11"


@pytest.mark.parametrize("module_name", DEPENDENCY_IMPORTS)
def test_core_dependency_can_be_imported(module_name: str) -> None:
    """第一阶段声明的核心依赖必须可以导入。"""
    importlib.import_module(module_name)


@pytest.mark.parametrize("relative_path", REQUIRED_DIRECTORIES)
def test_required_directory_exists(relative_path: str) -> None:
    """标准项目目录必须存在。"""
    assert (PROJECT_ROOT / relative_path).is_dir(), f"缺少目录：{relative_path}"


@pytest.mark.parametrize("relative_path", REQUIRED_FILES)
def test_required_file_exists(relative_path: str) -> None:
    """第一阶段关键文件必须存在。"""
    assert (PROJECT_ROOT / relative_path).is_file(), f"缺少文件：{relative_path}"
