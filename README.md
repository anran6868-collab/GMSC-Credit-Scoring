# GMSC 金融风控评分卡建模项目

本项目基于 Kaggle **Give Me Some Credit** 数据集，模拟完整的金融信用风险建模流程。预测目标为客户未来两年发生严重逾期的概率，不代表贷款审批、盈利预测或真实银行授信决策。

> 本项目仅用于金融风控建模学习和研究，不作为真实信贷决策依据。

## 技术路线

- 主路线：数据清洗 → 业务特征工程 → 专业分箱 → WOE/IV → Logistic Regression → PD → PDO 信用评分卡 → 风险策略。
- 对照路线：Random Forest、XGBoost，并使用 SHAP 进行解释。
- 展示路线：Streamlit + Plotly 八页面本地应用。
- 交付路线：Notebook、Markdown 知识手册、PDF/Word 报告和 Excel 分析结果。

## 项目结构

```text
GMSC-Credit-Scoring/
├── data/
│   ├── raw/                 # 用户提供的原始数据，只读
│   ├── processed/           # 清洗后的数据
│   └── features/            # 特征工程数据
├── src/
│   ├── data_processing/
│   ├── feature_engineering/
│   ├── modeling/
│   ├── scorecard/
│   └── evaluation/
├── notebooks/               # 教学、实验与展示
├── models/                  # 序列化模型与评分卡规则
├── app/                     # Streamlit 应用（阶段 6 开发）
├── reports/                 # 报告与分析结果
├── tests/                   # 自动化测试
├── docs/                    # 项目文档与知识手册
├── logs/                    # 运行日志（不提交日志内容）
├── requirements.txt
├── AGENTS.md
├── REQUIREMENTS.md
├── PROJECT_PROGRESS.md
├── DECISIONS.md
└── CHANGELOG.md
```

## 当前阶段

阶段 2（EDA 与数据清洗）已完成，当前等待用户验收。阶段 3 尚未开始；目前没有分箱、WOE/IV、模型训练或应用页面实现。

## Windows 环境安装

项目标准环境为 Python 3.11，虚拟环境位于项目根目录 `.venv`。

```powershell
# 使用 Python 3.11 创建环境（首次安装时）
<Python-3.11绝对路径> -m venv .venv

# 安装依赖
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 运行测试
.\.venv\Scripts\python.exe -m pytest
```

`启动GMSC系统.bat` 当前仅为安全占位文件；Streamlit 系统将在阶段 6 开发完成后启用。

## 原始数据

请用户自行从 Kaggle 下载以下文件并放入 `data/raw/`。当前项目中的文件名为：

- `GiveMeSomeCredit-training.csv`（也支持 `cs-training.csv`）
- `GiveMeSomeCredit-testing.csv`（也支持 `cs-test.csv`）
- `Data Dictionary.xls`

项目不会自动下载或修改原始数据，原始数据也不会提交到 Git。无标签的 Kaggle 测试文件不用于模型评价；项目的 70%/15%/15% 三组数据来自有标签的训练文件。

## 第二阶段复现

```powershell
.\.venv\Scripts\python.exe -m src.data_processing.run_stage2
.\.venv\Scripts\python.exe -m pytest -q
```

流程使用固定随机种子 42，先分层划分，再仅用训练集分析和拟合清洗规则。逐行清洗结果、划分清单与规则写入被 Git 忽略的 `data/processed/`；汇总表、20 张图和逐图中文解释位于 `reports/`。两本中文 Notebook 位于 `notebooks/`，包含真实数据运行输出。

## 七阶段开发计划

1. 环境与项目初始化。
2. EDA 与数据清洗。
3. 特征工程与 WOE/IV。
4. Logistic 评分卡与 PDO。
5. Random Forest、XGBoost 与 SHAP。
6. Streamlit 八页面系统。
7. 报告、展示材料与最终验收。

每个阶段完成后停止，等待用户验收后再进入下一阶段。
