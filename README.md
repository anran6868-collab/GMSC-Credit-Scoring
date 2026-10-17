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
│   ├── evaluation/
│   └── dashboard/             # 第六阶段只读展示服务
├── notebooks/               # 教学、实验与展示
├── models/                  # 序列化模型与评分卡规则
├── app/                     # Streamlit 八页面本地应用
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

七阶段计划已于 2026-10-18 全部完成并通过用户验收。Logistic 评分卡继续作为主模型，XGBoost 是验证集综合比较后的最佳树模型挑战者；测试集已在冻结后完成一次性审计。

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

## 第六阶段本地启动

双击桌面上的“启动 GMSC 系统”快捷方式，或双击项目根目录的 `启动GMSC系统.bat`。首次启动会自动打开浏览器；若网页已在运行，再次双击会直接打开原网页，不会另占一个端口。保持启动窗口开启即可继续使用，关闭窗口后网页服务停止。

也可以在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m src.dashboard.launcher
```

系统只在本机开放。首页、EDA、清洗、评分卡、模型评价、模型解释、客户评分和风险策略模拟共八页；页面不会训练模型。客户评分以 Logistic 的 PD、分数、等级及贡献为主，RF/XGBoost 只提供对照 PD；SHAP 只展示第五阶段已经计算的代表客户。风险策略页使用训练集真实评分和标签，属于可能偏乐观的样本内教学模拟。

运行需要本机保留前五阶段生成的 `models/` 序列化产物，以及 Git 忽略的 `data/processed/`、`data/features/` 文件。缺少文件时应用会显示中文提示，不会自动下载数据或重新训练。代码和已允许的汇总报告可进入版本库；原始数据、逐行数据、模型文件和日志不得提交。

## 第七阶段成果与复核

- 正式报告：[Word](reports/final/GMSC_建模总结报告.docx) 与 [PDF](reports/final/GMSC_建模总结报告.pdf)。
- 展示材料：[10 页 PDF 汇报稿](reports/final/GMSC_项目汇报稿.pdf)。
- 数据汇总：[综合 Excel](reports/excel/stage7_comprehensive_report.xlsx)及[第五阶段仓库版 Excel](reports/excel/stage5_model_comparison_public.xlsx)。数据质量、第三和第四阶段工作簿也保留在 `reports/excel/`。
- 教学材料：[第 8 本 Notebook](notebooks/08_完整案例与风险策略.ipynb) 与 [金融风控知识手册](docs/金融风控知识手册.md)。
- [第七阶段验收清单](reports/final/第七阶段验收清单.md)列出交付文件、已完成检查和待确认事项。

汇总来源为已冻结的第三至第五阶段报告和训练集策略数据，不重新训练或再次评价测试集。复核步骤：

```powershell
.\.venv\Scripts\python.exe -m src.reporting.stage7
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/execute_notebooks_inprocess.py notebooks/08_完整案例与风险策略.ipynb --working-directory .
```

## GitHub 仓库与本地产物

仓库保留源代码、测试、教学 Notebook、汇总表、图表和最终报告。第五阶段仓库版 Excel 保留 15 个汇总工作表，移除了包含真实个案源行号及原始特征值的 `SHAP个案`、`解释客户` 两张表。第 7、8 本 Notebook 的个案输出不保存在版本库；本地具备真实数据产物时可运行代码重新生成。完整第五阶段 Excel 留在本机，可通过 `scripts/build_stage5_excel.mjs` 从本地忽略的阶段表重建；加上 `--public` 可生成仓库版。

原始数据、分组后的逐行数据、序列化模型、个案 SHAP 明细和日志均不上传。GitHub 克隆版本因此不能直接打开需要本地模型与数据文件的 Streamlit 页面，也不能直接运行全部依赖数据的测试或 Notebook。按“原始数据”一节自行准备 Kaggle 文件后，在本地运行各阶段脚本重建产物；数据不由项目自动下载。

Excel 排版使用本地 `@oai/artifact-tool` 运行 `scripts/build_stage7_excel.mjs`。Word/PDF 排版模块为 `src.reporting.documents`，需要 `requirements.txt` 新增的文档依赖；当前受限环境中使用 Codex 随附的本地文档运行时，不上传数据。本机安装 Word 时，可运行 `scripts/export_stage7_word_pdf.ps1` 从 DOCX 导出完全同内容的正式 PDF；本次交付采用了该版本。文档生成后须逐页渲染检查。PDF 汇报稿为固定版式，不提供 PPTX。

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

## 第三阶段复现

```powershell
.\.venv\Scripts\python.exe -m src.feature_engineering.run_stage3
.\.venv\Scripts\python.exe -m pytest -q
```

流程保留 10 个原始变量，增加 7 个业务衍生变量并评估 3 个质量标记。OptimalBinning、WOE 和 IV 只使用训练集拟合；验证集只计算不使用标签的 PSI，测试集只应用规则并检查转换完整性。逐行特征、分箱和 WOE 数据写入被 Git 忽略的 `data/features/`，本地分箱器写入被忽略的 `models/`；综合 Excel、22 张图和逐变量中文解释位于 `reports/`。

## 第四阶段复现

```powershell
.\.venv\Scripts\python.exe -m src.modeling.run_stage4
.\.venv\Scripts\python.exe -m pytest -q
```

流程只使用训练集完成变量筛选、五折交叉验证、参数搜索和拟合，验证集用于比较原始比例与类别权重方案。最终评分卡采用基准分 600、好坏 Odds 20:1、PDO 20，并根据训练分数确定 A–E 等级。模型和逐行评分写入 Git 忽略目录；综合 Excel、8 张图、中文解释和第五本 Notebook 为可提交交付。第四阶段不读取测试集标签。

## 第五阶段复现

```powershell
.\.venv\Scripts\python.exe -m src.modeling.run_stage5
.\.venv\Scripts\python.exe scripts\execute_notebooks_inprocess.py notebooks\06_RF_XGBoost模型对比.ipynb notebooks\07_SHAP模型解释.ipynb --working-directory .
.\.venv\Scripts\python.exe -m pytest -q
```

流程使用训练集完成 RF/XGBoost 的 16 组参数抽样、五折 OOF 比较、阈值冻结、XGBoost 折内早停和最终拟合；验证集统一比较 Logistic、RF 与 XGBoost，并生成概率校准、PSI、复杂度、特征重要性和概率空间 SHAP。冻结清单写入后才一次读取测试标签进行最终审计，测试结果不产生排名、不用于返工调参。模型与逐行预测位于 Git 忽略目录；综合 Excel、12 张图、中文说明和两本已执行 Notebook 为可提交交付。

## 七阶段开发计划

1. 环境与项目初始化。
2. EDA 与数据清洗。
3. 特征工程与 WOE/IV。
4. Logistic 评分卡与 PDO。
5. Random Forest、XGBoost 与 SHAP。
6. Streamlit 八页面系统。
7. 报告、展示材料与最终验收。

每个阶段完成后停止，等待用户验收后再进入下一阶段。
