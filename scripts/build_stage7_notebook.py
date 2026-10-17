"""生成第八本中文教学 Notebook；业务计算均调用 src。"""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/08_完整案例与风险策略.ipynb"


def main() -> None:
    book = nbf.v4.new_notebook()
    book.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    book.cells = [
        nbf.v4.new_markdown_cell(
            "# 第八本教学案例：客户评分与风险策略\n\n"
            "本 Notebook 读取前五阶段已验收的真实 GMSC 产物，展示匿名验证集客户的评分拆解，"
            "并调用第六阶段的只读服务进行训练集样本内策略模拟。不会训练模型、再次评价测试集或输出客户原始记录。"
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n"
            "from src.dashboard.service import load_manifest, strategy_population, strategy_at_cutoff, strategy_curve\n\n"
            "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "manifest = load_manifest(ROOT)\n"
            "summary = json.loads((ROOT / 'reports/stage7_summary.json').read_text(encoding='utf-8'))\n"
            "assert summary['primary_model'] == 'Logistic 评分卡'\n"
            "assert summary['frozen_manifest_sha256'] == json.loads((ROOT / 'models/stage5_frozen_manifest.json').read_text(encoding='utf-8'))['sha256']\n"
            "print('已核对冻结清单；主模型：', summary['primary_model'])"
        ),
        nbf.v4.new_markdown_cell(
            "## 1 匿名真实客户的评分拆解\n\n"
            "从验证集既有评分中选择接近中位分的一条记录。只显示 PD、信用分、风险等级和变量贡献；"
            "不显示原始行号或客户原始字段。这里不重新拟合任何规则。"
        ),
        nbf.v4.new_code_cell(
            "scores = pd.read_csv(ROOT / 'data/features/validation_scorecard.csv')\n"
            "median_score = scores['score_raw'].median()\n"
            "case = scores.loc[(scores['score_raw'] - median_score).abs().idxmin()]\n"
            "overview = pd.DataFrame([{'严重逾期概率 PD': round(float(case['pd']), 6), '信用分': int(case['score']), '风险等级': case['risk_grade']}])\n"
            "print(overview.to_string(index=False))\n"
            "selected = summary['selected_features']\n"
            "contributions = pd.DataFrame({'变量': selected, '积分贡献': [float(case[f'points_{name}']) for name in selected]})\n"
            "assert abs(float(case['base_points']) + contributions['积分贡献'].sum() - float(case['score_raw'])) < 1e-8\n"
            "print('基础分与贡献之和已核对；基础分：', round(float(case['base_points']), 3))\n"
            "print(contributions.sort_values('积分贡献').to_string(index=False))"
        ),
        nbf.v4.new_markdown_cell(
            "## 2 最低通过分教学模拟\n\n"
            "以下口径使用训练集的真实评分与标签。通过客户坏率在无人通过时不可计算；"
            "被拒坏客户捕获率以训练集全部坏客户为分母。样本内结果可能偏乐观，不能当作真实审批策略。"
        ),
        nbf.v4.new_code_cell(
            "population = strategy_population(ROOT)\n"
            "result = strategy_at_cutoff(population, 600)\n"
            "assert result == summary['strategy_at_600_train_only']\n"
            "print(pd.DataFrame([result]).rename(columns={'cutoff': '最低通过分', 'passed': '通过人数', 'pass_rate': '通过率', 'pass_bad_rate': '通过客户坏率', 'rejected_bad_capture': '被拒坏客户捕获率'}).to_string(index=False))\n"
            "curve = strategy_curve(population, steps=30)\n"
            "print(curve[['cutoff', 'pass_rate', 'pass_bad_rate', 'rejected_bad_capture']].iloc[[0, len(curve)//2, len(curve)-1]].to_string(index=False))"
        ),
        nbf.v4.new_markdown_cell(
            "## 3 解读边界\n\n"
            "信用分和风险等级是已冻结评分卡的教学输出；RF/XGBoost 只是模型对照。"
            "上述策略没有验证集或测试集策略效果估计，也不构成真实授信审批结论。"
        ),
    ]
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(book, NOTEBOOK)
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
