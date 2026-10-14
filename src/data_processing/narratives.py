"""从真实统计结果生成可核对的中文图表说明。"""

from __future__ import annotations

import numpy as np

from .analysis import EDAResults
from .schema import FEATURES


FEATURE_NAMES = {
    "RevolvingUtilizationOfUnsecuredLines": "无担保循环额度使用率",
    "age": "年龄",
    "NumberOfTime30-59DaysPastDueNotWorse": "30–59天逾期次数",
    "DebtRatio": "债务比率",
    "MonthlyIncome": "月收入",
    "NumberOfOpenCreditLinesAndLoans": "开放信贷账户数",
    "NumberOfTimes90DaysLate": "90天及以上逾期次数",
    "NumberRealEstateLoansOrLines": "房地产贷款或额度数",
    "NumberOfTime60-89DaysPastDueNotWorse": "60–89天逾期次数",
    "NumberOfDependents": "家属人数",
}
FEATURE_MEANINGS = {
    "RevolvingUtilizationOfUnsecuredLines": "额度使用率反映无担保循环信用的使用程度；偏高可能提示资金周转压力。",
    "age": "年龄反映客户所处人生阶段，不能据此直接判断单个客户的信用状况。",
    "NumberOfTime30-59DaysPastDueNotWorse": "30–59天逾期记录反映较早期的还款延迟。",
    "DebtRatio": "债务比率可能反映偿债压力，但其极端值需结合月收入和字段定义解读。",
    "MonthlyIncome": "月收入可提示偿债资源；缺失收入是单独展示的群体。",
    "NumberOfOpenCreditLinesAndLoans": "开放账户数反映信贷使用广度，其风险方向未必单调。",
    "NumberOfTimes90DaysLate": "90天及以上逾期记录反映严重的既往还款问题。",
    "NumberRealEstateLoansOrLines": "房地产贷款或额度数反映信贷结构，不能单独理解为资产或负债水平。",
    "NumberOfTime60-89DaysPastDueNotWorse": "60–89天逾期记录反映较重的既往还款延迟。",
    "NumberOfDependents": "家属人数可能与家庭支出相关，但不等同于真实支出或偿债能力。",
}


def _block(stats: str, meaning: str, conclusion: str, limit: str) -> str:
    return (
        f"**统计结果：** {stats}\n\n"
        f"**风险含义：** {meaning}\n\n"
        f"**结论：** {conclusion}\n\n"
        f"**局限性：** {limit}"
    )


def build_narratives(results: EDAResults, missing_pattern_rows: int) -> dict[str, str]:
    """每张导出图表都有四段说明；文字仅引用实际统计值。"""
    narratives: dict[str, str] = {}
    quality = results.quality
    missing = quality.sort_values("missing_rate", ascending=False)
    top = missing.iloc[0]
    narratives["missing_rate"] = _block(
        f"最高缺失字段为{FEATURE_NAMES[top['feature']]}：{int(top['missing_count']):,}条，"
        f"占训练集的{top['missing_rate']:.2%}。",
        "缺失可能与客户信息记录方式有关；缺失本身需要保留供后续评估。",
        "填补参数只从训练集拟合，清洗前先保留缺失标记。",
        "缺失率不能说明缺失原因，也不能单凭此判断变量风险。",
    )
    narratives["missing_pattern"] = _block(
        f"图中展示按固定随机种子抽取的{missing_pattern_rows:,}条训练记录，"
        "全量缺失数量见缺失率表。",
        "同一行多个字段同时缺失可提示记录模式。",
        "用该图辅助检查缺失是否集中出现。",
        "抽样图不代表精确缺失率；精确比例以全量训练集统计为准。",
    )
    extreme = quality.loc[quality["feature"].isin([
        "MonthlyIncome", "DebtRatio", "RevolvingUtilizationOfUnsecuredLines", "age"
    ])]
    extreme_desc = "；".join(
        f"{FEATURE_NAMES[row.feature]}的99分位={row.p99:,.2f}、最大值={row['max']:,.2f}"
        for _, row in extreme.iterrows()
    )
    narratives["feature_distribution"] = _block(
        extreme_desc + "。图中横轴显示至99分位；债务比率使用log(1+x)尺度，尾部数量另在图中标注。",
        "偏态与极端尾部可能影响后续模型和分箱。",
        "第二阶段保留有业务可能性的极端值，第三阶段再检验分箱稳定性。",
        "截断显示只是绘图方式，原始值及清洗数据未因此被截断。",
    )
    special = results.anomalies.loc[results.anomalies["issue"] == "逾期字段出现96/98"].iloc[0]
    narratives["anomaly_counts"] = _block(
        f"逾期字段96/98特殊值涉及{int(special.row_count):,}条训练记录，"
        f"该组观察坏客户率为{special.bad_rate:.2%}。",
        "特殊编码可能携带风险信息，不能按普通逾期次数解释。",
        "对特殊值建立标记，并使用训练集规则处理原计数字段。",
        "仅凭该数据不能确定96/98的产生原因；高坏率也不证明因果关系。",
    )
    for feature in FEATURES:
        table = results.group_risks[feature]
        stable = table.loc[~table["unstable"]]
        if stable.empty:
            stats = f"{FEATURE_NAMES[feature]}共有{len(table)}组，均未满足稳定性提示阈值。"
            conclusion = "暂不能根据分组坏率判断风险趋势。"
        else:
            high = stable.loc[stable["bad_rate"].idxmax()]
            low = stable.loc[stable["bad_rate"].idxmin()]
            stats = (
                f"{FEATURE_NAMES[feature]}共{len(table)}组；稳定组中，"
                f"“{high['group']}”坏率{high['bad_rate']:.2%}（n={int(high['sample_count']):,}），"
                f"“{low['group']}”坏率{low['bad_rate']:.2%}（n={int(low['sample_count']):,}）。"
            )
            gap = (high["bad_rate"] - low["bad_rate"]) * 100
            conclusion = (
                f"这两组观察坏率相差{gap:.2f}个百分点；可作为第三阶段分箱与稳定性检查的线索。"
            )
        narratives[f"risk_{feature}"] = _block(
            stats,
            FEATURE_MEANINGS[feature],
            conclusion,
            "这是训练集描述性分组；小样本或坏客户少于20人的组已标记，"
            "组间差异不代表因果关系，也不等同于正式WOE分箱。",
        )
    corr = results.correlation.copy()
    stacked = corr.abs().where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    pair = stacked.idxmax()
    value = corr.loc[pair]
    narratives["correlation"] = _block(
        f"训练集绝对相关性最高的一对原始变量为{FEATURE_NAMES[pair[0]]}与"
        f"{FEATURE_NAMES[pair[1]]}，Spearman系数{value:.3f}。",
        "相关变量可能表达相近信息，后续建模需检查冗余。",
        "第三阶段再结合业务意义、IV和共线性证据决定是否保留。",
        "Spearman只反映单调关系；缺失和特殊逾期值已从相关计算中排除，相关不代表因果。",
    )
    for index, (name, table) in enumerate(results.interactions.items(), start=1):
        stable = table.loc[~table["unstable"]]
        if stable.empty:
            stats = f"{name}组合没有达到稳定性提示阈值的单元格。"
            conclusion = "暂不解读二维坏率差异。"
        else:
            high = stable.loc[stable["bad_rate"].idxmax()]
            low = stable.loc[stable["bad_rate"].idxmin()]
            stats = (
                f"{name}中，稳定单元格最高坏率为{high['bad_rate']:.2%}"
                f"（{high['group_a']} × {high['group_b']}，n={int(high['sample_count']):,}）；"
                f"最低为{low['bad_rate']:.2%}"
                f"（{low['group_a']} × {low['group_b']}，n={int(low['sample_count']):,}）。"
            )
            conclusion = "可将差异作为后续特征工程的候选线索，而非直接入模结论。"
        narratives[f"interaction_{index}"] = _block(
            stats,
            "两个变量同时出现时，客户群体的观察风险可能不同。",
            conclusion,
            "稀疏单元格不显示坏率；二维差异未控制其他变量，不能当作因果效应。",
        )
    segments = results.risk_segments
    high_segment = segments.loc[segments["bad_rate"].idxmax()]
    narratives["risk_segments"] = _block(
        f"“{high_segment['segment']}”群体坏率最高，为{high_segment['bad_rate']:.2%}，"
        f"样本数{int(high_segment['sample_count']):,}；训练集整体坏率"
        f"{results.overview.iloc[0]['bad_rate']:.2%}。",
        "历史逾期状态可帮助描述不同客户群体的风险差异。",
        "该群体划分只用于EDA，不构成最终评分等级或审批策略。",
        "分组未控制收入、年龄等共同影响因素，不能直接推断个体风险。",
    )
    top_auc = results.discrimination.iloc[0]
    narratives["single_feature_discrimination"] = _block(
        f"训练集原始单变量方向校正AUC最高的是{FEATURE_NAMES[top_auc['feature']]}："
        f"{top_auc['single_auc']:.3f}；对应KS为{top_auc['single_ks']:.3f}。",
        "单变量排序指标可提示初步风险区分能力。",
        "此排名仅用于确定后续分析重点，不在第二阶段筛选变量。",
        "统计来自同一训练集，未做验证；非单调关系、缺失与特殊值会影响指标，"
        "不能替代第三阶段IV分析或模型验证。",
    )
    return narratives
