"""GMSC 金融风控评分卡第六阶段本地展示系统。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.service import (
    MODEL_COLUMNS,
    OPTIONAL_FIELDS,
    cleaning_anomalies,
    load_manifest,
    load_models,
    read_report,
    read_summary,
    representative_shap,
    score_customer,
    strategy_at_cutoff,
    strategy_curve,
    strategy_population,
    validate_customer,
    validation_curves,
    validation_predictions,
)
from src.data_processing.schema import FEATURES
from src.feature_engineering.reporting import FEATURE_NAMES


st.set_page_config(page_title="GMSC 信用风险教学系统", page_icon="📊", layout="wide")

NAVIGATION = (
    "首页项目介绍",
    "EDA分析",
    "数据清洗",
    "评分卡分析",
    "模型评价",
    "模型解释",
    "客户信用评分",
    "风险策略模拟",
)
MODEL_COLORS = {"Logistic 评分卡": "#3867a7", "Random Forest": "#d39039", "XGBoost": "#1d957e"}
GRADE_TEXT = {"A": "低风险", "B": "较低风险", "C": "中风险", "D": "较高风险", "E": "高风险"}


@st.cache_data(show_spinner=False)
def report(name: str, sheet: str) -> pd.DataFrame:
    return read_report(name, sheet)


@st.cache_data(show_spinner=False)
def summary(stage: int) -> dict:
    return read_summary(stage)


@st.cache_data(show_spinner=False)
def shap_cases() -> pd.DataFrame:
    return representative_shap()


@st.cache_data(show_spinner=False)
def anomalies() -> pd.DataFrame:
    return cleaning_anomalies()


@st.cache_data(show_spinner=False)
def validation_data() -> pd.DataFrame:
    return validation_predictions()


@st.cache_data(show_spinner=False)
def population() -> pd.DataFrame:
    return strategy_population()


@st.cache_resource(show_spinner="正在加载已验收模型……")
def models():
    return load_models()


def chart(fig: go.Figure, *, height: int = 400) -> None:
    fig.update_layout(
        height=height,
        template="plotly_white",
        font=dict(family="Microsoft YaHei, SimHei, sans-serif", size=13),
        margin=dict(l=18, r=18, t=52, b=20),
        legend_title_text="",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def section(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def home() -> None:
    section("GMSC 金融风控评分卡", "基于真实 Give Me Some Credit 数据的本地教学与研究系统")
    st.info("本项目预测未来两年发生严重逾期的概率。结果仅供教学研究，不构成贷款审批或真实授信决策。")
    stage3 = summary(3)
    stage5 = summary(5)
    cols = st.columns(4)
    cols[0].metric("训练样本", f"{stage3['split_sizes']['train']:,}")
    cols[1].metric("验证样本", f"{stage3['split_sizes']['validation']:,}")
    cols[2].metric("原始字段", "10")
    cols[3].metric("候选特征", str(stage3["candidate_feature_count"]))
    st.subheader("建模与展示路线")
    st.markdown(
        "**分层划分 → 训练集 EDA 与清洗 → 特征工程与专业分箱 → WOE/IV → "
        "Logistic 评分卡 → RF/XGBoost 对照 → 冻结后测试审计 → 本地交互展示**"
    )
    left, right = st.columns(2)
    with left:
        st.markdown("#### 数据与方法")
        st.markdown(
            "数据来源：Kaggle *Give Me Some Credit*。有标签训练文件按目标变量分层随机划分为 "
            "70% 训练、15% 验证和 15% 测试。无标签竞赛测试文件不用于指标评价。"
        )
        st.markdown("主路线为 WOE + Logistic Regression；Random Forest 和 XGBoost 用作对照。")
    with right:
        st.markdown("#### 已冻结结论")
        st.markdown(
            f"主模型：**{stage5['primary_model']}**。验证集最佳树模型挑战者："
            f"**{stage5['best_tree_challenger']}**。模型选择不使用测试集。"
        )
        st.markdown("评分参数：基准分 600、好坏 Odds 20:1、PDO 20；A–E 边界由训练集确定。")


def eda() -> None:
    section("EDA 分析", "以下描述性统计与风险趋势均来自训练集；分组区间不等于正式评分卡分箱。")
    quality = report("quality", "缺失与分布")
    groups = report("quality", "单变量风险")
    options = [name for name in FEATURES if name in groups["feature"].unique()]
    feature = st.selectbox("选择原始变量", options, format_func=lambda name: FEATURE_NAMES.get(name, name))
    missing = quality.sort_values("missing_rate", ascending=False)
    missing["feature_cn"] = missing["feature"].map(FEATURE_NAMES)
    chart(px.bar(missing, x="feature_cn", y="missing_rate", title="原始字段缺失率", labels={"feature_cn": "字段", "missing_rate": "缺失率"}))
    selected = groups.loc[groups["feature"].eq(feature)].copy()
    selected["group"] = selected["group"].astype(str)
    left, right = st.columns(2)
    with left:
        chart(px.bar(selected, x="group", y="sample_count", title=f"{FEATURE_NAMES.get(feature, feature)}：分组样本数", labels={"group": "描述性分组", "sample_count": "样本数"}))
    with right:
        chart(px.line(selected, x="group", y="bad_rate", markers=True, title="分组严重逾期率", labels={"group": "描述性分组", "bad_rate": "坏客户率"}))
    st.caption("小样本或坏客户数不足的分组以 unstable 标记；相关关系不代表因果关系。")
    st.dataframe(selected[["group", "sample_count", "bad_count", "bad_rate", "risk_lift", "unstable"]], hide_index=True, use_container_width=True)


def cleaning() -> None:
    section("数据清洗", "训练集拟合中位数和特殊值处理规则；这里展示已有清洗结果，不重新拟合。")
    comparison = report("quality", "清洗前后")
    anomalies_table = anomalies()
    fig = go.Figure()
    names = comparison["feature"].map(FEATURE_NAMES)
    fig.add_bar(name="清洗前缺失", x=names, y=comparison["before_missing"])
    fig.add_bar(name="清洗后缺失", x=names, y=comparison["after_missing"])
    fig.update_layout(title="原始字段缺失数量变化", barmode="group", xaxis_title="字段", yaxis_title="训练集样本数")
    chart(fig)
    st.dataframe(comparison, hide_index=True, use_container_width=True)
    fig = go.Figure()
    fig.add_bar(name="清洗前", x=anomalies_table["issue"], y=anomalies_table["row_count"])
    fig.add_bar(name="清洗后", x=anomalies_table["issue"], y=anomalies_table["after_count"])
    fig.update_layout(title="异常与特殊取值处理前后", barmode="group", xaxis_title="项目", yaxis_title="训练集样本数")
    chart(fig)
    st.dataframe(anomalies_table, hide_index=True, use_container_width=True)
    st.caption("96/98 逾期特殊编码被标记并填补；极端额度使用率、债务比率与零收入按已验收规则保留。")


def scorecard_analysis() -> None:
    section("评分卡分析", "专业分箱、WOE 和 IV 仅由训练集拟合；下图展示已验收的真实结果。")
    bins = report("binning", "训练分箱明细")
    iv = report("binning", "IV_PSI与初筛").sort_values("iv", ascending=False)
    iv["feature_cn"] = iv["feature"].map(FEATURE_NAMES).fillna(iv["feature"])
    chart(px.bar(iv, x="feature_cn", y="iv", color="screening_status", title="候选变量 IV 排名", labels={"feature_cn": "变量", "iv": "IV"}))
    feature = st.selectbox("选择分箱变量", iv["feature"].tolist(), format_func=lambda name: FEATURE_NAMES.get(name, name))
    rows = bins.loc[bins["feature"].eq(feature) & ~bins["table_row"].eq("Totals")].copy()
    rows["Bin"] = rows["Bin"].astype(str)
    left, right = st.columns(2)
    with left:
        chart(px.bar(rows, x="Bin", y="Event rate", title="分箱坏客户率", labels={"Bin": "分箱", "Event rate": "坏客户率"}))
    with right:
        chart(px.bar(rows, x="Bin", y="WoE", title="分箱 WOE", labels={"Bin": "分箱", "WoE": "WOE"}))
    st.dataframe(rows[["Bin", "Count", "Event", "Event rate", "WoE", "IV"]], hide_index=True, use_container_width=True)
    st.caption("WOE 正值代表该箱好客户占比相对较高；IV 用于单变量参考，最终入模结合相关性、显著性和业务意义。")
    st.subheader("主评分卡规则")
    rules = report("scorecard", "评分卡规则")
    st.dataframe(rules.loc[rules["feature"].eq(feature)] if feature in rules["feature"].values else rules, hide_index=True, use_container_width=True)


def model_evaluation() -> None:
    section("模型评价", "模型选择基于验证集；测试集仅展示第五阶段冻结后的既有审计结果。")
    stage5 = summary(5)
    validation = pd.DataFrame(stage5["validation_metrics"])
    display = validation[["model", "auc", "ks", "pr_auc", "precision", "recall", "f1", "brier", "log_loss", "probability_psi"]]
    st.dataframe(display, hide_index=True, use_container_width=True)
    curves = validation_curves(validation_data())
    tabs = st.tabs(["ROC", "KS", "PR 曲线", "概率校准"])
    specs = [
        ("roc", "假阳性率", "真正率", "验证集 ROC"),
        ("ks", "PD 阈值", "真正率−假阳性率", "验证集 KS"),
        ("pr", "召回率", "精确率", "验证集 PR 曲线"),
        ("calibration", "平均预测 PD", "实际坏客户率", "验证集概率校准"),
    ]
    for tab, (key, x_title, y_title, title) in zip(tabs, specs):
        with tab:
            fig = go.Figure()
            for name in MODEL_COLUMNS:
                data = curves[name][key]
                fig.add_trace(go.Scatter(x=data["x"], y=data["y"], mode="lines+markers" if key == "calibration" else "lines", name=name, line_color=MODEL_COLORS[name]))
            if key in {"roc", "calibration"}:
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#999", dash="dash"), name="参考线"))
            fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
            chart(fig, height=500)
    st.caption("AUC、KS、PR-AUC 等数字取自第五阶段冻结报告；曲线由相同验证集逐行预测绘制。")
    with st.expander("查看冻结后的测试集最终审计"):
        audit = pd.DataFrame(stage5["test_metrics_frozen_audit"])
        st.dataframe(audit[["model", "auc", "ks", "pr_auc", "precision", "recall", "f1", "brier"]], hide_index=True, use_container_width=True)
        st.caption("此表不是第二轮选模结果，不据此调整模型或阈值。")


def model_explanation() -> None:
    section("模型解释", "Logistic 解释评分贡献；RF/XGBoost 的 SHAP 来自第五阶段已计算的代表客户。")
    coefficients = report("scorecard", "最终系数")
    rules = report("scorecard", "评分卡规则")
    st.subheader("Logistic 系数与分值")
    left, right = st.columns(2)
    with left:
        coefficients["feature_cn"] = coefficients["feature"].map(FEATURE_NAMES).fillna(coefficients["feature"])
        chart(px.bar(coefficients, x="feature_cn", y="coefficient", title="最终模型系数", labels={"feature_cn": "变量", "coefficient": "系数"}))
    with right:
        points = rules.groupby("feature", as_index=False)["points"].agg(["min", "max"]).reset_index()
        points["feature_cn"] = points["feature"].map(FEATURE_NAMES).fillna(points["feature"])
        chart(px.bar(points, x="feature_cn", y="min", title="各变量分箱最低贡献分", labels={"feature_cn": "变量", "min": "贡献分"}))
    st.caption("分值与变量的分箱有关；系数和单客户贡献需结合 WOE 理解。")
    st.subheader("代表客户概率空间 SHAP")
    shap = shap_cases()
    choice1, choice2 = st.columns(2)
    with choice1:
        model = st.selectbox("树模型", ["Random Forest", "XGBoost"])
    with choice2:
        case = st.selectbox("代表客户", shap["case_type"].drop_duplicates().tolist())
    selected = shap.loc[shap["model"].eq(model) & shap["case_type"].eq(case)].copy()
    selected = selected.sort_values("shap_probability_contribution")
    baseline = float(selected["base_probability"].iloc[0])
    prediction = float(selected["predicted_pd"].iloc[0])
    st.caption(f"源行号 {int(selected['source_row_id'].iloc[0])} · 基准概率 {baseline:.2%} · 模型 PD {prediction:.2%}")
    fig = px.bar(selected, x="shap_probability_contribution", y="feature_cn", orientation="h", color="shap_probability_contribution", color_continuous_scale=["#328b74", "#ececec", "#c65d58"], title="对严重逾期概率的贡献")
    fig.update_layout(xaxis_title="SHAP 概率贡献", yaxis_title="变量")
    chart(fig, height=650)
    st.dataframe(selected[["feature_cn", "raw_feature_value", "model_input_value", "shap_probability_contribution"]], hide_index=True, use_container_width=True)
    st.caption("正贡献提高此模型的严重逾期概率，负贡献降低概率。SHAP 为模型解释，不代表因果关系；此例不适用于其他客户。")


def customer_scoring() -> None:
    section("客户信用评分", "输入十个原始字段；评分与等级由 Logistic 主评分卡给出，树模型只提供对照 PD。")
    st.info("请勿输入真实可识别个人信息。空白月收入或家属人数将按已验收的训练集规则处理。")
    with st.form("customer_form"):
        columns = st.columns(2)
        values: dict[str, float | int | None] = {}
        for index, name in enumerate(FEATURES):
            with columns[index % 2]:
                label = FEATURE_NAMES.get(name, name)
                if name in OPTIONAL_FIELDS:
                    label += "（可留空）"
                is_int = name in {"age", "NumberOfTime30-59DaysPastDueNotWorse", "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate", "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents"}
                if is_int:
                    values[name] = st.number_input(label, min_value=0, value=None, step=1, key=f"customer_{name}")
                else:
                    values[name] = st.number_input(label, min_value=0.0, value=None, step=0.000001, format="%.9f", key=f"customer_{name}")
        submitted = st.form_submit_button("计算信用评分", type="primary")
    if submitted:
        try:
            raw = validate_customer(values)
            st.session_state["scoring_result"] = score_customer(raw, models())
        except (ValueError, FileNotFoundError, RuntimeError, TypeError) as exc:
            st.session_state.pop("scoring_result", None)
            st.error(str(exc))
    result = st.session_state.get("scoring_result")
    if not result:
        return
    st.caption("以下为上次提交的输入所对应的评分；修改字段后请再次点击“计算信用评分”。")
    cols = st.columns(3)
    cols[0].metric("未来两年严重逾期概率", f"{result['pd']:.2%}")
    cols[1].metric("信用评分", str(result["score"]))
    cols[2].metric("风险等级", f"{result['grade']} 级 · {GRADE_TEXT[result['grade']]}")
    st.subheader("主要评分因素")
    points = result["points"].copy()
    points["feature_cn"] = points["feature"].map(FEATURE_NAMES).fillna(points["feature"])
    chart(px.bar(points, x="points", y="feature_cn", orientation="h", title="Logistic 各变量贡献分", labels={"points": "相对于基准分的贡献", "feature_cn": "变量"}), height=500)
    negative = points.loc[points["points"].lt(0)].head(3)
    if negative.empty:
        st.write("此客户没有负向分值贡献；请结合完整变量贡献理解结果。")
    else:
        st.write("主要扣分项：" + "、".join(negative["feature_cn"].tolist()) + "。这些是模型内贡献，并非因果判断。")
    st.caption(f"基准分 {result['base_points']:.2f} ＋ 各变量贡献分 = 未取整总分 {result['score_raw']:.2f}。")
    st.subheader("模型 PD 对照")
    st.dataframe(pd.DataFrame([{"model": name, "pd": value} for name, value in result["probabilities"].items()]), hide_index=True, use_container_width=True)
    st.caption("RF 和 XGBoost 不生成此评分卡的信用分或 A–E 等级。此页面不计算当前客户的 SHAP。")


def risk_strategy() -> None:
    section("风险策略模拟", "基于训练集真实标签与已冻结的 Logistic 评分；结果是样本内教学模拟，可能偏乐观。")
    data = population()
    lower = int(data["score"].min())
    upper = int(data["score"].max()) + 1
    cutoff = st.slider("最低通过分（整数）", min_value=lower, max_value=upper, value=int(data["score"].median()), step=1)
    result = strategy_at_cutoff(data, cutoff)
    cols = st.columns(3)
    cols[0].metric("通过率", f"{result['pass_rate']:.2%}")
    cols[1].metric("通过客户坏率", "无可计算样本" if result["pass_bad_rate"] is None else f"{result['pass_bad_rate']:.2%}")
    cols[2].metric("被拒坏客户捕获率", f"{result['rejected_bad_capture']:.2%}")
    curve = strategy_curve(data)
    fig = go.Figure()
    for name, label, color in (
        ("pass_rate", "通过率", "#3867a7"),
        ("pass_bad_rate", "通过客户坏率", "#d39039"),
        ("rejected_bad_capture", "被拒坏客户捕获率", "#c65d58"),
    ):
        fig.add_trace(go.Scatter(x=curve["cutoff"], y=curve[name], mode="lines", name=label, line_color=color))
    fig.add_vline(x=cutoff, line_dash="dash", line_color="#444")
    fig.update_layout(title="最低通过分与策略指标", xaxis_title="最低通过分", yaxis_title="比例", yaxis_tickformat=".0%")
    chart(fig, height=500)
    st.caption("通过：整数信用分 ≥ 最低通过分。被拒坏客户捕获率 = 被拒的坏客户数 ÷ 全部坏客户数；不是审批或盈利建议。")


PAGES = {
    "首页项目介绍": home,
    "EDA分析": eda,
    "数据清洗": cleaning,
    "评分卡分析": scorecard_analysis,
    "模型评价": model_evaluation,
    "模型解释": model_explanation,
    "客户信用评分": customer_scoring,
    "风险策略模拟": risk_strategy,
}


def main() -> None:
    st.markdown(
        """<style>
        .block-container {padding-top: 2rem; max-width: 1280px;}
        h1 {color: #173c5a; letter-spacing: -.02em;}
        h2, h3 {color: #204c65;}
        div[data-testid="stMetric"] {background: rgba(125, 153, 174, .13); border: 1px solid rgba(125, 153, 174, .35); border-radius: 12px; padding: 14px;}
        </style>""",
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.markdown("## GMSC 评分卡")
        page = st.radio("页面导航", NAVIGATION, label_visibility="collapsed")
        st.caption("本地教学研究 · 已验收模型只读")
    if page != "客户信用评分":
        st.session_state.pop("scoring_result", None)
    try:
        load_manifest()
        PAGES[page]()
    except (FileNotFoundError, ValueError, KeyError, RuntimeError, OSError) as exc:
        st.error(f"页面无法加载：{exc}")
        st.info("请核对本地已验收产物及字段；应用不会自动下载数据或重新训练。")


if __name__ == "__main__":
    main()
