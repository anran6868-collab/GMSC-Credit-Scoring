"""运行第二阶段：划分、训练集 EDA、清洗及报告。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .analysis import analyze_training
from .cleaning import CleaningRules
from .narratives import FEATURE_NAMES, build_narratives
from .plots import create_eda_charts
from .schema import inspect_unlabeled_file, load_labeled_training
from .split import SEED, stratified_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_excel(path: Path, results, split_sizes: dict[str, int], comparison: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    group_risk = pd.concat(results.group_risks.values(), ignore_index=True)
    interactions = pd.concat(
        [table.assign(interaction=name) for name, table in results.interactions.items()],
        ignore_index=True,
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame([split_sizes]).to_excel(writer, sheet_name="划分规模", index=False)
        results.overview.to_excel(writer, sheet_name="训练集概览", index=False)
        results.quality.to_excel(writer, sheet_name="缺失与分布", index=False)
        results.anomalies.to_excel(writer, sheet_name="异常与特殊值", index=False)
        comparison.to_excel(writer, sheet_name="清洗前后", index=False)
        group_risk.to_excel(writer, sheet_name="单变量风险", index=False)
        results.correlation.to_excel(writer, sheet_name="相关性")
        interactions.to_excel(writer, sheet_name="二维风险", index=False)
        results.risk_segments.to_excel(writer, sheet_name="风险群体", index=False)
        results.discrimination.to_excel(writer, sheet_name="初步区分能力", index=False)


def _comparison(raw: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in raw.columns.intersection(cleaned.columns):
        if feature in {"source_row_id", "SeriousDlqin2yrs"}:
            continue
        rows.append({
            "feature": feature,
            "before_missing": int(raw[feature].isna().sum()),
            "after_missing": int(cleaned[feature].isna().sum()),
            "changed_values": int((raw[feature].fillna(-999999) != cleaned[feature].fillna(-999999)).sum()),
        })
    return pd.DataFrame(rows)


def run_stage2(project_root: Path = PROJECT_ROOT) -> dict[str, object]:
    raw_dir = project_root / "data" / "raw"
    if not (raw_dir / "Data Dictionary.xls").is_file():
        raise ValueError("未找到 Data Dictionary.xls，请将数据字典放入 data/raw 目录。")
    processed_dir = project_root / "data" / "processed"
    figure_dir = project_root / "reports" / "figures" / "eda"
    report_dir = project_root / "reports" / "excel"
    data, source = load_labeled_training(raw_dir)
    competition_test = inspect_unlabeled_file(raw_dir)
    splits = stratified_split(data, seed=SEED)
    results = analyze_training(splits.train)
    rules = CleaningRules.fit(splits.train)
    clean = {
        "train": rules.transform(splits.train),
        "validation": rules.transform(splits.validation),
        "test": rules.transform(splits.test),
    }
    paths, sample_size = create_eda_charts(splits.train, results, figure_dir)
    narratives = build_narratives(results, sample_size)
    if paths.keys() != narratives.keys():
        raise RuntimeError("图表和业务解释未一一对应，请检查生成逻辑。")

    processed_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in clean.items():
        frame.to_csv(processed_dir / f"{name}_clean.csv", index=False)
    manifest = pd.concat([
        frame[["source_row_id"]].assign(split=name)
        for name, frame in {
            "train": splits.train, "validation": splits.validation, "test": splits.test
        }.items()
    ], ignore_index=True)
    manifest.to_csv(processed_dir / "split_manifest.csv", index=False)
    (processed_dir / "cleaning_rules.json").write_text(
        json.dumps(rules.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    split_sizes = {name: len(frame) for name, frame in clean.items()}
    _write_excel(report_dir / "data_quality.xlsx", results, split_sizes,
                 _comparison(splits.train, clean["train"]))
    explanation_lines = [
        "# GMSC 第二阶段核心图表业务解释",
        "",
        "所有图表与统计只使用分层划分后的训练集。坏客户指 SeriousDlqin2yrs=1。",
        "描述性分组不作为正式 WOE 分箱，相关关系不代表因果关系。",
        "",
    ]
    chart_titles = {
        "missing_rate": "字段缺失率",
        "missing_pattern": "缺失模式",
        "feature_distribution": "主要连续变量分布",
        "anomaly_counts": "异常与特殊值",
        "correlation": "原始变量相关性",
        "risk_segments": "历史逾期客户群体",
        "single_feature_discrimination": "单变量初步区分能力",
    }
    chart_titles.update({f"risk_{feature}": f"{FEATURE_NAMES[feature]}分组风险" for feature in FEATURE_NAMES})
    chart_titles.update({f"interaction_{i}": f"{name}二维风险" for i, name in enumerate(results.interactions, 1)})
    for key, path in paths.items():
        explanation_lines.extend([
            f"## {chart_titles[key]}", "", f"![{chart_titles[key]}](figures/eda/{path.name})", "",
            narratives[key], "",
        ])
    (project_root / "reports" / "eda_explanations.md").write_text(
        "\n".join(explanation_lines), encoding="utf-8"
    )
    summary = {
        "source": source.name,
        "labeled_rows": len(data),
        "split_sizes": split_sizes,
        "train_bad_rate": float(results.overview.iloc[0]["bad_rate"]),
        "competition_test": competition_test,
        "chart_count": len(paths),
        "report": str(report_dir / "data_quality.xlsx"),
    }
    return summary


if __name__ == "__main__":
    print(json.dumps(run_stage2(), ensure_ascii=False, indent=2))
