"""运行第三阶段：业务特征、OptimalBinning、WOE、IV 与无标签 PSI。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_processing.schema import TARGET

from .binning import WOEBinningPipeline
from .features import CANDIDATE_FEATURES, FeatureEngineer
from .reporting import create_stage3_charts, write_analysis_excel, write_explanations
from .stability import calculate_psi, high_correlation_pairs, screen_candidates


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_NAMES = ("train", "validation", "test")


def _load_processed(processed_dir: Path) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for split in SPLIT_NAMES:
        path = processed_dir / f"{split}_clean.csv"
        if not path.is_file():
            raise ValueError(f"未找到第二阶段产物：{path}")
        frame = pd.read_csv(path)
        required = {"source_row_id", TARGET}
        if not required.issubset(frame.columns):
            raise ValueError(f"{path.name} 缺少 source_row_id 或目标字段。")
        if frame["source_row_id"].duplicated().any():
            raise ValueError(f"{path.name} 的 source_row_id 存在重复。")
        data[split] = frame
    ids = [set(data[split]["source_row_id"]) for split in SPLIT_NAMES]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise ValueError("训练、验证、测试集合的 source_row_id 不互斥。")
    return data


def _with_metadata(source: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    if len(source) != len(values) or not source.index.equals(values.index):
        raise ValueError("转换结果与源数据行顺序不一致。")
    return pd.concat(
        [source.loc[:, ["source_row_id", TARGET]].reset_index(drop=True), values.reset_index(drop=True)],
        axis=1,
    )


def _validate_training_tables(tables: pd.DataFrame, train_rows: int) -> None:
    for feature, table in tables.groupby("feature", sort=False):
        total = table.loc[table["table_row"].eq("Totals")]
        if len(total) != 1 or int(total.iloc[0]["Count"]) != train_rows:
            raise RuntimeError(f"变量 {feature} 的训练分箱样本数无法对回 {train_rows}。")
        nonempty = table.loc[~table["table_row"].eq("Totals") & table["Count"].gt(0)]
        if not np.isfinite(nonempty["WoE"].astype(float)).all():
            raise RuntimeError(f"变量 {feature} 的非空箱存在非有限 WOE。")


def run_stage3(project_root: Path = PROJECT_ROOT) -> dict[str, object]:
    processed = _load_processed(project_root / "data" / "processed")
    engineer = FeatureEngineer()
    features = {name: engineer.transform(frame) for name, frame in processed.items()}

    pipeline = WOEBinningPipeline().fit(features["train"], processed["train"][TARGET])
    woe = {name: pipeline.transform(frame, metric="woe") for name, frame in features.items()}
    bins = {name: pipeline.transform(frame, metric="bins") for name, frame in features.items()}
    bin_indices = {name: pipeline.transform(frame, metric="indices") for name, frame in features.items()}

    feature_dir = project_root / "data" / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        _with_metadata(processed[name], features[name]).to_csv(
            feature_dir / f"{name}_features.csv", index=False
        )
        _with_metadata(processed[name], bins[name]).to_csv(
            feature_dir / f"{name}_bins.csv", index=False
        )
        _with_metadata(processed[name], woe[name]).to_csv(
            feature_dir / f"{name}_woe.csv", index=False
        )

    model_path = project_root / "models" / "stage3_binning.joblib"
    pipeline.save(model_path)

    binning_tables = pipeline.binning_tables()
    _validate_training_tables(binning_tables, len(processed["train"]))
    iv_summary = pipeline.summary()
    psi_summary, psi_detail = calculate_psi(bin_indices["train"], bin_indices["validation"])
    correlation, correlation_pairs = high_correlation_pairs(woe["train"])
    screening = screen_candidates(iv_summary, psi_summary)

    run_manifest = pd.DataFrame([
        {
            "split": name,
            "rows": len(processed[name]),
            "source_id_unique": processed[name]["source_row_id"].is_unique,
            "target_used_for_fit": name == "train",
            "target_used_for_report": name == "train",
            "purpose": {
                "train": "拟合分箱、WOE、IV和相关性",
                "validation": "仅无标签PSI与转换完整性",
                "test": "仅转换完整性",
            }[name],
        }
        for name in SPLIT_NAMES
    ])
    report_path = project_root / "reports" / "excel" / "stage3_woe_iv_analysis.xlsx"
    write_analysis_excel(
        report_path,
        configuration=pipeline.configuration_table(),
        binning_tables=binning_tables,
        screening=screening,
        psi_detail=psi_detail,
        correlation=correlation,
        correlation_pairs=correlation_pairs,
        run_manifest=run_manifest,
    )
    chart_paths = create_stage3_charts(
        project_root / "reports" / "figures" / "stage3", screening, binning_tables
    )
    explanation_path = project_root / "reports" / "stage3_explanations.md"
    write_explanations(explanation_path, screening, binning_tables, correlation_pairs, chart_paths)

    summary = {
        "candidate_feature_count": len(CANDIDATE_FEATURES),
        "derived_feature_count": 7,
        "split_sizes": {name: len(frame) for name, frame in processed.items()},
        "stage4_candidate_count": int(screening["screening_status"].eq("阶段四候选").sum()),
        "high_iv_review_count": int(screening["screening_status"].eq("高IV待复核").sum()),
        "unstable_review_count": int(screening["screening_status"].eq("稳定性待复核").sum()),
        "high_correlation_pair_count": len(correlation_pairs),
        "chart_count": len(chart_paths),
        "report": str(report_path),
        "artifact": str(model_path),
    }
    (project_root / "reports" / "stage3_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(run_stage3(), ensure_ascii=False, indent=2))
