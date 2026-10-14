"""第四阶段训练集变量筛选：IV、相关性、VIF、方向与显著性。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.data_processing.schema import FEATURES


@dataclass(frozen=True)
class SelectionConfig:
    min_iv: float = 0.02
    max_psi: float = 0.10
    correlation_threshold: float = 0.70
    max_vif: float = 5.0
    max_pvalue: float = 0.05
    iv_tie_margin: float = 0.02


def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    if X.empty:
        raise ValueError("VIF输入不得为空。")
    design = sm.add_constant(X.astype(float), has_constant="add")
    values = design.to_numpy()
    rows = []
    for index, feature in enumerate(X.columns, start=1):
        try:
            value = float(variance_inflation_factor(values, index)) if X.shape[1] > 1 else 1.0
        except (np.linalg.LinAlgError, ValueError):
            value = float("inf")
        rows.append({"feature": feature, "vif": value})
    return pd.DataFrame(rows).sort_values(["vif", "feature"], ascending=[False, True]).reset_index(drop=True)


def glm_diagnostics(X: pd.DataFrame, y: pd.Series) -> tuple[object, pd.DataFrame]:
    design = sm.add_constant(X.astype(float), has_constant="add")
    model = sm.GLM(np.asarray(y, dtype=int), design, family=sm.families.Binomial()).fit()
    interval = model.conf_int()
    rows = []
    for feature in design.columns:
        rows.append({
            "feature": feature,
            "coefficient": float(model.params[feature]),
            "std_error": float(model.bse[feature]),
            "z_value": float(model.tvalues[feature]),
            "p_value": float(model.pvalues[feature]),
            "ci_lower": float(interval.loc[feature, 0]),
            "ci_upper": float(interval.loc[feature, 1]),
        })
    return model, pd.DataFrame(rows)


def _correlation_components(correlation: pd.DataFrame, threshold: float) -> list[set[str]]:
    graph = {column: set() for column in correlation.columns}
    for left_index, left in enumerate(correlation.columns):
        for right in correlation.columns[left_index + 1:]:
            value = correlation.loc[left, right]
            if pd.notna(value) and abs(float(value)) >= threshold:
                graph[left].add(right)
                graph[right].add(left)
    components: list[set[str]] = []
    unseen = set(graph)
    while unseen:
        root = min(unseen)
        stack = [root]
        component: set[str] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(graph[node] - component)
        unseen -= component
        if len(component) > 1:
            components.append(component)
    return components


def _choose_component_representative(component: set[str], iv: pd.Series, margin: float) -> str:
    maximum = max(float(iv[name]) for name in component)
    near_best = [name for name in component if maximum - float(iv[name]) < margin]
    originals = [name for name in near_best if name in FEATURES]
    pool = originals or near_best
    return sorted(pool, key=lambda name: (-float(iv[name]), name))[0]


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    screening: pd.DataFrame,
    config: SelectionConfig | None = None,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = config or SelectionConfig()
    summary = screening.set_index("feature")
    missing = sorted(set(X_train.columns).difference(summary.index))
    if missing:
        raise ValueError(f"变量初筛表缺少字段：{missing}")
    decisions: list[dict[str, object]] = []
    selected = []
    for feature in X_train.columns:
        iv = float(summary.loc[feature, "iv"])
        psi = float(summary.loc[feature, "psi"])
        if iv < cfg.min_iv:
            decisions.append({"feature": feature, "action": "排除", "stage": "IV_PSI", "reason": f"IV={iv:.6f}<0.02"})
        elif psi >= cfg.max_psi:
            decisions.append({"feature": feature, "action": "排除", "stage": "IV_PSI", "reason": f"PSI={psi:.6f}>=0.10"})
        else:
            selected.append(feature)
            decisions.append({"feature": feature, "action": "保留", "stage": "IV_PSI", "reason": f"IV={iv:.6f}, PSI={psi:.6f}"})

    correlation = X_train.loc[:, selected].corr(method="spearman")
    iv = summary["iv"]
    for component in _correlation_components(correlation, cfg.correlation_threshold):
        representative = _choose_component_representative(component, iv, cfg.iv_tie_margin)
        for feature in sorted(component):
            if feature != representative and feature in selected:
                selected.remove(feature)
                decisions.append({
                    "feature": feature,
                    "action": "排除",
                    "stage": "相关性",
                    "reason": f"与相关组高度相关；保留代表变量 {representative}",
                })

    iteration = 0
    final_vif = pd.DataFrame()
    final_glm = pd.DataFrame()
    while True:
        iteration += 1
        final_vif = vif_table(X_train.loc[:, selected])
        highest = final_vif.iloc[0]
        if float(highest["vif"]) > cfg.max_vif:
            tied = final_vif.loc[np.isclose(final_vif["vif"], highest["vif"], equal_nan=False), "feature"].tolist()
            remove = sorted(tied, key=lambda name: (float(iv[name]), name))[0]
            selected.remove(remove)
            decisions.append({"feature": remove, "action": "排除", "stage": f"VIF迭代{iteration}", "reason": f"VIF={float(highest['vif']):.6f}>5"})
            continue
        _, final_glm = glm_diagnostics(X_train.loc[:, selected], y_train)
        variables = final_glm.loc[final_glm["feature"].ne("const")].copy()
        wrong = variables.loc[variables["coefficient"].ge(0) | ~np.isfinite(variables["coefficient"])]
        if not wrong.empty:
            remove = wrong.sort_values(["p_value", "feature"], ascending=[False, True]).iloc[0]["feature"]
            selected.remove(remove)
            decisions.append({"feature": remove, "action": "排除", "stage": f"方向迭代{iteration}", "reason": "WOE系数非负，不符合坏客户PD方向"})
            continue
        insignificant = variables.loc[variables["p_value"].isna() | variables["p_value"].gt(cfg.max_pvalue)]
        if not insignificant.empty:
            remove = insignificant.assign(sort_p=insignificant["p_value"].fillna(np.inf)).sort_values(
                ["sort_p", "feature"], ascending=[False, True]
            ).iloc[0]["feature"]
            selected.remove(remove)
            decisions.append({"feature": remove, "action": "排除", "stage": f"显著性迭代{iteration}", "reason": "训练集GLM p值>0.05"})
            continue
        break
    for feature in selected:
        decisions.append({"feature": feature, "action": "最终保留", "stage": "完成", "reason": "通过IV、PSI、相关性、VIF、方向和显著性检查"})
    return selected, pd.DataFrame(decisions), final_vif, final_glm
