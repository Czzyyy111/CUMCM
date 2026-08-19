"""问题三：分层公共因子相关情景 + 问题二 CVaR 随机 MILP。

本脚本保留问题二的土地、适种、轮作、最小面积、分散度、销售与 CVaR 模型，
只把销量、亩产、成本和价格的独立抽样升级为分层公共因子 Copula 情景。

主要输出
--------
1. ``outputs/result3.xlsx``：问题三最终种植方案；
2. ``outputs/问题三_分层公共因子_CVaR分析结果.xlsx``：完整指标、诊断与源表；
3. ``source_data``：论文图形使用的可审计 CSV；
4. ``logs``：求解器、数据隔离、相关性和约束诊断；
5. ``参数基准与显示含义.md``：参数含义与论文解释边界。

默认执行正式规模。``--quick`` 仅用于流程检查，不应直接写入论文。
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pulp


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
Q2_DIR = ROOT_DIR / "Q2"
if str(Q2_DIR) not in sys.path:
    sys.path.insert(0, str(Q2_DIR))

import Q2_Stochastic_CVaR as q2  # noqa: E402


# =============================================================================
# 全局配置
# =============================================================================

OUTPUT_DIR = SCRIPT_DIR / "outputs"
LOG_DIR = SCRIPT_DIR / "logs"
SOURCE_DIR = SCRIPT_DIR / "source_data"
FIGURE_DIR = SCRIPT_DIR / "figures_nature"

YEARS = q2.YEARS
CROPS = tuple(range(1, 42))
CORRELATION_STRENGTHS = (0.0, 0.5, 1.0, 1.5)
BASE_FACTOR_SEED_OFFSET = 910_000
BOOTSTRAP_SEED = 202407

GROUP_DEFINITIONS: dict[str, set[int]] = {
    "粮食豆类": set(range(1, 6)),
    "主粮": {6, 7, 16},
    "杂粮薯类": {8, 9, 10, 11, 12, 13, 14, 15},
    "豆荚类蔬菜": {17, 18, 19},
    "茄果瓜类": {21, 22, 24, 29, 31},
    "叶菜甘蓝类": {23, 25, 26, 27, 28, 30, 32, 33, 34, 35},
    "根茎类": {20, 36, 37},
    "食用菌": {38, 39, 40, 41},
}
GROUP_NAMES = tuple(GROUP_DEFINITIONS)
CROP_TO_GROUP = {
    crop: group
    for group, members in GROUP_DEFINITIONS.items()
    for crop in members
}
if set(CROP_TO_GROUP) != set(CROPS):
    raise RuntimeError("作物组定义必须完整覆盖 1—41 号作物且不得重复。")


# 基准载荷。正气候因子表示气候有利；正通胀因子表示农资上涨较快；
# 正市场/作物组因子表示需求景气。
LOADINGS = {
    "yield_field_weather": 0.65,
    "yield_rice_weather": 0.45,
    "yield_vegetable_weather": 0.35,
    "yield_fungi_weather": 0.20,
    "cost_inflation": 0.70,
    "demand_market": 0.35,
    "demand_group": 0.45,
    "vegetable_price_weather": -0.20,
    "vegetable_price_inflation": 0.30,
    "vegetable_price_market": 0.20,
    "vegetable_price_group": 0.30,
    "fungi_price_inflation": 0.20,
    "fungi_price_market": 0.20,
    "fungi_price_group": 0.35,
}
AR_PHI = {
    "气候因子": 0.20,
    "农资通胀因子": 0.60,
    "市场景气因子": 0.40,
    "作物组市场因子": 0.30,
}


@dataclass
class FactorScenarioBundle:
    scenarios: q2.ScenarioSet
    factors: pd.DataFrame
    mode: str
    kappa: float


# =============================================================================
# STAGE A  数值工具、分组和 Q2 方案读取
# =============================================================================

def normal_ppf(probability: np.ndarray) -> np.ndarray:
    """Acklam 近似的标准正态分位函数；纯 NumPy、最大误差约 1e-9。"""
    p = np.clip(np.asarray(probability, dtype=float), 1e-12, 1.0 - 1e-12)
    out = np.empty_like(p)
    a = np.array([
        -3.969683028665376e01, 2.209460984245205e02,
        -2.759285104469687e02, 1.383577518672690e02,
        -3.066479806614716e01, 2.506628277459239e00,
    ])
    b = np.array([
        -5.447609879822406e01, 1.615858368580409e02,
        -1.556989798598866e02, 6.680131188771972e01,
        -1.328068155288572e01,
    ])
    c = np.array([
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e00, -2.549732539343734e00,
        4.374664141464968e00, 2.938163982698783e00,
    ])
    d = np.array([
        7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e00, 3.754408661907416e00,
    ])
    plow = 0.02425
    phigh = 1.0 - plow
    low = p < plow
    high = p > phigh
    middle = ~(low | high)
    if np.any(low):
        q = np.sqrt(-2.0 * np.log(p[low]))
        out[low] = (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if np.any(high):
        q = np.sqrt(-2.0 * np.log(1.0 - p[high]))
        out[high] = -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if np.any(middle):
        q = p[middle] - 0.5
        r = q * q
        out[middle] = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    return out


def normal_cdf(z: np.ndarray) -> np.ndarray:
    """标准正态分布函数的向量化近似，误差小于约 8e-8。"""
    x = np.asarray(z, dtype=float)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.2316419 * ax)
    poly = t * (
        0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
    )
    tail = np.exp(-0.5 * ax * ax) / math.sqrt(2.0 * math.pi) * poly
    cdf_pos = 1.0 - tail
    result = np.where(x >= 0.0, cdf_pos, 1.0 - cdf_pos)
    return np.clip(result, 1e-10, 1.0 - 1e-10)


def scaled_coefficients(coefficients: list[float], kappa: float) -> tuple[np.ndarray, float]:
    """载荷乘 sqrt(kappa)，返回缩放载荷与保持单位方差的特有噪声系数。"""
    scaled = np.asarray(coefficients, dtype=float) * math.sqrt(max(kappa, 0.0))
    common_variance = float(np.dot(scaled, scaled))
    if common_variance >= 0.98:
        scaled *= math.sqrt(0.98 / common_variance)
        common_variance = float(np.dot(scaled, scaled))
    return scaled, math.sqrt(max(1.0 - common_variance, 0.0))


def crop_yield_weather_loading(crop: int) -> float:
    if crop <= 15:
        return LOADINGS["yield_field_weather"]
    if crop == 16:
        return LOADINGS["yield_rice_weather"]
    if crop in q2.VEGETABLE_CROPS:
        return LOADINGS["yield_vegetable_weather"]
    return LOADINGS["yield_fungi_weather"]


def load_q2_reference_candidate(stat_for_key: dict) -> q2.Candidate:
    """从问题二分析工作簿读取最终方案，不重新利用测试集调参。"""
    candidates = sorted(
        (Q2_DIR / "outputs").glob("问题二_随机规划_CVaR分析结果*.xlsx"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("未找到问题二分析工作簿。")
    workbook = candidates[0]
    detail = pd.read_excel(workbook, sheet_name="地块种植方案")
    x_values = {key: 0.0 for key in stat_for_key}
    for _, row in detail.iterrows():
        season = 1 if "1" in str(row["季次"]) or "一" in str(row["季次"]) else 2
        key = (str(row["地块名称"]), int(row["年份"]), season, int(row["作物编号"]))
        if key in x_values:
            x_values[key] = float(row["种植面积/亩"])
    y_values = {key: int(value > q2.EPS) for key, value in x_values.items()}
    return q2.Candidate(
        risk_weight=0.2,
        x_values=x_values,
        y_values=y_values,
        status="问题二既有最终方案",
        solution_status="已冻结",
        objective=float("nan"),
        elapsed_seconds=0.0,
        mip_gap=float("nan"),
    )


def decision_stat_map(data: q2.InputData) -> dict:
    """构造与问题二模型一致的决策键和统计参数，用于求解前读取 MIP 启动方案。"""
    result = {}
    for plot in data.plots.values():
        for year in YEARS:
            for season in q2.SEASONS:
                for crop in q2.allowed_crops(plot, season):
                    key = (plot.name, year, season, crop)
                    result[key] = q2.get_stat(data, plot, crop, season)
    return result


# =============================================================================
# STAGE B  分层公共因子情景生成
# =============================================================================

def ar1_paths(innovations: np.ndarray, phi: float) -> np.ndarray:
    """把标准正态创新转成边际仍为 N(0,1) 的 AR(1) 路径。"""
    output = np.empty_like(innovations)
    output[:, 0] = innovations[:, 0]
    innovation_scale = math.sqrt(max(1.0 - phi * phi, 0.0))
    for t in range(1, innovations.shape[1]):
        output[:, t] = phi * output[:, t - 1] + innovation_scale * innovations[:, t]
    return output


def generate_latent_factors(n: int, seed: int, method: str) -> tuple[np.ndarray, np.ndarray]:
    dimensions = len(YEARS) * (3 + len(GROUP_NAMES))
    u = q2.unit_samples(n, dimensions, seed + BASE_FACTOR_SEED_OFFSET, method)
    innovations = normal_ppf(u).reshape(n, len(YEARS), 3 + len(GROUP_NAMES))
    global_factors = np.empty((n, len(YEARS), 3), dtype=float)
    global_factors[:, :, 0] = ar1_paths(innovations[:, :, 0], AR_PHI["气候因子"])
    global_factors[:, :, 1] = ar1_paths(innovations[:, :, 1], AR_PHI["农资通胀因子"])
    global_factors[:, :, 2] = ar1_paths(innovations[:, :, 2], AR_PHI["市场景气因子"])
    group_factors = np.empty((n, len(YEARS), len(GROUP_NAMES)), dtype=float)
    for group_index in range(len(GROUP_NAMES)):
        group_factors[:, :, group_index] = ar1_paths(
            innovations[:, :, 3 + group_index], AR_PHI["作物组市场因子"]
        )
    return global_factors, group_factors


def build_scenarios_from_uniforms(
    data: q2.InputData,
    u: np.ndarray,
    name: str,
    method: str,
    seed: int,
) -> q2.ScenarioSet:
    """严格复用问题二的逆分布映射与年度复合口径。"""
    n = u.shape[0]
    demand = np.zeros((n, len(YEARS), 42), dtype=float)
    yield_mult = np.ones_like(demand)
    cost_mult = np.ones_like(demand)
    price_mult = np.ones_like(demand)
    for crop_index, crop in enumerate(CROPS):
        current_demand = np.full(n, data.sales_demand[crop], dtype=float)
        current_cost = np.ones(n, dtype=float)
        current_price = np.ones(n, dtype=float)
        is_morel = "羊肚菌" in data.crop_names[crop]
        for t, _year in enumerate(YEARS):
            u_d, u_y, u_c, u_p = (u[:, t, crop_index, j] for j in range(4))
            if crop in {6, 7}:
                current_demand *= 1.0 + 0.05 + 0.05 * u_d
                demand[:, t, crop] = current_demand
            else:
                demand[:, t, crop] = data.sales_demand[crop] * (0.95 + 0.10 * u_d)
            yield_mult[:, t, crop] = 0.90 + 0.20 * u_y
            current_cost *= 1.0 + q2.triangular_inverse(u_c, 0.03, 0.05, 0.07)
            cost_mult[:, t, crop] = current_cost
            if crop <= 16:
                price_mult[:, t, crop] = 1.0
            elif crop in q2.VEGETABLE_CROPS:
                current_price *= 1.0 + q2.triangular_inverse(u_p, 0.03, 0.05, 0.07)
                price_mult[:, t, crop] = current_price
            else:
                decline = np.full(n, 0.05) if is_morel else 0.01 + 0.04 * u_p
                current_price *= 1.0 - decline
                price_mult[:, t, crop] = current_price
    return q2.ScenarioSet(name, method, seed, demand, yield_mult, cost_mult, price_mult)


def generate_factor_scenarios(
    data: q2.InputData,
    n: int,
    seed: int,
    method: str,
    name: str,
    kappa: float = 1.0,
    include_global: bool = True,
    include_group: bool = True,
) -> FactorScenarioBundle:
    """生成分层公共因子 Copula 情景；kappa=0 时精确回到问题二独立抽样。"""
    dimensions = len(YEARS) * len(CROPS) * 4
    base_u = q2.unit_samples(n, dimensions, seed, method).reshape(n, len(YEARS), len(CROPS), 4)
    global_factors, group_factors = generate_latent_factors(n, seed, method)
    mode = (
        "完整分层因子" if include_global and include_group
        else "仅全局因子" if include_global
        else "仅作物组因子" if include_group
        else "问题二独立情景"
    )
    if kappa <= 0.0 or (not include_global and not include_group):
        correlated_u = base_u.copy()
    else:
        base_z = normal_ppf(base_u)
        correlated_z = np.empty_like(base_z)
        for crop_index, crop in enumerate(CROPS):
            group_index = GROUP_NAMES.index(CROP_TO_GROUP[crop])
            weather = global_factors[:, :, 0]
            inflation = global_factors[:, :, 1]
            market = global_factors[:, :, 2]
            group = group_factors[:, :, group_index]

            demand_coeffs = [
                LOADINGS["demand_market"] if include_global else 0.0,
                LOADINGS["demand_group"] if include_group else 0.0,
            ]
            coeff, residual = scaled_coefficients(demand_coeffs, kappa)
            correlated_z[:, :, crop_index, 0] = (
                coeff[0] * market + coeff[1] * group + residual * base_z[:, :, crop_index, 0]
            )

            yield_coeff = crop_yield_weather_loading(crop) if include_global else 0.0
            coeff, residual = scaled_coefficients([yield_coeff], kappa)
            correlated_z[:, :, crop_index, 1] = coeff[0] * weather + residual * base_z[:, :, crop_index, 1]

            cost_coeff = LOADINGS["cost_inflation"] if include_global else 0.0
            coeff, residual = scaled_coefficients([cost_coeff], kappa)
            correlated_z[:, :, crop_index, 2] = coeff[0] * inflation + residual * base_z[:, :, crop_index, 2]

            if crop <= 16 or crop == 41:
                correlated_z[:, :, crop_index, 3] = base_z[:, :, crop_index, 3]
            elif crop in q2.VEGETABLE_CROPS:
                price_coeffs = [
                    LOADINGS["vegetable_price_weather"] if include_global else 0.0,
                    LOADINGS["vegetable_price_inflation"] if include_global else 0.0,
                    LOADINGS["vegetable_price_market"] if include_global else 0.0,
                    LOADINGS["vegetable_price_group"] if include_group else 0.0,
                ]
                coeff, residual = scaled_coefficients(price_coeffs, kappa)
                correlated_z[:, :, crop_index, 3] = (
                    coeff[0] * weather + coeff[1] * inflation + coeff[2] * market
                    + coeff[3] * group + residual * base_z[:, :, crop_index, 3]
                )
            else:
                price_coeffs = [
                    LOADINGS["fungi_price_inflation"] if include_global else 0.0,
                    LOADINGS["fungi_price_market"] if include_global else 0.0,
                    LOADINGS["fungi_price_group"] if include_group else 0.0,
                ]
                coeff, residual = scaled_coefficients(price_coeffs, kappa)
                correlated_z[:, :, crop_index, 3] = (
                    coeff[0] * inflation + coeff[1] * market + coeff[2] * group
                    + residual * base_z[:, :, crop_index, 3]
                )
        correlated_u = normal_cdf(correlated_z)

    scenarios = build_scenarios_from_uniforms(
        data, correlated_u, name, f"{method}+{mode}+Copula", seed
    )
    group_weights = np.array([
        sum(data.sales_demand[crop] for crop in GROUP_DEFINITIONS[group])
        for group in GROUP_NAMES
    ], dtype=float)
    group_weights /= group_weights.sum()
    rows = []
    for scenario_index in range(n):
        for t, year in enumerate(YEARS):
            row = {
                "情景编号": scenario_index + 1,
                "年份": year,
                "气候因子": global_factors[scenario_index, t, 0],
                "农资通胀因子": global_factors[scenario_index, t, 1],
                "市场景气因子": global_factors[scenario_index, t, 2],
                "作物组因子加权均值": float(group_factors[scenario_index, t] @ group_weights),
                "相关强度κ": kappa,
                "情景模式": mode,
            }
            for group_index, group_name in enumerate(GROUP_NAMES):
                row[f"组因子_{group_name}"] = group_factors[scenario_index, t, group_index]
            rows.append(row)
    return FactorScenarioBundle(scenarios, pd.DataFrame(rows), mode, kappa)


def factor_route_table(factors: pd.DataFrame, reduction: pd.DataFrame) -> pd.DataFrame:
    representative_ids = set(reduction["代表情景原编号"].astype(int).unique())
    route = factors.groupby("情景编号", as_index=False)[
        ["气候因子", "农资通胀因子", "市场景气因子", "作物组因子加权均值"]
    ].mean()
    route["是否优化代表情景"] = np.where(route["情景编号"].isin(representative_ids), "是", "否")
    for column in ["气候因子", "农资通胀因子", "市场景气因子", "作物组因子加权均值"]:
        std = route[column].std(ddof=0)
        route[f"{column}标准分数"] = (route[column] - route[column].mean()) / max(std, 1e-12)
    return route


# =============================================================================
# STAGE C  相关性、边际分布和求解辅助
# =============================================================================

def mean_upper_correlation(matrix: np.ndarray) -> float:
    ranks = pd.DataFrame(matrix).rank(axis=0, method="average").to_numpy(float)
    corr = np.corrcoef(ranks, rowvar=False)
    upper = corr[np.triu_indices_from(corr, k=1)]
    finite = upper[np.isfinite(upper)]
    return float(finite.mean()) if len(finite) else float("nan")


def spearman_pair(x: np.ndarray, y: np.ndarray) -> float:
    ranks = pd.DataFrame({"x": x, "y": y}).rank(method="average").to_numpy(float)
    if np.std(ranks[:, 0]) <= 1e-12 or np.std(ranks[:, 1]) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(ranks[:, 0], ranks[:, 1])[0, 1])


def realized_correlation_table(data: q2.InputData, scenarios: q2.ScenarioSet) -> pd.DataFrame:
    metrics: dict[str, list[float]] = defaultdict(list)
    field = np.array(sorted(q2.GRAIN_CROPS), dtype=int)
    all_crops = np.array(CROPS, dtype=int)
    vegetable = np.array(sorted(q2.VEGETABLE_CROPS), dtype=int)
    group_indices = [np.array(sorted(members), dtype=int) for members in GROUP_DEFINITIONS.values() if len(members) >= 2]
    for t in range(len(YEARS)):
        metrics["不同作物成本增长"].append(mean_upper_correlation(scenarios.cost_mult[:, t, all_crops]))
        metrics["露地作物亩产"].append(mean_upper_correlation(scenarios.yield_mult[:, t, field]))
        group_corr = []
        for group in group_indices:
            ratios = scenarios.demand[:, t, group] / np.array([data.sales_demand[int(c)] for c in group])[None, :]
            group_corr.append(mean_upper_correlation(ratios))
        metrics["同组作物销量"].append(float(np.nanmean(group_corr)))
        cp, yp, dp = [], [], []
        for crop in vegetable:
            cp.append(spearman_pair(
                scenarios.cost_mult[:, t, crop], scenarios.price_mult[:, t, crop]
            ))
            yp.append(spearman_pair(
                scenarios.yield_mult[:, t, crop], scenarios.price_mult[:, t, crop]
            ))
            dp.append(spearman_pair(
                scenarios.demand[:, t, crop], scenarios.price_mult[:, t, crop]
            ))
        metrics["成本—蔬菜价格"].append(float(np.nanmean(cp)))
        metrics["亩产—蔬菜价格"].append(float(np.nanmean(yp)))
        metrics["销量—蔬菜价格"].append(float(np.nanmean(dp)))
    targets = {
        "不同作物成本增长": 0.49,
        "露地作物亩产": 0.65 ** 2,
        "同组作物销量": 0.35 ** 2 + 0.45 ** 2,
        "成本—蔬菜价格": 0.70 * 0.30,
        "亩产—蔬菜价格": 0.35 * (-0.20),
        "销量—蔬菜价格": 0.35 * 0.20 + 0.45 * 0.30,
    }
    rows = []
    for relation, values in metrics.items():
        realized = float(np.nanmean(values))
        rows.append({
            "相关关系": relation,
            "基准隐含相关系数": targets[relation],
            "测试情景实现Spearman相关": realized,
            "绝对偏差": abs(realized - targets[relation]),
            "符号是否一致": "是" if realized == 0 or np.sign(realized) == np.sign(targets[relation]) else "否",
        })
    return pd.DataFrame(rows)


def marginal_audit_table(
    data: q2.InputData, independent: q2.ScenarioSet, correlated: q2.ScenarioSet
) -> pd.DataFrame:
    crops = np.array(CROPS, dtype=int)
    variable_price = np.array(sorted(q2.VEGETABLE_CROPS | (q2.FUNGI_CROPS - {41})), dtype=int)
    base_demand = np.array([data.sales_demand[int(c)] for c in crops])
    sources = {
        "销量相对2023基准": (
            independent.demand[:, :, crops] / base_demand[None, None, :],
            correlated.demand[:, :, crops] / base_demand[None, None, :],
        ),
        "亩产乘数": (independent.yield_mult[:, :, crops], correlated.yield_mult[:, :, crops]),
        "累计成本乘数": (independent.cost_mult[:, :, crops], correlated.cost_mult[:, :, crops]),
        "可变价格乘数": (
            independent.price_mult[:, :, variable_price], correlated.price_mult[:, :, variable_price]
        ),
    }
    rows = []
    for variable, (base, corr) in sources.items():
        for label, values in (("问题二独立", base.ravel()), ("问题三相关", corr.ravel())):
            rows.append({
                "变量": variable,
                "情景类型": label,
                "均值": float(np.mean(values)),
                "标准差": float(np.std(values, ddof=1)),
                "P10": float(np.quantile(values, 0.10)),
                "P50": float(np.quantile(values, 0.50)),
                "P90": float(np.quantile(values, 0.90)),
                "最小值": float(np.min(values)),
                "最大值": float(np.max(values)),
            })
    return pd.DataFrame(rows)


def solve_candidate_grid(
    data: q2.InputData,
    scenarios: q2.ScenarioSet,
    risk_weights: tuple[float, ...],
    time_limit: int,
    label: str,
    initial_start: q2.Candidate,
) -> tuple[list[q2.Candidate], dict, pd.DataFrame]:
    """独立构建每个风险权重模型，并以问题二方案作为首个可靠 MIP 启动点。"""
    subdir = LOG_DIR / label
    subdir.mkdir(parents=True, exist_ok=True)
    candidates: list[q2.Candidate] = []
    diagnostics: list[dict] = []
    common_stat_for_key = None
    risk_neutral_or_first: q2.Candidate | None = None
    for index, risk_weight in enumerate(risk_weights):
        model, x, y, stat_for_key, expected_profit, lower_cvar, water_binary_count = (
            q2.build_stochastic_model(data, scenarios)
        )
        if common_stat_for_key is None:
            common_stat_for_key = stat_for_key
        elif common_stat_for_key.keys() != stat_for_key.keys():
            raise RuntimeError("不同候选模型的决策变量集合不一致。")
        model.setObjective((1.0 - risk_weight) * expected_profit + risk_weight * lower_cvar)
        solver_log = subdir / f"HiGHS_风险权重_{risk_weight:.1f}.log"
        solver = pulp.HiGHS(
            msg=False, timeLimit=time_limit, gapRel=q2.SOLVER_GAP, threads=4,
            log_file=str(solver_log),
        )
        if not solver.available():
            raise RuntimeError("PuLP 未找到 HiGHS，请安装 Q3/requirements.txt。")
        mip_start = initial_start if index == 0 else risk_neutral_or_first
        started = time.perf_counter()
        mip_start_note = q2.solve_with_optional_mip_start(model, solver, x, y, mip_start)
        elapsed = time.perf_counter() - started
        pulp_status = pulp.LpStatus.get(model.status, str(model.status))
        solution_status = pulp.LpSolution.get(model.sol_status, str(model.sol_status))
        highs_status = model.solverModel.modelStatusToString(model.solverModel.getModelStatus())
        info = model.solverModel.getInfo()
        if model.sol_status not in {pulp.LpSolutionOptimal, pulp.LpSolutionIntegerFeasible}:
            raise RuntimeError(
                f"{label} λ={risk_weight:.1f} 未得到整数可行解：{highs_status}/{solution_status}"
            )
        status = "达到时间上限，已有整数可行解" if "Time limit" in highs_status else highs_status
        candidate = q2.Candidate(
            risk_weight=risk_weight,
            x_values={key: max(0.0, float(pulp.value(var) or 0.0)) for key, var in x.items()},
            y_values={key: int(round(float(pulp.value(var) or 0.0))) for key, var in y.items()},
            status=status,
            solution_status=solution_status,
            objective=float(pulp.value(model.objective)) * q2.PROFIT_SCALE,
            elapsed_seconds=elapsed,
            mip_gap=float(info.mip_gap),
        )
        candidates.append(candidate)
        diagnostics.append({
            "实验模块": label,
            "风险权重": risk_weight,
            "PuLP状态": pulp_status,
            "整数解状态": solution_status,
            "HiGHS状态": highs_status,
            "目标函数值/元": candidate.objective,
            "最终相对间隙": candidate.mip_gap,
            "求解耗时/秒": elapsed,
            "变量总数": len(model.variables()),
            "约束总数": len(model.constraints),
            "二进制变量数": len(y) + water_binary_count,
            "支持集策略": "独立完整MILP；未固定其他候选的作物支持集",
            "MIP启动点": mip_start_note,
        })
        print(
            f"  {label} λ={risk_weight:.1f}：{status}，目标={candidate.objective:,.0f}元，"
            f"间隙={candidate.mip_gap:.2%}，耗时={elapsed:.1f}秒"
        )
        if index == 0:
            risk_neutral_or_first = candidate
    if common_stat_for_key is None:
        raise RuntimeError(f"{label} 没有生成候选方案。")
    return candidates, common_stat_for_key, pd.DataFrame(diagnostics)


def solve_one_candidate(
    data: q2.InputData,
    scenarios: q2.ScenarioSet,
    risk_weight: float,
    time_limit: int,
    label: str,
    initial_start: q2.Candidate,
) -> tuple[q2.Candidate, dict, pd.DataFrame]:
    candidates, stat_for_key, diag = solve_candidate_grid(
        data, scenarios, (risk_weight,), time_limit, label, initial_start
    )
    return candidates[0], stat_for_key, diag


def solve_main_candidates(
    data: q2.InputData,
    scenarios: q2.ScenarioSet,
    lambdas: tuple[float, ...],
    time_limit: int,
    initial_start: q2.Candidate,
) -> tuple[list[q2.Candidate], dict, pd.DataFrame]:
    return solve_candidate_grid(
        data, scenarios, lambdas, time_limit, "完整分层因子风险权重", initial_start
    )


def bootstrap_plan_difference(
    q2_eval: q2.Evaluation,
    q3_eval: q2.Evaluation,
    repetitions: int,
) -> pd.DataFrame:
    if len(q2_eval.total_profit) != len(q3_eval.total_profit):
        raise ValueError("配对 Bootstrap 要求两套方案使用同一批测试情景。")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(q2_eval.total_profit)
    mean_diff = np.empty(repetitions)
    cvar_diff = np.empty(repetitions)
    for b in range(repetitions):
        index = rng.integers(0, n, size=n)
        q2_profit = q2_eval.total_profit[index]
        q3_profit = q3_eval.total_profit[index]
        mean_diff[b] = np.mean(q3_profit - q2_profit)
        cvar_diff[b] = q2.lower_tail_cvar(q3_profit) - q2.lower_tail_cvar(q2_profit)
    rows = []
    for metric, values in (("期望利润差（Q3-Q2）", mean_diff), ("CVaR90差（Q3-Q2）", cvar_diff)):
        rows.append({
            "指标": metric,
            "Bootstrap次数": repetitions,
            "估计值/元": float(np.mean(values)),
            "95%区间下限/元": float(np.quantile(values, 0.025)),
            "95%区间上限/元": float(np.quantile(values, 0.975)),
        })
    return pd.DataFrame(rows)


def loading_table() -> pd.DataFrame:
    rows = [
        ("亩产-露地粮食", "气候因子", 0.65, "正值表示有利气候提高亩产"),
        ("亩产-水稻", "气候因子", 0.45, "灌溉削弱共同气候暴露"),
        ("亩产-蔬菜", "气候因子", 0.35, "水浇地与设施农业的折中暴露"),
        ("亩产-食用菌", "气候因子", 0.20, "设施环境削弱天气冲击"),
        ("种植成本", "农资通胀因子", 0.70, "化肥、人工、能源等共同上涨"),
        ("预期销量", "市场景气因子", 0.35, "整体需求景气"),
        ("预期销量", "作物组市场因子", 0.45, "同类作物行情共同变化"),
        ("蔬菜价格", "气候因子", -0.20, "丰产增加供给并压低价格"),
        ("蔬菜价格", "农资通胀因子", 0.30, "成本向售价部分传导"),
        ("蔬菜价格", "市场景气因子", 0.20, "需求旺盛推高价格"),
        ("蔬菜价格", "作物组市场因子", 0.30, "同类蔬菜行情共同变化"),
        ("普通食用菌价格", "农资通胀因子", 0.20, "成本部分抵消价格下降"),
        ("普通食用菌价格", "市场景气因子", 0.20, "消费景气传导"),
        ("普通食用菌价格", "作物组市场因子", 0.35, "食用菌组内行情"),
    ]
    return pd.DataFrame(rows, columns=["变量通道", "公共因子", "基准载荷", "显示含义"])


# =============================================================================
# STAGE D  输出表构造
# =============================================================================

def metrics_row(label: str, evaluation: q2.Evaluation, **extra) -> dict:
    return {"方案或实验": label, **extra, **q2.evaluation_metrics(evaluation)}


def annual_comparison_table(q2_eval: q2.Evaluation, q3_eval: q2.Evaluation) -> pd.DataFrame:
    rows = []
    for label, evaluation in (("问题二既有方案", q2_eval), ("问题三最终方案", q3_eval)):
        for t, year in enumerate(YEARS):
            values = evaluation.annual_profit[:, t]
            rows.append({
                "方案": label,
                "年份": year,
                "平均利润/元": float(values.mean()),
                "P10利润/元": float(np.quantile(values, 0.10)),
                "P50利润/元": float(np.quantile(values, 0.50)),
                "P90利润/元": float(np.quantile(values, 0.90)),
            })
    return pd.DataFrame(rows)


def write_parameter_markdown(
    selected: q2.Candidate,
    q2_eval: q2.Evaluation,
    q3_eval: q2.Evaluation,
    correlation_diag: pd.DataFrame,
    quick: bool,
) -> None:
    mean_delta = (q3_eval.total_profit.mean() - q2_eval.total_profit.mean()) / 1e4
    cvar_delta = (q2.lower_tail_cvar(q3_eval.total_profit) - q2.lower_tail_cvar(q2_eval.total_profit)) / 1e4
    corr_lines = "\n".join(
        f"- {row['相关关系']}：基准隐含相关 {row['基准隐含相关系数']:.3f}，"
        f"测试情景实现 Spearman 相关 {row['测试情景实现Spearman相关']:.3f}。"
        for _, row in correlation_diag.iterrows()
    )
    text = f"""# 问题三分层公共因子参数基准与显示含义

## 1. 使用边界

附件只有 2023 年汇总基准，没有逐年原始价格、成本、销量与亩产序列，因此本文不把载荷解释为统计估计值。载荷是根据农业生产和市场传导方向设置的中等强度机理先验；其可信性依靠低、中、高相关强度、消融实验和样本外测试检验。

运行模式：{'快速流程检查，结果不可直接用于论文' if quick else '正式规模运行'}。

## 2. 公共因子及正负方向

1. **气候因子**：正值表示气候有利。它提高亩产，并通过供给增加对蔬菜价格产生负向作用。
2. **农资通胀因子**：正值表示化肥、人工、能源和运输等成本上涨较快。它直接提高种植成本，并部分传导到蔬菜和普通食用菌价格。
3. **市场景气因子**：正值表示整体需求和购买力较强，同时推高预期销量及可变价格。
4. **作物组市场因子**：表示同类作物的共同市场行情。八组分别为粮食豆类、主粮、杂粮薯类、豆荚类蔬菜、茄果瓜类、叶菜甘蓝类、根茎类和食用菌。
5. **作物特有噪声**：表示未被公共因子解释的局地天气、个体行情和经营差异。

潜在扰动先按公共因子线性组合，再通过标准正态分布函数映射为 0—1 分位数，最后进入问题二原有的均匀分布或三角分布逆函数。因此，问题三改变的是变量之间的联合变化方式，而不是题面给定的单变量波动区间。

## 3. 基准载荷

| 变量通道 | 公共因子 | 基准载荷 | 显示含义 |
|---|---|---:|---|
"""
    for _, row in loading_table().iterrows():
        text += f"| {row['变量通道']} | {row['公共因子']} | {row['基准载荷']:.2f} | {row['显示含义']} |\n"
    text += f"""

粮食价格继续按题面保持稳定；羊肚菌价格继续固定每年下降 5%，两者不因公共因子产生额外随机价格变化。

## 4. 年度持续性

| 因子 | AR(1) 系数 | 含义 |
|---|---:|---|
| 气候因子 | 0.20 | 年际天气仅有弱持续性 |
| 农资通胀因子 | 0.60 | 成本冲击具有较强惯性 |
| 市场景气因子 | 0.40 | 市场趋势具有中等延续性 |
| 作物组市场因子 | 0.30 | 类别行情具有短期持续性 |

## 5. 相关强度参数

设置 κ=0、0.5、1.0、1.5。κ=0 精确退化为问题二独立情景；κ=1 为基准分层因子；其余两档用于敏感性分析。代码将公共载荷乘以 sqrt(κ)，因为两个变量的相关系数近似等于相应载荷的乘积。特有噪声系数同步调整，以保持每个潜在变量的单位方差。

## 6. 相关结构实现诊断

{corr_lines}

实现值采用测试情景的 Spearman 秩相关。由于变量经过非线性分位映射并包含年度复合，理论载荷乘积与实现秩相关不要求完全相等；重点检查方向、数量级和边际分布是否保持。

## 7. CVaR 与最终方案

继续使用 90% 下尾 CVaR，并对 λ∈{{0,0.2,0.4,0.6,0.8}} 独立求解完整 MILP。风险权重只由验证集选择，测试集不参与参数选择。最终采用 λ={selected.risk_weight:.1f}。

在同一批基准相关测试情景中，问题三方案相对问题二既有方案的期望利润差为 {mean_delta:.2f} 万元，下尾 CVaR90 差为 {cvar_delta:.2f} 万元。正值表示问题三方案更高。该比较使用共同随机数；若差异小于 MIP 间隙或 Bootstrap 区间覆盖 0，应表述为“结果接近”，不能宣称严格改进。

## 8. 消融实验解释

1. **问题二独立情景**：不使用任何公共因子，为基线。
2. **仅全局因子**：只保留气候、农资通胀和整体市场景气。
3. **仅作物组因子**：只保留八类作物内部的市场共同波动。
4. **完整分层因子**：同时使用全局因子和作物组因子，为问题三主模型。

消融实验冻结问题三最终种植方案，并使用相同基础随机数依次关闭全局或作物组因子。这样比较隔离了各相关模块对利润分布和尾部风险评价的贡献，不受附加 MILP 求解间隙干扰。问题三方案本身只在完整分层因子模型中重新优化。

## 9. 论文表达限制

- 公共因子载荷是可审计机理假设，不是由单年数据估计的经验相关系数。
- 模拟分位带是设定分布下的经验区间，不是真实历史置信区间。
- 若求解器达到时限，应同时报告整数可行解、best bound 与 MIP 相对间隙。
- 问题二方案必须在问题三相同测试情景中重新评价，不能直接比较两问各自不同随机样本下的指标。
"""
    (SCRIPT_DIR / "参数基准与显示含义.md").write_text(text, encoding="utf-8")


def write_outputs(
    data: q2.InputData,
    selected: q2.Candidate,
    q2_reference: q2.Candidate,
    frontier: pd.DataFrame,
    training_objective: pd.DataFrame,
    sensitivity: pd.DataFrame,
    ablation: pd.DataFrame,
    route: pd.DataFrame,
    correlation_diag: pd.DataFrame,
    marginal_audit: pd.DataFrame,
    annual_comparison: pd.DataFrame,
    bootstrap: pd.DataFrame,
    solver_diag: pd.DataFrame,
    hard_checks: dict,
    q2_eval: q2.Evaluation,
    q3_eval: q2.Evaluation,
    quick: bool,
) -> None:
    for directory in (OUTPUT_DIR, LOG_DIR, SOURCE_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    q2.write_template(data, selected, OUTPUT_DIR / "result3.xlsx")
    detail, crop_summary, structure = q2.plan_tables(data, selected)
    q3_profit = pd.DataFrame({
        "测试情景编号": np.arange(1, len(q3_eval.total_profit) + 1),
        "问题二方案七年利润/元": q2_eval.total_profit,
        "问题三方案七年利润/元": q3_eval.total_profit,
        "配对利润差Q3-Q2/元": q3_eval.total_profit - q2_eval.total_profit,
        "问题二方案滞销率": q2_eval.unsold_rate,
        "问题三方案滞销率": q3_eval.unsold_rate,
    })
    method = pd.DataFrame([
        ("主模型", "分层公共因子 Copula 情景 + 共享事前决策随机 MILP + 90%下尾CVaR"),
        ("保留内容", "完全复用问题二硬约束、边际分布、利润函数和训练/验证/测试隔离"),
        ("全局因子", "气候、农资通胀、市场景气"),
        ("作物组因子", "八类作物组市场因子"),
        ("相关强度", "κ=0/0.5/1.0/1.5；主模型κ=1"),
        ("风险权重", f"验证集重新选择 λ={selected.risk_weight:.1f}，未直接沿用问题二权重"),
        ("测试比较", "问题二与问题三方案在同一批相关测试情景中配对评价"),
        ("运行标记", "快速流程检查（不可直接用于论文）" if quick else "正式规模运行"),
    ], columns=["项目", "说明"])
    diagnostics = pd.DataFrame(
        [{"检查项": key, "结果": value} for key, value in hard_checks.items()]
        + [
            {"检查项": "原始地块数", "结果": len(data.plots)},
            {"检查项": "原始作物数", "结果": len(data.crop_names)},
            {"检查项": "2023种植记录数", "结果": len(data.planting_2023)},
            {"检查项": "最终风险权重", "结果": selected.risk_weight},
            {"检查项": "问题二参考方案是否冻结", "结果": "是"},
            {"检查项": "测试集是否参与选权", "结果": "否"},
            {"检查项": "正式结果标记", "结果": not quick},
        ]
    )
    workbook = OUTPUT_DIR / "问题三_分层公共因子_CVaR分析结果.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        frontier.to_excel(writer, sheet_name="风险厌恶结果", index=False)
        sensitivity.to_excel(writer, sheet_name="相关强度敏感性", index=False)
        ablation.to_excel(writer, sheet_name="消融实验", index=False)
        route.to_excel(writer, sheet_name="公共因子路线", index=False)
        loading_table().to_excel(writer, sheet_name="公共因子载荷", index=False)
        correlation_diag.to_excel(writer, sheet_name="实现相关性诊断", index=False)
        marginal_audit.to_excel(writer, sheet_name="边际分布保持", index=False)
        annual_comparison.to_excel(writer, sheet_name="年度利润比较", index=False)
        bootstrap.to_excel(writer, sheet_name="配对Bootstrap", index=False)
        q3_profit.to_excel(writer, sheet_name="配对测试利润", index=False)
        detail.to_excel(writer, sheet_name="地块种植方案", index=False)
        crop_summary.to_excel(writer, sheet_name="作物面积汇总", index=False)
        structure.to_excel(writer, sheet_name="作物结构", index=False)
        training_objective.to_excel(writer, sheet_name="训练目标函数", index=False)
        solver_diag.to_excel(writer, sheet_name="求解诊断", index=False)
        method.to_excel(writer, sheet_name="方法说明", index=False)
        diagnostics.to_excel(writer, sheet_name="约束与泄漏检查", index=False)
    q2.style_workbook(workbook)

    source_tables = {
        "风险厌恶结果.csv": frontier,
        "相关强度利润比较.csv": sensitivity,
        "消融实验结果.csv": ablation,
        "公共因子路径.csv": route,
        "公共因子载荷.csv": loading_table(),
        "实现相关性诊断.csv": correlation_diag,
        "边际分布保持.csv": marginal_audit,
        "年度利润比较.csv": annual_comparison,
        "配对Bootstrap.csv": bootstrap,
        "配对测试利润.csv": q3_profit,
        "作物结构.csv": structure,
        "求解诊断.csv": solver_diag,
    }
    for filename, frame in source_tables.items():
        frame.to_csv(SOURCE_DIR / filename, index=False, encoding="utf-8-sig")

    summary = {
        "运行模式": "快速流程检查" if quick else "正式规模",
        "最终风险权重": selected.risk_weight,
        "问题二方案测试期望利润/元": float(q2_eval.total_profit.mean()),
        "问题三方案测试期望利润/元": float(q3_eval.total_profit.mean()),
        "问题二方案测试CVaR90/元": q2.lower_tail_cvar(q2_eval.total_profit),
        "问题三方案测试CVaR90/元": q2.lower_tail_cvar(q3_eval.total_profit),
        "问题三相对问题二期望利润差/元": float((q3_eval.total_profit - q2_eval.total_profit).mean()),
        "问题三相对问题二CVaR90差/元": q2.lower_tail_cvar(q3_eval.total_profit) - q2.lower_tail_cvar(q2_eval.total_profit),
        **hard_checks,
    }
    (LOG_DIR / "运行摘要.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log_lines = ["问题三 分层公共因子 + CVaR 诊断日志", "=" * 68, ""]
    log_lines.extend(f"{key}: {value}" for key, value in summary.items())
    log_lines.extend([
        "",
        "有效性与泄漏检查：",
        "1. 训练、验证和测试使用相互独立的固定随机种子。",
        "2. 公共因子仅改变联合分位结构，逆分布和波动区间完全复用问题二。",
        "3. 风险权重只由验证集选择；测试集不参与模型求解和参数选择。",
        "4. 问题二方案与问题三方案在同一批测试情景中配对评价。",
        "5. 相关强度、全局因子和作物组因子实验均保留全部测试记录。",
        "6. 消融实验冻结问题三最终方案，并用共同随机数逐步关闭全局或作物组因子。",
        "7. 相关载荷属于机理先验，不表述为由 2023 年单年数据统计估计。",
    ])
    (LOG_DIR / "问题三_诊断日志.txt").write_text("\n".join(log_lines), encoding="utf-8")
    write_parameter_markdown(selected, q2_eval, q3_eval, correlation_diag, quick)


# =============================================================================
# STAGE E  主流程
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="问题三：分层公共因子相关情景 + CVaR随机MILP")
    parser.add_argument("--quick", action="store_true", help="小规模流程检查，不作为正式论文结果")
    parser.add_argument("--skip-figures", action="store_true", help="跳过论文图形生成")
    parser.add_argument("--time-limit", type=int, default=60, help="每个MILP的求解时限（秒）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for directory in (OUTPUT_DIR, LOG_DIR, SOURCE_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    train_n, val_n, test_n = (20, 200, 500) if args.quick else (200, 2000, 5000)
    optimization_n = min(20 if args.quick else 40, train_n)
    lambdas = (0.0, 0.4, 0.8) if args.quick else q2.LAMBDA_GRID
    time_limit = min(args.time_limit, 18) if args.quick else args.time_limit

    print("[STAGE A] 读取问题二模型接口与题目附件……")
    data = q2.load_input_data()
    q2_reference = load_q2_reference_candidate(decision_stat_map(data))
    print(f"  地块={len(data.plots)}，作物={len(data.crop_names)}，2023记录={len(data.planting_2023)}")

    print("[STAGE B] 生成基准分层公共因子训练、验证和测试情景……")
    train_bundle = generate_factor_scenarios(data, train_n, q2.TRAIN_SEED, "拉丁超立方", "Q3训练集")
    optimization_train, reduction = q2.reduce_training_scenarios(
        data, train_bundle.scenarios, optimization_n
    )
    validation_bundle = generate_factor_scenarios(
        data, val_n, q2.VALIDATION_SEED, "蒙特卡洛", "Q3验证集"
    )
    test_bundle = generate_factor_scenarios(
        data, test_n, q2.TEST_SEED, "蒙特卡洛", "Q3测试集"
    )
    independent_test = generate_factor_scenarios(
        data, test_n, q2.TEST_SEED, "蒙特卡洛", "Q2同种子独立测试集", kappa=0.0
    )
    route = factor_route_table(train_bundle.factors, reduction)
    correlation_diag = realized_correlation_table(data, test_bundle.scenarios)
    marginal_audit = marginal_audit_table(data, independent_test.scenarios, test_bundle.scenarios)
    print(f"  训练={train_n}（代表={optimization_n}），验证={val_n}，测试={test_n}")

    print("[STAGE C] 对完整分层因子模型独立求解 CVaR 风险权重……")
    main_candidates, stat_for_key, main_solver_diag = solve_main_candidates(
        data, optimization_train, lambdas, time_limit, q2_reference
    )
    train_evals = {
        candidate.risk_weight: q2.evaluate_plan(candidate, stat_for_key, optimization_train)
        for candidate in main_candidates
    }
    validation_evals = {
        candidate.risk_weight: q2.evaluate_plan(candidate, stat_for_key, validation_bundle.scenarios)
        for candidate in main_candidates
    }
    selected, frontier = q2.choose_candidate(main_candidates, validation_evals)
    test_evals = {
        candidate.risk_weight: q2.evaluate_plan(candidate, stat_for_key, test_bundle.scenarios)
        for candidate in main_candidates
    }
    training_objective = q2.training_objective_table(main_candidates, train_evals)
    test_frontier = pd.DataFrame([
        {"风险权重λ": candidate.risk_weight, **q2.evaluation_metrics(test_evals[candidate.risk_weight], "测试集_")}
        for candidate in main_candidates
    ])
    frontier = frontier.merge(training_objective, on="风险权重λ", how="left").merge(
        test_frontier, on="风险权重λ", how="left"
    )
    q2_full_test = q2.evaluate_plan(q2_reference, stat_for_key, test_bundle.scenarios)
    q3_full_test = test_evals[selected.risk_weight]
    print(
        f"  入选 λ={selected.risk_weight:.1f}；Q3测试均值={q3_full_test.total_profit.mean():,.0f}元；"
        f"CVaR90={q2.lower_tail_cvar(q3_full_test.total_profit):,.0f}元"
    )

    print("[STAGE D1] 相关强度敏感性：固定问题二和问题三方案进行同场景评价……")
    sensitivity_rows = []
    for kappa in CORRELATION_STRENGTHS:
        test_at_strength = generate_factor_scenarios(
            data, test_n, q2.TEST_SEED, "蒙特卡洛", f"κ={kappa}测试集", kappa=kappa
        ).scenarios
        for label, plan in (
            ("问题二既有方案", q2_reference),
            ("问题三基准方案", selected),
        ):
            evaluation = q2.evaluate_plan(plan, stat_for_key, test_at_strength)
            sensitivity_rows.append(metrics_row(label, evaluation, **{"相关强度κ": kappa}))
    sensitivity = pd.DataFrame(sensitivity_rows)

    print("[STAGE D2] 固定问题三方案进行公共因子模块消融……")
    ablation_rows = []
    ablation_specs = (
        ("问题二独立情景", 0.0, False, False),
        ("仅全局因子", 1.0, True, False),
        ("仅作物组因子", 1.0, False, True),
        ("完整分层因子", 1.0, True, True),
    )
    for label, kappa, include_global, include_group in ablation_specs:
        scenario = generate_factor_scenarios(
            data, test_n, q2.TEST_SEED, "蒙特卡洛", f"{label}消融测试集",
            kappa=kappa, include_global=include_global, include_group=include_group,
        ).scenarios
        evaluation = q2.evaluate_plan(selected, stat_for_key, scenario)
        ablation_rows.append(metrics_row(label, evaluation, **{"保留的相关模块": label}))
    ablation = pd.DataFrame(ablation_rows)

    print("[STAGE D3] 配对比较、约束复核与诊断……")
    bootstrap = bootstrap_plan_difference(
        q2_full_test, q3_full_test, 200 if args.quick else 800
    )
    annual_comparison = annual_comparison_table(q2_full_test, q3_full_test)
    hard_checks = q2.validate_hard_constraints(data, selected)
    solver_diag = main_solver_diag.copy()
    if correlation_diag["符号是否一致"].ne("是").any():
        print("  警告：至少一个实现相关性的符号与基准机理不一致，请查看诊断表。")

    print("[STAGE E] 导出种植方案、分析工作簿、源数据、日志与参数说明……")
    write_outputs(
        data, selected, q2_reference, frontier, training_objective, sensitivity,
        ablation, route, correlation_diag, marginal_audit, annual_comparison,
        bootstrap, solver_diag, hard_checks, q2_full_test, q3_full_test, args.quick,
    )
    if not args.skip_figures:
        print("[STAGE F] 使用 Python / matplotlib 生成 Nature 风格论文图……")
        subprocess.run([sys.executable, str(SCRIPT_DIR / "Q3_nature_figures.py")], check=True)
    print(f"完成。问题三结果目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
