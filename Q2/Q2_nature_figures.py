"""问题二论文级中文图形：风险收益前沿、利润分布、典型情景、种植结构、训练目标与情景路线。

图 1—5 读取主模型输出；图 6 使用与主模型相同的固定种子复现 200 条 LHS 训练路径，
不重新优化、不更改任何利润记录。输出 PNG、SVG、PDF 与 600 dpi TIFF。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator

from Q2_Stochastic_CVaR import (
    TRAIN_SCENARIOS,
    TRAIN_SEED,
    generate_scenarios,
    load_input_data,
    reduce_training_scenarios,
)


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR / "source_data"
OUTPUT_DIR = SCRIPT_DIR / "figures_nature"
LOG_FILE = SCRIPT_DIR / "logs" / "运行摘要.json"
YEARS = np.arange(2024, 2031)

COLORS = {
    "neutral": "#7A8088",
    "neutral_light": "#D9DDE2",
    "blue": "#276FBF",
    "blue_dark": "#154C79",
    "blue_light": "#C9DDF2",
    "gold": "#D39A32",
    "gold_light": "#F2E4C4",
    "rose": "#C66B7D",
    "teal": "#4C9A91",
    "violet": "#8C78B8",
    "ink": "#25282C",
    "muted": "#72777F",
    "grid": "#E5E8EB",
    "tail": "#E8B4B8",
}


def apply_publication_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.2,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def load_sources():
    build_scenario_route_source()
    required = {
        "frontier": SOURCE_DIR / "风险收益前沿.csv",
        "profit": SOURCE_DIR / "测试情景利润.csv",
        "annual": SOURCE_DIR / "测试情景年度利润.csv",
        "typical": SOURCE_DIR / "典型情景年度利润.csv",
        "structure": SOURCE_DIR / "年度作物结构.csv",
        "training_objective": SOURCE_DIR / "训练目标函数.csv",
        "plan_difference": SOURCE_DIR / "方案差异诊断.csv",
        "scenario_route": SOURCE_DIR / "情景扰动路线.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少图形源数据：" + "；".join(missing))
    frames = {name: pd.read_csv(path) for name, path in required.items()}
    if LOG_FILE.exists():
        summary = json.loads(LOG_FILE.read_text(encoding="utf-8"))
    else:
        summary = {}
    return frames, summary


def build_scenario_route_source() -> pd.DataFrame:
    """复现正式训练池并构造四类可比的标准化情景扰动指标。"""
    data = load_input_data()
    scenarios = generate_scenarios(
        data, TRAIN_SCENARIOS, TRAIN_SEED, "拉丁超立方", "训练集"
    )
    _, mapping = reduce_training_scenarios(data, scenarios, 40)
    representative_ids = set(mapping["代表情景原编号"].astype(int).unique())
    crops = np.asarray(sorted(data.crop_names), dtype=int)

    demand = scenarios.demand[:, :, crops]
    demand_center = np.median(demand, axis=0)
    demand_deviation = np.mean(demand / demand_center[None, :, :] - 1.0, axis=(1, 2)) * 100

    yield_deviation = np.mean(
        scenarios.yield_mult[:, :, crops] - 1.0, axis=(1, 2)
    ) * 100

    cost = scenarios.cost_mult[:, :, crops]
    cost_center = np.median(cost, axis=0)
    cost_deviation = np.mean(cost / cost_center[None, :, :] - 1.0, axis=(1, 2)) * 100

    price = scenarios.price_mult[:, :, crops]
    price_center = np.median(price, axis=0)
    variable_price = np.ptp(price, axis=0) > 1e-12
    if not np.any(variable_price):
        raise RuntimeError("价格情景没有可变维度，无法构造价格扰动路线。")
    price_deviation = np.mean(
        price[:, variable_price] / price_center[variable_price][None, :] - 1.0,
        axis=1,
    ) * 100

    raw = np.column_stack([
        demand_deviation, yield_deviation, cost_deviation, price_deviation
    ])
    scale = raw.std(axis=0, ddof=0)
    if np.any(scale <= 1e-12):
        raise RuntimeError("至少一个情景扰动指标没有跨情景差异。")
    standardized = (raw - raw.mean(axis=0)) / scale

    route = pd.DataFrame({
        "训练情景编号": np.arange(1, scenarios.n + 1),
        "是否优化代表情景": ["是" if i in representative_ids else "否" for i in range(1, scenarios.n + 1)],
        "销量相对中心路径偏差/%": demand_deviation,
        "亩产量相对基准偏差/%": yield_deviation,
        "成本相对中心路径偏差/%": cost_deviation,
        "可变价格相对中心路径偏差/%": price_deviation,
        "销量扰动标准分数": standardized[:, 0],
        "亩产扰动标准分数": standardized[:, 1],
        "成本扰动标准分数": standardized[:, 2],
        "价格扰动标准分数": standardized[:, 3],
    })
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    route.to_csv(SOURCE_DIR / "情景扰动路线.csv", index=False, encoding="utf-8-sig")
    return route


def clean_axis(ax, y_grid: bool = False) -> None:
    ax.spines["left"].set_color(COLORS["ink"])
    ax.spines["bottom"].set_color(COLORS["ink"])
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6, zorder=0) if y_grid else ax.grid(False)


def panel_label(ax, label: str) -> None:
    ax.text(-0.10, 1.04, label, transform=ax.transAxes, fontsize=9, fontweight="bold",
            ha="left", va="bottom", color=COLORS["ink"])


def export_figure(fig, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUTPUT_DIR / stem
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.04,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def figure_risk_return(frontier: pd.DataFrame) -> None:
    """主证据：展示候选风险权重的样本外收益—下尾保护关系。"""
    df = frontier.sort_values("风险权重λ").copy()
    mean = df["测试集_期望利润/元"].to_numpy(float) / 1e4
    cvar = df["测试集_下尾CVaR90/元"].to_numpy(float) / 1e4
    selected = df["是否入选"].astype(str).eq("是").to_numpy()

    fig, ax = plt.subplots(figsize=(7.20, 3.85))
    collapsed = np.ptp(mean) < 0.01 and np.ptp(cvar) < 0.01
    if collapsed:
        weights = df["风险权重λ"].to_numpy(float)
        ax.plot(weights, mean, color=COLORS["blue_dark"], lw=2.0, marker="o", ms=4.2,
                label="测试集期望利润")
        ax.plot(weights, cvar, color=COLORS["rose"], lw=1.7, marker="s", ms=3.8,
                ls="--", label="最差10%情景平均利润")
        ax.scatter(weights[selected], mean[selected], s=72, color=COLORS["gold"],
                   edgecolor="white", linewidth=0.8, zorder=4, label="验证集入选权重")
        ax.set_xlabel("CVaR 风险权重")
        ax.set_ylabel("七年累计利润（万元）")
        ax.legend(loc="best")
        ax.text(0.02, 0.05, "各风险权重已独立优化但得到同一方案\n风险收益前沿在当前假设下退化",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=7.2,
                color=COLORS["muted"])
        title = "独立优化后风险权重候选仍表现稳定"
    else:
        ax.plot(cvar, mean, color=COLORS["neutral"], lw=1.3, zorder=1)
        ax.scatter(cvar[~selected], mean[~selected], s=44, facecolor="white",
                   edgecolor=COLORS["blue"], linewidth=1.3, zorder=3)
        ax.scatter(cvar[selected], mean[selected], s=78, facecolor=COLORS["gold"],
                   edgecolor="white", linewidth=0.8, zorder=4)
        for _, row in df.iterrows():
            x = row["测试集_下尾CVaR90/元"] / 1e4
            y = row["测试集_期望利润/元"] / 1e4
            suffix = "（入选）" if row["是否入选"] == "是" else ""
            ax.annotate(f"风险权重 {row['风险权重λ']:.1f}{suffix}", (x, y), xytext=(5, 6),
                        textcoords="offset points", fontsize=7.0,
                        color=COLORS["blue_dark"] if not suffix else COLORS["ink"])
        ax.set_xlabel("测试集下尾风险收益（最差10%情景平均利润，万元）")
        ax.set_ylabel("测试集期望利润（万元）")
        title = "不同风险权重种植方案的独立测试收益—下尾表现"
    ax.set_title(title, loc="left", fontweight="bold", pad=10)
    clean_axis(ax, y_grid=True)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    footer = ("五个候选均独立求解；若重合则表示当前情景与约束下风险前沿退化"
              if collapsed else "每个点对应一套共享于所有情景的七年种植方案")
    fig.text(0.985, 0.025, footer,
             ha="right", va="bottom", fontsize=6.7, color=COLORS["muted"])
    fig.subplots_adjust(left=0.12, right=0.97, top=0.82, bottom=0.20)
    export_figure(fig, "图1_风险收益前沿")


def figure_profit_distribution(profit: pd.DataFrame, annual: pd.DataFrame) -> None:
    """分布证据：全部测试情景的累计利润与年度分位带。"""
    values = profit["七年累计利润/元"].to_numpy(float) / 1e4
    p10, median, mean = np.quantile(values, 0.10), np.median(values), values.mean()
    annual_values = annual[[f"{year}年利润/元" for year in YEARS]].to_numpy(float) / 1e4
    q10, q50, q90 = np.quantile(annual_values, [0.10, 0.50, 0.90], axis=0)

    fig = plt.figure(figsize=(7.20, 4.05))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.30)
    ax1, ax2 = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    bins = np.histogram_bin_edges(values, bins="fd")
    counts, edges, patches = ax1.hist(values, bins=bins, color=COLORS["blue_light"],
                                      edgecolor="white", linewidth=0.4)
    for patch, left in zip(patches, edges[:-1]):
        if left < p10:
            patch.set_facecolor(COLORS["tail"])
            patch.set_hatch("///")
    ax1.axvline(p10, color=COLORS["rose"], lw=1.4, ls="--", label=f"第10百分位：{p10:.0f} 万元")
    ax1.axvline(median, color=COLORS["blue_dark"], lw=1.4, label=f"中位数：{median:.0f} 万元")
    ax1.axvline(mean, color=COLORS["gold"], lw=1.4, ls=":", label=f"均值：{mean:.0f} 万元")
    ax1.set_xlabel("七年累计利润（万元）")
    ax1.set_ylabel("测试情景数")
    ax1.legend(loc="upper left")
    clean_axis(ax1, y_grid=True)
    panel_label(ax1, "甲")

    ax2.fill_between(YEARS, q10, q90, color=COLORS["blue_light"], alpha=0.9,
                     label="第10—90百分位区间")
    ax2.plot(YEARS, q50, color=COLORS["blue_dark"], lw=2.0, marker="o", ms=4,
             label="年度利润中位数")
    ax2.axhline(0, color=COLORS["neutral"], lw=0.8, ls="--")
    ax2.set_xticks(YEARS)
    ax2.set_xticklabels(YEARS, rotation=45, ha="right")
    ax2.set_xlabel("年份")
    ax2.set_ylabel("年度利润（万元）")
    ax2.legend(loc="best")
    clean_axis(ax2, y_grid=True)
    panel_label(ax2, "乙")

    fig.suptitle("全新测试情景揭示最终方案的利润分布与年度波动",
                 x=0.08, y=0.99, ha="left", fontsize=10, fontweight="bold")
    fig.text(0.985, 0.018, f"测试情景数：{len(values)}；左图使用全部记录，阴影为最低10%利润情景",
             ha="right", va="bottom", fontsize=6.6, color=COLORS["muted"])
    fig.subplots_adjust(left=0.09, right=0.985, top=0.82, bottom=0.21)
    export_figure(fig, "图2_测试利润分布与年度区间")


def figure_typical_scenarios(typical: pd.DataFrame) -> None:
    """典型情景：最差、P10、中位、P90 和最佳测试路径的年度利润。"""
    order = ["最差情景", "下行典型情景（P10）", "中位情景（P50）", "上行情景（P90）", "最佳情景"]
    colors = [COLORS["rose"], "#B9876A", COLORS["blue_dark"], COLORS["teal"], COLORS["violet"]]
    styles = [":", "--", "-", "--", ":"]
    widths = [1.2, 1.5, 2.4, 1.5, 1.2]

    fig, ax = plt.subplots(figsize=(7.20, 3.95))
    all_values = []
    for label, color, style, width in zip(order, colors, styles, widths):
        frame = typical[typical["典型情景"] == label].sort_values("年份")
        y = frame["年度利润/元"].to_numpy(float) / 1e4
        all_values.extend(y.tolist())
        total = frame["七年累计利润/元"].iloc[0] / 1e4
        ax.plot(YEARS, y, color=color, ls=style, lw=width, marker="o", ms=3.8,
                label=f"{label}（累计 {total:.0f} 万元）")
    values = np.asarray(all_values, dtype=float)
    span = max(float(np.ptp(values)), 1.0)
    tick_step = 10.0 if span <= 100 else (20.0 if span <= 200 else 50.0)
    lower = np.floor((values.min() - 0.08 * span) / tick_step) * tick_step
    upper = np.ceil((values.max() + 0.08 * span) / tick_step) * tick_step
    ax.set_ylim(lower, upper)
    ax.yaxis.set_major_locator(MultipleLocator(tick_step))
    ax.set_xticks(YEARS)
    ax.set_xlabel("年份")
    ax.set_ylabel("年度利润（万元）")
    ax.set_title("五类典型测试情景呈现最终种植方案的年度利润路径", loc="left",
                 fontweight="bold", pad=10)
    ax.legend(loc="best", ncol=2, columnspacing=1.2, handlelength=2.4)
    clean_axis(ax, y_grid=True)
    ax.text(0.01, 0.03, "纵轴采用局部放大，未从 0 起",
            transform=ax.transAxes, fontsize=6.6, color=COLORS["muted"],
            ha="left", va="bottom")
    fig.text(0.985, 0.025, "典型情景按七年累计利润最接近相应经验分位点选取；纵轴刻度已局部放大",
             ha="right", va="bottom", fontsize=6.7, color=COLORS["muted"])
    fig.subplots_adjust(left=0.10, right=0.985, top=0.82, bottom=0.17)
    export_figure(fig, "图3_典型情景年度利润")


def figure_planting_structure(structure: pd.DataFrame) -> None:
    categories = ["豆类", "其他粮食", "其他蔬菜", "食用菌"]
    colors = [COLORS["teal"], COLORS["gold"], COLORS["blue"], COLORS["violet"]]
    hatches = ["///", "", "...", "\\\\"]
    table = structure.pivot(index="年份", columns="作物大类", values="种植面积/亩").reindex(
        index=YEARS, columns=categories, fill_value=0
    )
    fig, ax = plt.subplots(figsize=(7.20, 3.85))
    bottom = np.zeros(len(YEARS))
    for category, color, hatch in zip(categories, colors, hatches):
        values = table[category].fillna(0).to_numpy(float)
        ax.bar(YEARS, values, bottom=bottom, width=0.68, color=color, label=category,
               hatch=hatch, edgecolor="white", linewidth=0.5)
        bottom += values
    ax.set_xticks(YEARS)
    ax.set_xlabel("年份")
    ax.set_ylabel("种植面积（亩·季）")
    ax.set_title("风险控制后的七年作物类别种植结构", loc="left", fontweight="bold", pad=10)
    ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    clean_axis(ax, y_grid=False)
    fig.text(0.985, 0.025, "面积为同一年各季种植面积之和；全部地块方案均纳入汇总",
             ha="right", va="bottom", fontsize=6.7, color=COLORS["muted"])
    fig.subplots_adjust(left=0.10, right=0.985, top=0.78, bottom=0.17)
    export_figure(fig, "图4_年度作物种植结构")


def figure_training_objective(training_objective: pd.DataFrame, frontier: pd.DataFrame) -> None:
    """优化证据：展示每个风险权重下的训练期望、下尾 CVaR 与加权目标。"""
    df = training_objective.sort_values("风险权重λ").copy()
    weights = df["风险权重λ"].to_numpy(float)
    mean_profit = df["训练集_期望利润/元"].to_numpy(float) / 1e4
    cvar_profit = df["训练集_下尾CVaR90/元"].to_numpy(float) / 1e4
    objective = df["训练集_CVaR加权目标/元"].to_numpy(float) / 1e4
    selected_lambda = float(frontier.loc[frontier["是否入选"] == "是", "风险权重λ"].iloc[0])
    selected = np.isclose(weights, selected_lambda)

    fig, ax = plt.subplots(figsize=(7.20, 3.95))
    ax.plot(weights, mean_profit, color=COLORS["blue_dark"], lw=1.8,
            marker="o", ms=4.2, label="训练集期望利润")
    ax.plot(weights, cvar_profit, color=COLORS["rose"], lw=1.6,
            ls="--", marker="s", ms=3.8, label="训练集下尾 CVaR90")
    ax.plot(weights, objective, color=COLORS["gold"], lw=2.2,
            marker="D", ms=4.1, label="CVaR 加权目标值")
    ax.scatter(weights[selected], objective[selected], s=76, color=COLORS["gold"],
               edgecolor="white", linewidth=0.9, zorder=5)
    if selected.any():
        ax.annotate(f"验证集入选 λ={selected_lambda:.1f}",
                    (weights[selected][0], objective[selected][0]),
                    xytext=(7, 8), textcoords="offset points", fontsize=7.0,
                    color=COLORS["ink"])
    ax.set_xticks(weights)
    ax.set_xlabel("CVaR 风险权重 λ")
    ax.set_ylabel("七年累计利润或目标值（万元）")
    ax.set_title("风险权重改变训练阶段的收益—下尾风险加权目标", loc="left",
                 fontweight="bold", pad=10)
    ax.legend(loc="best", ncol=3, columnspacing=1.0, handlelength=2.3)
    clean_axis(ax, y_grid=True)
    ax.yaxis.set_major_locator(MaxNLocator(7))
    fig.text(
        0.985, 0.025,
        "目标值 = (1−λ)×训练集期望利润 + λ×训练集下尾CVaR90；不同λ对应不同偏好尺度",
        ha="right", va="bottom", fontsize=6.7, color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.11, right=0.985, top=0.82, bottom=0.18)
    export_figure(fig, "图5_风险权重与CVaR加权目标")


def figure_scenario_routes(route: pd.DataFrame) -> None:
    """情景构造证据：展示训练池及代表情景在四类不确定因素上的路线。"""
    columns = [
        "销量扰动标准分数", "亩产扰动标准分数",
        "成本扰动标准分数", "价格扰动标准分数",
    ]
    values = route[columns].to_numpy(float)
    representative = route["是否优化代表情景"].eq("是").to_numpy()
    x = np.arange(len(columns))
    q10, median, q90 = np.quantile(values, [0.10, 0.50, 0.90], axis=0)

    fig, ax = plt.subplots(figsize=(7.20, 4.05))
    ax.fill_between(x, q10, q90, color=COLORS["neutral_light"], alpha=0.75,
                    zorder=0, label="第10—90百分位区间")
    for row in values[~representative]:
        ax.plot(x, row, color=COLORS["neutral"], lw=0.45, alpha=0.10,
                zorder=1, rasterized=True)
    for row in values[representative]:
        ax.plot(x, row, color=COLORS["rose"], lw=0.75, alpha=0.48, zorder=2)
    ax.plot(x, median, color=COLORS["blue_dark"], lw=2.1, marker="o", ms=4.0,
            zorder=3)
    for position in x:
        ax.axvline(position, color=COLORS["grid"], lw=0.6, zorder=0)

    ax.axhline(0, color=COLORS["neutral"], lw=0.75, ls="--", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([
        "销量扰动\n相对中心路径", "亩产扰动\n相对基准",
        "成本扰动\n相对中心路径", "价格扰动\n可变价格作物",
    ])
    ax.set_ylabel("标准化扰动指数 $z$")
    ax.set_title("拉丁超立方训练情景的四因素扰动路线", loc="left",
                 fontweight="bold", pad=10)
    clean_axis(ax, y_grid=True)
    ax.yaxis.set_major_locator(MaxNLocator(7))
    legend_handles = [
        Line2D([0], [0], color=COLORS["neutral"], lw=1.0, alpha=0.45,
               label="全部200条训练情景"),
        Line2D([0], [0], color=COLORS["rose"], lw=1.8,
               label="40条优化代表情景"),
        Line2D([0], [0], color=COLORS["blue_dark"], lw=2.1, marker="o",
               ms=4.0, label="各因素中位路线"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, 1.01), columnspacing=1.2, handlelength=2.3)
    fig.text(
        0.985, 0.022,
        "纵轴为各因素在训练池中的标准分数；粮食固定价格不进入价格扰动指标；未删除任何训练情景",
        ha="right", va="bottom", fontsize=6.6, color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.10, right=0.985, top=0.82, bottom=0.20)
    export_figure(fig, "图6_训练情景四因素扰动路线")


def write_qa_notes(frames: dict[str, pd.DataFrame], summary: dict) -> None:
    selected = summary.get("最终风险权重", "未记录")
    mode = summary.get("运行模式", "未记录")
    profit = frames["profit"]
    collapsed = (
        frames["frontier"]["测试集_期望利润/元"].max() - frames["frontier"]["测试集_期望利润/元"].min() < 1.0
        and frames["frontier"]["测试集_下尾CVaR90/元"].max() - frames["frontier"]["测试集_下尾CVaR90/元"].min() < 1.0
    )
    text = f"""# 问题二图形说明与质量检查

## 图形契约

- 核心结论：候选方案的期望收益—下尾保护关系由独立情景评价，最终方案在全新测试情景中展示利润分布、年度波动和典型路径。
- 图形类型：定量比较图；图 1 为样本外主证据，图 2—4 为分布、典型场景与结构验证，图 5 为训练优化目标审计，图 6 为情景构造与削减覆盖检查。
- 绘图后端：Python / matplotlib，全流程未使用其他绘图后端。
- 最终尺寸：约 183 mm 双栏宽度；SVG/PDF 保留可编辑文字；TIFF 为 600 dpi、LZW 压缩；PNG 为 300 dpi 预览。

## 数据完整性

- 运行模式：{mode}。
- 最终风险权重：{selected}。
- 测试利润记录：{len(profit)} 条，图 2 使用全部记录，无抽样、删除或人为平滑。
- 典型情景仅用于展示：按累计利润最接近最差、P10、P50、P90、最佳位置选取，不参与模型优化或参数选择。
- 风险收益前沿保留全部 {len(frames['frontier'])} 个风险权重候选。
- 候选方案是否重合：{'是；但五个权重均为独立完整MILP，未固定其他权重的支持集' if collapsed else '否；候选之间存在可见的风险—收益差异'}。
- 方案差异诊断保留全部 {len(frames['plan_difference'])} 个候选与 λ=0 的面积和作物支持集比较。
- 图 6 使用全部 {len(frames['scenario_route'])} 条 LHS 训练路径，其中 {frames['scenario_route']['是否优化代表情景'].eq('是').sum()} 条为优化代表情景；未抽样删除训练路径。

## 建议图注

**图 1｜候选种植方案的样本外风险—收益前沿。** 横轴为测试利润最低 10% 情景的平均值，纵轴为测试集平均七年累计利润；高亮点为独立验证集选择的最终方案。

**图 2｜最终方案在独立测试集中的利润分布与年度区间。** 甲为七年累计利润直方图，阴影表示最低 10% 情景；乙为各年度利润的第 10—90 百分位区间及中位数。

**图 3｜最终方案在五类典型测试情景中的年度利润。** 情景按七年累计利润最接近最差、P10、中位、P90 和最佳经验分位位置确定。

**图 4｜最终方案的年度作物类别种植结构。** 面积按豆类、其他粮食、其他蔬菜和食用菌汇总，单位“亩·季”为同年各季面积之和。

**图 5｜不同风险权重对应的训练阶段 CVaR 加权目标。** 同时展示训练集期望利润、下尾 CVaR90 和按相应 λ 计算的加权目标值；各点均来自独立求解的完整随机 MILP。

**图 6｜拉丁超立方训练情景的四因素扰动路线。** 灰线为全部 200 条训练路径，红线突出 40 条等频分层优化代表路径，蓝线为各因素中位数；纵轴为各因素跨情景标准分数。价格指标仅汇总存在随机价格变化的作物，粮食固定价格不参与标准化。

## 审稿风险与解释边界

- 图中分布来自题面区间及明确分布假设下的模拟情景，不应表述为真实历史频率或统计置信区间。
- 测试情景与训练、验证情景随机种子隔离，测试时冻结方案且不逐情景重新优化。
- 风险权重由验证集选择；测试集只用于最终报告，避免样本内乐观偏差。
- 若独立求解后的候选仍重合，λ=0.4 仅是预先声明的中等风险并列规则，不应解释为 CVaR 已观察到额外风险改善。
- 图 3 为辨认年度路径差异而局部放大纵轴，坐标范围未从 0 开始，图内已显式标注。
- 图 5 的加权目标随 λ 改变了评价尺度，适合解释风险偏好，不应把不同 λ 的目标值差额直接称为实际利润损失。
- 图 6 的标准分数只用于比较路线覆盖，不能反推原始百分比；原始偏差百分比已保留在源数据表中。
- 图中没有显著性检验；阴影和分位线均为模拟利润的经验分位数。
"""
    (OUTPUT_DIR / "图形说明与QA.md").write_text(text, encoding="utf-8")


def main() -> None:
    apply_publication_style()
    frames, summary = load_sources()
    figure_risk_return(frames["frontier"])
    figure_profit_distribution(frames["profit"], frames["annual"])
    figure_typical_scenarios(frames["typical"])
    figure_planting_structure(frames["structure"])
    figure_training_objective(frames["training_objective"], frames["frontier"])
    figure_scenario_routes(frames["scenario_route"])
    write_qa_notes(frames, summary)
    print(f"已生成 6 张论文级中文图形：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
