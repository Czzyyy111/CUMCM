"""问题三论文级图形：相关强度、公共因子路线、风险权重、消融与诊断。

所有面板只读取主模型导出的 CSV，不重新求解、不筛除测试记录。输出 SVG、PDF、
300 dpi PNG 和 600 dpi LZW TIFF；SVG 保留可编辑文字。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR / "source_data"
OUTPUT_DIR = SCRIPT_DIR / "figures_nature"
SUMMARY_FILE = SCRIPT_DIR / "logs" / "运行摘要.json"

COLORS = {
    "q2": "#484878",
    "q3": "#0F4D92",
    "optimized": "#B64342",
    "global": "#42949E",
    "group": "#9A4D8E",
    "neutral": "#767676",
    "neutral_light": "#D8D8D8",
    "neutral_pale": "#EEEEEE",
    "gold": "#D39A32",
    "teal": "#3A8F88",
    "ink": "#272727",
    "grid": "#E4E7EA",
    "rose_light": "#F6CFCB",
    "blue_light": "#C9DDF2",
}


def apply_publication_style() -> None:
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = [
        'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Arial', 'DejaVu Sans', 'Liberation Sans'
    ]
    mpl.rcParams.update({
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.3,
        "ytick.labelsize": 7.3,
        "legend.fontsize": 7.0,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def export_figure(fig, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUTPUT_DIR / stem
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(
        base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.04,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def clean_axis(ax, grid: bool = True) -> None:
    ax.spines["left"].set_color(COLORS["ink"])
    ax.spines["bottom"].set_color(COLORS["ink"])
    if grid:
        ax.grid(axis="y", color=COLORS["grid"], lw=0.55, zorder=0)
    else:
        ax.grid(False)


def panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.04, label, transform=ax.transAxes, fontsize=9.2,
            fontweight="bold", ha="left", va="bottom", color=COLORS["ink"])


def set_zoomed_ylim(ax, values: np.ndarray, lower_pad: float = 0.18,
                    upper_pad: float = 0.28) -> None:
    """按当前序列的实际跨度缩放纵轴，并为直接数值标签留出空间。"""
    values = np.asarray(values, dtype=float)
    span = float(np.ptp(values))
    if span <= 1e-12:
        span = max(abs(float(values.mean())) * 0.01, 1.0)
    ax.set_ylim(float(values.min()) - lower_pad * span,
                float(values.max()) + upper_pad * span)


def annotate_line_values(ax, x: np.ndarray, y: np.ndarray, displayed: np.ndarray,
                         color: str, vertical_offset: float = 7.0,
                         decimals: int = 2) -> None:
    """在每个数据点旁标注原始绝对值，并将首尾标签向图内收拢。"""
    for index, (xi, yi, value) in enumerate(zip(x, y, displayed)):
        if index == 0:
            horizontal_offset, alignment = 3, "left"
        elif index == len(x) - 1:
            horizontal_offset, alignment = -3, "right"
        else:
            horizontal_offset, alignment = 0, "center"
        ax.annotate(
            f"{value:.{decimals}f}", (xi, yi),
            xytext=(horizontal_offset, vertical_offset), textcoords="offset points",
            ha=alignment, va="bottom" if vertical_offset >= 0 else "top",
            fontsize=6.4, color=color, clip_on=False,
        )


def load_sources() -> tuple[dict[str, pd.DataFrame], dict]:
    required = {
        "sensitivity": "相关强度利润比较.csv",
        "route": "公共因子路径.csv",
        "frontier": "风险厌恶结果.csv",
        "ablation": "消融实验结果.csv",
        "loadings": "公共因子载荷.csv",
        "correlation": "实现相关性诊断.csv",
        "annual": "年度利润比较.csv",
        "profit": "配对测试利润.csv",
    }
    paths = {key: SOURCE_DIR / filename for key, filename in required.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少图形源数据：" + "；".join(missing))
    frames = {key: pd.read_csv(path) for key, path in paths.items()}
    summary = json.loads(SUMMARY_FILE.read_text(encoding="utf-8")) if SUMMARY_FILE.exists() else {}
    return frames, summary


def figure_correlation_strength(frame: pd.DataFrame) -> None:
    """以 κ=0 为参照放大趋势，点旁保留各方案的绝对利润。"""
    order = ["问题二既有方案", "问题三基准方案"]
    colors = [COLORS["q2"], COLORS["q3"]]
    styles = ["--", "-"]
    markers = ["s", "o"]
    fig, axes = plt.subplots(1, 2, figsize=(7.20, 3.55), sharex=True)
    plotted_values = [[], []]
    for series_index, (label, color, style, marker) in enumerate(zip(order, colors, styles, markers)):
        df = frame[frame["方案或实验"] == label].sort_values("相关强度κ")
        x = df["相关强度κ"].to_numpy(float)
        mean = df["期望利润/元"].to_numpy(float) / 1e4
        cvar = df["下尾CVaR90/元"].to_numpy(float) / 1e4
        mean_change = mean - mean[0]
        cvar_change = cvar - cvar[0]
        axes[0].plot(x, mean_change, color=color, ls=style, lw=1.9,
                     marker=marker, ms=4.0, label=label)
        axes[1].plot(x, cvar_change, color=color, ls=style, lw=1.9,
                     marker=marker, ms=4.0, label=label)
        mean_label_offset = -8.0 if series_index == 0 else 7.0
        cvar_label_offset = 7.0 if series_index == 0 else -8.0
        annotate_line_values(axes[0], x, mean_change, mean, color, mean_label_offset)
        annotate_line_values(axes[1], x, cvar_change, cvar, color, cvar_label_offset)
        plotted_values[0].extend(mean_change.tolist())
        plotted_values[1].extend(cvar_change.tolist())
    for ax in axes:
        ax.axvline(1.0, color=COLORS["gold"], lw=1.0, ls="--", alpha=0.8)
        ax.axhline(0.0, color=COLORS["neutral"], lw=0.7, alpha=0.7)
        ax.set_xticks(sorted(frame["相关强度κ"].unique()))
        ax.set_xlabel("相关强度 κ")
        clean_axis(ax, grid=True)
        ax.yaxis.set_major_locator(MaxNLocator(7))
    set_zoomed_ylim(axes[0], np.asarray(plotted_values[0]), lower_pad=0.22, upper_pad=0.33)
    set_zoomed_ylim(axes[1], np.asarray(plotted_values[1]), lower_pad=0.23, upper_pad=0.15)
    axes[0].set_ylabel("相对 κ=0 的期望利润变化（万元）")
    axes[1].set_ylabel("相对 κ=0 的 CVaR90 变化（万元）")
    axes[0].set_title("平均收益变化", loc="left", fontweight="bold")
    axes[1].set_title("联合尾部变化", loc="left", fontweight="bold")
    panel_label(axes[0], "甲")
    panel_label(axes[1], "乙")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.54, 0.99))
    fig.suptitle("公共相关强度改变两套固定方案的收益与联合尾部", x=0.07, y=1.08,
                 ha="left", fontsize=10.2, fontweight="bold")
    fig.text(0.985, 0.015, "纵轴为相对 κ=0 的变化；点旁数字为绝对利润（万元）；竖虚线为主模型 κ=1",
             ha="right", va="bottom", fontsize=6.5, color=COLORS["neutral"])
    fig.subplots_adjust(left=0.09, right=0.985, top=0.80, bottom=0.18, wspace=0.30)
    export_figure(fig, "图1_相关强度与问题二同场景利润比较")


def figure_factor_routes(route: pd.DataFrame) -> None:
    """参考用户平行路线图：灰色完整训练池、红色代表情景、蓝色中位路线。"""
    columns = [
        "气候因子标准分数", "农资通胀因子标准分数",
        "市场景气因子标准分数", "作物组因子加权均值标准分数",
    ]
    values = route[columns].to_numpy(float)
    representative = route["是否优化代表情景"].eq("是").to_numpy()
    x = np.arange(len(columns))
    q10, median, q90 = np.quantile(values, [0.10, 0.50, 0.90], axis=0)
    fig, ax = plt.subplots(figsize=(7.20, 4.05))
    ax.fill_between(x, q10, q90, color=COLORS["neutral_light"], alpha=0.72, zorder=0)
    for row in values[~representative]:
        ax.plot(x, row, color=COLORS["neutral"], lw=0.42, alpha=0.10, zorder=1, rasterized=True)
    for row in values[representative]:
        ax.plot(x, row, color=COLORS["optimized"], lw=0.85, alpha=0.48, zorder=2)
    ax.plot(x, median, color=COLORS["q3"], lw=2.2, marker="o", ms=4.2, zorder=3)
    for position in x:
        ax.axvline(position, color=COLORS["grid"], lw=0.6, zorder=0)
    ax.axhline(0, color=COLORS["neutral"], lw=0.7, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([
        "气候条件\n有利程度", "农资成本\n上涨压力",
        "整体市场\n需求景气", "作物组行情\n加权状态",
    ])
    ax.set_ylabel("七年平均公共因子标准分数")
    ax.set_title("分层公共因子训练路径与优化代表情景覆盖", loc="left",
                 fontweight="bold", pad=10)
    clean_axis(ax, grid=True)
    handles = [
        Line2D([0], [0], color=COLORS["neutral"], lw=1.0, alpha=0.45, label=f"全部{len(route)}条训练路径"),
        Line2D([0], [0], color=COLORS["optimized"], lw=1.8,
               label=f"{representative.sum()}条优化代表路径"),
        Line2D([0], [0], color=COLORS["q3"], lw=2.2, marker="o", ms=4, label="各因子中位路线"),
    ]
    ax.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01))
    fig.text(0.985, 0.018, "灰线保留完整训练池；红线来自综合压力等频分层；未为绘图删除路径",
             ha="right", fontsize=6.5, color=COLORS["neutral"])
    fig.subplots_adjust(left=0.105, right=0.985, top=0.80, bottom=0.20)
    export_figure(fig, "图2_公共因子载荷分布路线")


def figure_risk_aversion(frontier: pd.DataFrame) -> None:
    """相关情景下不同风险厌恶权重的训练与测试表现。"""
    df = frontier.sort_values("风险权重λ")
    x = df["风险权重λ"].to_numpy(float)
    selected = df["是否入选"].eq("是").to_numpy()
    mean_test = df["测试集_期望利润/元"].to_numpy(float) / 1e4
    cvar_test = df["测试集_下尾CVaR90/元"].to_numpy(float) / 1e4
    objective = df["训练集_CVaR加权目标/元"].to_numpy(float) / 1e4

    fig, axes = plt.subplots(1, 3, figsize=(7.20, 3.95), sharex=True)
    series = [
        (mean_test, COLORS["q3"], "o", "测试期望利润", "万元"),
        (cvar_test, COLORS["optimized"], "s", "测试下尾 CVaR90", "万元"),
        (objective, COLORS["gold"], "D", "训练加权目标", "万元"),
    ]
    for index, (ax, (values, color, marker, title, unit)) in enumerate(zip(axes, series)):
        ax.plot(x, values, color=color, lw=1.9, marker=marker, ms=3.8)
        ax.scatter(x[selected], values[selected], s=58, color=COLORS["gold"],
                   edgecolor="white", lw=0.8, zorder=5)
        annotate_line_values(ax, x, values, values, color, vertical_offset=7.0, decimals=2)
        set_zoomed_ylim(ax, values, lower_pad=0.18, upper_pad=0.38)
        ax.set_xticks(x)
        ax.set_xlabel("风险权重 λ")
        ax.set_ylabel(unit)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=8.6)
        ax.yaxis.set_major_locator(MaxNLocator(6))
        clean_axis(ax, grid=True)
        panel_label(ax, "甲乙丙"[index])
    fig.suptitle("引入公共相关性后重新选择风险厌恶权重", x=0.07, y=0.99,
                 ha="left", fontsize=10.2, fontweight="bold")
    selected_lambda = x[selected][0] if selected.any() else np.nan
    fig.text(0.985, 0.015,
             f"三幅纵轴分别按各自数据范围放大；点旁为准确值；金色点为最终 λ={selected_lambda:.1f}",
             ha="right", fontsize=6.5, color=COLORS["neutral"])
    fig.subplots_adjust(left=0.085, right=0.985, top=0.80, bottom=0.18, wspace=0.38)
    export_figure(fig, "图3_公共干扰下风险厌恶因子结果")


def figure_ablation(ablation: pd.DataFrame) -> None:
    order = ["问题二独立情景", "仅全局因子", "仅作物组因子", "完整分层因子"]
    colors = [COLORS["q2"], COLORS["global"], COLORS["group"], COLORS["q3"]]
    df = ablation.set_index("方案或实验").loc[order]
    metrics = [
        ("期望利润/元", "期望利润（万元）", 1e4, 2),
        ("下尾CVaR90/元", "下尾CVaR90（万元）", 1e4, 2),
        ("平均滞销率", "平均滞销率（%）", 0.01, 3),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.20, 4.15), sharey=True)
    y = np.arange(len(order))
    for ax, (column, title, divisor, decimals) in zip(axes, metrics):
        values = df[column].to_numpy(float) / divisor
        bars = ax.barh(y, values, color=colors, edgecolor="white", lw=0.5)
        for bar, alpha in zip(bars, [0.72, 0.76, 0.82, 1.0]):
            bar.set_alpha(alpha)
        span = max(float(np.ptp(values)), abs(values).mean() * 0.001, 1e-6)
        ax.set_xlim(values.min() - 0.10 * span, values.max() + 0.16 * span)
        for yi, value in zip(y, values):
            ax.text(value + 0.02 * span, yi, f"{value:.{decimals}f}",
                    va="center", ha="left", fontsize=6.8)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=8.7)
        ax.grid(axis="x", color=COLORS["grid"], lw=0.55)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_locator(MaxNLocator(4))
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order)
    axes[0].invert_yaxis()
    axes[2].xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    fig.suptitle("公共因子模块消融揭示全局与组内相关性的风险贡献", x=0.04, y=0.98,
                 ha="left", fontsize=10.2, fontweight="bold")
    fig.text(0.985, 0.015, "冻结最终问题三方案并共享底层随机数；仅依次关闭全局或作物组相关模块",
             ha="right", fontsize=6.5, color=COLORS["neutral"])
    fig.subplots_adjust(left=0.18, right=0.985, top=0.80, bottom=0.13, wspace=0.34)
    export_figure(fig, "图4_分层公共因子消融实验")


def figure_loading_and_correlation(loadings: pd.DataFrame, correlation: pd.DataFrame) -> None:
    """参数透明度：固定载荷矩阵与测试情景实现相关性。"""
    factor_order = ["气候因子", "农资通胀因子", "市场景气因子", "作物组市场因子"]
    channel_order = list(dict.fromkeys(loadings["变量通道"].tolist()))
    matrix = loadings.pivot_table(index="变量通道", columns="公共因子", values="基准载荷", fill_value=0.0)
    matrix = matrix.reindex(index=channel_order, columns=factor_order, fill_value=0.0)

    fig = plt.figure(figsize=(7.20, 5.15))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.58)
    ax1, ax2 = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    image = ax1.imshow(matrix.to_numpy(float), cmap="RdBu_r", vmin=-0.75, vmax=0.75, aspect="auto")
    ax1.set_xticks(np.arange(len(factor_order)))
    ax1.set_xticklabels(["气候", "农资通胀", "市场景气", "作物组行情"], rotation=32, ha="right")
    ax1.set_yticks(np.arange(len(channel_order)))
    ax1.set_yticklabels(channel_order)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iat[i, j]
            if abs(value) > 1e-12:
                ax1.text(j, i, f"{value:.2f}", ha="center", va="center",
                         color="white" if abs(value) > 0.42 else COLORS["ink"], fontsize=6.5)
    ax1.set_title("基准公共因子载荷", loc="left", fontweight="bold")
    ax1.tick_params(length=0)
    for spine in ax1.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax1, fraction=0.036, pad=0.03)
    cbar.ax.set_title("载荷", fontsize=6.5, pad=3)
    cbar.ax.tick_params(labelsize=6.5)
    panel_label(ax1, "甲")

    corr = correlation.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(corr))
    target = corr["基准隐含相关系数"].to_numpy(float)
    realized = corr["测试情景实现Spearman相关"].to_numpy(float)
    for yi, left, right in zip(y, target, realized):
        ax2.plot([left, right], [yi, yi], color=COLORS["neutral_light"], lw=2.0)
    ax2.scatter(target, y, facecolor="white", edgecolor=COLORS["q2"], s=35, lw=1.1, label="基准隐含相关")
    ax2.scatter(realized, y, color=COLORS["q3"], s=32, label="测试实现秩相关")
    ax2.axvline(0, color=COLORS["neutral"], lw=0.8, ls="--")
    ax2.set_yticks(y)
    ax2.set_yticklabels(corr["相关关系"])
    ax2.tick_params(axis="y", labelsize=6.8, pad=1)
    ax2.set_xlabel("相关系数")
    ax2.set_title("目标方向与模拟实现", loc="left", fontweight="bold")
    ax2.legend(loc="lower right")
    clean_axis(ax2, grid=False)
    panel_label(ax2, "乙")
    fig.suptitle("公共因子载荷与相关结构实现诊断", x=0.05, y=0.99,
                 ha="left", fontsize=10.2, fontweight="bold")
    fig.text(0.985, 0.015, "实现值为完整测试情景的年度平均Spearman相关；非线性分位映射后不要求与载荷乘积完全相等",
             ha="right", fontsize=6.4, color=COLORS["neutral"])
    fig.subplots_adjust(left=0.17, right=0.98, top=0.88, bottom=0.12)
    export_figure(fig, "图5_公共因子载荷与实现相关性")


def figure_annual_comparison(annual: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.20, 3.85))
    settings = [
        ("问题二既有方案", COLORS["q2"], "--", COLORS["neutral_light"], -9.0),
        ("问题三最终方案", COLORS["q3"], "-", COLORS["blue_light"], 7.0),
    ]
    lower_bounds, upper_bounds = [], []
    for label, color, style, band, label_offset in settings:
        df = annual[annual["方案"] == label].sort_values("年份")
        x = df["年份"].to_numpy(int)
        mean = df["平均利润/元"].to_numpy(float) / 1e4
        p10 = df["P10利润/元"].to_numpy(float) / 1e4
        p90 = df["P90利润/元"].to_numpy(float) / 1e4
        ax.fill_between(x, p10, p90, color=band, alpha=0.42)
        ax.plot(x, mean, color=color, ls=style, lw=2.0, marker="o", ms=3.8, label=label)
        annotate_line_values(ax, x, mean, mean, color, vertical_offset=label_offset, decimals=2)
        lower_bounds.extend(p10.tolist())
        upper_bounds.extend(p90.tolist())
    ax.set_xticks(sorted(annual["年份"].unique()))
    ax.set_xlabel("年份")
    ax.set_ylabel("年度利润（万元）")
    lower = np.floor(min(lower_bounds) / 5.0) * 5.0 - 5.0
    upper = np.ceil(max(upper_bounds) / 5.0) * 5.0 + 5.0
    ax.set_ylim(lower, upper)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.set_title("问题二与问题三方案在同一相关测试场景中的年度表现", loc="left",
                 fontweight="bold", pad=10)
    ax.legend(loc="upper left")
    clean_axis(ax, grid=True)
    fig.text(0.985, 0.018, "纵轴刻度间隔缩小为10万元；点旁为年度平均利润准确值；阴影为第10—90百分位区间",
             ha="right", fontsize=6.5, color=COLORS["neutral"])
    fig.subplots_adjust(left=0.10, right=0.985, top=0.82, bottom=0.17)
    export_figure(fig, "图6_问题二与问题三年度利润比较")


def write_qa_notes(frames: dict[str, pd.DataFrame], summary: dict) -> None:
    route = frames["route"]
    profit = frames["profit"]
    selected = summary.get("最终风险权重", "未记录")
    text = f"""# 问题三图形说明与质量检查

## 图形契约

- 核心结论：分层公共因子改变联合尾部风险，并使最优种植方案对相关冲击作出结构性响应。
- 图形类型：定量比较组图。图1为相关强度主证据；图2为情景生成与削减覆盖；图3为风险厌恶选择；图4为消融；图5为参数与相关性诊断；图6为年度配对比较。
- 绘图后端：Python / matplotlib；未使用其他绘图后端。
- 最终尺寸：约183 mm双栏宽度；SVG/PDF保留可编辑文字；PNG为300 dpi预览；TIFF为600 dpi并采用LZW压缩。

## 数据完整性

- 运行模式：{summary.get('运行模式', '未记录')}。
- 最终风险权重：λ={selected}。
- 相关强度图保留全部 {len(frames['sensitivity'])} 条“强度—方案”汇总记录。
- 公共因子路线使用全部 {len(route)} 条训练路径，其中 {route['是否优化代表情景'].eq('是').sum()} 条为优化代表情景；未删除或抽样隐藏路径。
- 配对测试利润保留全部 {len(profit)} 条测试情景；问题二与问题三使用相同情景编号。
- 消融图保留四种预设模型，不按结果筛选。

## 建议图注

**图1｜不同相关强度下问题二与问题三方案的同场景利润比较。** κ=0退化为独立情景，κ=1为主模型。纵轴显示相对κ=0的变化以放大趋势，点旁标注对应绝对利润；两套方案均保持固定，并在每个κ下使用同一组底层随机数。

**图2｜分层公共因子训练路径及优化代表情景覆盖。** 灰线为完整训练池，红线为综合压力等频分层选出的代表路径，蓝线为中位路线；各轴为七年平均公共因子标准分数。

**图3｜引入公共相关性后的风险厌恶权重结果。** 甲、乙、丙分别展示测试期望利润、测试下尾CVaR90和训练加权目标，各纵轴按自身数据范围缩放，点旁标注准确值；金色点由独立验证集规则选择。

**图4｜分层公共因子消融实验。** 冻结最终问题三方案，使用同一组底层随机数生成独立、仅全局、仅作物组和完整分层四类测试情景，仅改变相关模块。利润与CVaR标注保留两位小数，滞销率保留三位小数，以显示原始数据中的细微差异。

**图5｜基准公共因子载荷及其实现相关性。** 甲给出变量通道对公共因子的固定载荷，乙比较载荷乘积隐含相关与测试情景实现的Spearman相关。

**图6｜问题二与问题三方案的年度配对比较。** 实线/虚线为各年度平均利润，点旁标注准确值，纵轴刻度间隔为10万元，阴影为第10—90百分位区间；两套方案在完全相同的相关测试路径中评价。

## 审稿风险与解释边界

- 载荷是根据农业和市场机理设置的中等强度先验，不是由2023年单年数据估计的相关系数。
- 分位区间来自设定分布下的模拟情景，不是真实历史置信区间。
- 相关强度变化同时改变联合尾部，不改变题面给定的单变量边界。
- 若MIP间隙较大，几万元以内的方案差异不得解释为严格最优性改进。
- 图2借鉴平行坐标路径的视觉结构，但统计含义由当前公共因子数据重新定义，没有复用参考图中的数值。
"""
    (OUTPUT_DIR / "图形说明与QA.md").write_text(text, encoding="utf-8")


def main() -> None:
    apply_publication_style()
    frames, summary = load_sources()
    figure_correlation_strength(frames["sensitivity"])
    figure_factor_routes(frames["route"])
    figure_risk_aversion(frames["frontier"])
    figure_ablation(frames["ablation"])
    figure_loading_and_correlation(frames["loadings"], frames["correlation"])
    figure_annual_comparison(frames["annual"])
    write_qa_notes(frames, summary)
    print(f"已生成6张问题三论文级图形：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
