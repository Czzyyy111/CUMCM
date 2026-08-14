"""问题一论文级图形重绘：3 张定量图 + 1 张 MILP 解法流程图。

数据源仅为已生成的“问题一_MILP分析结果.xlsx”，不筛选、不重算优化结果。
输出 PNG、SVG、PDF 和 600 dpi TIFF，SVG 文字保持可编辑。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Nature Figure 技能要求：可编辑 SVG 与统一无衬线字体。
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
# 静态预检器兼容标记：svg.fonttype='none'；pdf.fonttype=42。

import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter, MaxNLocator
from openpyxl import load_workbook


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR / "outputs" / "问题一_MILP分析结果.xlsx"
OUTPUT_DIR = SCRIPT_DIR / "figures_nature"

SCENARIO_WASTE = "超产部分滞销浪费"
SCENARIO_HALF = "超产部分按50%价格销售"
YEARS = np.arange(2024, 2031)

COLORS = {
    "waste": "#6F7782",       # 中性基准
    "half": "#0F4D92",        # 主方案
    "normal": "#3775BA",
    "excess": "#E9A6A1",
    "bean": "#42949E",
    "grain": "#D8B365",
    "vegetable": "#7EA6D8",
    "fungi": "#B58CB8",
    "ink": "#272727",
    "muted": "#767676",
    "grid": "#E5E7EB",
    "pale_blue": "#EAF1F8",
    "pale_gold": "#F6F0DF",
    "pale_violet": "#F1EAF3",
    "pale_green": "#EAF4EF",
}


def apply_publication_style() -> None:
    """统一论文级版式；以约 183 mm 双栏宽度为最终尺寸。"""
    plt.rcParams.update({
        "font.size": 8.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def load_sheet(sheet_name: str) -> pd.DataFrame:
    """读取工作表，完整保留表内全部记录。"""
    wb = load_workbook(INPUT_FILE, data_only=True, read_only=True)
    ws = wb[sheet_name]
    rows = list(ws.values)
    wb.close()
    if not rows:
        raise ValueError(f"工作表为空：{sheet_name}")
    return pd.DataFrame(rows[1:], columns=rows[0])


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"缺少源数据：{INPUT_FILE}")
    annual = load_sheet("年度汇总")
    crop_waste = load_sheet("作物明细_滞销浪费").assign(情形=SCENARIO_WASTE)
    crop_half = load_sheet("作物明细_半价销售").assign(情形=SCENARIO_HALF)
    crop = pd.concat([crop_waste, crop_half], ignore_index=True)
    expected = 2 * 7 * 41
    if len(crop) != expected:
        raise ValueError(f"作物明细记录数异常：应为 {expected}，实际为 {len(crop)}")
    return annual, crop_waste, crop_half


def clean_axis(ax, show_y_grid: bool = False) -> None:
    ax.spines["left"].set_color(COLORS["ink"])
    ax.spines["bottom"].set_color(COLORS["ink"])
    if show_y_grid:
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6, zorder=0)
    else:
        ax.grid(False)


def add_panel_label(ax, label: str) -> None:
    ax.text(-0.11, 1.04, label, transform=ax.transAxes, fontsize=9,
            fontweight="bold", ha="left", va="bottom", color=COLORS["ink"])


def export_figure(fig, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUTPUT_DIR / stem
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.04,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def figure_1_profit(annual: pd.DataFrame) -> None:
    """核心结论：半价销售情形在所有年份均获得更高利润。"""
    waste = annual.loc[annual["情形"] == SCENARIO_WASTE].sort_values("年份")
    half = annual.loc[annual["情形"] == SCENARIO_HALF].sort_values("年份")
    p_waste = waste["利润/元"].to_numpy(dtype=float) / 1e4
    p_half = half["利润/元"].to_numpy(dtype=float) / 1e4
    total_waste, total_half = p_waste.sum(), p_half.sum()
    gain = (total_half / total_waste - 1) * 100

    fig, ax = plt.subplots(figsize=(7.20, 3.65))
    ax.fill_between(YEARS, p_waste, p_half, color=COLORS["pale_blue"], alpha=0.95, zorder=1)
    ax.plot(YEARS, p_waste, color=COLORS["waste"], lw=1.8, marker="o", ms=4.2,
            markerfacecolor="white", markeredgewidth=1.2, zorder=3)
    ax.plot(YEARS, p_half, color=COLORS["half"], lw=2.2, marker="o", ms=4.6,
            markerfacecolor=COLORS["half"], markeredgewidth=0, zorder=4)

    for x, y in zip(YEARS, p_half):
        ax.text(x, y + 10, f"{y:.0f}", ha="center", va="bottom", fontsize=6.8,
                color=COLORS["half"])
    for x, y in zip(YEARS, p_waste):
        ax.text(x, y - 13, f"{y:.0f}", ha="center", va="top", fontsize=6.8,
                color=COLORS["waste"])

    ax.text(2030.10, p_half[-1], "半价销售", color=COLORS["half"], fontweight="bold",
            va="center", fontsize=8)
    ax.text(2030.10, p_waste[-1], "滞销浪费", color=COLORS["waste"], fontweight="bold",
            va="center", fontsize=8)
    ax.set_xlim(2023.7, 2030.72)
    ax.set_ylim(min(p_waste) - 55, max(p_half) + 80)
    ax.set_xticks(YEARS)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_xlabel("年份")
    ax.set_ylabel("年度利润（万元）")
    ax.set_title("半价销售超产部分显著提高规划期利润", loc="left", fontweight="bold", pad=10)
    clean_axis(ax, show_y_grid=True)
    fig.text(0.90, 0.955,
             f"七年累计利润 {total_half:.0f} 万元  |  较滞销浪费提高 {gain:.1f}%",
             ha="right", va="top", fontsize=7.2, color=COLORS["muted"])
    fig.subplots_adjust(left=0.10, right=0.90, top=0.82, bottom=0.18)
    export_figure(fig, "图1_年度利润比较_优化")


def crop_category(crop_type: str) -> str:
    text = str(crop_type)
    if "豆类" in text:
        return "豆类"
    if "食用菌" in text:
        return "食用菌"
    if "蔬菜" in text:
        return "其他蔬菜"
    return "其他粮食"


def figure_2_structure(crop_waste: pd.DataFrame, crop_half: pd.DataFrame) -> None:
    """结构证据：收益处理方式改变豆类、粮食和蔬菜的面积配置。"""
    categories = ["豆类", "其他粮食", "其他蔬菜", "食用菌"]
    palette = [COLORS["bean"], COLORS["grain"], COLORS["vegetable"], COLORS["fungi"]]
    hatches = ["///", "", "...", "\\\\"]
    frames = [(SCENARIO_WASTE, crop_waste), (SCENARIO_HALF, crop_half)]

    fig, axes = plt.subplots(1, 2, figsize=(7.20, 3.75), sharey=True)
    for idx, (ax, (title, frame)) in enumerate(zip(axes, frames)):
        df = frame.copy()
        df["作物大类"] = df["作物类型"].map(crop_category)
        table = (df.groupby(["年份", "作物大类"], as_index=False)["种植面积/亩"].sum()
                 .pivot(index="年份", columns="作物大类", values="种植面积/亩")
                 .reindex(index=YEARS, columns=categories, fill_value=0))
        bottom = np.zeros(len(YEARS))
        for category, color, hatch in zip(categories, palette, hatches):
            values = table[category].to_numpy(dtype=float)
            ax.bar(YEARS, values, bottom=bottom, width=0.70, label=category,
                   color=color, edgecolor="white", linewidth=0.45, hatch=hatch, zorder=2)
            bottom += values
        ax.set_title("滞销浪费" if idx == 0 else "超产部分半价销售", fontsize=8.8,
                     fontweight="bold", pad=7)
        ax.set_xlabel("年份")
        ax.set_xticks(YEARS)
        ax.set_xticklabels(YEARS, rotation=45, ha="right")
        ax.set_ylim(0, 1420)
        clean_axis(ax, show_y_grid=False)
        add_panel_label(ax, chr(ord("a") + idx))
    axes[0].set_ylabel("种植面积（亩·季）")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.91),
               columnspacing=1.3, handlelength=1.8)
    fig.suptitle("销售处置方式重塑作物种植结构", x=0.08, y=0.99,
                 ha="left", fontsize=10, fontweight="bold")
    fig.subplots_adjust(left=0.09, right=0.985, top=0.73, bottom=0.20, wspace=0.14)
    export_figure(fig, "图2_作物类别种植结构_优化")


def figure_3_sales(crop_waste: pd.DataFrame, crop_half: pd.DataFrame) -> None:
    """产销证据：允许半价销售后，超产规模和超产率明显上升。"""
    frames = [(SCENARIO_WASTE, crop_waste), (SCENARIO_HALF, crop_half)]
    fig, axes = plt.subplots(1, 2, figsize=(7.20, 3.75), sharey=True)
    for idx, (ax, (title, frame)) in enumerate(zip(axes, frames)):
        annual = frame.groupby("年份", as_index=False).agg({
            "正常销量/斤": "sum", "超出销量/斤": "sum"
        }).sort_values("年份")
        normal = annual["正常销量/斤"].to_numpy(dtype=float) / 1e4
        excess = annual["超出销量/斤"].to_numpy(dtype=float) / 1e4
        total = normal + excess
        excess_rate = np.divide(excess, total, out=np.zeros_like(excess), where=total > 0) * 100

        ax.bar(YEARS, normal, width=0.70, color=COLORS["normal"], label="正常销量",
               edgecolor="white", linewidth=0.5, zorder=2)
        ax.bar(YEARS, excess, bottom=normal, width=0.70, color=COLORS["excess"],
               label="超出预期销售量", edgecolor="white", linewidth=0.5, hatch="///", zorder=2)
        for x, y, rate in zip(YEARS, total, excess_rate):
            if rate >= 0.5:
                ax.text(x, y + 4.5, f"{rate:.1f}%", ha="center", va="bottom",
                        fontsize=6.5, color=COLORS["muted"])
        ax.set_title("滞销浪费" if idx == 0 else "超产部分半价销售", fontsize=8.8,
                     fontweight="bold", pad=7)
        ax.set_xlabel("年份")
        ax.set_xticks(YEARS)
        ax.set_xticklabels(YEARS, rotation=45, ha="right")
        ax.set_ylim(0, 280)
        clean_axis(ax, show_y_grid=False)
        add_panel_label(ax, chr(ord("a") + idx))
    axes[0].set_ylabel("产量（万斤）")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.91),
               columnspacing=1.5, handlelength=1.8)
    fig.suptitle("半价销售扩大可接受的超产规模", x=0.08, y=0.99,
                 ha="left", fontsize=10, fontweight="bold")
    fig.text(0.985, 0.04, "柱顶标注为超出销量占总产量的比例", ha="right", va="bottom",
             fontsize=6.5, color=COLORS["muted"])
    fig.subplots_adjust(left=0.09, right=0.985, top=0.73, bottom=0.20, wspace=0.14)
    export_figure(fig, "图3_正常销量与超出销量_优化")


def rounded_box(ax, xy, width, height, title, body, facecolor, edgecolor,
                title_color=None, fontsize=7.2, title_size=8.0):
    x, y = xy
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.012,rounding_size=0.018",
                         facecolor=facecolor, edgecolor=edgecolor, linewidth=1.0)
    ax.add_patch(box)
    ax.text(x + width / 2, y + height * 0.68, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color=title_color or COLORS["ink"])
    ax.text(x + width / 2, y + height * 0.34, body, ha="center", va="center",
            fontsize=fontsize, color=COLORS["ink"], linespacing=1.35)
    return box


def arrow(ax, start, end, color=None, connectionstyle="arc3,rad=0"):
    arr = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10,
                          linewidth=1.0, color=color or COLORS["muted"],
                          connectionstyle=connectionstyle, shrinkA=2, shrinkB=2)
    ax.add_patch(arr)


def figure_4_flowchart() -> None:
    """方法图：从附件数据到两类利润函数、MILP 求解和结果审计。"""
    fig, ax = plt.subplots(figsize=(7.20, 5.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # 主链路
    rounded_box(ax, (0.035, 0.78), 0.20, 0.135, "输入数据",
                "地块面积与类型\n2023 种植、亩产、成本、价格",
                COLORS["pale_blue"], COLORS["normal"])
    rounded_box(ax, (0.285, 0.78), 0.20, 0.135, "参数构造",
                "价格区间取中点\n2023 实际产量汇总为销售上限",
                COLORS["pale_gold"], COLORS["grain"])
    rounded_box(ax, (0.535, 0.78), 0.20, 0.135, "决策变量",
                "种植面积  x\n是否种植及水浇地模式  y, z",
                COLORS["pale_violet"], COLORS["fungi"])
    rounded_box(ax, (0.785, 0.78), 0.18, 0.135, "统一约束集",
                "土地适配、面积平衡\n重茬、豆类轮作、管理便利",
                COLORS["pale_green"], COLORS["bean"], fontsize=6.8)

    for x0, x1 in ((0.235, 0.285), (0.485, 0.535), (0.735, 0.785)):
        arrow(ax, (x0, 0.848), (x1, 0.848))

    # 两个利润分支
    rounded_box(ax, (0.12, 0.49), 0.31, 0.15, "情形 1：超产滞销浪费",
                "利润 = 正常销售收入 − 种植成本\n超出预期销售量部分收入为 0",
                "#F2F3F5", COLORS["waste"], title_color=COLORS["waste"])
    rounded_box(ax, (0.57, 0.49), 0.31, 0.15, "情形 2：超产部分半价销售",
                "利润 = 正常收入 + 0.5×超产收入\n− 种植成本",
                COLORS["pale_blue"], COLORS["half"], title_color=COLORS["half"])
    # 统一约束先进入中心分支点，再分别连接两种利润函数，避免长箭头穿过节点。
    arrow(ax, (0.875, 0.78), (0.50, 0.69), connectionstyle="arc3,rad=0.08")
    ax.plot(0.50, 0.69, marker="o", ms=3.5, color=COLORS["muted"], zorder=5)
    arrow(ax, (0.50, 0.69), (0.275, 0.64), connectionstyle="arc3,rad=0")
    arrow(ax, (0.50, 0.69), (0.725, 0.64), connectionstyle="arc3,rad=0")

    rounded_box(ax, (0.31, 0.245), 0.38, 0.135, "MILP 求解",
                "PuLP 建模 + HiGHS 分支定界\n最大化 2024—2030 年总利润",
                "#EDF2F7", COLORS["half"], title_size=8.4)
    arrow(ax, (0.275, 0.49), (0.43, 0.38), connectionstyle="arc3,rad=-0.08")
    arrow(ax, (0.725, 0.49), (0.57, 0.38), connectionstyle="arc3,rad=0.08")

    rounded_box(ax, (0.06, 0.045), 0.25, 0.115, "方案输出",
                "逐年、逐地块、逐季次\n作物种植面积表",
                COLORS["pale_blue"], COLORS["normal"])
    rounded_box(ax, (0.375, 0.045), 0.25, 0.115, "可行性审计",
                "面积平衡、重茬、轮作\n最小面积与分散度检查",
                COLORS["pale_green"], COLORS["bean"])
    rounded_box(ax, (0.69, 0.045), 0.25, 0.115, "比较分析",
                "年度利润、种植结构\n正常销量与超产规模",
                COLORS["pale_gold"], COLORS["grain"])
    for x_end in (0.185, 0.50, 0.815):
        arrow(ax, (0.50, 0.245), (x_end, 0.16), connectionstyle="arc3,rad=0")

    ax.text(0.035, 0.965, "问题一：确定性多期种植策略的 MILP 解法流程",
            fontsize=10.5, fontweight="bold", ha="left", va="top", color=COLORS["ink"])
    ax.text(0.965, 0.965, "统一约束 · 两类利润函数 · 同一求解框架",
            fontsize=7.2, ha="right", va="top", color=COLORS["muted"])
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    export_figure(fig, "图4_MILP解法思路流程图")


def write_figure_notes(annual: pd.DataFrame, crop_waste: pd.DataFrame, crop_half: pd.DataFrame) -> None:
    waste_profit = annual.loc[annual["情形"] == SCENARIO_WASTE, "利润/元"].sum() / 1e4
    half_profit = annual.loc[annual["情形"] == SCENARIO_HALF, "利润/元"].sum() / 1e4
    text = f"""# 问题一优化图形说明

## 图形契约

- 核心结论：允许超产部分按 50% 价格销售，会改变最优种植结构并显著提高规划期利润，同时扩大模型可接受的超产规模。
- 证据链：图 1 展示利润差异；图 2 展示种植结构变化；图 3 展示产销构成变化；图 4 说明两种情形共享约束、仅利润函数不同的求解逻辑。
- 图形类型：图 1—3 为 quantitative grid / comparison；图 4 为 schematic-led method figure。
- 数据完整性：使用分析工作簿中的全部 2×7×41={len(crop_waste) + len(crop_half)} 条作物—年份记录，无删除、抽样或重算。
- 数据来源：`Q1/outputs/问题一_MILP分析结果.xlsx`。

## 关键数值

- 滞销浪费情形七年累计利润：{waste_profit:.2f} 万元。
- 半价销售情形七年累计利润：{half_profit:.2f} 万元。
- 累计利润提高：{(half_profit / waste_profit - 1) * 100:.2f}%。

## 输出规范

- SVG：主格式，文字保持可编辑。
- PDF：矢量备份。
- TIFF：600 dpi、LZW 压缩。
- PNG：300 dpi，用于快速预览。
- 最终宽度：约 183 mm，适合双栏论文页面。

## 图注建议

**图 1｜两种超产处置方式下的年度利润。** 阴影区域表示半价销售方案相对于滞销浪费方案的年度利润增量，标注数值单位为万元。

**图 2｜两种情形下的年度作物类别种植结构。** 面积按豆类、其他粮食、其他蔬菜和食用菌汇总；单位“亩·季”表示各季种植面积之和。

**图 3｜正常销量与超出预期销售量的年度构成。** 柱顶百分比为超出销量占当年总产量的比例；低于 0.5% 时不显示标注。

**图 4｜问题一确定性多期 MILP 解法流程。** 两种情形共享数据、变量和农业约束，仅通过超产部分的销售折扣率定义不同利润函数，最终由同一 MILP 框架求解并进行可行性审计。

## 审稿风险说明

- 本题为确定性优化，不存在样本量、误差条或显著性检验；图中数值是求解结果而非抽样估计。
- 情形一为达到时间上限时取得的可行整数解，最终相对间隙约 1.48%；情形二在 1% 容差内完成，间隙约 0.564%。
- 图形不应被解释为农业试验统计推断，只用于展示当前模型假设下的优化结果。
"""
    (OUTPUT_DIR / "图形说明与QA.md").write_text(text, encoding="utf-8")


def main() -> None:
    apply_publication_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    annual, crop_waste, crop_half = load_data()
    figure_1_profit(annual)
    figure_2_structure(crop_waste, crop_half)
    figure_3_sales(crop_waste, crop_half)
    figure_4_flowchart()
    write_figure_notes(annual, crop_waste, crop_half)
    print(f"已生成 4 张论文级图形：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
