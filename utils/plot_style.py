# utils/plot_style.py
# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from cycler import cycler

# =========================
# 4 组“官方推荐浅色系”配色（来自你截图）
# =========================
SCHEMES = {
    # 方案1（粉紫绿蓝系）：粉嫩、柔和（折线/柱状/流程）
    "scheme1": ["#f6cfaf", "#b8b9d2", "#b5d4be", "#afd4e3"],

    # 方案2（蓝绿粉系）：清新、明亮（流程/组合图表）
    "scheme2": ["#479ca6", "#d2edea", "#ffe8da", "#fbb1c8", "#fbefed"],

    # 方案3（蓝绿紫粉系）：梦幻、柔美（饼图/面积/散点）
    "scheme3": ["#99b9e9", "#b0d992", "#bdb5e1", "#ffcccc", "#fcf1f0"],

    # 方案4（绿橙蓝系）：更丰富、适合多序列（组合图/多类别）
    "scheme4": ["#eaf1e1", "#c1de9c", "#fdebdd", "#fdb093",
                "#ddeff4", "#66bdce", "#d6e0ed", "#79a7c7"],
}

# 给“关键曲线”准备一组更深一点的强调色（不刺眼，但更高级）
ACCENTS = {
    "scheme1": ["#e6a77d", "#8a8bb4", "#7fb08f", "#6ea8c4"],
    "scheme2": ["#2f7d86", "#7bc8c0", "#f0b59b", "#e77aa0", "#d5c2c0"],
    "scheme3": ["#5a8dd6", "#72b04f", "#8c79c9", "#ff8fa3", "#d9c7c7"],
    "scheme4": ["#7fb08f", "#8bbf50", "#f3b58a", "#f07b54",
                "#7ab6c7", "#2a8ea3", "#9db3c5", "#2f6f9a"],
}

# =========================
# 基础工具
# =========================
def get_colors(scheme: str = "scheme4"):
    """返回浅色调色板"""
    if scheme not in SCHEMES:
        raise ValueError(f"Unknown scheme={scheme}. Use one of {list(SCHEMES.keys())}")
    return SCHEMES[scheme]

def get_accents(scheme: str = "scheme4"):
    """返回强调色（用于主线/关键点）"""
    if scheme not in ACCENTS:
        raise ValueError(f"Unknown scheme={scheme}. Use one of {list(ACCENTS.keys())}")
    return ACCENTS[scheme]

def make_cmap(scheme: str = "scheme4", name: str = None):
    """用当前 scheme 生成连续渐变 colormap（做密度图/渐变填充更有层次）"""
    cols = get_colors(scheme)
    cmap_name = name or f"{scheme}_cmap"
    return LinearSegmentedColormap.from_list(cmap_name, cols)

def save_svg(fig, path: str):
    """统一 SVG 输出"""
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

# =========================
# “美赛高级感”样式：颜色更多 + 2.5D 层次（阴影/描边/透明渐变）
# =========================
def set_mcm_pastel_style(
    scheme: str = "scheme4",
    *,
    figsize=(7.8, 4.5),
    grid=True
):
    """
    在脚本开头调用一次即可。
    - scheme: scheme1~scheme4
    """
    colors = get_colors(scheme)
    mpl.rcParams.update({
        "figure.figsize": figsize,
        "figure.dpi": 140,

        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,

        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,

        # 清爽轴风格
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,

        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,

        # 网格更“高级”：浅、细、虚线
        "axes.grid": grid,
        "grid.alpha": 0.18,
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,

        # 线条更“立体”：稍粗
        "lines.linewidth": 2.0,
        "legend.frameon": False,
    })

    # 全局颜色循环：让多条曲线/多柱子自动换你指定的浅色系
    mpl.rcParams["axes.prop_cycle"] = cycler(color=colors)

def line_3d_effect(line, shadow_alpha: float = 0.22):
    """
    给折线加“2.5D”立体效果：阴影 + 白色描边 + 正常线
    用法：ln,=ax.plot(...); line_3d_effect(ln)
    """
    line.set_path_effects([
        pe.SimpleLineShadow(offset=(1.2, -1.2), alpha=shadow_alpha),
        pe.Stroke(linewidth=line.get_linewidth() + 2.2, foreground="white", alpha=0.75),
        pe.Normal(),
    ])

def bar_3d_effect(patch, shadow_alpha: float = 0.18):
    """
    给柱子/面积块加“浮起”感：阴影
    用法：bars=ax.bar(...); for b in bars: bar_3d_effect(b)
    """
    patch.set_path_effects([
        pe.SimplePatchShadow(offset=(1.2, -1.2), alpha=shadow_alpha),
        pe.Normal(),
    ])

def gradient_fill_between(ax, x, y1, y2, base_color: str, layers: int = 10, alpha_max: float = 0.22):
    """
    用多层透明 fill_between 模拟“渐变/层次感”（SVG 友好，不依赖复杂渐变）
    """
    # 从深到浅叠加
    for k in range(layers):
        a = alpha_max * (1 - k / layers) ** 1.6
        ax.fill_between(x, y1, y2, color=base_color, alpha=a, linewidth=0)

def info_card(ax, text: str, loc=(0.02, 0.98)):
    """图内信息卡片（美赛常用、外行觉得像报告仪表盘）"""
    ax.text(
        loc[0], loc[1], text,
        transform=ax.transAxes,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.40", fc="white", ec="0.75", alpha=0.95)
    )
