# -*- coding: utf-8 -*-
"""
Q1 对齐（±5min 多对一聚合）+ MAE/RMSE + 匹配覆盖率 + 出图

输入：你预处理后的
- A1_clean.csv（国控站小时）
- A2_clean.csv（企业高频，含 hour_nearest, delta_nearest_min 等字段）

输出：
- aligned_hourly.csv（对齐后的小时表，含匹配质量）
- q1_metrics.json（Coverage / MAE / RMSE / Bias / mean(n_points)）
- fig_timeseries.svg / fig_scatter.svg / fig_residual.svg
"""
import json
import pandas as pd
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from utils.plot_style import (
    set_mcm_pastel_style, get_colors, get_accents, make_cmap,
    line_3d_effect, bar_3d_effect, gradient_fill_between,
    info_card, save_svg
)

# 选一个你喜欢的配色方案：scheme1~scheme4
set_mcm_pastel_style("scheme1")

colors = get_colors("scheme1")
accents = get_accents("scheme1")
cmap = make_cmap("scheme1")

# =========================
# 路径：改成你自己的
# =========================
A1_CLEAN = r"..\data\outputs_c_preprocess\A1_clean.csv"
A2_CLEAN = r"..\data\outputs_c_preprocess\A2_clean.csv"
OUT_DIR  = r"..\data\q1_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

TIME_COL = "Time"
HOUR_COL_A1 = "hour"          # A1_clean 里应当已经有
HOUR_COL_A2 = "hour_nearest"  # A2_clean 里应当已经有
DELTA_COL   = "delta_nearest_min"
PM25_COL    = "PM2.5"

TOL_MIN = 5  # ±5分钟

# =========================
# 读入
# =========================
a1 = pd.read_csv(A1_CLEAN, encoding="utf-8-sig")
a2 = pd.read_csv(A2_CLEAN, encoding="utf-8-sig")

# 时间解析
a1[TIME_COL] = pd.to_datetime(a1[TIME_COL], errors="coerce")
a2[TIME_COL] = pd.to_datetime(a2[TIME_COL], errors="coerce")

# hour 列解析（兼容你之前预处理）
if HOUR_COL_A1 in a1.columns:
    a1[HOUR_COL_A1] = pd.to_datetime(a1[HOUR_COL_A1], errors="coerce")
else:
    a1[HOUR_COL_A1] = a1[TIME_COL].dt.floor("h")

if HOUR_COL_A2 in a2.columns:
    a2[HOUR_COL_A2] = pd.to_datetime(a2[HOUR_COL_A2], errors="coerce")
else:
    # 如果没有 hour_nearest，则自己算（不推荐；你预处理中已有）
    hour_floor = a2[TIME_COL].dt.floor("h")
    hour_ceil  = a2[TIME_COL].dt.ceil("h")
    delta_floor = (a2[TIME_COL] - hour_floor).dt.total_seconds().abs() / 60.0
    delta_ceil  = (hour_ceil - a2[TIME_COL]).dt.total_seconds().abs() / 60.0
    a2[HOUR_COL_A2] = np.where(delta_floor <= delta_ceil, hour_floor, hour_ceil)
    a2[DELTA_COL]   = np.minimum(delta_floor, delta_ceil)

# 数值化
a2[DELTA_COL] = pd.to_numeric(a2[DELTA_COL], errors="coerce")
a2[PM25_COL]  = pd.to_numeric(a2[PM25_COL], errors="coerce")
a1[PM25_COL]  = pd.to_numeric(a1[PM25_COL], errors="coerce")

# =========================
# 1) ±5min 筛选（窗口候选集）
# =========================
a2_win = a2[(a2[DELTA_COL] <= TOL_MIN)].copy()

# =========================
# 2) 多对一聚合：按 hour_nearest 分组
#    输出 n_points/min_delta/mean/median/std/IQR
# =========================
def iqr(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) == 0:
        return np.nan
    return float(x.quantile(0.75) - x.quantile(0.25))

grouped = a2_win.groupby(HOUR_COL_A2)

agg = grouped.agg(
    n_points=(PM25_COL, "count"),
    min_delta=(DELTA_COL, "min"),
    mean_pm25=(PM25_COL, "mean"),
    median_pm25=(PM25_COL, "median"),
    std_pm25=(PM25_COL, "std"),
).reset_index()

# IQR 需要自定义
iqr_vals = grouped[PM25_COL].apply(iqr).reset_index(name="iqr_pm25")
agg = agg.merge(iqr_vals, on=HOUR_COL_A2, how="left")

# =========================
# 3) 自适应选择：n>=3 用 median，否则用 mean
# =========================
agg["pm25_company"] = np.where(
    agg["n_points"] >= 3,
    agg["median_pm25"],
    agg["mean_pm25"]
)

# 整理列名
agg = agg.rename(columns={HOUR_COL_A2: "hour"}).copy()

# =========================
# 4) 与国控站小时对齐（merge）
# =========================
station = a1[[HOUR_COL_A1, PM25_COL]].rename(columns={HOUR_COL_A1: "hour", PM25_COL: "pm25_station"}).copy()

aligned = station.merge(
    agg[["hour", "pm25_company", "n_points", "min_delta", "std_pm25", "iqr_pm25"]],
    on="hour",
    how="left"
)

# 缺失规则：n_points为空（即 |S|=0）就是缺失，不参与 MAE/RMSE
aligned["matched"] = aligned["n_points"].fillna(0).astype(int) > 0

# =========================
# 5) 计算误差指标（仅对 matched hours）
# =========================
eval_df = aligned[aligned["matched"]].dropna(subset=["pm25_station", "pm25_company"]).copy()
eval_df["residual"] = eval_df["pm25_company"] - eval_df["pm25_station"]
eval_df["abs_err"] = eval_df["residual"].abs()
eval_df["sq_err"] = eval_df["residual"] ** 2

mae = float(eval_df["abs_err"].mean()) if len(eval_df) else np.nan
rmse = float(np.sqrt(eval_df["sq_err"].mean())) if len(eval_df) else np.nan
bias = float(eval_df["residual"].mean()) if len(eval_df) else np.nan

total_hours = int(len(aligned))
matched_hours = int(aligned["matched"].sum())
coverage = float(matched_hours / total_hours) if total_hours else np.nan
mean_n_points = float(eval_df["n_points"].mean()) if len(eval_df) else np.nan

metrics = {
    "total_hours": total_hours,
    "matched_hours": matched_hours,
    "coverage": coverage,
    "mean_n_points_over_matched": mean_n_points,
    "MAE": mae,
    "RMSE": rmse,
    "Bias(company-station)": bias,
    "tolerance_minutes": TOL_MIN,
    "rule": "if n_points>=3 use median else mean; missing if n_points==0"
}

# =========================
# 6) 输出文件
# =========================
aligned_out = os.path.join(OUT_DIR, "aligned_hourly.csv")
metrics_out = os.path.join(OUT_DIR, "q1_metrics.json")

aligned.to_csv(aligned_out, index=False, encoding="utf-8-sig")
with open(metrics_out, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print("Saved:", aligned_out)
print("Saved:", metrics_out)
print("Metrics:", metrics)

# =========================
# 7) 画图（3张）
# =========================
# =========================
# 7) 画图（Premium 3张，统一 SVG）
# =========================

def plot_timeseries_premium(eval_df, coverage, mean_n_points, mae, rmse, bias, out_dir,
                            colors, accents):
    fig, ax = plt.subplots()

    t = eval_df["hour"]
    y1 = eval_df["pm25_station"].to_numpy()
    y2 = eval_df["pm25_company"].to_numpy()

    # 2.5D 层次：用你 scheme 的浅色做渐变叠加
    gradient_fill_between(ax, t, y1, y2, base_color=colors[-1], layers=10, alpha_max=0.22)

    l1, = ax.plot(t, y1, label="National station", color=accents[0])
    l2, = ax.plot(t, y2, label="Company (aligned)", color=accents[min(3, len(accents)-1)])

    line_3d_effect(l1)
    line_3d_effect(l2)

    ax.set_title("PM2.5 Time Series (Aligned Hours)")
    ax.set_xlabel("Time")
    ax.set_ylabel("PM2.5")
    ax.grid(True)

    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    info = (
        f"Coverage: {coverage*100:.2f}%\n"
        f"Mean n_points: {mean_n_points:.2f}\n"
        f"MAE: {mae:.2f}   RMSE: {rmse:.2f}\n"
        f"Bias: {bias:.2f}"
    )
    info_card(ax, info)

    ax.legend(loc="upper right")
    fig.tight_layout()
    save_svg(fig, os.path.join(out_dir, "fig_timeseries_premium.svg"))


def plot_scatter_premium(eval_df, out_dir, cmap, accents):
    x = eval_df["pm25_station"].to_numpy()
    y = eval_df["pm25_company"].to_numpy()

    r = float(np.corrcoef(x, y)[0, 1])
    a, b = np.polyfit(x, y, 1)

    fig, ax = plt.subplots()

    # 用你 scheme 生成的 cmap，而不是 viridis
    hb = ax.hexbin(x, y, gridsize=38, mincnt=1, cmap=cmap, linewidths=0.0, alpha=0.95)

    # 少量高光点增强“高级感”
    idx = np.random.choice(len(x), size=min(350, len(x)), replace=False)
    ax.scatter(x[idx], y[idx], s=14, color="white", alpha=0.22, linewidths=0)

    mn = float(np.nanmin([x.min(), y.min()]))
    mx = float(np.nanmax([x.max(), y.max()]))

    ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1.3, color=accents[-1], label="1:1 line")
    xx = np.linspace(mn, mx, 200)
    ax.plot(xx, a * xx + b, linewidth=2.2, color=accents[0], label=f"Fit: y={a:.2f}x+{b:.2f}")

    ax.set_title("PM2.5 Agreement (Density Scatter)")
    ax.set_xlabel("National station PM2.5")
    ax.set_ylabel("Company PM2.5 (aligned)")
    ax.grid(True, alpha=0.12)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)

    cb = fig.colorbar(hb, ax=ax, pad=0.01)
    cb.set_label("Point density")

    info_card(ax, f"n={len(x)}\nr={r:.3f}")
    ax.legend(loc="lower right")

    fig.tight_layout()
    save_svg(fig, os.path.join(out_dir, "fig_scatter_premium.svg"))


def plot_residual_premium(eval_df, mae, rmse, bias, out_dir, colors, accents):
    t = eval_df["hour"]
    res = eval_df["residual"].to_numpy()

    fig, ax = plt.subplots()

    # 分位带：用 scheme 的浅色做背景层次
    q05, q95 = np.nanquantile(res, [0.05, 0.95])
    ax.axhspan(q05, q95, color=colors[1], alpha=0.20, linewidth=0)

    ln, = ax.plot(t, res, color=accents[0], alpha=0.95)
    line_3d_effect(ln, shadow_alpha=0.18)

    # 散点用 cmap 上色（更“立体”）
    sc = ax.scatter(t, res, c=res, cmap=cmap, s=14, alpha=0.55, linewidths=0)

    ax.axhline(0, linestyle="--", color=accents[-1], linewidth=1.2)

    ax.set_title("Residual over Time (Company - National)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Residual")
    ax.grid(True)

    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("Residual magnitude")

    info_card(ax, f"Bias={bias:.2f}\nMAE={mae:.2f}\nRMSE={rmse:.2f}")

    fig.tight_layout()
    save_svg(fig, os.path.join(out_dir, "fig_residual_premium.svg"))


# ---- 只输出 premium 三张 ----
plot_timeseries_premium(eval_df, coverage, mean_n_points, mae, rmse, bias, OUT_DIR, colors, accents)
plot_scatter_premium(eval_df, OUT_DIR, cmap, accents)
plot_residual_premium(eval_df, mae, rmse, bias, OUT_DIR, colors, accents)

print("Premium figures saved to:", OUT_DIR)

