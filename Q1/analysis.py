# -*- coding: utf-8 -*-
"""
Q1 补充分析：相关性(皮尔逊r) + 误差(Bias/MAE/RMSE) + 图（可选）
输入：aligned_hourly.csv（你上一段代码输出的对齐表）
输出：
- q1_corr_metrics.json
- fig_corr_scatter.svg（散点+1:1线+线性回归线）
- fig_residual_hist.svg（残差直方图）
- fig_residual_vs_station.svg（残差 vs 国控浓度）
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.plot_style import (
    set_mcm_pastel_style, get_colors, get_accents, make_cmap,
    line_3d_effect, bar_3d_effect, info_card, save_svg
)

# 选一种：scheme1/2/3/4（我建议 scheme4 颜色最多）
SCHEME = "scheme1"
set_mcm_pastel_style(SCHEME)

colors  = get_colors(SCHEME)
accents = get_accents(SCHEME)
cmap    = make_cmap(SCHEME)

# =========================
# 路径：改成你自己的
# =========================
ALIGNED_CSV = r"..\data\q1_outputs\aligned_hourly.csv"
OUT_DIR     = r"..\data\q1_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# 读取
# =========================
df = pd.read_csv(ALIGNED_CSV, encoding="utf-8-sig")

# 只取成功匹配 & 两列不缺失
df["hour"] = pd.to_datetime(df["hour"], errors="coerce")
df["pm25_station"] = pd.to_numeric(df["pm25_station"], errors="coerce")
df["pm25_company"] = pd.to_numeric(df["pm25_company"], errors="coerce")

# matched 列可能是 True/False，也可能是 0/1；都兼容
if "matched" in df.columns:
    matched_mask = df["matched"].astype(bool)
else:
    # 没有 matched 列就用 n_points>0 作为匹配
    matched_mask = pd.to_numeric(df.get("n_points"), errors="coerce").fillna(0) > 0

use = df[matched_mask].dropna(subset=["pm25_station", "pm25_company"]).copy()
n = int(len(use))
if n == 0:
    raise RuntimeError("没有可用的匹配样本（matched hours = 0），请检查对齐结果。")

# =========================
# 基础误差指标
# =========================
use["residual"] = use["pm25_company"] - use["pm25_station"]
mae  = float(use["residual"].abs().mean())
rmse = float(np.sqrt((use["residual"] ** 2).mean()))
bias = float(use["residual"].mean())

# =========================
# 皮尔逊相关系数 r（不依赖 scipy）
# =========================
x = use["pm25_station"].to_numpy()
y = use["pm25_company"].to_numpy()
r = float(np.corrcoef(x, y)[0, 1])

# 如有 scipy，可选算 p-value（没有也不影响）
p_value = None
try:
    from scipy.stats import pearsonr
    r2, p2 = pearsonr(x, y)
    r = float(r2)
    p_value = float(p2)
except Exception:
    pass

# 线性回归（最小二乘）：y = a*x + b
a, b = np.polyfit(x, y, 1)
a, b = float(a), float(b)

metrics = {
    "n_matched_pairs": n,
    "MAE": mae,
    "RMSE": rmse,
    "Bias(company-station)": bias,
    "pearson_r": r,
    "pearson_p_value_if_available": p_value,
    "linear_fit_y=a*x+b": {"a": a, "b": b}
}

with open(os.path.join(OUT_DIR, "q1_corr_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print("Saved metrics:", os.path.join(OUT_DIR, "q1_corr_metrics.json"))
print("Metrics:", metrics)

# =========================
# 图1：密度散点 + 1:1线 + 回归线（Premium）
# =========================
fig, ax = plt.subplots()

# 用 hexbin 做“密度层次感”
hb = ax.hexbin(x, y, gridsize=40, mincnt=1, cmap=cmap, linewidths=0.0, alpha=0.95)

mn = float(np.nanmin([x.min(), y.min()]))
mx = float(np.nanmax([x.max(), y.max()]))

# 1:1 线
l11, = ax.plot([mn, mx], [mn, mx], linestyle="--", color=accents[-1], label="1:1 line")

# 回归线
xx = np.linspace(mn, mx, 200)
yy = a * xx + b
lfit, = ax.plot(xx, yy, color=accents[0], label=f"fit: y={a:.3f}x+{b:.2f}")
line_3d_effect(lfit)

ax.set_title("PM2.5 Agreement (Density Scatter)")
ax.set_xlabel("Station PM2.5")
ax.set_ylabel("Company PM2.5 (aligned)")
ax.grid(True, alpha=0.12)

# 等比例坐标更“专业”
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(mn, mx)
ax.set_ylim(mn, mx)

# 色条（让外行感觉“高级仪表”）
cb = fig.colorbar(hb, ax=ax, pad=0.01)
cb.set_label("Point density")

# 信息卡片
txt = f"n={n}\nr={r:.3f}"
if p_value is not None:
    txt += f"\np={p_value:.2e}"
info_card(ax, txt)

ax.legend(loc="lower right")
fig.tight_layout()
save_svg(fig, os.path.join(OUT_DIR, "fig_corr_scatter_premium.svg"))
print("Saved fig_corr_scatter_premium.svg")
# =========================
# 图2：残差直方图（Premium）
# =========================
fig, ax = plt.subplots()

res = use["residual"].to_numpy()

counts, bins, patches = ax.hist(res, bins=40, edgecolor="white", linewidth=0.6)

# 给柱子加“浮起”阴影 + 多色渐变
# 用柱子的高度做颜色映射（越高越深）
h = np.array([p.get_height() for p in patches], dtype=float)
h_norm = (h - h.min()) / (h.max() - h.min() + 1e-9)

for p, t in zip(patches, h_norm):
    p.set_facecolor(cmap(0.25 + 0.7 * t))  # 颜色更丰富
    bar_3d_effect(p, shadow_alpha=0.16)

ax.axvline(0, linestyle="--", color=accents[-1], linewidth=1.3, label="0 line")
ax.axvline(res.mean(), color=accents[0], linewidth=2.0, label=f"mean={res.mean():.2f}")

ax.set_title("Residual Histogram (Company - Station)")
ax.set_xlabel("Residual")
ax.set_ylabel("Count")
ax.grid(True, alpha=0.12)

ax.legend(loc="upper right")
fig.tight_layout()
save_svg(fig, os.path.join(OUT_DIR, "fig_residual_hist_premium.svg"))
print("Saved fig_residual_hist_premium.svg")
# =========================
# 图3：残差 vs 国控浓度（Premium）
# =========================
fig, ax = plt.subplots()

xs = use["pm25_station"].to_numpy()
rs = use["residual"].to_numpy()

sc = ax.scatter(xs, rs, c=rs, cmap=cmap, s=18, alpha=0.75, linewidths=0)

ax.axhline(0, linestyle="--", color=accents[-1], linewidth=1.3)

# 可选：加一条残差-浓度回归线，外行更容易理解“趋势”
aa, bb = np.polyfit(xs, rs, 1)
xx = np.linspace(xs.min(), xs.max(), 200)
ln, = ax.plot(xx, aa * xx + bb, color=accents[0], linewidth=2.2, label=f"trend: {aa:.3f}x+{bb:.2f}")
line_3d_effect(ln)

ax.set_title("Residual vs Station PM2.5")
ax.set_xlabel("Station PM2.5")
ax.set_ylabel("Residual (Company - Station)")
ax.grid(True, alpha=0.12)

cb = fig.colorbar(sc, ax=ax, pad=0.01)
cb.set_label("Residual magnitude")

info_card(ax, f"n={len(xs)}\ntrend slope={aa:.3f}")
ax.legend(loc="upper right")

fig.tight_layout()
save_svg(fig, os.path.join(OUT_DIR, "fig_residual_vs_station_premium.svg"))
print("Saved fig_residual_vs_station_premium.svg")
