# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.plot_style import (
    set_mcm_pastel_style, get_colors, get_accents, make_cmap,
    line_3d_effect, bar_3d_effect, gradient_fill_between,
    info_card, save_svg
)

# =========================
# Global style
# =========================
SCHEME = "scheme1"
set_mcm_pastel_style(SCHEME)

colors = get_colors(SCHEME)
accents = get_accents(SCHEME)
cmap = make_cmap(SCHEME)
# --- Safe color pickers (avoid index error for scheme1) ---
def C(i, fallback="C0"):
    return colors[i % len(colors)] if colors else fallback

def A(i, fallback="C1"):
    return accents[i % len(accents)] if accents else fallback
# =========================
# Load data
# =========================
aligned = pd.read_csv("../data/q2_output/aligned_from_raw.csv", low_memory=False)
test = pd.read_csv("../data/q2_output/test_with_pred.csv", low_memory=False)

# keep matched rows only (if exists)
if "matched" in aligned.columns:
    aligned = aligned[aligned["matched"] == True].copy()

# build E if missing
if "E" not in aligned.columns and ("mc_PM2.5_mean" in aligned.columns and "PM2.5" in aligned.columns):
    aligned["E"] = pd.to_numeric(aligned["mc_PM2.5_mean"], errors="coerce") - pd.to_numeric(aligned["PM2.5"], errors="coerce")

aligned = aligned.dropna(subset=["E"]).copy()
test = test.dropna(subset=["E", "E_pred"]).copy()

# =========================
# Metrics for info_card
# =========================
rmse = float(np.sqrt(np.mean((test["E"].values - test["E_pred"].values) ** 2)))
mae = float(np.mean(np.abs(test["E"].values - test["E_pred"].values)))

card = (
    f"Holdout RMSE: {rmse:.2f}\n"
    f"Holdout MAE:  {mae:.2f}\n"
    f"N(aligned): {len(aligned)}\n"
    f"N(test):    {len(test)}"
)

# Utility: subsample scatter for speed/clarity
def subsample(df, n=6000, seed=0):
    if len(df) > n:
        return df.sample(n, random_state=seed)
    return df

# =========================
# Figure 1: E vs mc_PM10_mean (binned mean)
# =========================
if "mc_PM10_mean" in aligned.columns:
    df1 = aligned[["E", "mc_PM10_mean"]].dropna().copy()

    # quantile bins -> mean trend line
    df1["bin"] = pd.qcut(df1["mc_PM10_mean"], q=20, duplicates="drop")
    g1 = df1.groupby("bin").agg(
        pm10=("mc_PM10_mean", "mean"),
        E_mean=("E", "mean")
    ).reset_index(drop=True).sort_values("pm10")

    scat = subsample(df1, n=6000, seed=0)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.scatter(
        scat["mc_PM10_mean"], scat["E"],
        s=14, alpha=0.22, edgecolors="none",
        color=C(1)
    )

    ln, = ax.plot(
        g1["pm10"], g1["E_mean"],
        marker="o", markersize=4.5,
        color=A(2)
    )
    line_3d_effect(ln)

    ax.set_xlabel("Micro-station PM10 mean (±5 min window)")
    ax.set_ylabel("E = Micro PM2.5 − National PM2.5")
    ax.set_title("Bias vs PM10 (binned mean)")

    info_card(ax, card, loc=(0.68, 0.98))
    save_svg(fig, "../data/q2_output/fig_E_vs_mcPM10_binned.svg")

# =========================
# Figure 2: Humidity effect with PM2.5 interaction
# =========================
if ("mc_Humidity_mean" in aligned.columns) and ("PM2.5" in aligned.columns):
    df2 = aligned[["mc_Humidity_mean", "PM2.5", "E"]].dropna().copy()
    pm_thr = df2["PM2.5"].median()

    df2["grp"] = np.where(df2["PM2.5"] >= pm_thr, "High PM2.5", "Low PM2.5")
    df2["hbin"] = pd.qcut(df2["mc_Humidity_mean"], q=15, duplicates="drop")

    g2 = df2.groupby(["grp", "hbin"]).agg(
        h=("mc_Humidity_mean", "mean"),
        E_mean=("E", "mean")
    ).reset_index(drop=False)

    scat2 = subsample(df2, n=6000, seed=1)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.scatter(
        scat2["mc_Humidity_mean"], scat2["E"],
        s=14, alpha=0.12, edgecolors="none",
        color=C(2)
    )

    # two lines with accent colors (safe)
    color_map = {"High PM2.5": A(1), "Low PM2.5": A(2)}
    for name, sub in g2.groupby("grp"):
        sub = sub.sort_values("h")
        ln, = ax.plot(
            sub["h"], sub["E_mean"],
            marker="o", markersize=4.5,
            label=name,
            color=color_map.get(name, A(0))
        )
        line_3d_effect(ln)

    ax.set_xlabel("Micro-station Humidity mean (±5 min window)")
    ax.set_ylabel("E = Micro PM2.5 − National PM2.5")
    ax.set_title("Humidity effect with PM2.5 interaction (binned means)")
    ax.legend()

    info_card(ax, card, loc=(0.68, 0.98))
    save_svg(fig, "../data/q2_output/fig_E_vs_humidity_interaction.svg")

# =========================
# Table: High/Low humidity × High/Low PM2.5 summary
# =========================
if ("mc_Humidity_mean" in aligned.columns) and ("PM2.5" in aligned.columns):
    df = aligned[["E", "mc_Humidity_mean", "PM2.5"]].dropna().copy()
    hum_thr = df["mc_Humidity_mean"].median()
    pm_thr2 = df["PM2.5"].median()

    df["Hum"] = np.where(df["mc_Humidity_mean"] >= hum_thr, "High Hum", "Low Hum")
    df["PM"] = np.where(df["PM2.5"] >= pm_thr2, "High PM2.5", "Low PM2.5")

    tab = df.groupby(["Hum", "PM"])["E"].agg(["count", "mean", "median", "std"]).reset_index()
    tab.to_csv("../data/q2_output/humidity_pm25_interaction_summary.csv", index=False, encoding="utf-8-sig")

print("Done: fig_E_vs_mcPM10_binned.svg, fig_E_vs_humidity_interaction.svg, humidity_pm25_interaction_summary.csv")
