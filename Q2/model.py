# -*- coding: utf-8 -*-
"""
Q2 (解释版，无target leakage):
用 C_Appendix_1.xlsx (国站, 小时) + C_Appendix_2.xlsx (微站, 分钟/不规则+气象)
±5分钟窗口对齐 -> E = 微站PM2.5 - 国站PM2.5
用 气象/交叉气体/漂移/对齐质量 解释 E 的成因（不使用 mc_PM2.5_* 作为输入）
"""

import os
import math
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
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


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def add_time_features(df: pd.DataFrame, time_col: str = "Time") -> pd.DataFrame:
    dt = pd.to_datetime(df[time_col])
    df = df.copy()
    df["dt"] = dt
    df["t_days"] = (dt - dt.min()).dt.total_seconds() / 86400.0
    df["hour_of_day"] = dt.dt.hour
    df["dow"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["sin_hour"] = np.sin(2 * np.pi * df["hour_of_day"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour_of_day"] / 24.0)
    df["sin_month"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)
    return df

def evaluate_cv(model, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scoring = {
        "neg_rmse": "neg_root_mean_squared_error",
        "neg_mae": "neg_mean_absolute_error",
        "r2": "r2"
    }
    cv = cross_validate(model, X, y, cv=tscv, scoring=scoring, n_jobs=-1, return_train_score=False)
    return {
        "CV_RMSE": float(-cv["test_neg_rmse"].mean()),
        "CV_MAE": float(-cv["test_neg_mae"].mean()),
        "CV_R2": float(cv["test_r2"].mean())
    }

def time_train_test_split(df, time_col="dt", test_ratio=0.2):
    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)
    n_test = max(1, int(math.ceil(n * test_ratio)))
    train = df.iloc[:-n_test].copy()
    test = df.iloc[-n_test:].copy()
    return train, test

def iqr(x):
    x = np.asarray(x)
    if len(x) == 0:
        return np.nan
    return float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25))

def align_with_window(st_df, mc_df, tol_minutes=5):
    st_df = st_df.sort_values("dt").reset_index(drop=True)
    mc_df = mc_df.sort_values("dt").reset_index(drop=True)
    mc_times = mc_df["dt"].values.astype("datetime64[ns]")
    out_rows = []
    tol = np.timedelta64(tol_minutes, "m")
    mc_cols = [c for c in mc_df.columns if c not in ["Time", "dt"]]

    for _, row in st_df.iterrows():
        t = row["dt"].to_datetime64()
        left = np.searchsorted(mc_times, t - tol, side="left")
        right = np.searchsorted(mc_times, t + tol, side="right")
        win = mc_df.iloc[left:right]

        if len(win) == 0:
            out = row.to_dict()
            out["matched"] = False
            out["n_points"] = 0
            out["min_delta_sec"] = np.nan
            out_rows.append(out)
            continue

        deltas = np.abs((win["dt"].values.astype("datetime64[ns]") - t).astype("timedelta64[s]").astype(int))
        out = row.to_dict()
        out["matched"] = True
        out["n_points"] = int(len(win))
        out["min_delta_sec"] = float(np.min(deltas))

        for c in mc_cols:
            vals = pd.to_numeric(win[c], errors="coerce").values
            out[f"mc_{c}_mean"] = float(np.nanmean(vals)) if np.isfinite(np.nanmean(vals)) else np.nan
            out[f"mc_{c}_std"]  = float(np.nanstd(vals)) if np.isfinite(np.nanstd(vals)) else np.nan
            out[f"mc_{c}_iqr"]  = iqr(vals)

        out_rows.append(out)

    return pd.DataFrame(out_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--appendix1", type=str, default="../data/C_Appendix_1.xlsx")
    parser.add_argument("--appendix2", type=str, default="../data/C_Appendix_2.xlsx")
    parser.add_argument("--outdir", type=str, default="../data/q2_output")
    parser.add_argument("--tol_minutes", type=int, default=5)
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()

    ensure_dir(args.outdir)

    st = pd.read_excel(args.appendix1, sheet_name=0)
    mc = pd.read_excel(args.appendix2, sheet_name=0)

    st["Time"] = pd.to_datetime(st["Time"])
    mc["Time"] = pd.to_datetime(mc["Time"])

    st = add_time_features(st, time_col="Time")
    mc = add_time_features(mc, time_col="Time")

    aligned = align_with_window(st_df=st, mc_df=mc, tol_minutes=args.tol_minutes)
    aligned.to_csv(os.path.join(args.outdir, "aligned_from_raw.csv"), index=False, encoding="utf-8-sig")

    aligned = aligned[aligned["matched"] == True].copy()

    # 目标：E = 微站PM2.5 - 国站PM2.5（定义不变）
    aligned["E"] = aligned["mc_PM2.5_mean"] - aligned["PM2.5"]

    aligned["log_st_pm25"] = np.log1p(aligned["PM2.5"].clip(lower=0))
    if "mc_Humidity_mean" in aligned.columns:
        aligned["pm25_hum"] = aligned["PM2.5"] * aligned["mc_Humidity_mean"]

    base_time = ["t_days", "sin_hour", "cos_hour", "sin_month", "cos_month", "is_weekend"]
    align_quality = ["n_points", "min_delta_sec"]

    # 国站污染物（保留 PM2.5 是合理的：量程效应/浓度水平会影响偏差）
    st_poll = [c for c in ["PM2.5", "PM10", "CO", "NO2", "SO2", "O3", "log_st_pm25"] if c in aligned.columns]

    # 微站污染物（关键：剔除 PM2.5，避免 target leakage）
    mc_poll = []
    for c in ["PM10", "CO", "NO2", "SO2", "O3"]:
        for suf in ["mean", "std", "iqr"]:
            col = f"mc_{c}_{suf}"
            if col in aligned.columns:
                mc_poll.append(col)

    # 微站气象
    mc_met = []
    for c in ["WindSpeed", "Pressure", "Precipitation", "Temperature", "Humidity"]:
        for suf in ["mean", "std", "iqr"]:
            col = f"mc_{c}_{suf}"
            if col in aligned.columns:
                mc_met.append(col)

    extra = [c for c in ["pm25_hum"] if c in aligned.columns]

    numeric_cols = []
    for part in [st_poll, mc_poll, mc_met, base_time, align_quality, extra]:
        numeric_cols += part
    numeric_cols = list(dict.fromkeys([c for c in numeric_cols if c in aligned.columns]))

    aligned = aligned.dropna(subset=["E"]).copy()

    train_df, test_df = time_train_test_split(aligned, time_col="dt", test_ratio=0.2)
    X_train = train_df[numeric_cols]
    y_train = train_df["E"].values
    X_test = test_df[numeric_cols]
    y_test = test_df["E"].values

    # Ridge + Spline（解释性）
    spline_features = [c for c in ["PM2.5", "mc_Humidity_mean", "t_days"] if c in numeric_cols]
    other_numeric = [c for c in numeric_cols if c not in spline_features]

    pre_linear = ColumnTransformer(
        transformers=[
            ("spline", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("spline", SplineTransformer(n_knots=6, degree=3, include_bias=False)),
                ("sc", StandardScaler())
            ]), spline_features),
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler())
            ]), other_numeric),
        ],
        remainder="drop"
    )

    model_linear = Pipeline(steps=[
        ("pre", pre_linear),
        ("reg", Ridge(alpha=2.0, random_state=0))
    ])

    # 非线性：HGBR
    model_tree = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="median")),
        ("reg", HistGradientBoostingRegressor(
            max_depth=4, learning_rate=0.06, max_iter=800,
            random_state=0
        ))
    ])

    m1 = evaluate_cv(model_linear, X_train, y_train, n_splits=args.n_splits)
    m2 = evaluate_cv(model_tree, X_train, y_train, n_splits=args.n_splits)

    best_name, best_model = ("Ridge+Spline", model_linear) if m1["CV_RMSE"] <= m2["CV_RMSE"] else ("HGBR", model_tree)
    best_cv = m1 if best_name == "Ridge+Spline" else m2

    best_model.fit(X_train, y_train)
    pred_test = best_model.predict(X_test)

    metrics = {
        "model_linear_CV_RMSE": m1["CV_RMSE"],
        "model_linear_CV_MAE": m1["CV_MAE"],
        "model_linear_CV_R2": m1["CV_R2"],
        "model_tree_CV_RMSE": m2["CV_RMSE"],
        "model_tree_CV_MAE": m2["CV_MAE"],
        "model_tree_CV_R2": m2["CV_R2"],
        "best_model": best_name,
        "best_CV_RMSE": best_cv["CV_RMSE"],
        "best_CV_MAE": best_cv["CV_MAE"],
        "test_RMSE": rmse(y_test, pred_test),
        "test_MAE": float(mean_absolute_error(y_test, pred_test)),
    }
    pd.DataFrame([metrics]).to_csv(os.path.join(args.outdir, "metrics.csv"), index=False, encoding="utf-8-sig")

    test_out = test_df[["dt", "E"] + numeric_cols].copy()
    test_out["E_pred"] = pred_test
    test_out["residual"] = test_out["E"] - test_out["E_pred"]
    test_out.to_csv(os.path.join(args.outdir, "test_with_pred.csv"), index=False, encoding="utf-8-sig")

    perm = permutation_importance(
        best_model, X_test, y_test,
        n_repeats=20, random_state=0, n_jobs=-1,
        scoring="neg_root_mean_squared_error"
    )
    imp = pd.DataFrame({
        "feature": numeric_cols,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std
    }).sort_values("importance_mean", ascending=False)

    imp.to_csv(os.path.join(args.outdir, "permutation_importance.csv"), index=False, encoding="utf-8-sig")

    # =========================
    # 证据图（utils.plot_style 统一模板）
    # =========================

    # 关键：避免 scheme1 下 colors/accents 索引越界（scheme1 只有 4 个颜色/强调色）
    def C(i, fallback="C0"):
        return colors[i % len(colors)] if colors else fallback

    def A(i, fallback="C1"):
        return accents[i % len(accents)] if accents else fallback

    card_base = (
        f"Best: {best_name}\n"
        f"CV RMSE: {best_cv['CV_RMSE']:.2f}\n"
        f"Test RMSE: {rmse(y_test, pred_test):.2f}\n"
        f"Test MAE: {float(mean_absolute_error(y_test, pred_test)):.2f}\n"
        f"N(train): {len(train_df)}\nN(test):  {len(test_df)}\n"
        f"tol=±{args.tol_minutes} min"
    )

    # (1) E over time
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ln1, = ax.plot(train_df["dt"], train_df["E"], linewidth=1.0, label="E (train)", color=A(0))
    ln2, = ax.plot(test_out["dt"], test_out["E"], linewidth=1.2, label="E (test)", color=A(1))
    line_3d_effect(ln1)
    line_3d_effect(ln2)

    ax.axvline(test_out["dt"].min(), linestyle="--", linewidth=1.2, label="test start", color=C(2))
    ax.set_xlabel("Time")
    ax.set_ylabel("E = micro − station")
    ax.set_title("Bias over time (drift evidence)")
    ax.legend()

    info_card(ax, card_base)
    save_svg(fig, os.path.join(args.outdir, "fig_E_over_time.svg"))

    # (2) E vs Station PM2.5（散点）
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.scatter(aligned["PM2.5"], aligned["E"], s=14, alpha=0.22, edgecolors="none", color=C(1))
    ax.set_xlabel("Station PM2.5")
    ax.set_ylabel("E")
    ax.set_title("E vs Station PM2.5 (range effect)")
    info_card(ax, card_base)
    save_svg(fig, os.path.join(args.outdir, "fig_E_vs_station_pm25.svg"))

    # (3) E vs Humidity（散点）
    if "mc_Humidity_mean" in aligned.columns:
        fig, ax = plt.subplots(figsize=(7.8, 5.0))
        ax.scatter(aligned["mc_Humidity_mean"], aligned["E"], s=14, alpha=0.18, edgecolors="none", color=C(2))
        ax.set_xlabel("Micro Humidity (mean)")
        ax.set_ylabel("E")
        ax.set_title("E vs Humidity (meteorology effect)")
        info_card(ax, card_base)
        save_svg(fig, os.path.join(args.outdir, "fig_E_vs_humidity.svg"))

    # (4) 日变化：Diurnal（折线+渐变填充）
    hour_mean = aligned.groupby("hour_of_day")["E"].mean()
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    ln, = ax.plot(hour_mean.index, hour_mean.values, marker="o", color=A(2))
    line_3d_effect(ln)
    gradient_fill_between(ax, hour_mean.index, 0, hour_mean.values, base_color=A(2), layers=10, alpha_max=0.18)

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean E")
    ax.set_title("Diurnal pattern of bias")
    info_card(ax, card_base, loc=(0.68, 0.98))

    save_svg(fig, os.path.join(args.outdir, "fig_diurnal_mean_E.svg"))

    # (5) Permutation importance Top12（barh + 3D effect）
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    topk = imp.head(12).iloc[::-1]
    bars = ax.barh(topk["feature"], topk["importance_mean"], color=C(3))
    for b in bars:
        bar_3d_effect(b)

    ax.set_xlabel("Permutation importance (decrease in score)")
    ax.set_title(f"Top feature importance (best: {best_name})")
    info_card(ax, card_base)
    save_svg(fig, os.path.join(args.outdir, "fig_feature_importance_top12.svg"))

    # (6) PDP
    pdp_priority = ["mc_Humidity_mean", "mc_Temperature_mean", "mc_WindSpeed_mean",
                    "mc_NO2_mean", "mc_O3_mean", "t_days", "PM2.5"]
    pdp_features = [c for c in pdp_priority if c in numeric_cols][:3]
    if len(pdp_features) > 0:
        fig, ax = plt.subplots(figsize=(9.2, 4.2))
        PartialDependenceDisplay.from_estimator(best_model, X_test, pdp_features, ax=ax)
        ax.set_title("Partial dependence (direction of key factors)")
        info_card(ax, card_base)
        save_svg(fig, os.path.join(args.outdir, "fig_partial_dependence.svg"))

    # 贡献度：Permutation importance（先算，再画）
    perm = permutation_importance(
        best_model, X_test, y_test,
        n_repeats=50, random_state=0, n_jobs=-1,
        scoring="neg_root_mean_squared_error"
    )

    imp = pd.DataFrame({
        "feature": numeric_cols,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std
    }).sort_values("importance_mean", ascending=False)

    imp.to_csv(os.path.join(args.outdir, "permutation_importance.csv"),
               index=False, encoding="utf-8-sig")

    # Top12 可视化（只保留这一份）
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    topk = imp.head(12).iloc[::-1]
    bars = ax.barh(topk["feature"], topk["importance_mean"], color=C(3))
    for b in bars:
        bar_3d_effect(b)

    ax.set_xlabel("Permutation importance (decrease in score)")
    ax.set_title(f"Top feature importance (best: {best_name})")
    info_card(ax, card_base, loc=(0.68, 0.50))
    save_svg(fig, os.path.join(args.outdir, "fig_feature_importance_top12.svg"))
    plt.close(fig)

    # PDP（方向性）：优先题干因素
    pdp_priority = ["mc_Humidity_mean", "mc_Temperature_mean", "mc_WindSpeed_mean", "mc_NO2_mean", "mc_O3_mean", "t_days", "PM2.5"]
    pdp_features = [c for c in pdp_priority if c in numeric_cols][:3]
    if len(pdp_features) > 0:
        fig, ax = plt.subplots(figsize=(9, 4))
        PartialDependenceDisplay.from_estimator(best_model, X_test, pdp_features, ax=ax)
        plt.title("Partial dependence (direction of key factors)")
        save_svg(fig,os.path.join(args.outdir, "fig_partial_dependence.svg"))

    print("===== Q2 Explain (no leak) Done =====")
    print("CV (Ridge+Spline):", m1)
    print("CV (HGBR):        ", m2)
    print("Best model:", best_name)
    print("Test RMSE:", rmse(y_test, pred_test))
    print("Test MAE :", float(mean_absolute_error(y_test, pred_test)))
    print("Outputs:", os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()
