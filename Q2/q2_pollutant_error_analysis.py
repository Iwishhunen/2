# -*- coding: utf-8 -*-
"""
Q2 跨污染物误差分析（与 model.py 模型/风格一致）
输入：aligned_from_raw.csv
输出：对 PM2.5/PM10/CO/NO2/SO2/O3 的误差 E_X = mc_X_mean - X
      训练同款模型（Ridge+Spline vs HGBR），并输出同款证据图（info_card 全放图外）
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 直接复用你原来的 model.py（同目录下 import 即可）
import model as q2m

# 复用同一套画图工具（model.py 里已经 import 了这些）
from utils.plot_style import (
    set_mcm_pastel_style, get_colors, get_accents, make_cmap,
    line_3d_effect, bar_3d_effect, gradient_fill_between,
    save_svg
)

# =========================
# 图外 info_card：永不遮挡
# =========================
def info_card_outside(fig, ax, text: str, pad_right=0.30, x=1.02, y=1.0):
    """
    把信息卡片放到图外右侧。
    - pad_right: 给右侧预留空间比例（0.25~0.35 常用）
    - x,y: 以 axes 坐标系为基准，x>1 表示在图外右边
    """
    # 给右侧留白，避免被裁切
    fig.subplots_adjust(right=1 - pad_right)

    ax.text(
        x, y, text,
        transform=ax.transAxes,
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="0.7", alpha=0.95)
    )

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def safe_C(colors, i, fallback="C0"):
    return colors[i % len(colors)] if colors else fallback

def safe_A(accents, i, fallback="C1"):
    return accents[i % len(accents)] if accents else fallback

# =========================
# 特征列生成：严格沿用 model.py 的逻辑，只是做“目标污染物泄露剔除”
# =========================
def build_numeric_cols(aligned: pd.DataFrame, target: str):
    """
    target: "PM2.5"/"PM10"/"CO"/"NO2"/"SO2"/"O3"
    返回：numeric_cols（与 model.py 结构一致，但剔除 mc_target_* 以免泄露）
    """
    base_time = ["t_days", "sin_hour", "cos_hour", "sin_month", "cos_month", "is_weekend"]
    align_quality = ["n_points", "min_delta_sec"]

    # 国站污染物（保留目标自身用于量程效应；PM2.5 额外保留 log）
    st_poll = [c for c in ["PM2.5", "PM10", "CO", "NO2", "SO2", "O3"] if c in aligned.columns]
    if "log_st_pm25" in aligned.columns:
        st_poll += ["log_st_pm25"]

    # 微站污染物：沿用 model.py 的做法（mean/std/iqr）
    # 但“剔除目标污染物 target”，避免 target leakage
    mc_poll = []
    for c in ["PM2.5", "PM10", "CO", "NO2", "SO2", "O3"]:
        if c == target:
            continue
        for suf in ["mean", "std", "iqr"]:
            col = f"mc_{c}_{suf}"
            if col in aligned.columns:
                mc_poll.append(col)

    # 微站气象：同 model.py
    mc_met = []
    for c in ["WindSpeed", "Pressure", "Precipitation", "Temperature", "Humidity"]:
        for suf in ["mean", "std", "iqr"]:
            col = f"mc_{c}_{suf}"
            if col in aligned.columns:
                mc_met.append(col)

    # 交互项：沿用你 PM2.5 的写法，但对每个 target 都建一个 X_hum（如果列存在）
    extra = []
    xhum = f"{target.lower().replace('.', '')}_hum"
    if xhum in aligned.columns:
        extra.append(xhum)
    # 也保留你原来的 pm25_hum（如果你想在非PM2.5目标里也作为协变量）
    if "pm25_hum" in aligned.columns and "pm25_hum" not in extra:
        extra.append("pm25_hum")

    numeric_cols = []
    for part in [st_poll, mc_poll, mc_met, base_time, align_quality, extra]:
        numeric_cols += part

    # 去重 + 存在性过滤
    numeric_cols = list(dict.fromkeys([c for c in numeric_cols if c in aligned.columns]))
    return numeric_cols

# =========================
# 画图：完全保持 model.py 的 6 张证据图结构，只是每张都用 info_card_outside
# =========================
def plot_evidence_bundle(aligned, train_df, test_df, test_out,
                         imp, best_model, best_name, best_cv,
                         numeric_cols, card_base, outdir, target, colors, accents):
    C = lambda i: safe_C(colors, i, "C0")
    A = lambda i: safe_A(accents, i, "C1")

    Ecol = f"E_{target}"
    st_col = target  # 国站列
    title_prefix = f"[{target}] "

    # (1) E over time
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ln1, = ax.plot(train_df["dt"], train_df[Ecol], linewidth=1.0, label=f"{Ecol} (train)", color=A(0))
    ln2, = ax.plot(test_out["dt"], test_out[Ecol], linewidth=1.2, label=f"{Ecol} (test)", color=A(1))
    line_3d_effect(ln1); line_3d_effect(ln2)

    ax.axvline(test_out["dt"].min(), linestyle="--", linewidth=1.2, label="test start", color=C(2))
    ax.set_xlabel("Time")
    ax.set_ylabel(f"{Ecol} = micro − station")
    ax.set_title(title_prefix + "Bias over time (drift evidence)")
    ax.legend()
    info_card_outside(fig, ax, card_base)
    save_svg(fig, os.path.join(outdir, f"fig_{target}_E_over_time.svg"))

    # (2) E vs Station target
    if st_col in aligned.columns:
        fig, ax = plt.subplots(figsize=(7.8, 5.0))
        ax.scatter(aligned[st_col], aligned[Ecol], s=14, alpha=0.22, edgecolors="none", color=C(1))
        ax.set_xlabel(f"Station {st_col}")
        ax.set_ylabel(Ecol)
        ax.set_title(title_prefix + f"{Ecol} vs Station {st_col} (range effect)")
        info_card_outside(fig, ax, card_base)
        save_svg(fig, os.path.join(outdir, f"fig_{target}_E_vs_station_{st_col.replace('.', '')}.svg"))

    # (3) E vs Humidity
    if "mc_Humidity_mean" in aligned.columns:
        fig, ax = plt.subplots(figsize=(7.8, 5.0))
        ax.scatter(aligned["mc_Humidity_mean"], aligned[Ecol], s=14, alpha=0.18, edgecolors="none", color=C(2))
        ax.set_xlabel("Micro Humidity (mean)")
        ax.set_ylabel(Ecol)
        ax.set_title(title_prefix + f"{Ecol} vs Humidity (meteorology effect)")
        info_card_outside(fig, ax, card_base)
        save_svg(fig, os.path.join(outdir, f"fig_{target}_E_vs_humidity.svg"))

    # (4) Diurnal mean E
    if "hour_of_day" in aligned.columns:
        hour_mean = aligned.groupby("hour_of_day")[Ecol].mean()
        fig, ax = plt.subplots(figsize=(7.8, 4.4))
        ln, = ax.plot(hour_mean.index, hour_mean.values, marker="o", color=A(2))
        line_3d_effect(ln)
        gradient_fill_between(ax, hour_mean.index, 0, hour_mean.values, base_color=A(2), layers=10, alpha_max=0.18)

        ax.set_xlabel("Hour of day")
        ax.set_ylabel(f"Mean {Ecol}")
        ax.set_title(title_prefix + "Diurnal pattern of bias")
        info_card_outside(fig, ax, card_base, pad_right=0.28)
        save_svg(fig, os.path.join(outdir, f"fig_{target}_diurnal_mean_E.svg"))

    # (5) Permutation importance Top12（barh + 3D effect）
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    topk = imp.head(12).iloc[::-1]
    bars = ax.barh(topk["feature"], topk["importance_mean"], color=C(3))
    for b in bars:
        bar_3d_effect(b)
    ax.set_xlabel("Permutation importance (decrease in score)")
    ax.set_title(title_prefix + f"Top feature importance (best: {best_name})")
    info_card_outside(fig, ax, card_base, pad_right=0.33)
    save_svg(fig, os.path.join(outdir, f"fig_{target}_feature_importance_top12.svg"))

    # (6) PDP：同 model.py 的优先顺序（若列存在就取前三）
    pdp_priority = ["mc_Humidity_mean", "mc_Temperature_mean", "mc_WindSpeed_mean",
                    "mc_NO2_mean", "mc_O3_mean", "t_days", "PM2.5", "PM10", "CO", "NO2", "SO2", "O3"]
    pdp_features = [c for c in pdp_priority if c in numeric_cols][:3]
    if len(pdp_features) > 0:
        fig, ax = plt.subplots(figsize=(9.2, 4.2))
        q2m.PartialDependenceDisplay.from_estimator(best_model, test_df[numeric_cols], pdp_features, ax=ax)
        ax.set_title(title_prefix + "Partial dependence (direction of key factors)")
        info_card_outside(fig, ax, card_base, pad_right=0.33)
        save_svg(fig, os.path.join(outdir, f"fig_{target}_partial_dependence.svg"))

# =========================
# 主程序
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned_csv", type=str, default="../data/q2_output/aligned_from_raw.csv")
    parser.add_argument("--outdir", type=str, default="../data/q2_output/pollutant_error_analysis")
    parser.add_argument("--tol_minutes", type=int, default=5)
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()

    ensure_dir(args.outdir)

    # 保证风格一致
    set_mcm_pastel_style("scheme1")
    colors = get_colors("scheme1")
    accents = get_accents("scheme1")
    _ = make_cmap("scheme1")

    aligned = pd.read_csv(args.aligned_csv, low_memory=False)

    # matched 过滤（与 model.py 一致）
    if "matched" in aligned.columns:
        aligned = aligned[aligned["matched"] == True].copy()

    # dt/time features：aligned_from_raw.csv 里通常已经有 dt/t_days/hour_of_day 等
    aligned["dt"] = pd.to_datetime(aligned["dt"], errors="coerce")
    aligned = aligned.dropna(subset=["dt"]).copy()

    # PM2.5 log & pm25_hum（沿用 model.py）
    if "PM2.5" in aligned.columns:
        aligned["log_st_pm25"] = np.log1p(pd.to_numeric(aligned["PM2.5"], errors="coerce").clip(lower=0))
    if "PM2.5" in aligned.columns and "mc_Humidity_mean" in aligned.columns:
        aligned["pm25_hum"] = aligned["PM2.5"] * aligned["mc_Humidity_mean"]

    # 目标污染物集合（只处理数据里存在的）
    targets = ["PM2.5", "PM10", "CO", "NO2", "SO2", "O3"]
    targets = [t for t in targets if (t in aligned.columns and f"mc_{t}_mean" in aligned.columns)]
    ANOMALY_DATE = {
        "SO2": ["2019-01-23"],
        "PM10": ["2019-02-19"],
    }

    perf_rows = []

    for target in targets:
        Ecol = f"E_{target}"

        # 交互项 X_hum（可选，但不改变模型结构）
        if target in aligned.columns and "mc_Humidity_mean" in aligned.columns:
            aligned[target.lower().replace(".", "") + "_hum"] = aligned[target] * aligned["mc_Humidity_mean"]

        # 定义误差
        aligned[Ecol] = aligned[f"mc_{target}_mean"] - aligned[target]

        # 生成特征列（与 model.py 同结构，只是剔除 mc_target_*）
        numeric_cols = build_numeric_cols(aligned, target)

        # 去掉缺失
        work = aligned.dropna(subset=[Ecol]).copy()
        if target in ANOMALY_DATE:
            bad_dates = set(pd.to_datetime(ANOMALY_DATE[target]).date)
            mask_bad = work["dt"].dt.date.isin(bad_dates)
            n_bad = int(mask_bad.sum())
            if n_bad > 0:
                work = work.loc[~mask_bad].copy()
                print(f"[{target}] removed anomaly rows: {n_bad} (dates={sorted(bad_dates)})")

        # 时间切分（复用 model.py）
        train_df, test_df = q2m.time_train_test_split(work, time_col="dt", test_ratio=0.2)
        X_train = train_df[numeric_cols]
        y_train = train_df[Ecol].values
        X_test  = test_df[numeric_cols]
        y_test  = test_df[Ecol].values

        # =========================
        # 模型：完全照搬 model.py
        # =========================
        # Ridge + Spline（解释性）
        spline_features = [c for c in ["PM2.5", "mc_Humidity_mean", "t_days"] if c in numeric_cols]
        other_numeric = [c for c in numeric_cols if c not in spline_features]

        pre_linear = q2m.ColumnTransformer(
            transformers=[
                ("spline", q2m.Pipeline([
                    ("imp", q2m.SimpleImputer(strategy="median")),
                    ("spline", q2m.SplineTransformer(n_knots=6, degree=3, include_bias=False)),
                    ("sc", q2m.StandardScaler())
                ]), spline_features),
                ("num", q2m.Pipeline([
                    ("imp", q2m.SimpleImputer(strategy="median")),
                    ("sc", q2m.StandardScaler())
                ]), other_numeric),
            ],
            remainder="drop"
        )

        model_linear = q2m.Pipeline(steps=[
            ("pre", pre_linear),
            ("reg", q2m.Ridge(alpha=2.0, random_state=0))
        ])

        # 非线性：HGBR
        model_tree = q2m.Pipeline(steps=[
            ("imp", q2m.SimpleImputer(strategy="median")),
            ("reg", q2m.HistGradientBoostingRegressor(
                max_depth=4, learning_rate=0.06, max_iter=800,
                random_state=0
            ))
        ])

        # CV（复用 model.py）
        m1 = q2m.evaluate_cv(model_linear, X_train, y_train, n_splits=args.n_splits)
        m2 = q2m.evaluate_cv(model_tree,   X_train, y_train, n_splits=args.n_splits)

        best_name, best_model = ("Ridge+Spline", model_linear) if m1["CV_RMSE"] <= m2["CV_RMSE"] else ("HGBR", model_tree)
        best_cv = m1 if best_name == "Ridge+Spline" else m2

        best_model.fit(X_train, y_train)
        pred_test = best_model.predict(X_test)

        # 输出 metrics（每个污染物单独一份）
        metrics = {
            "target": target,
            "model_linear_CV_RMSE": m1["CV_RMSE"],
            "model_linear_CV_MAE": m1["CV_MAE"],
            "model_linear_CV_R2":  m1["CV_R2"],
            "model_tree_CV_RMSE":  m2["CV_RMSE"],
            "model_tree_CV_MAE":   m2["CV_MAE"],
            "model_tree_CV_R2":    m2["CV_R2"],
            "best_model": best_name,
            "best_CV_RMSE": best_cv["CV_RMSE"],
            "best_CV_MAE":  best_cv["CV_MAE"],
            "test_RMSE": q2m.rmse(y_test, pred_test),
            "test_MAE": float(q2m.mean_absolute_error(y_test, pred_test)),
            "N_train": len(train_df),
            "N_test":  len(test_df),
            "tol_minutes": args.tol_minutes
        }
        perf_rows.append(metrics)

        # 预测输出（每个污染物单独一份）
        test_out = test_df[["dt", Ecol] + numeric_cols].copy()
        test_out[f"{Ecol}_pred"] = pred_test
        test_out["residual"] = test_out[Ecol] - test_out[f"{Ecol}_pred"]
        test_out.to_csv(os.path.join(args.outdir, f"test_with_pred_{target}.csv"),
                        index=False, encoding="utf-8-sig")

        # 重要性（Permutation importance）——同 model.py
        perm = q2m.permutation_importance(
            best_model, X_test, y_test,
            n_repeats=20, random_state=0, n_jobs=-1,
            scoring="neg_root_mean_squared_error"
        )
        imp = pd.DataFrame({
            "feature": numeric_cols,
            "importance_mean": perm.importances_mean,
            "importance_std":  perm.importances_std
        }).sort_values("importance_mean", ascending=False)

        imp.to_csv(os.path.join(args.outdir, f"permutation_importance_{target}.csv"),
                   index=False, encoding="utf-8-sig")

        # info 卡片内容（与 model.py 同格式）
        card_base = (
            f"Target: {target}\n"
            f"Best: {best_name}\n"
            f"CV RMSE: {best_cv['CV_RMSE']:.2f}\n"
            f"Test RMSE: {q2m.rmse(y_test, pred_test):.2f}\n"
            f"Test MAE: {float(q2m.mean_absolute_error(y_test, pred_test)):.2f}\n"
            f"N(train): {len(train_df)}\nN(test):  {len(test_df)}\n"
            f"tol=±{args.tol_minutes} min"
        )

        # 画图输出目录（每个污染物一个子文件夹，更清晰）
        subdir = os.path.join(args.outdir, f"{target}")
        ensure_dir(subdir)

        plot_evidence_bundle(
            aligned=work, train_df=train_df, test_df=test_df, test_out=test_out,
            imp=imp, best_model=best_model, best_name=best_name, best_cv=best_cv,
            numeric_cols=numeric_cols, card_base=card_base,
            outdir=subdir, target=target, colors=colors, accents=accents
        )

        print(f"[{target}] done. best={best_name}, testRMSE={q2m.rmse(y_test, pred_test):.3f}")

    # 总汇总表
    perf = pd.DataFrame(perf_rows)
    perf.to_csv(os.path.join(args.outdir, "perf_summary_all_pollutants.csv"),
                index=False, encoding="utf-8-sig")
    print("All done. Outputs:", os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()
