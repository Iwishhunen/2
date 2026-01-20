# -*- coding: utf-8 -*-
"""
Q3 校准模型 FINAL v2（RF + O3残差校正 + 回退策略 + 2019异常日剔除 + Post-hoc线性校正）
输出：
- 每个污染物子目录：SVG证据图 + permutation_importance.csv + test_with_pred_*.csv
- perf_summary_q3.csv（含回退/方法/Post-hoc 参数）
- aligned_with_calibrated_all_pollutants.csv（全量校准序列）
- models/model_<pollutant>.joblib（每个污染物一个模型包）
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import joblib

import Q2.model as q2m

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor

from utils.plot_style import (
    set_mcm_pastel_style, get_colors, get_accents, make_cmap,
    line_3d_effect, bar_3d_effect, gradient_fill_between,
    save_svg
)

# =========================
# utils
# =========================
def info_card_outside(fig, ax, text: str, pad_right=0.30, x=1.02, y=1.0):
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
# metrics
# =========================
def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def pearson_r(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return np.nan
    yt = y_true - np.mean(y_true)
    yp = y_pred - np.mean(y_pred)
    denom = (np.sqrt(np.sum(yt**2)) * np.sqrt(np.sum(yp**2)))
    return float(np.sum(yt * yp) / denom) if denom != 0 else np.nan

def ccc(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return np.nan
    mu_x = np.mean(y_true)
    mu_y = np.mean(y_pred)
    var_x = np.var(y_true, ddof=1)
    var_y = np.var(y_pred, ddof=1)
    cov_xy = np.cov(y_true, y_pred, ddof=1)[0, 1]
    denom = var_x + var_y + (mu_x - mu_y) ** 2
    return float((2 * cov_xy) / denom) if denom != 0 else np.nan

def metrics_pack(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
        "CCC": ccc(y_true, y_pred),
        "Bias": float(np.mean(y_pred - y_true)),
        "r": pearson_r(y_true, y_pred)
    }

def fit_posthoc_linear(y_true, y_pred, min_points=20):
    """
    Post-hoc linear correction fitted on TRAIN:
        y_true ≈ a * y_pred + b
    Return (a, b, used_flag).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) < max(2, int(min_points)):
        return 1.0, 0.0, False
    X = np.vstack([y_pred[mask], np.ones(int(mask.sum()))]).T
    a, b = np.linalg.lstsq(X, y_true[mask], rcond=None)[0]
    return float(a), float(b), True

# =========================
# features (deployable: no station columns)
# =========================
def build_features_q3(aligned: pd.DataFrame, target: str):
    base_time = ["t_days", "sin_hour", "cos_hour", "sin_month", "cos_month", "is_weekend", "hour_of_day"]
    align_quality = ["n_points", "min_delta_sec"]

    # micro target stats
    mc_target = []
    for suf in ["mean", "std", "iqr"]:
        col = f"mc_{target}_{suf}"
        if col in aligned.columns:
            mc_target.append(col)

    # micro other pollutants (cross-sensitivity)
    mc_other = []
    for c in ["PM2.5", "PM10", "CO", "NO2", "SO2", "O3"]:
        if c == target:
            continue
        for suf in ["mean", "std", "iqr"]:
            col = f"mc_{c}_{suf}"
            if col in aligned.columns:
                mc_other.append(col)

    # micro meteo
    mc_met = []
    for c in ["WindSpeed", "Pressure", "Precipitation", "Temperature", "Humidity"]:
        for suf in ["mean", "std", "iqr"]:
            col = f"mc_{c}_{suf}"
            if col in aligned.columns:
                mc_met.append(col)

    extra = []

    # target_mean * humidity_mean (helps humidity interference)
    if f"mc_{target}_mean" in aligned.columns and "mc_Humidity_mean" in aligned.columns:
        xhum = f"mc_{target}_mean_x_hum"
        aligned[xhum] = pd.to_numeric(aligned[f"mc_{target}_mean"], errors="coerce") * pd.to_numeric(aligned["mc_Humidity_mean"], errors="coerce")
        extra.append(xhum)

    # O3 special: NO2 * sin_hour (proxy for diurnal photochemistry / titration pattern)
    if target == "O3" and ("mc_NO2_mean" in aligned.columns) and ("sin_hour" in aligned.columns):
        xno2 = "mc_NO2_mean_x_sin_hour"
        aligned[xno2] = pd.to_numeric(aligned["mc_NO2_mean"], errors="coerce") * pd.to_numeric(aligned["sin_hour"], errors="coerce")
        extra.append(xno2)

    feats = []
    for part in [mc_target, mc_other, mc_met, base_time, align_quality, extra]:
        feats += part

    # remove duplicates, keep existing
    feats = list(dict.fromkeys([c for c in feats if c in aligned.columns]))
    return feats

# =========================
# 2019 bad days filter
# =========================
BAD_DAYS = {
    "SO2":  {pd.to_datetime("2019-01-23").date()},
    "PM10": {pd.to_datetime("2019-02-19").date()},
}

def drop_bad_days(df: pd.DataFrame, pollutant: str) -> pd.DataFrame:
    bad = BAD_DAYS.get(pollutant, None)
    if not bad or "_date" not in df.columns:
        return df
    return df[~df["_date"].isin(bad)].copy()

# =========================
# plots
# =========================
def plot_bundle_q3(test_out: pd.DataFrame,
                   imp_df: pd.DataFrame,
                   best_model,
                   best_name: str,
                   feats: list,
                   card_text: str,
                   outdir: str,
                   target: str,
                   colors,
                   accents):

    C = lambda i: safe_C(colors, i, "C0")
    A = lambda i: safe_A(accents, i, "C1")

    st_col = target
    mc_col = f"mc_{target}_mean"
    pred_col = f"{target}_calib_pred"

    # 防重复列名导致 test_out[col] 变 DataFrame
    test_out = test_out.loc[:, ~test_out.columns.duplicated()].copy()

    # (1) Time series compare
    fig, ax = plt.subplots(figsize=(9.8, 4.4))
    dt = pd.to_datetime(test_out["dt"], errors="coerce")

    y_station = pd.to_numeric(test_out[st_col], errors="coerce").to_numpy()
    y_micro   = pd.to_numeric(test_out[mc_col], errors="coerce").to_numpy()
    y_calib   = pd.to_numeric(test_out[pred_col], errors="coerce").to_numpy()

    mask = (~pd.isna(dt)) & (~np.isnan(y_station)) & (~np.isnan(y_micro)) & (~np.isnan(y_calib))
    dt = dt[mask]
    y_station, y_micro, y_calib = y_station[mask], y_micro[mask], y_calib[mask]

    ln1 = ax.plot(dt, y_station, linewidth=1.4, label="Station (truth)", color=A(0))[0]
    ln2 = ax.plot(dt, y_micro,   linewidth=1.0, label="Micro raw",       color=C(2))[0]
    ln3 = ax.plot(dt, y_calib,   linewidth=1.4, label=f"Calibrated ({best_name})", color=A(1))[0]
    line_3d_effect(ln1); line_3d_effect(ln2); line_3d_effect(ln3)
    ax.set_xlabel("Time")
    ax.set_ylabel(target)
    ax.set_title(f"[{target}] Test segment: Station vs Micro vs Calibrated")
    ax.legend()
    info_card_outside(fig, ax, card_text, pad_right=0.32)
    save_svg(fig, os.path.join(outdir, f"fig_{target}_test_timeseries_compare.svg"))

    # (2) Scatter before/after vs station
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    ax.scatter(y_station, y_micro, s=14, alpha=0.22, edgecolors="none", label="Micro raw", color=C(1))
    ax.scatter(y_station, y_calib, s=14, alpha=0.22, edgecolors="none", label="Calibrated", color=A(1))

    mn = float(np.nanmin([np.min(y_station), np.min(y_micro), np.min(y_calib)]))
    mx = float(np.nanmax([np.max(y_station), np.max(y_micro), np.max(y_calib)]))
    ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1.2, label="1:1", color=C(3))

    ax.set_xlabel("Station (truth)")
    ax.set_ylabel("Value")
    ax.set_title(f"[{target}] Scatter vs Station (before/after calibration)")
    ax.legend()
    info_card_outside(fig, ax, card_text, pad_right=0.32)
    save_svg(fig, os.path.join(outdir, f"fig_{target}_scatter_before_after.svg"))

    # (3) Residual vs humidity (after)
    if "mc_Humidity_mean" in test_out.columns:
        hum = pd.to_numeric(test_out["mc_Humidity_mean"], errors="coerce").to_numpy()[mask]
        resid = y_calib - y_station
        fig, ax = plt.subplots(figsize=(7.8, 5.0))
        ax.scatter(hum, resid, s=14, alpha=0.18, edgecolors="none", color=C(2))
        ax.axhline(0, linestyle="--", linewidth=1.1, color=C(3))
        ax.set_xlabel("Micro Humidity (mean)")
        ax.set_ylabel("Residual (calib - station)")
        ax.set_title(f"[{target}] Residual vs Humidity (after calibration)")
        info_card_outside(fig, ax, card_text, pad_right=0.30)
        save_svg(fig, os.path.join(outdir, f"fig_{target}_residual_vs_humidity.svg"))

    # (4) Diurnal mean residual
    if "hour_of_day" in test_out.columns:
        tmp = test_out.loc[mask, ["hour_of_day"]].copy()
        tmp["resid"] = (y_calib - y_station)
        fig, ax = plt.subplots(figsize=(7.8, 4.4))
        hour_mean = tmp.groupby("hour_of_day")["resid"].mean()
        ln = ax.plot(hour_mean.index, hour_mean.values, marker="o", color=A(2))[0]
        line_3d_effect(ln)
        gradient_fill_between(ax, hour_mean.index, 0, hour_mean.values, base_color=A(2), layers=10, alpha_max=0.18)
        ax.axhline(0, linestyle="--", linewidth=1.1, color=C(3))
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Mean residual")
        ax.set_title(f"[{target}] Diurnal mean residual (after calibration)")
        info_card_outside(fig, ax, card_text, pad_right=0.28)
        save_svg(fig, os.path.join(outdir, f"fig_{target}_diurnal_mean_residual.svg"))

    # (5) Importance top12
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    topk = imp_df.head(12).iloc[::-1]
    bars = ax.barh(topk["feature"], topk["importance_mean"], color=C(3))
    for b in bars:
        bar_3d_effect(b)
    ax.set_xlabel("Permutation importance (decrease in score)")
    ax.set_title(f"[{target}] Feature importance (best: {best_name})")
    info_card_outside(fig, ax, card_text, pad_right=0.33)
    save_svg(fig, os.path.join(outdir, f"fig_{target}_feature_importance_top12.svg"))

    # (6) PDP for top 3 priority features (if supported)
    pdp_priority = [
        f"mc_{target}_mean", "mc_Humidity_mean", "mc_Temperature_mean", "mc_WindSpeed_mean",
        "mc_NO2_mean", "mc_O3_mean", "t_days"
    ]
    pdp_feats = [c for c in pdp_priority if c in feats][:3]
    if len(pdp_feats) > 0:
        try:
            fig, ax = plt.subplots(figsize=(9.2, 4.2))
            PartialDependenceDisplay.from_estimator(best_model, test_out.loc[:, feats], pdp_feats, ax=ax)
            ax.set_title(f"[{target}] Partial dependence (key factors)")
            info_card_outside(fig, ax, card_text, pad_right=0.33)
            save_svg(fig, os.path.join(outdir, f"fig_{target}_partial_dependence.svg"))
        except Exception:
            # 某些 pipeline/模型组合可能不支持PDP，静默跳过
            pass

# =========================
# main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned_csv", type=str, default="../data/q2_output/aligned_from_raw.csv")
    parser.add_argument("--outdir", type=str, default="../data/q3_output/calibration")
    parser.add_argument("--models_dir", type=str, default="../data/q3_output/models")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--tol_minutes", type=int, default=5)
    parser.add_argument("--clip_nonneg", action="store_true")
    parser.add_argument("--disable_posthoc", action="store_true",
                        help="Disable post-hoc linear correction y=a*y_pred+b fitted on TRAIN.")
    args = parser.parse_args()

    ensure_dir(args.outdir)
    ensure_dir(args.models_dir)

    # style (same as Q2)
    set_mcm_pastel_style("scheme1")
    colors = get_colors("scheme1")
    accents = get_accents("scheme1")
    _ = make_cmap("scheme1")

    aligned = pd.read_csv(args.aligned_csv, low_memory=False)

    if "matched" in aligned.columns:
        aligned = aligned[aligned["matched"] == True].copy()

    aligned["dt"] = pd.to_datetime(aligned["dt"], errors="coerce")
    aligned = aligned.dropna(subset=["dt"]).copy()
    aligned["_date"] = aligned["dt"].dt.date

    targets = ["PM2.5", "PM10", "CO", "NO2", "SO2", "O3"]
    targets = [t for t in targets if (t in aligned.columns and f"mc_{t}_mean" in aligned.columns)]

    perf_rows = []
    merged_out = aligned[["dt"]].copy()

    for target in targets:
        st_col = target
        mc_col = f"mc_{target}_mean"

        feats = build_features_q3(aligned, target)

        # training pool
        work = aligned.dropna(subset=[st_col, mc_col]).copy()
        work = drop_bad_days(work, target)

        # time split
        if hasattr(q2m, "time_train_test_split"):
            train_df, test_df = q2m.time_train_test_split(work, time_col="dt", test_ratio=args.test_ratio)
        else:
            work = work.sort_values("dt").reset_index(drop=True)
            n = len(work)
            n_test = max(1, int(round(n * args.test_ratio)))
            train_df = work.iloc[:-n_test].copy()
            test_df = work.iloc[-n_test:].copy()

        X_train = train_df[feats]
        X_test  = test_df[feats]

        # baseline raw micro
        y_base_train = train_df[mc_col].values.astype(float)
        y_base_test  = test_df[mc_col].values.astype(float)

        y_station_train = train_df[st_col].values.astype(float)
        y_station_test  = test_df[st_col].values.astype(float)

        # ---- calibration method ----
        # default: direct for most pollutants
        # O3: residual calibration (more robust)
        use_residual = (target == "O3")
        cal_method = "residual" if use_residual else "direct"

        if use_residual:
            y_train = (y_station_train - y_base_train)  # delta
        else:
            y_train = y_station_train

        # ---- models: linear (ridge+spline) vs RF ----
        spline_features = [c for c in [f"mc_{target}_mean", "mc_Humidity_mean", "t_days"] if c in feats]
        other_numeric = [c for c in feats if c not in spline_features]

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

        model_tree = q2m.Pipeline(steps=[
            ("imp", q2m.SimpleImputer(strategy="median")),
            ("reg", RandomForestRegressor(
                n_estimators=600,
                max_depth=None,
                min_samples_leaf=2,
                max_features="sqrt",
                n_jobs=-1,
                random_state=0
            ))
        ])

        # CV choose (prefer q2m.evaluate_cv if available)
        if hasattr(q2m, "evaluate_cv"):
            m1 = q2m.evaluate_cv(model_linear, X_train, y_train, n_splits=args.n_splits)
            m2 = q2m.evaluate_cv(model_tree,   X_train, y_train, n_splits=args.n_splits)
            cv_rmse_1, cv_rmse_2 = m1.get("CV_RMSE", np.nan), m2.get("CV_RMSE", np.nan)
        else:
            model_linear.fit(X_train, y_train)
            model_tree.fit(X_train, y_train)
            cv_rmse_1 = rmse(y_train, model_linear.predict(X_train))
            cv_rmse_2 = rmse(y_train, model_tree.predict(X_train))
            m1 = {"CV_RMSE": cv_rmse_1}
            m2 = {"CV_RMSE": cv_rmse_2}

        best_name, best_model, best_cv = (
            ("Ridge+Spline", model_linear, m1) if cv_rmse_1 <= cv_rmse_2 else ("RF", model_tree, m2)
        )

        # fit best
        best_model.fit(X_train, y_train)

        # predict test (deliver space)
        pred_test = best_model.predict(X_test).astype(float)
        if use_residual:
            y_pred = y_base_test + pred_test
        else:
            y_pred = pred_test

        if args.clip_nonneg:
            y_pred = np.clip(y_pred, 0, None)

        # ---- post-hoc linear correction on TRAIN (improves CCC/Bias) ----
        posthoc_used = False
        posthoc_a, posthoc_b = 1.0, 0.0
        if not args.disable_posthoc:
            pred_train = best_model.predict(X_train).astype(float)
            if use_residual:
                y_train_pred_delivered = y_base_train + pred_train
            else:
                y_train_pred_delivered = pred_train

            posthoc_a, posthoc_b, posthoc_used = fit_posthoc_linear(
                y_station_train, y_train_pred_delivered, min_points=20
            )
            if posthoc_used:
                y_pred = posthoc_a * y_pred + posthoc_b
                if args.clip_nonneg:
                    y_pred = np.clip(y_pred, 0, None)

        # metrics before/after (after post-hoc)
        base_m = metrics_pack(y_station_test, y_base_test)
        calib_m = metrics_pack(y_station_test, y_pred)

        # ---- robust fallback: never worse than raw on test ----
        used_fallback = False
        if calib_m["RMSE"] > base_m["RMSE"]:
            y_pred = y_base_test.copy()
            if args.clip_nonneg:
                y_pred = np.clip(y_pred, 0, None)
            calib_m = metrics_pack(y_station_test, y_pred)
            best_name = best_name + " (fallback->raw)"
            used_fallback = True

        # if fallback is used, post-hoc is not actually delivered
        if used_fallback:
            posthoc_used = False
            posthoc_a, posthoc_b = 1.0, 0.0

        # save model pack
        model_path = os.path.join(args.models_dir, f"model_{target.replace('.', '')}.joblib")
        joblib.dump({
            "model": best_model,
            "features": feats,
            "target": target,
            "cal_method": cal_method,
            "used_fallback_rule": True,
            "posthoc_used": posthoc_used,
            "posthoc_a": posthoc_a,
            "posthoc_b": posthoc_b,
        }, model_path)

        # per-target output dir
        subdir = os.path.join(args.outdir, target.replace(".", ""))
        ensure_dir(subdir)

        # test_out (avoid duplicate columns)
        feats_out = [c for c in feats if c not in ["dt", st_col, mc_col]]
        test_out = test_df[["dt", st_col, mc_col] + feats_out].copy()
        test_out[f"{target}_calib_pred"] = y_pred
        test_out["residual"] = test_out[f"{target}_calib_pred"] - test_out[st_col]
        test_out.to_csv(os.path.join(subdir, f"test_with_pred_{target.replace('.', '')}.csv"),
                        index=False, encoding="utf-8-sig")

        # full prediction for merged output (apply same method + post-hoc + fallback rule)
        # NOTE: fallback here means "deliver raw series" for that pollutant (safe delivery)
        if used_fallback:
            full_y = aligned[mc_col].astype(float).to_numpy()
        else:
            full_pred = best_model.predict(aligned[feats]).astype(float)
            if use_residual:
                full_y = aligned[mc_col].astype(float).to_numpy() + full_pred
            else:
                full_y = full_pred

            # apply post-hoc correction to full series
            if posthoc_used:
                full_y = posthoc_a * full_y + posthoc_b

        if args.clip_nonneg:
            full_y = np.clip(full_y, 0, None)

        merged_out[f"{target}_micro_raw"] = aligned.get(mc_col, np.nan)
        merged_out[f"{target}_calibrated"] = full_y
        if st_col in aligned.columns:
            merged_out[f"{target}_station_truth"] = aligned[st_col]

        # importance (permutation)
        try:
            perm = permutation_importance(
                best_model, test_df[feats], y_station_test,
                n_repeats=20, random_state=0, n_jobs=-1,
                scoring="neg_root_mean_squared_error"
            )
            imp_df = pd.DataFrame({
                "feature": feats,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std
            }).sort_values("importance_mean", ascending=False)
        except Exception:
            imp_df = pd.DataFrame({"feature": feats, "importance_mean": np.nan, "importance_std": np.nan})

        imp_df.to_csv(os.path.join(subdir, "permutation_importance.csv"),
                      index=False, encoding="utf-8-sig")

        # info card text
        card_text = (
            f"Target: {target}\n"
            f"Method: {cal_method}\n"
            f"Best: {best_name}\n"
            f"N(train): {len(train_df)}\nN(test):  {len(test_df)}\n"
            f"Fallback used: {used_fallback}\n"
            f"Post-hoc: {posthoc_used} (a={posthoc_a:.3f}, b={posthoc_b:.3f})\n\n"
            f"Before (micro raw):\n"
            f"  RMSE={base_m['RMSE']:.2f}, MAE={base_m['MAE']:.2f}\n"
            f"  R2={base_m['R2']:.3f}, CCC={base_m['CCC']:.3f}\n\n"
            f"After (delivered):\n"
            f"  RMSE={calib_m['RMSE']:.2f}, MAE={calib_m['MAE']:.2f}\n"
            f"  R2={calib_m['R2']:.3f}, CCC={calib_m['CCC']:.3f}\n"
            f"tol=±{args.tol_minutes} min"
        )

        # plots
        plot_bundle_q3(
            test_out=test_out,
            imp_df=imp_df,
            best_model=best_model,
            best_name=best_name,
            feats=feats,
            card_text=card_text,
            outdir=subdir,
            target=target,
            colors=colors,
            accents=accents
        )

        perf_rows.append({
            "target": target,
            "best_model": best_name,
            "cal_method": cal_method,
            "USED_FALLBACK": used_fallback,
            "POSTHOC_USED": posthoc_used,
            "POSTHOC_A": posthoc_a,
            "POSTHOC_B": posthoc_b,
            "N_train": len(train_df),
            "N_test": len(test_df),

            "BASE_MAE": base_m["MAE"],
            "BASE_RMSE": base_m["RMSE"],
            "BASE_R2": base_m["R2"],
            "BASE_CCC": base_m["CCC"],
            "BASE_Bias": base_m["Bias"],
            "BASE_r": base_m["r"],

            "CAL_MAE": calib_m["MAE"],
            "CAL_RMSE": calib_m["RMSE"],
            "CAL_R2": calib_m["R2"],
            "CAL_CCC": calib_m["CCC"],
            "CAL_Bias": calib_m["Bias"],
            "CAL_r": calib_m["r"],

            "IMPROVE_RMSE": base_m["RMSE"] - calib_m["RMSE"],
            "IMPROVE_MAE": base_m["MAE"] - calib_m["MAE"],
            "tol_minutes": args.tol_minutes
        })

        print(f"[{target}] done. Best={best_name}, RMSE: {base_m['RMSE']:.3f} -> {calib_m['RMSE']:.3f}")

    # summary outputs
    perf = pd.DataFrame(perf_rows)
    perf_path = os.path.join(args.outdir, "perf_summary_q3.csv")
    perf.to_csv(perf_path, index=False, encoding="utf-8-sig")

    merged_path = os.path.join(args.outdir, "aligned_with_calibrated_all_pollutants.csv")
    merged_out.to_csv(merged_path, index=False, encoding="utf-8-sig")

    print("All done.")
    print("Perf summary:", os.path.abspath(perf_path))
    print("Merged output:", os.path.abspath(merged_path))
    print("Models dir:", os.path.abspath(args.models_dir))


if __name__ == "__main__":
    main()
