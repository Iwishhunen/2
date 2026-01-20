# -*- coding: utf-8 -*-
"""
汇总 permutation_importance_*.csv：
- Top特征长表 + 宽表
- Top特征频次统计
- 频次SVG图

用法示例：
python summarize_perm_importance.py ^
  --rootdir "D:\\study\\数学建模\\美赛\\第二次训练\\codes\\data\\q2_output\\pollutant_error_analysis" ^
  --topk 12 --topn_plot 20
"""

import argparse
from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def parse_pollutant_from_filename(path: Path) -> str:
    """
    从文件名 permutation_importance_{pollutant}.csv 解析 pollutant
    若失败则用父目录名兜底
    """
    m = re.match(r"permutation_importance_(.+)\.csv$", path.name)
    if m:
        return m.group(1)
    return path.parent.name


def load_one_csv(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp, low_memory=False)
    # 兼容列名
    if "feature" not in df.columns or "importance_mean" not in df.columns:
        raise ValueError(f"Bad columns in {fp}")
    if "importance_std" not in df.columns:
        df["importance_std"] = np.nan
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rootdir", type=str, required=True, help="包含 permutation_importance_*.csv 的根目录（可递归）")
    ap.add_argument("--outdir", type=str, default=None, help="输出目录，默认= rootdir")
    ap.add_argument("--topk", type=int, default=12, help="每个污染物取前K个特征")
    ap.add_argument("--topn_plot", type=int, default=20, help="频次图展示前N个特征")
    ap.add_argument("--positive_only", action="store_true", help="仅统计 importance_mean>0 的特征（推荐）")
    ap.add_argument("--no_positive_only", dest="positive_only", action="store_false", help="不限制正重要性")
    ap.set_defaults(positive_only=True)
    args = ap.parse_args()

    root = Path(args.rootdir)
    outdir = Path(args.outdir) if args.outdir else root
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(root.rglob("permutation_importance_*.csv"))
    if not files:
        raise FileNotFoundError(f"No permutation_importance_*.csv found under: {root}")

    long_rows = []
    wide_rows = []

    for fp in files:
        pollutant = parse_pollutant_from_filename(fp)
        df = load_one_csv(fp).copy()

        # 按重要性降序
        df = df.sort_values("importance_mean", ascending=False).reset_index(drop=True)

        # 只取“有效重要性”（默认 >0）
        if args.positive_only:
            df = df[df["importance_mean"] > 0].copy().reset_index(drop=True)

        top = df.head(args.topk).copy()
        top["pollutant"] = pollutant
        top["rank"] = np.arange(1, len(top) + 1)

        # 长表
        long_rows.append(top[["pollutant", "rank", "feature", "importance_mean", "importance_std"]])

        # 宽表：每个污染物一行
        row = {"pollutant": pollutant}
        for i, r in top.iterrows():
            k = int(r["rank"])
            row[f"top{k}_feature"] = r["feature"]
            row[f"top{k}_importance_mean"] = float(r["importance_mean"])
            row[f"top{k}_importance_std"] = float(r["importance_std"]) if pd.notna(r["importance_std"]) else np.nan
        wide_rows.append(row)

    long_df = pd.concat(long_rows, ignore_index=True)
    wide_df = pd.DataFrame(wide_rows).sort_values("pollutant").reset_index(drop=True)

    # 输出总表
    long_path = outdir / "table_top_features_long.csv"
    wide_path = outdir / "table_top_features_wide.csv"
    long_df.to_csv(long_path, index=False, encoding="utf-8-sig")
    wide_df.to_csv(wide_path, index=False, encoding="utf-8-sig")

    # ===== 频次统计 =====
    # 统计每个 feature 在各污染物 TopK 中出现次数，并给出平均rank、平均importance
    freq = (
        long_df.groupby("feature")
        .agg(
            count=("feature", "size"),
            avg_rank=("rank", "mean"),
            avg_importance=("importance_mean", "mean"),
        )
        .reset_index()
        .sort_values(["count", "avg_importance"], ascending=[False, False])
        .reset_index(drop=True)
    )

    # 还原：出现在哪些污染物
    feats_to_pollutants = (
        long_df.groupby("feature")["pollutant"]
        .apply(lambda s: ",".join(sorted(set(s))))
        .reset_index(name="pollutants")
    )
    freq = freq.merge(feats_to_pollutants, on="feature", how="left")

    freq_path = outdir / "feature_frequency_topk.csv"
    freq.to_csv(freq_path, index=False, encoding="utf-8-sig")

    # ===== 频次SVG图 =====
    topn = min(args.topn_plot, len(freq))
    plot_df = freq.head(topn).copy()
    plot_df = plot_df.iloc[::-1]  # 反转，频次最高在最上

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.35 * topn)))
    bars = ax.barh(plot_df["feature"], plot_df["count"])
    ax.set_xlabel(f"Frequency in Top{args.topk} across pollutants")
    ax.set_title(f"Top feature frequency (Top{args.topk}, N_files={len(files)})")

    # 标注数字（不指定颜色，默认matplotlib）
    for b, c in zip(bars, plot_df["count"].values):
        ax.text(b.get_width() + 0.05, b.get_y() + b.get_height() / 2, str(int(c)),
                va="center", ha="left", fontsize=9)

    fig.tight_layout()
    fig_path = outdir / "fig_feature_frequency.svg"
    fig.savefig(fig_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    print("Done.")
    print("Saved:")
    print(f"  {long_path}")
    print(f"  {wide_path}")
    print(f"  {freq_path}")
    print(f"  {fig_path}")


if __name__ == "__main__":
    main()
