# -*- coding: utf-8 -*-
"""
Problem C 数据预处理（Appendix 1 & 2）
输出：
- outputs/A1_clean.csv   国控站小时数据（清洗后）
- outputs/A2_clean.csv   企业点位高频数据（清洗后 + 小时锚点/偏移特征）
- outputs/clean_report.json  简单质量报告（缺失/异常统计）

依赖：
pip install pandas numpy openpyxl
"""

import os
import json
import numpy as np
import pandas as pd

# =========================
# 路径配置
# =========================
APP1_PATH = "../data/C_Appendix_1.xlsx"
APP2_PATH = "../data/C_Appendix_2.xlsx"
OUT_DIR = "../data/outputs_c_preprocess"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# 可调参数
# =========================
WINSORIZE = True          # 是否做温和去极值（建议开）
WINSOR_Q_LOW = 0.005
WINSOR_Q_HIGH = 0.995

TIME_COL = "Time"

# 污染物列（两份表共有）
POLLUTANT_COLS = ["PM2.5", "PM10", "CO", "NO2", "SO2", "O3"]

# Appendix 2 才有的气象列
METEO_COLS = ["WindSpeed", "Pressure", "Precipitation", "Temperature", "Humidity"]

# 物理/合理范围（超出则置为 NaN）
RANGES = {
    # 污染物：非负即可（上界可不硬卡，交给 winsorize）
    "PM2.5": (0, None),
    "PM10": (0, None),
    "CO": (0, None),
    "NO2": (0, None),
    "SO2": (0, None),
    "O3": (0, None),

    # 气象
    "WindSpeed": (0, None),          # m/s
    "Pressure": (8.0e4, 1.1e5),      # Pa（保守范围）
    "Precipitation": (0, None),      # mm/m^2
    "Humidity": (0, 100),            # rh%
    "Temperature": (-60, 60),        # 未注明单位，按摄氏度保守范围
}

# =========================
# 工具函数
# =========================
def read_first_sheet_xlsx(path: str) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    sheet = xl.sheet_names[0]
    df = pd.read_excel(path, sheet_name=sheet)
    # 统一列名（去空格）
    df.columns = [str(c).strip() for c in df.columns]
    return df

def ensure_datetime(df: pd.DataFrame, time_col: str = TIME_COL) -> pd.DataFrame:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    # 丢弃无法解析时间的行
    df = df.dropna(subset=[time_col])
    return df

def coerce_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def normalize_pressure_to_pa(df):
    if "Pressure" not in df.columns:
        return df
    s = pd.to_numeric(df["Pressure"], errors="coerce")
    med = s.median()

    # 经验判别：把 Pressure 统一换算成 Pa
    if pd.isna(med):
        return df

    if 800 <= med <= 1200:          # hPa
        df["Pressure"] = s * 100.0
    elif 80 <= med <= 120:          # kPa
        df["Pressure"] = s * 1000.0
    else:                           # 认为已经是 Pa 或其它（先不动）
        df["Pressure"] = s
    return df

def drop_or_aggregate_duplicates(df: pd.DataFrame, time_col: str = TIME_COL) -> pd.DataFrame:
    """
    同一时间戳多行：按数值列求均值（更稳），非数值列保留首个
    """
    df = df.copy()
    if df.duplicated(subset=[time_col]).any():
        num_cols = [c for c in df.columns if c != time_col and pd.api.types.is_numeric_dtype(df[c])]
        other_cols = [c for c in df.columns if c not in num_cols and c != time_col]
        agg = {c: "mean" for c in num_cols}
        for c in other_cols:
            agg[c] = "first"
        df = df.groupby(time_col, as_index=False).agg(agg)
    return df

def apply_ranges(df: pd.DataFrame, ranges: dict) -> pd.DataFrame:
    """
    超出物理范围：置 NaN（不直接删行，避免破坏时间序列）
    """
    df = df.copy()
    for c, (lo, hi) in ranges.items():
        if c not in df.columns:
            continue
        if lo is not None:
            df.loc[df[c] < lo, c] = np.nan
        if hi is not None:
            df.loc[df[c] > hi, c] = np.nan
    return df

def winsorize_cols(df: pd.DataFrame, cols, q_low=0.005, q_high=0.995) -> pd.DataFrame:
    """
    温和去极值：把极端值压到分位点，不改变 NaN
    """
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c].dropna()
        if s.empty:
            continue
        lo = s.quantile(q_low)
        hi = s.quantile(q_high)
        df[c] = df[c].clip(lower=lo, upper=hi)
    return df

def add_hour_features_station(df: pd.DataFrame, time_col: str = TIME_COL) -> pd.DataFrame:
    """
    国控站小时数据：增加 hour（整点）作为键
    """
    df = df.copy()
    df["hour"] = df[time_col].dt.floor("H")
    return df

def add_hour_features_company(df: pd.DataFrame, time_col: str = TIME_COL) -> pd.DataFrame:
    """
    企业点位高频数据：增加 floor/ceil 小时候选 + 与整点的偏移（分钟）
    之后做 ±5min 匹配时，直接用这些列判定即可。
    """
    df = df.copy()
    t = df[time_col]
    df["hour_floor"] = t.dt.floor("H")
    df["hour_ceil"] = t.dt.ceil("H")

    # 距离对应整点的时间差（分钟，绝对值）
    df["delta_floor_min"] = (t - df["hour_floor"]).dt.total_seconds().abs() / 60.0
    df["delta_ceil_min"] = (df["hour_ceil"] - t).dt.total_seconds().abs() / 60.0

    # 最近整点（两者取更小的）
    df["hour_nearest"] = np.where(df["delta_floor_min"] <= df["delta_ceil_min"], df["hour_floor"], df["hour_ceil"])
    df["delta_nearest_min"] = np.minimum(df["delta_floor_min"], df["delta_ceil_min"])
    return df

def make_report(df: pd.DataFrame, name: str, key_cols) -> dict:
    rep = {"name": name, "rows": int(df.shape[0]), "cols": int(df.shape[1])}
    miss = {c: float(df[c].isna().mean()) for c in key_cols if c in df.columns}
    rep["missing_rate"] = miss
    return rep


# =========================
# 主流程
# =========================
def preprocess():
    # ---- 读入
    a1 = read_first_sheet_xlsx(APP1_PATH)
    a2 = read_first_sheet_xlsx(APP2_PATH)

    # ---- 时间解析 + 排序
    a1 = ensure_datetime(a1, TIME_COL).sort_values(TIME_COL).reset_index(drop=True)
    a2 = ensure_datetime(a2, TIME_COL).sort_values(TIME_COL).reset_index(drop=True)

    # ---- 数值化
    a1 = coerce_numeric(a1, POLLUTANT_COLS)
    a2 = coerce_numeric(a2, POLLUTANT_COLS + METEO_COLS)

    # ---- 同一时间戳重复行处理
    a1 = drop_or_aggregate_duplicates(a1, TIME_COL)
    a2 = drop_or_aggregate_duplicates(a2, TIME_COL)
    a2 = normalize_pressure_to_pa(a2)
    # ---- 物理范围清洗（置 NaN）
    a1 = apply_ranges(a1, RANGES)
    a2 = apply_ranges(a2, RANGES)


    # ---- 温和去极值（建议开，尤其 PM2.5）
    if WINSORIZE:
        a1 = winsorize_cols(a1, POLLUTANT_COLS, WINSOR_Q_LOW, WINSOR_Q_HIGH)
        a2 = winsorize_cols(a2, POLLUTANT_COLS + METEO_COLS, WINSOR_Q_LOW, WINSOR_Q_HIGH)

    # ---- 生成小时锚点/偏移特征
    a1 = add_hour_features_station(a1, TIME_COL)
    a2 = add_hour_features_company(a2, TIME_COL)

    # ---- 导出
    out_a1 = os.path.join(OUT_DIR, "A1_clean.csv")
    out_a2 = os.path.join(OUT_DIR, "A2_clean.csv")
    a1.to_csv(out_a1, index=False, encoding="utf-8-sig")
    a2.to_csv(out_a2, index=False, encoding="utf-8-sig")

    # ---- 简单报告
    report = {
        "params": {
            "winsorize": WINSORIZE,
            "winsor_q_low": WINSOR_Q_LOW,
            "winsor_q_high": WINSOR_Q_HIGH,
        },
        "A1": make_report(a1, "Appendix1_station_hourly", POLLUTANT_COLS),
        "A2": make_report(a2, "Appendix2_company_highfreq", POLLUTANT_COLS + METEO_COLS),
        "notes": [
            "超出物理范围的值已置为 NaN（未删除整行）",
            "企业点位已生成 hour_floor/hour_ceil/hour_nearest 及对应 delta_*_min，方便做 ±5min 匹配",
        ],
    }
    with open(os.path.join(OUT_DIR, "clean_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Done.")
    print("A1_clean:", out_a1)
    print("A2_clean:", out_a2)
    print("report :", os.path.join(OUT_DIR, "clean_report.json"))

if __name__ == "__main__":
    preprocess()
