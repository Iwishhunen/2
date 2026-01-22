# -*- coding: utf-8 -*-
"""

outdir/
  ├─ sites.csv
  ├─ distance_km.csv
  ├─ simulated_multisite_long.csv
  ├─ scores_by_site_all_pollutants.csv        # 长表：site_id + pollutant + SCI/TCI/MCI/RI
  ├─ scores_by_site_avg_all.csv               # 每站点跨污染物平均（若启用 ALL）
  ├─ anomaly_report_q4.json                   # 总报告（含每污染物异常站点、最差站点时间定位）
  ├─ PM25/                                   # 以 Q3 命名规则：PM2.5 -> PM25
  │    ├─ scores_by_site_PM25.csv
  │    ├─ spatial_deviation_PM25.csv
  │    ├─ anomaly_report_PM25.json
  │    ├─ fig_PM25_RI_by_site.svg
  │    ├─ fig_PM25_indicators_normalized.svg
  │    └─ fig_PM25_spatial_deviation_worst_site.svg
  ├─ PM10/
  ├─ CO/
  ├─ NO2/
  ├─ SO2/
  └─ O3/

一致性指标定义（与“问题四思路”对应的可执行版本）：
- SCI（Spatial Consistency Index）：|x_i(t) - mean_neighbors(t)| / (|mean_neighbors(t)|+eps) 的时均
- TCI（Temporal Consistency Index）：序列差分的 robust-z 跳变占比（越小越好）
- MCI（Meteo Consistency Index）：在空间邻居中挑气象最相似的 k_meteo 个做参照，再算偏差
- RI（Reliability Index）：1 - (α*SCI_n + β*TCI_n + γ*MCI_n)，其中 *_n 为[0,1]归一化

说明：
- 这是在“没有国家站真值”的情况下，用“多站点内部一致性”评估相对可靠性，符合 Q4 目标。
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Q2/Q3 统一风格：utils.plot_style
# =========================
SCHEME = "scheme1"
try:
    from utils.plot_style import (
        set_mcm_pastel_style, get_colors, get_accents, make_cmap,
        line_3d_effect, bar_3d_effect, gradient_fill_between,
        save_svg
    )
    _HAS_STYLE = True
except Exception:
    _HAS_STYLE = False

    def set_mcm_pastel_style(_scheme="scheme1"):
        pass

    def get_colors(_scheme="scheme1"):
        return []

    def get_accents(_scheme="scheme1"):
        return []

    def make_cmap(_scheme="scheme1"):
        return None

    def save_svg(fig, path: str):
        fig.savefig(path, format="svg", bbox_inches="tight")
        plt.close(fig)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def info_card_outside(fig, ax, text: str, pad_right=0.30, x=1.02, y=1.0):
    """与 Q3 同款：信息卡放图右侧。"""
    fig.subplots_adjust(right=1 - pad_right)
    ax.text(
        x, y, text,
        transform=ax.transAxes,
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="0.7", alpha=0.95)
    )


def safe_C(colors, i, fallback="C0"):
    return colors[i % len(colors)] if colors else fallback


# =========================
# Fields
# =========================
POLLUTANTS = ["PM2.5", "PM10", "CO", "NO2", "SO2", "O3"]
METEOS = ["Temperature", "Humidity", "WindSpeed", "Pressure", "Precipitation"]


# =========================
# Params
# =========================
@dataclass
class MCParams:
    n_sites: int = 12
    sigma_lat: float = 0.01
    sigma_lon: float = 0.01
    spatial_decay_k: float = 0.15  # corr = exp(-k * d_km)
    eps: float = 1e-6


@dataclass
class ScoreParams:
    k_neighbors: int = 5
    k_meteo: int = 3
    alpha: float = 0.5
    beta: float = 0.25
    gamma: float = 0.25
    ri_threshold: float = 0.6
    jump_z: float = 3.0  # TCI：跳变阈值（robust z）


# =========================
# Utils
# =========================
def robust_z(x: np.ndarray, eps: float = 1e-9):
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + eps
    return 0.6745 * (x - med) / mad


def minmax01(x: np.ndarray, eps: float = 1e-12):
    x = np.asarray(x, dtype=float)
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or (mx - mn) < eps:
        return np.zeros_like(x, dtype=float)
    return (x - mn) / (mx - mn + eps)


def contiguous_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    segs = []
    in_run = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_run:
            in_run = True
            start = i
        if in_run and (not v):
            segs.append((start, i - 1))
            in_run = False
    if in_run:
        segs.append((start, len(mask) - 1))
    return segs


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def pol_dirname(pol: str) -> str:
    """完全复刻 Q3：把 'PM2.5' -> 'PM25'（去掉点号）"""
    return pol.replace(".", "")


# =========================
# Load Appendix2
# =========================
def load_appendix2(path_xlsx: str, freq: str = "H") -> pd.DataFrame:
    df = pd.read_excel(path_xlsx)

    time_col = "Time" if "Time" in df.columns else None
    if time_col is None:
        for c in ["time", "datetime", "dt", "DateTime", "Timestamp"]:
            if c in df.columns:
                time_col = c
                break
    if time_col is None:
        raise ValueError("Cannot find time column in Appendix2 (e.g., 'Time').")

    df["dt"] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)

    keep_cols = [c for c in (POLLUTANTS + METEOS) if c in df.columns]
    df = df[["dt"] + keep_cols].copy()

    for c in keep_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(how="all", subset=keep_cols).copy()

    if freq and str(freq).lower() != "none":
        df = (df.set_index("dt")
              .resample(freq)
              .mean(numeric_only=True)
              .reset_index())
    return df


def estimate_ar1_rho(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return 0.7
    x0 = x[:-1]
    x1 = x[1:]
    if np.nanstd(x0) < 1e-12 or np.nanstd(x1) < 1e-12:
        return 0.7
    rho = np.corrcoef(x0, x1)[0, 1]
    if not np.isfinite(rho):
        rho = 0.7
    return float(np.clip(rho, 0.2, 0.98))


def extract_stats(base: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    stats = {}
    for c in [col for col in base.columns if col != "dt"]:
        x = base[c].to_numpy(dtype=float)
        mu = float(np.nanmean(x))
        sd = float(np.nanstd(x))
        rho = estimate_ar1_rho(x)
        stats[c] = {"mu": mu, "sd": max(sd, 1e-6), "rho": rho}
    return stats


# =========================
# Monte Carlo multisite
# =========================
def gen_sites_latlon(n_sites: int,
                     center_lat: float = 30.0, center_lon: float = 114.0,
                     sigma_lat: float = 0.01, sigma_lon: float = 0.01,
                     seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lats = center_lat + rng.normal(0, sigma_lat, size=n_sites)
    lons = center_lon + rng.normal(0, sigma_lon, size=n_sites)
    return pd.DataFrame({"site_id": np.arange(n_sites), "lat": lats, "lon": lons})


def build_distance_matrix_km(sites: pd.DataFrame) -> np.ndarray:
    lat = sites["lat"].to_numpy()
    lon = sites["lon"].to_numpy()
    n = len(sites)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        D[i, :] = haversine_km(lat[i], lon[i], lat, lon)
    return D


def spatial_corr_from_distance(D_km: np.ndarray, k: float) -> np.ndarray:
    C = np.exp(-k * D_km)
    np.fill_diagonal(C, 1.0)
    C = (C + C.T) / 2
    C += np.eye(C.shape[0]) * 1e-8
    return C


def simulate_ar1_multivariate(T: int, mu_vec: np.ndarray, sd_vec: np.ndarray,
                              rho: float, corr: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(mu_vec)
    Sigma = corr * np.outer(sd_vec, sd_vec) * (1 - rho ** 2)
    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(Sigma)
        w = np.clip(w, 1e-10, None)
        L = V @ np.diag(np.sqrt(w))

    X = np.zeros((T, n), dtype=float)
    X[0, :] = mu_vec + rng.normal(0, sd_vec, size=n)
    for t in range(1, T):
        z = rng.normal(0, 1, size=n)
        eps = L @ z
        X[t, :] = mu_vec + rho * (X[t - 1, :] - mu_vec) + eps
    return X


def generate_multisite(base: pd.DataFrame, stats: Dict[str, Dict[str, float]],
                       mc: MCParams, seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)

    sites = gen_sites_latlon(
        mc.n_sites, center_lat=30.0, center_lon=114.0,
        sigma_lat=mc.sigma_lat, sigma_lon=mc.sigma_lon,
        seed=seed
    )
    D = build_distance_matrix_km(sites)
    Corr_spatial = spatial_corr_from_distance(D, mc.spatial_decay_k)

    T = len(base)
    out_rows = []

    # meteo: base + site_bias + noise
    meteo_data = {}
    for m in [c for c in METEOS if c in base.columns]:
        x0 = base[m].to_numpy(dtype=float)
        sd = stats[m]["sd"]
        site_offset = rng.normal(0, 0.2 * sd, size=mc.n_sites)
        noise = rng.normal(0, 0.2 * sd, size=(T, mc.n_sites))
        meteo_data[m] = x0[:, None] + site_offset[None, :] + noise

    # pollutants: multivariate AR(1) + spatial corr
    pollutant_data = {}
    for p in [c for c in POLLUTANTS if c in base.columns]:
        mu = stats[p]["mu"]
        sd = stats[p]["sd"]
        rho = stats[p]["rho"]

        mu_vec = np.full(mc.n_sites, mu, dtype=float)
        mu_vec += rng.normal(0, 0.1 * sd, size=mc.n_sites)
        sd_vec = np.full(mc.n_sites, sd, dtype=float) * rng.uniform(0.85, 1.15, size=mc.n_sites)

        X = simulate_ar1_multivariate(
            T, mu_vec, sd_vec, rho, Corr_spatial,
            seed=seed + (hash(p) % 10000)
        )
        X = np.clip(X, 0, None)
        pollutant_data[p] = X

    dt = base["dt"].to_numpy()
    for i in range(mc.n_sites):
        row = pd.DataFrame({"dt": dt, "site_id": i})
        row["lat"] = sites.loc[i, "lat"]
        row["lon"] = sites.loc[i, "lon"]
        for m, arr in meteo_data.items():
            row[m] = arr[:, i]
        for p, arr in pollutant_data.items():
            row[p] = arr[:, i]
        out_rows.append(row)

    sim = pd.concat(out_rows, ignore_index=True)
    return sim, sites, D


# =========================
# Optional anomaly injection
# =========================
def inject_anomaly(sim_long: pd.DataFrame,
                   pollutant_cols: List[str],
                   anom_site: int = 5,
                   anom_start: int = 200,
                   anom_end: int = 400,
                   anom_type: str = "sensitivity",
                   strength: float = 0.5,
                   add_offset: float = 20.0,
                   spike_mag: float = 80.0,
                   seed: int = 0) -> Tuple[pd.DataFrame, Dict]:
    rng = np.random.default_rng(seed)
    rep = {"anom_site": int(anom_site), "anom_start": int(anom_start), "anom_end": int(anom_end), "anom_type": anom_type}

    df = sim_long.copy()
    df["_t"] = df.groupby("site_id").cumcount()
    mask_site = (df["site_id"] == anom_site)

    if anom_type == "spike":
        t0 = int(np.clip((anom_start + anom_end) // 2, 0, df["_t"].max()))
        mask_t = (df["_t"] == t0)
        for p in pollutant_cols:
            sign = rng.choice([-1.0, 1.0])
            df.loc[mask_site & mask_t, p] = np.clip(df.loc[mask_site & mask_t, p] + sign * spike_mag, 0, None)
        rep["spike_t"] = t0
        rep["spike_mag"] = float(spike_mag)
    else:
        mask_t = (df["_t"] >= anom_start) & (df["_t"] <= anom_end)
        if anom_type == "sensitivity":
            for p in pollutant_cols:
                df.loc[mask_site & mask_t, p] = np.clip(df.loc[mask_site & mask_t, p] * (1.0 + strength), 0, None)
            rep["strength"] = float(strength)
        elif anom_type == "zero":
            for p in pollutant_cols:
                df.loc[mask_site & mask_t, p] = np.clip(df.loc[mask_site & mask_t, p] + add_offset, 0, None)
            rep["add_offset"] = float(add_offset)
        else:
            raise ValueError("Unknown anom_type. Use: sensitivity | zero | spike")

    df = df.drop(columns=["_t"])
    return df, rep


# =========================
# SCI / TCI / MCI
# =========================
def neighbors_by_distance(D: np.ndarray, k: int) -> Dict[int, List[int]]:
    n = D.shape[0]
    neigh = {}
    for i in range(n):
        idx = np.argsort(D[i, :])
        idx = idx[idx != i][:k]
        neigh[i] = idx.tolist()
    return neigh


def compute_sci(sim_long: pd.DataFrame, sites: pd.DataFrame, D: np.ndarray,
                pollutant: str, k_neighbors: int, eps: float = 1e-6) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(sites)
    neigh = neighbors_by_distance(D, k_neighbors)
    pivot = sim_long.pivot(index="dt", columns="site_id", values=pollutant).sort_index()

    X = pivot.values
    E = np.zeros_like(X, dtype=float)
    for i in range(n):
        nb = neigh[i]
        ref = np.nanmean(X[:, nb], axis=1)
        E[:, i] = np.abs(X[:, i] - ref) / (np.abs(ref) + eps)

    e_t = pd.DataFrame(E, index=pivot.index, columns=[f"site_{i}" for i in range(n)]).reset_index()
    sci_site = pd.DataFrame({"site_id": np.arange(n), "SCI": np.nanmean(E, axis=0)})
    return sci_site, e_t


def compute_tci(sim_long: pd.DataFrame, pollutant: str, jump_z: float = 3.0) -> pd.DataFrame:
    rows = []
    for sid, g in sim_long.groupby("site_id"):
        x = g.sort_values("dt")[pollutant].to_numpy(dtype=float)
        d = np.diff(x)
        z = np.abs(robust_z(d))
        tci = float(np.nanmean(z > jump_z))
        rows.append({"site_id": int(sid), "TCI": tci})
    return pd.DataFrame(rows)


def compute_mci(sim_long: pd.DataFrame, sites: pd.DataFrame, D: np.ndarray,
                pollutant: str, k_neighbors: int, k_meteo: int,
                eps: float = 1e-6) -> pd.DataFrame:
    n = len(sites)
    neigh = neighbors_by_distance(D, k_neighbors)

    piv_p = sim_long.pivot(index="dt", columns="site_id", values=pollutant).sort_index()
    met_cols = [c for c in METEOS if c in sim_long.columns]
    piv_mets = {m: sim_long.pivot(index="dt", columns="site_id", values=m).sort_index() for m in met_cols}

    met_std = {}
    for m, pv in piv_mets.items():
        v = float(np.nanstd(pv.values))
        met_std[m] = v if v > 1e-9 else 1.0

    Xp = piv_p.values
    T = Xp.shape[0]
    mci = np.zeros(n, dtype=float)

    for i in range(n):
        nb = neigh[i]
        if len(nb) == 0:
            mci[i] = np.nan
            continue
        e_list = []
        for t in range(T):
            dmet = np.zeros(len(nb), dtype=float)
            for mi, j in enumerate(nb):
                s = 0.0
                for m, pv in piv_mets.items():
                    vi = pv.values[t, i]
                    vj = pv.values[t, j]
                    if np.isfinite(vi) and np.isfinite(vj):
                        s += ((vi - vj) / (met_std[m] + eps)) ** 2
                dmet[mi] = s
            k_sel = min(k_meteo, len(nb))
            sel = np.argsort(dmet)[:k_sel]
            nb_sel = [nb[int(s)] for s in sel]
            ref = np.nanmean(Xp[t, nb_sel])
            e = np.abs(Xp[t, i] - ref) / (np.abs(ref) + eps)
            e_list.append(e)
        mci[i] = float(np.nanmean(e_list))

    return pd.DataFrame({"site_id": np.arange(n), "MCI": mci})


# =========================
# RI + detect + localize
# =========================
def compute_ri(sci: pd.DataFrame, tci: pd.DataFrame, mci: pd.DataFrame, sp: ScoreParams) -> pd.DataFrame:
    df = sci.merge(tci, on="site_id", how="left").merge(mci, on="site_id", how="left")

    df["SCI_n"] = minmax01(df["SCI"].to_numpy())
    df["TCI_n"] = minmax01(df["TCI"].to_numpy())
    df["MCI_n"] = minmax01(df["MCI"].to_numpy())

    df["RI"] = 1.0 - (sp.alpha * df["SCI_n"] + sp.beta * df["TCI_n"] + sp.gamma * df["MCI_n"])
    return df


def detect_anomaly_sites(scores: pd.DataFrame, sp: ScoreParams) -> Dict:
    out = {}
    out["by_threshold"] = scores.loc[scores["RI"] < sp.ri_threshold, "site_id"].astype(int).tolist()

    # KMeans optional（同 Q3 的“尽量稳健但不强依赖”风格）
    try:
        from sklearn.cluster import KMeans
        X = scores[["SCI_n", "TCI_n", "MCI_n"]].fillna(0.0).to_numpy()
        km = KMeans(n_clusters=2, random_state=0, n_init="auto")
        lab = km.fit_predict(X)

        tmp = scores.copy()
        tmp["cluster"] = lab
        c_mean = tmp.groupby("cluster")["RI"].mean()
        abnormal_cluster = int(c_mean.idxmin())
        out["by_kmeans"] = tmp.loc[tmp["cluster"] == abnormal_cluster, "site_id"].astype(int).tolist()
        out["kmeans_cluster_mean_RI"] = {str(k): float(v) for k, v in c_mean.items()}
    except Exception as e:
        out["by_kmeans"] = []
        out["kmeans_error"] = str(e)

    return out


def localize_anomaly_window(e_t: pd.DataFrame, site_id: int, min_len: int = 3) -> List[Dict]:
    col = f"site_{site_id}"
    if col not in e_t.columns:
        return []
    s = e_t[col].to_numpy(dtype=float)
    z = np.abs(robust_z(s))
    mask = z > 3.0

    segs = contiguous_segments(mask)
    segs = [seg for seg in segs if (seg[1] - seg[0] + 1) >= min_len]

    dt = pd.to_datetime(e_t["dt"], errors="coerce").to_numpy()
    windows = []
    for a, b in segs:
        windows.append({
            "start_index": int(a),
            "end_index": int(b),
            "start_dt": str(dt[a]),
            "end_dt": str(dt[b]),
            "len": int(b - a + 1),
        })
    return windows


# =========================
# Plots (Q2/Q3 unified style)
# =========================
def plot_bundle_q4(scores: pd.DataFrame,
                   e_t: pd.DataFrame,
                   pollutant: str,
                   outdir_pol: str,
                   colors: List,
                   sp: ScoreParams,
                   detected: Dict,
                   worst_site: int,
                   windows: List[Dict]):
    pol_tag = pol_dirname(pollutant)

    # ---- fig 1: RI by site ----
    srt = scores.sort_values("RI", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    ax.bar(srt["site_id"].astype(str), srt["RI"], color=safe_C(colors, 0))
    ax.set_xlabel("site_id")
    ax.set_ylabel("RI (higher = better)")
    ax.set_title(f"{pollutant} | Reliability Index (RI) by site")

    card = "\n".join([
        f"Target: {pollutant}",
        f"N sites: {len(scores)}",
        f"k_neighbors: {sp.k_neighbors}",
        f"k_meteo: {sp.k_meteo}",
        f"alpha,beta,gamma: {sp.alpha:.2f}, {sp.beta:.2f}, {sp.gamma:.2f}",
        f"RI<thr({sp.ri_threshold}): {len(detected.get('by_threshold', []))}",
        f"Worst site: {worst_site}",
    ])
    info_card_outside(fig, ax, card, pad_right=0.32)
    save_svg(fig, os.path.join(outdir_pol, f"fig_{pol_tag}_RI_by_site.svg"))

    # ---- fig 2: indicators normalized ----
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    ax.plot(scores["site_id"], scores["SCI_n"], marker="o", label="SCI (norm)", color=safe_C(colors, 1))
    ax.plot(scores["site_id"], scores["TCI_n"], marker="o", label="TCI (norm)", color=safe_C(colors, 2))
    ax.plot(scores["site_id"], scores["MCI_n"], marker="o", label="MCI (norm)", color=safe_C(colors, 3))
    ax.set_xlabel("site_id")
    ax.set_ylabel("Normalized indicator (lower=better)")
    ax.set_title(f"{pollutant} | Consistency indicators (normalized)")
    ax.legend()
    save_svg(fig, os.path.join(outdir_pol, f"fig_{pol_tag}_indicators_normalized.svg"))

    # ---- fig 3: spatial deviation series on worst site ----
    col = f"site_{worst_site}"
    if col in e_t.columns:
        tt = pd.to_datetime(e_t["dt"])
        yy = e_t[col].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(11.2, 4.8))
        ax.plot(tt, yy, linewidth=1.4, color=safe_C(colors, 0))
        ax.set_xlabel("Time")
        ax.set_ylabel("Spatial deviation e(t)")
        ax.set_title(f"{pollutant} | Spatial deviation on worst site={worst_site}")

        z = np.abs(robust_z(yy))
        mask = z > 3.0
        if np.any(mask):
            ax.scatter(tt[mask], yy[mask], s=18, color=safe_C(colors, 2), label="outliers (|z|>3)")
            ax.legend()

        # 标注前几段异常窗口（最多3段，避免图太乱）
        if windows:
            for w in windows[:3]:
                try:
                    t0 = pd.to_datetime(w["start_dt"])
                    t1 = pd.to_datetime(w["end_dt"])
                    ax.axvspan(t0, t1, alpha=0.12)
                except Exception:
                    pass

        card2 = "\n".join([
            f"Worst site: {worst_site}",
            f"Windows: {len(windows)}",
            *( [f"#{i+1}: {windows[i]['start_dt']} ~ {windows[i]['end_dt']}" for i in range(min(3, len(windows)))] )
        ])
        info_card_outside(fig, ax, card2, pad_right=0.35)

        save_svg(fig, os.path.join(outdir_pol, f"fig_{pol_tag}_spatial_deviation_worst_site.svg"))


# =========================
# Auto path helpers (PyCharm-friendly)
# =========================
def auto_find_appendix2() -> str:
    candidates = [
        os.path.join("..", "data", "C_Appendix_2.xlsx"),
        os.path.join(".", "data", "C_Appendix_2.xlsx"),
        os.path.join(".", "C_Appendix_2.xlsx"),
        os.path.join(os.path.dirname(__file__), "C_Appendix_2.xlsx"),
        "/mnt/data/C_Appendix_2.xlsx",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Cannot find C_Appendix_2.xlsx. Put it in ../data/ or ./data/ or script folder, or edit main()."
    )


def auto_outdir() -> str:
    # 复刻 Q3：默认用 ../data/q4_output/consistency
    p1 = os.path.join("..", "data", "q4_output", "consistency")
    p2 = os.path.join(".", "data", "q4_output", "consistency")
    if os.path.exists(os.path.join("..", "data")) or os.path.exists(".."):
        return p1
    return p2


# =========================
# Main (one-click)
# =========================
def main():
    # ---- Style ----
    set_mcm_pastel_style(SCHEME)
    colors = get_colors(SCHEME)

    # =====================================================
    # ✅ 你只需要改这里（完全 PyCharm 友好）
    # =====================================================
    APPENDIX2_PATH = auto_find_appendix2()
    OUTDIR = auto_outdir()

    DEFAULT_FREQ = "H"            # 推荐 "H"；若太少可改 "none"
    DEFAULT_N_SITES = 12
    DEFAULT_SEED = 42

    # scoring
    DEFAULT_K_NEIGHBORS = 5
    DEFAULT_K_METEO = 3
    ALPHA, BETA, GAMMA = 0.5, 0.25, 0.25
    RI_THRESHOLD = 0.6

    # 选择污染物： "ALL" 或单个（如 "PM2.5"）
    SCORE_POLLUTANT = "ALL"

    # 注入异常（用于写作展示/验证算法能抓出来）
    INJECT_ANOMALY = True
    ANOM_SITE = 5
    ANOM_TYPE = "sensitivity"     # sensitivity | zero | spike
    ANOM_START = 200
    ANOM_END = 400
    ANOM_STRENGTH = 0.5
    ANOM_OFFSET = 20.0
    ANOM_SPIKE = 80.0
    # =====================================================

    ensure_dir(OUTDIR)

    # ---- Load base ----
    base = load_appendix2(APPENDIX2_PATH, freq=DEFAULT_FREQ)
    if len(base) < 50:
        raise ValueError("After resampling, too few rows. Try DEFAULT_FREQ='none'.")

    stats = extract_stats(base)

    # ---- MC generate ----
    mc = MCParams(n_sites=DEFAULT_N_SITES)
    sim_long, sites, D = generate_multisite(base, stats, mc, seed=DEFAULT_SEED)

    pollutant_cols = [p for p in POLLUTANTS if p in sim_long.columns]

    injected_info = None
    if INJECT_ANOMALY:
        sim_long, injected_info = inject_anomaly(
            sim_long,
            pollutant_cols=pollutant_cols,
            anom_site=ANOM_SITE,
            anom_start=ANOM_START,
            anom_end=ANOM_END,
            anom_type=ANOM_TYPE,
            strength=ANOM_STRENGTH,
            add_offset=ANOM_OFFSET,
            spike_mag=ANOM_SPIKE,
            seed=DEFAULT_SEED + 7
        )

    # ---- Root outputs (Q3-like summary files) ----
    sites.to_csv(os.path.join(OUTDIR, "sites.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(D, columns=[f"site_{i}" for i in range(D.shape[0])]).to_csv(
        os.path.join(OUTDIR, "distance_km.csv"), index=False, encoding="utf-8-sig"
    )
    sim_long.to_csv(os.path.join(OUTDIR, "simulated_multisite_long.csv"), index=False, encoding="utf-8-sig")

    # ---- Select pollutants to score ----
    if str(SCORE_POLLUTANT).upper() == "ALL":
        score_pollutants = pollutant_cols
    else:
        if SCORE_POLLUTANT not in pollutant_cols:
            raise ValueError(f"SCORE_POLLUTANT={SCORE_POLLUTANT} not in data. Available={pollutant_cols}")
        score_pollutants = [SCORE_POLLUTANT]

    sp = ScoreParams(
        k_neighbors=DEFAULT_K_NEIGHBORS,
        k_meteo=DEFAULT_K_METEO,
        alpha=ALPHA, beta=BETA, gamma=GAMMA,
        ri_threshold=RI_THRESHOLD
    )

    # ---- Per-pollutant: compute + save into subdir (exactly like Q3) ----
    all_rows = []
    per_pollutant_reports = {}

    for pol in score_pollutants:
        pol_subdir = os.path.join(OUTDIR, pol_dirname(pol))  # Q3 style
        ensure_dir(pol_subdir)

        sci_site, e_t = compute_sci(sim_long, sites, D, pol, sp.k_neighbors)
        tci_site = compute_tci(sim_long, pol, sp.jump_z)
        mci_site = compute_mci(sim_long, sites, D, pol, sp.k_neighbors, sp.k_meteo)
        scores = compute_ri(sci_site, tci_site, mci_site, sp)
        scores["pollutant"] = pol

        detected = detect_anomaly_sites(scores, sp)
        worst_site = int(scores.sort_values("RI").iloc[0]["site_id"])
        windows = localize_anomaly_window(e_t, worst_site, min_len=3)

        # ---- save per pollutant outputs (Q3-like) ----
        scores_path = os.path.join(pol_subdir, f"scores_by_site_{pol_dirname(pol)}.csv")
        e_t_path = os.path.join(pol_subdir, f"spatial_deviation_{pol_dirname(pol)}.csv")
        rep_path = os.path.join(pol_subdir, f"anomaly_report_{pol_dirname(pol)}.json")

        scores.to_csv(scores_path, index=False, encoding="utf-8-sig")
        e_t.to_csv(e_t_path, index=False, encoding="utf-8-sig")

        rep = {
            "pollutant": pol,
            "n_sites": int(DEFAULT_N_SITES),
            "freq": DEFAULT_FREQ,
            "k_neighbors": int(sp.k_neighbors),
            "k_meteo": int(sp.k_meteo),
            "weights": {"alpha": sp.alpha, "beta": sp.beta, "gamma": sp.gamma},
            "ri_threshold": float(sp.ri_threshold),
            "detected": detected,
            "worst_site_by_RI": worst_site,
            "localized_windows_on_worst_site": windows,
        }
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)

        # ---- figures (Q2/Q3 style) ----
        plot_bundle_q4(
            scores=scores,
            e_t=e_t,
            pollutant=pol,
            outdir_pol=pol_subdir,
            colors=colors,
            sp=sp,
            detected=detected,
            worst_site=worst_site,
            windows=windows
        )

        # collect for root summary
        all_rows.append(scores)
        per_pollutant_reports[pol] = rep

        print(f"[Q4] {pol} done. worst_site={worst_site}, anomalies(thr)={len(detected.get('by_threshold', []))}")

    # ---- Root summary tables (Q3-like) ----
    scores_all = pd.concat(all_rows, ignore_index=True)
    scores_all.to_csv(os.path.join(OUTDIR, "scores_by_site_all_pollutants.csv"), index=False, encoding="utf-8-sig")

    scores_avg = None
    if len(score_pollutants) > 1:
        scores_avg = (scores_all.groupby("site_id")[["SCI", "TCI", "MCI", "SCI_n", "TCI_n", "MCI_n", "RI"]]
                      .mean()
                      .reset_index())
        scores_avg.to_csv(os.path.join(OUTDIR, "scores_by_site_avg_all.csv"), index=False, encoding="utf-8-sig")

    # ---- Root report ----
    report_q4 = {
        "input_appendix2": os.path.abspath(APPENDIX2_PATH),
        "outdir": os.path.abspath(OUTDIR),
        "freq": DEFAULT_FREQ,
        "n_sites": int(DEFAULT_N_SITES),
        "seed": int(DEFAULT_SEED),
        "scoring_pollutants": score_pollutants,
        "global_weights": {"alpha": sp.alpha, "beta": sp.beta, "gamma": sp.gamma},
        "ri_threshold": float(sp.ri_threshold),
        "injected_anomaly": injected_info,
        "per_pollutant": per_pollutant_reports,
    }
    with open(os.path.join(OUTDIR, "anomaly_report_q4.json"), "w", encoding="utf-8") as f:
        json.dump(report_q4, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("Q4 done (Q3-like structure + Q2/Q3 style).")
    print("Appendix2:", os.path.abspath(APPENDIX2_PATH))
    print("Outdir   :", os.path.abspath(OUTDIR))
    print("Root files:",
          "\n - sites.csv\n - distance_km.csv\n - simulated_multisite_long.csv\n"
          " - scores_by_site_all_pollutants.csv\n - anomaly_report_q4.json")
    print("=" * 72)


if __name__ == "__main__":
    main()
