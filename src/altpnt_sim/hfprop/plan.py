"""送信所×受信点×時刻の周波数計画と、使える送信所の組からの位置誤差.

距離誤差モデル（1 送信所あたり、km）:
  地表波（距離 <= ground_wave_km）: σ = ground_wave_sigma_km（VOACAP は地表波を扱わないので固定値）
  上空波: σ² = (dP/dh' · σ_h)² + σ_toa²,  さらにマルチパス確率 MPROB で (1 + 2·MPROB) 倍
    P(D, h') … 鏡面反射モデルの群路長（h' は VOACAP の仮想反射高度 V HITE、n ホップ）
    σ_h      … 反射高度の推定誤差。補正方式（tier）で変わる: model 20 km / realtime 5 km
    tier=differential … 近くの参照点で差分補正した残差 σ_diff を距離誤差として直接使う
    σ_toa    … 包絡線 TOA の CRLB（帯域内 SNR から）
位置誤差: 球面上の方位から作る TDOA ヤコビアン（行を dP/dD でスケール）と
  各送信所の σ から重み付き最小二乗の共分散を作り、sqrt(trace) を返す（drms, km）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..constants import C
from ..nav.tdoa import tdoa_cov

R_KM = 6371.0
H_SIGMA_KM = {"model": 20.0, "realtime": 5.0}


# ------------------------------------------------------------------ 球面幾何
def gc_dist_az(lat1, lon1, lat2, lon2):
    """大圏距離 [km] と 1→2 の方位 [rad]."""
    p1, l1, p2, l2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dl = l2 - l1
    c = math.acos(max(-1.0, min(1.0, math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(dl))))
    az = math.atan2(math.sin(dl) * math.cos(p2), math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl))
    return c * R_KM, az


def destination(lat, lon, bearing_deg, dist_km):
    p, l, b, d = math.radians(lat), math.radians(lon), math.radians(bearing_deg), dist_km / R_KM
    p2 = math.asin(math.sin(p) * math.cos(d) + math.cos(p) * math.sin(d) * math.cos(b))
    l2 = l + math.atan2(math.sin(b) * math.sin(d) * math.cos(p), math.cos(d) - math.sin(p) * math.sin(p2))
    return math.degrees(p2), (math.degrees(l2) + 540) % 360 - 180


def group_path_km(D: float, h: float, hops: int = 1) -> float:
    """鏡面反射（球面）モデルの群路長."""
    th = D / (2 * hops * R_KM)
    s = math.sqrt(R_KM**2 + (R_KM + h) ** 2 - 2 * R_KM * (R_KM + h) * math.cos(th))
    return 2 * hops * s


def hops_of(mode: str | None) -> int:
    try:
        return max(1, int((mode or "1").strip()[0]))
    except ValueError:
        return 1


# ------------------------------------------------------------------ リンク計画
@dataclass
class LinkHour:
    hour_utc: int
    usable: bool
    kind: str                 # "ground" | "sky" | "none"
    freq_mhz: float | None
    rel: float | None
    snr_dbhz: float | None
    mode: str | None
    vhite_km: float | None
    mprob: float | None
    muf_mhz: float | None
    dist_km: float
    dPdD: float               # 群路長の地表距離微分（ヤコビアンのスケール）
    dPdh: float               # 群路長の反射高度微分


def choose_row(rows: list[dict], rel_min: float) -> dict | None:
    """信頼度が rel_min 以上の候補から、REL − 0.2·MPROB が最大、同点なら SNR が高い周波数."""
    ok = [r for r in rows if (r.get("rel") or 0) >= rel_min and r.get("vhite_km")]
    if not ok:
        return None
    return max(ok, key=lambda r: ((r.get("rel") or 0) - 0.2 * (r.get("mprob") or 0), r.get("snr_dbhz") or -999))


def plan_link(hours: list[dict], dist_km: float, rel_min: float = 0.8, ground_wave_km: float = 300.0,
              muf_factor: float = 1.0) -> list[LinkHour]:
    """muf_factor < 1 は電離層嵐で MUF が下がった状態: MUF×muf_factor を超える周波数は使わない."""
    out = []
    for b in hours:
        if dist_km <= ground_wave_km:
            out.append(LinkHour(b["hour_utc"], True, "ground", None, None, None, None, None, None, b["muf_mhz"],
                                dist_km, 1.0, 0.0))
            continue
        rows = b["rows"] if muf_factor >= 1 else [r for r in b["rows"] if r["freq_mhz"] <= muf_factor * b["muf_mhz"]]
        r = choose_row(rows, rel_min)
        if r is None:
            out.append(LinkHour(b["hour_utc"], False, "none", None, None, None, None, None, None, b["muf_mhz"],
                                dist_km, 1.0, 0.0))
            continue
        h, n = r["vhite_km"], hops_of(r.get("mode"))
        dPdD = (group_path_km(dist_km + 1, h, n) - group_path_km(dist_km - 1, h, n)) / 2
        dPdh = (group_path_km(dist_km, h + 1, n) - group_path_km(dist_km, h - 1, n)) / 2
        out.append(LinkHour(b["hour_utc"], True, "sky", r["freq_mhz"], r["rel"], r.get("snr_dbhz"), r.get("mode"),
                            h, r.get("mprob") or 0.0, b["muf_mhz"], dist_km, dPdD, dPdh))
    return out


def range_sigma_km(lh: LinkHour, tier: str, bandwidth_hz: float = 5000.0, integ_s: float = 1.0,
                   ground_wave_sigma_km: float = 0.3, diff_sigma_km: float = 0.2) -> float:
    if lh.kind == "ground":
        return ground_wave_sigma_km
    inband = (lh.snr_dbhz or 0) - 10 * math.log10(bandwidth_hz)
    beta = bandwidth_hz / math.sqrt(12)
    toa = C / (2 * math.pi * beta * math.sqrt(2 * 10 ** (inband / 10) * bandwidth_hz * integ_s)) / 1e3
    iono = diff_sigma_km if tier == "differential" else lh.dPdh * H_SIGMA_KM[tier]
    return math.hypot(iono, toa) * (1 + 2 * (lh.mprob or 0))


# ------------------------------------------------------------------ 位置誤差
def position_error(rx: tuple[float, float], stations: list[tuple[float, float]], links: list[LinkHour],
                   tier: str, **kw) -> tuple[float, int]:
    """(drms 位置誤差 km, 使えた送信所数). 3 局未満なら (nan, n)."""
    idx = [i for i, lh in enumerate(links) if lh.usable]
    if len(idx) < 3:
        return float("nan"), len(idx)
    rows, sig = [], []
    for i in idx:
        _, az = gc_dist_az(rx[0], rx[1], *stations[i])
        g = -links[i].dPdD * np.array([math.sin(az), math.cos(az)])   # 受信点が動いたときの群路長の勾配（東, 北）
        rows.append(g)
        sig.append(range_sigma_km(links[i], tier, **kw))
    G = np.array(rows)
    D = np.delete(np.eye(len(idx)), 0, 0)
    D[:, 0] = -1
    H = D @ G
    Cz = tdoa_cov(len(idx), np.array(sig))
    try:
        P = np.linalg.inv(H.T @ np.linalg.solve(Cz, H))
    except np.linalg.LinAlgError:
        return float("inf"), len(idx)
    return float(math.sqrt(max(np.trace(P), 0.0))), len(idx)
