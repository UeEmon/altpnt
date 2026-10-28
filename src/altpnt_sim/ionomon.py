"""電離層の監視網（送信所の近くに置いた受信局で、ほかの送信所の電波を受けて電離層を観測し、補正値を作る）.

仕組み
  - 各送信所の近くに受信局を置く（位置は既知）。受信局は他局の送信を受信し、群路長を測る
  - 送信所 A・B の両方に受信局があれば、A→B と B→A を両方測れる（相互観測）
      和 → 経路の群路長（送信所の時計のずれが消える）→ 経路の反射点（中点）での反射高さのずれ
      差 → 送信所 A・B の時計のずれ（HF による双方向の時刻比較。経路の相反性が前提）
  - 反射高さのずれは地理的に相関する場とみなし、観測した経路の中点の値から、
    ブイ ⇔ 各送信所の経路の中点での値を推定する（クリギング）。これが補正値
  - 補正値は更新間隔ごとに作り、遅延（配信にかかる時間）のあとでブイ網の推定に使う

モデル
  反射高さのずれ δh(p, t) = σh × 単位の場。空間相関 exp(−距離 / L)、時間相関 exp(−Δt / τ)（Gauss-Markov）
  観測点は経路の大圏中点（2 ホップ以上でも中点で代表させる簡略化）
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .hfprop.plan import R_KM, group_path_km


def midpoint(a, b):
    """大圏の中点 (lat, lon)."""
    p1, l1, p2, l2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    bx = math.cos(p2) * math.cos(l2 - l1)
    by = math.cos(p2) * math.sin(l2 - l1)
    p = math.atan2(math.sin(p1) + math.sin(p2), math.sqrt((math.cos(p1) + bx) ** 2 + by**2))
    lo = l1 + math.atan2(by, math.cos(p1) + bx)
    return math.degrees(p), (math.degrees(lo) + 540) % 360 - 180


def gc_km(a, b):
    p1, l1, p2, l2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin((l2 - l1) / 2) ** 2
    return 2 * R_KM * math.asin(math.sqrt(min(1.0, h)))


def dPdh(d_km, h_km=300.0):
    """群路長の反射高さに対する感度（鏡面反射モデル、ホップ数は 3,500 km ごと）."""
    hops = max(1, math.ceil(d_km / 3500.0))
    return (group_path_km(d_km, h_km + 1.0, hops) - group_path_km(d_km, h_km - 1.0, hops)) / 2.0


@dataclass
class MonitorPath:
    a: int
    b: int
    dist_km: float
    mid: tuple
    dPdh: float
    reciprocal: bool


def monitor_paths(stations, names, sites, min_km=400.0, max_km=4000.0, rx_only=None):
    """監視経路. sites = 受信局を併設した送信所の名前、rx_only = {名前: (lat, lon)} 受信だけの局（送信所なし）.

    送信所どうしは、どちらかに受信局があれば経路を観測できる（両方にあれば相互観測）。
    受信だけの局は全送信所を受信する（片方向。送信所の時計は同期済みとみなす）。
    """
    have = {names.index(s) for s in sites if s in names}
    out = []
    for a, b in combinations(range(len(stations)), 2):
        if a not in have and b not in have:
            continue
        d = gc_km(stations[a], stations[b])
        if not (min_km <= d <= max_km):
            continue
        out.append(MonitorPath(a, b, d, midpoint(stations[a], stations[b]), dPdh(d), a in have and b in have))
    extra = {nm: ll for nm, ll in (rx_only or {}).items() if not (nm in names and names.index(nm) in have)}
    for k, (nm, ll) in enumerate(extra.items()):   # 送信所に併設済みの受信局は二重に数えない
        for a in range(len(stations)):
            d = gc_km(stations[a], ll)
            if min_km <= d <= max_km:
                out.append(MonitorPath(a, -1 - k, d, midpoint(stations[a], ll), dPdh(d), False))
    return out


@dataclass
class MonitorCorrector:
    """ブイ経路 N 本と監視経路の、反射高さのずれ（単位化）の場と、補正の計算."""
    buoy_mid: list
    paths: list
    corr_km: float
    sigma_h_km: float
    tau_s: float
    dt: float
    T: int
    sigma_obs_m: float = 100.0
    update_s: float = 60.0
    latency_s: float = 60.0
    seed: int = 0
    Z: np.ndarray = field(init=False)
    Y: np.ndarray = field(init=False)

    def __post_init__(self):
        pts = list(self.buoy_mid) + [p.mid for p in self.paths]
        n = len(pts)
        D = np.array([[gc_km(pts[i], pts[j]) for j in range(n)] for i in range(n)])
        self.C = np.exp(-D / self.corr_km)
        self.nb = len(self.buoy_mid)
        Lc = np.linalg.cholesky(self.C + 1e-9 * np.eye(n))
        a = math.exp(-self.dt / self.tau_s)
        rng = np.random.default_rng(self.seed)
        W = np.empty((self.T, n)); w = rng.standard_normal(n)
        drive = rng.standard_normal((self.T, n))
        for i in range(self.T):
            W[i] = w
            w = a * w + math.sqrt(1 - a * a) * drive[i]
        self.Z = W @ Lc.T                                  # 単位化した反射高さのずれ（列: ブイ経路 → 監視経路）
        # 監視経路の観測雑音（単位化）: 群路長の測定誤差 / (dP/dh × σh)。相互観測なら 2 回の平均
        nm = np.array([self.sigma_obs_m / 1e3 / (p.dPdh * self.sigma_h_km) / (math.sqrt(2) if p.reciprocal else 1.0)
                       for p in self.paths])
        self.Rm = np.diag(nm**2)
        self.Y = self.Z[:, self.nb:] + rng.standard_normal((self.T, len(self.paths))) * nm
        Cbm = self.C[:self.nb, self.nb:]
        Cmm = self.C[self.nb:, self.nb:]
        self.K = Cbm @ np.linalg.inv(Cmm + self.Rm) if self.paths else np.zeros((self.nb, 0))
        self.gain = np.diag(self.K @ Cbm.T) if self.paths else np.zeros(self.nb)   # 推定できる分散の割合
        self.k_upd = max(1, int(round(self.update_s / self.dt)))
        self.k_lat = int(round(self.latency_s / self.dt))
        age = self.latency_s + self.update_s / 2
        self.resid_unit = np.sqrt(np.clip(1 - math.exp(-2 * age / self.tau_s) * self.gain, 1e-6, 1.0))

    def correction(self, i):
        """時刻 i にブイ網の推定で使える補正値（単位化、ブイ経路ごと）. まだ届いていなければ None."""
        i_obs = ((i - self.k_lat) // self.k_upd) * self.k_upd
        if i_obs < 0 or not self.paths:
            return None
        a = math.exp(-(i - i_obs) * self.dt / self.tau_s)
        return a * (self.K @ self.Y[i_obs])

    def buoy_truth(self, i):
        return self.Z[i, :self.nb]

    def message_bps(self, n_values=None, bits_per_value=16, header_bits=64):
        """補正値の放送に必要なビットレート（ブイ経路ごとの値、または格子点の値を更新間隔ごとに送る）."""
        n = n_values if n_values is not None else self.nb
        return (header_bits + n * (bits_per_value + 8)) / self.update_s
