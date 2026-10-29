"""宇宙天気・妨害イベント（長距離シナリオ用）.

events は GPS ロスト後の経過時間 [h] で指定する:
  - {type: flare, start_h, duration_h}
      太陽フレア（デリンジャー現象）: 昼側の経路は HF が D 層で吸収されて使えない。
      経路の中点で太陽天頂角が 90° 未満なら昼側とみなす。
  - {type: storm, start_h, duration_h, ramp_h, dh_km, muf_factor, sigma_factor}
      電離層嵐（負相）: 反射高度が dh_km 上がり（EKF は知らない系統誤差）、MUF が muf_factor 倍に下がり、
      ランダムな残差も sigma_factor 倍に増える。立ち上がり・回復は ramp_h かけて直線的に変化する。
  - {type: outage, station, start_h, duration_h}
      送信所の停止（妨害・障害）。
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

# 補正方式ごとに嵐の系統誤差がどれだけ残るか（リアルタイム推定は一部を追従し、近くの参照ブイ差分はほぼ打ち消す）
STORM_LEAK = {"model": 1.0, "realtime": 0.25, "differential": 0.05}


def subsolar_point(when: datetime) -> tuple[float, float]:
    """太陽直下点（緯度, 経度）。均時差は無視（誤差 ±4° 程度）."""
    doy = when.timetuple().tm_yday
    dec = 23.44 * math.sin(2 * math.pi * (284 + doy) / 365.0)
    ut = when.hour + when.minute / 60 + when.second / 3600
    lon = (12.0 - ut) * 15.0
    return dec, (lon + 540) % 360 - 180


def sunlit(lat, lon, when: datetime) -> bool:
    slat, slon = map(math.radians, subsolar_point(when))
    p, l = math.radians(lat), math.radians(lon)
    return math.sin(p) * math.sin(slat) + math.cos(p) * math.cos(slat) * math.cos(l - slon) > 0.0


def gc_midpoint(lat1, lon1, lat2, lon2):
    p1, l1, p2, l2 = map(math.radians, (lat1, lon1, lat2, lon2))
    bx, by = math.cos(p2) * math.cos(l2 - l1), math.cos(p2) * math.sin(l2 - l1)
    p = math.atan2(math.sin(p1) + math.sin(p2), math.hypot(math.cos(p1) + bx, by))
    l = l1 + math.atan2(by, math.cos(p1) + bx)
    return math.degrees(p), (math.degrees(l) + 540) % 360 - 180


class Timeline:
    def __init__(self, events: list[dict] | None, t: np.ndarray, names: list[str], year: int, month: int,
                 depart_utc: float):
        self.events = events or []
        self.t = t
        self.names = names
        self.t0 = datetime(year, month, 15, tzinfo=timezone.utc).timestamp() + depart_utc * 3600
        th = t / 3600.0
        self.storm = np.zeros(len(t))
        self.storm_cfg = None
        self.flare = np.zeros(len(t), bool)
        self.outage = np.zeros((len(t), len(names)), bool)
        for ev in self.events:
            a, d = ev["start_h"], ev["duration_h"]
            if ev["type"] == "storm":
                self.storm_cfg = ev
                r = max(ev.get("ramp_h", 1.0), 1e-6)
                up = np.clip((th - a) / r, 0, 1)
                down = np.clip((a + d - th) / r, 0, 1)
                self.storm = np.maximum(self.storm, np.minimum(up, down))
            elif ev["type"] == "flare":
                self.flare |= (th >= a) & (th < a + d)
            elif ev["type"] == "outage":
                self.outage[(th >= a) & (th < a + d), names.index(ev["station"])] = True
            else:
                raise ValueError(f"未知のイベント: {ev['type']}")

    def when(self, i: int) -> datetime:
        return datetime.fromtimestamp(self.t0 + self.t[i], tz=timezone.utc)

    def blocked(self, i: int, s: int, lat: float, lon: float, st: tuple[float, float], kind: str) -> bool:
        """送信所 s が時刻 i に使えないか（停止、またはフレア中の昼側の上空波経路）."""
        if self.outage[i, s]:
            return True
        if self.flare[i] and kind == "sky":
            return sunlit(*gc_midpoint(st[0], st[1], lat, lon), self.when(i))
        return False

    def storm_bias_m(self, i: int, dPdh: float, tier: str) -> float:
        if self.storm_cfg is None or self.storm[i] == 0:
            return 0.0
        return self.storm[i] * self.storm_cfg.get("dh_km", 40.0) * 1e3 * dPdh * STORM_LEAK[tier]

    def sigma_factor(self, i: int) -> float:
        if self.storm_cfg is None:
            return 1.0
        return 1.0 + (self.storm_cfg.get("sigma_factor", 2.0) - 1.0) * self.storm[i]
