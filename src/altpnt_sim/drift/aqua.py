"""aqua-drift-simulator 連携.

https://github.com/UeEmon/aqua-drift-simulator の backend パッケージ `aqua_drift` を使う。
  pip install -e ".[aqua]"    （または backend ディレクトリを pip install -e）

1. AquaDriftAdapter
   aqua_drift の潮流場 v(p) = base + G (p - p_ref)（p は NM, 速度は kt; CurrentFieldConfig）を
   本シミュレータの VelocityField（局所 ENU [m] → [m/s]）として提供する。
   aqua_drift.physics.current_at は 1 点ずつの純 Python 計算なので、場が線形であることを使い
   同じ係数で numpy 一括評価する（tests/test_aqua.py で current_at と一致を確認）。
   aqua_drift の観測者は潮流のみで流れる（風圧流・拡散なし）。leeway > 0 で風圧流を追加できる。

2. AquaDropModel
   aqua_drift.wind の高度別風プロファイルと自由落下（重力 + 2 乗空気抵抗）で、
   投下点（航空機位置・対地速度・高度）から着水点と落下時間を求める。
   投下補正は aqua_drift と同じく「直前の投下の着水点から推定した平均風」で次の投下位置をずらす。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # aqua_drift は任意依存
    from aqua_drift import physics as _aphys
    from aqua_drift import wind as _awind
    from aqua_drift.models import CurrentFieldConfig, Position, WindConfig
    HAVE_AQUA = True
except ImportError:  # pragma: no cover
    HAVE_AQUA = False

KT = 0.5144444444444445
NM = 1852.0
FT = 0.3048


def _require():
    if not HAVE_AQUA:
        raise ImportError("aqua_drift が見つかりません: pip install -e '.[aqua]' を実行してください")


class AquaDriftAdapter:
    """aqua_drift の潮流場を局所 ENU [m] の VelocityField として包む."""

    def __init__(self, current_field: "CurrentFieldConfig | dict | None" = None,
                 origin_latlon: tuple[float, float] | None = None, depth_m: float = 0.0,
                 leeway: float = 0.0, wind_mps: tuple[float, float] = (0.0, 0.0)):
        _require()
        if isinstance(current_field, dict):
            current_field = CurrentFieldConfig.model_validate(current_field)
        self.field = current_field or CurrentFieldConfig()
        ref = self.field.reference_position
        lat, lon = origin_latlon or (ref.latitude, ref.longitude)
        self.origin = Position(latitude=lat, longitude=lon, depth_ft=depth_m / FT)
        e, n, d = _aphys.local_offset_m(ref, self.origin)          # ENU 原点の p_ref からの位置
        self._off_nm = np.array([e, n, d]) / NM
        b = self.field.base_velocity
        self._base_kt = np.array([b.east_kt, b.north_kt])
        self._G = np.asarray(self.field.gradient_per_nm, float)    # (3,3) kt/NM
        self._leeway_v = leeway * np.asarray(wind_mps, float)

    def velocity(self, xy: np.ndarray, t: float = 0.0) -> np.ndarray:
        xy = np.atleast_2d(xy)
        p_nm = np.column_stack([xy / NM, np.zeros(len(xy))]) + self._off_nm
        v_kt = self._base_kt + p_nm @ self._G[:2, :].T
        return v_kt * KT + self._leeway_v

    def to_latlon(self, xy: np.ndarray) -> np.ndarray:
        """局所 ENU [m] → (lat, lon)（aqua_drift.physics.move_position と同じ球面近似）."""
        xy = np.atleast_2d(xy)
        lat0 = np.radians(self.origin.latitude)
        lat = self.origin.latitude + np.degrees(xy[:, 1] / _aphys.EARTH_RADIUS_M)
        lon = self.origin.longitude + np.degrees(xy[:, 0] / (_aphys.EARTH_RADIUS_M * np.cos(lat0)))
        return np.column_stack([lat, lon])


@dataclass
class DropResult:
    release_xy: np.ndarray      # (N,2) 実際に放出した位置
    release_t: np.ndarray       # (N,)
    splash_xy: np.ndarray       # (N,2) 着水点（真値）
    splash_t: np.ndarray        # (N,)
    miss_m: np.ndarray          # (N,) 予定投下点からの着水ずれ
    wind_est_mps: np.ndarray    # (N,2) 各投下で補正に使った平均風推定


class AquaDropModel:
    """aqua_drift.wind による投下・自由落下・投下補正."""

    def __init__(self, wind_config: "WindConfig | dict | None" = None, release_correction: bool = True,
                 splash_obs_sigma_m: float = 100.0, rng: np.random.Generator | None = None):
        _require()
        if isinstance(wind_config, dict):
            wind_config = WindConfig.model_validate(wind_config)
        self.profile = _awind.WindProfile(wind_config or WindConfig())
        self.correction = release_correction
        self.splash_obs_sigma = splash_obs_sigma_m
        self.rng = rng or np.random.default_rng(0)

    def fall(self, alt_m: float, v_ground: np.ndarray, wind_mps: tuple[float, float] | None = None):
        wind = self.profile.at if wind_mps is None else _awind.uniform(*wind_mps)
        f = _awind.fall(alt_m / FT, float(v_ground[0]), float(v_ground[1]), wind, self.profile.terminal_mps)
        return np.array([f.east_m, f.north_m]), f.time_s

    def surface_wind_mps(self) -> tuple[float, float]:
        return self.profile.at(0.0)

    def deploy(self, traj: np.ndarray, planned_xy: np.ndarray, t_over: np.ndarray, alt_m: float) -> DropResult:
        """traj: (T,5) [t,x,y,vx,vy] 等間隔. planned_xy: 予定投下点, t_over: 予定点の通過時刻.

        補正: 推定平均風での落下変位 d を求め、放出を d の進行方向成分だけ早める（横風成分は
        経路変更が必要なので残る）。推定平均風は直前の投下の着水点（観測 σ=splash_obs_sigma）から
        aqua_drift.wind.estimate_mean_wind で求め、初回は無風を仮定する。
        """
        t = traj[:, 0]
        dt = t[1] - t[0]
        N = len(planned_xy)
        out = {k: np.zeros((N, 2)) for k in ("rel", "spl", "west")}
        rt, st = np.zeros(N), np.zeros(N)
        w_est = np.zeros(2)
        for k in range(N):
            i0 = int(round(t_over[k] / dt))
            v = traj[i0, 3:5]
            u = v / np.linalg.norm(v)
            ti = t_over[k]
            if self.correction:
                d_est, _ = self.fall(alt_m, v, tuple(w_est))
                ti = t_over[k] - float(d_est @ u) / np.linalg.norm(v)
            i = int(np.clip(round(ti / dt), 0, len(t) - 1))
            rel, vg = traj[i, 1:3], traj[i, 3:5]
            d_true, tf = self.fall(alt_m, vg)
            out["rel"][k], out["spl"][k], out["west"][k] = rel, rel + d_true, w_est
            rt[k], st[k] = t[i], t[i] + tf
            # 着水点の観測（ブイ初回 HF 測位相当）から平均風を推定し次の投下に使う
            obs = d_true + self.splash_obs_sigma * self.rng.standard_normal(2)
            mw = _awind.estimate_mean_wind(alt_m / FT, float(vg[0]), float(vg[1]), float(obs[0]),
                                           float(obs[1]), self.profile.terminal_mps)
            w_est = np.array([mw.east_mps, mw.north_mps])
        miss = np.linalg.norm(out["spl"] - planned_xy, axis=1)
        return DropResult(out["rel"], rt, out["spl"], st, miss, out["west"])
