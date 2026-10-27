"""海洋漂流エンジン.

海面漂流ソノブイ群の移動を計算する。流速場は
  一様海流 + 渦（ガウス渦）+ 潮流（M2 分潮の回転楕円）+ 風圧流（リーウェイ）
の重ね合わせとし、未解像の乱流は確率的ランダムウォーク（拡散係数 K）で表す。
全ブイを (N,2) 配列で一括処理する（ベクトル化）。

aqua-drift-simulator の潮流場は drift/aqua.py の AquaDriftAdapter（同じ velocity(xy, t) を提供）で使う。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

M2_PERIOD_S = 12.4206 * 3600.0


class VelocityField(Protocol):
    def velocity(self, xy: np.ndarray, t: float) -> np.ndarray:  # (N,2) -> (N,2) [m/s]
        ...


@dataclass
class OceanCurrentField:
    mean_current: tuple[float, float] = (0.25, 0.10)      # [m/s]
    eddy_centers: np.ndarray = field(default_factory=lambda: np.array([[20e3, -15e3]]))
    eddy_strength: np.ndarray = field(default_factory=lambda: np.array([0.4]))   # 最大接線速度 [m/s]（正=反時計回り）
    eddy_radius: np.ndarray = field(default_factory=lambda: np.array([25e3]))    # [m]
    tidal_amp: tuple[float, float] = (0.15, 0.06)         # 長軸・短軸 [m/s]
    tidal_phase: float = 0.0
    wind: tuple[float, float] = (8.0, 3.0)                # 海上風 [m/s]
    leeway: float = 0.03                                  # 風圧流係数（ブイ形状依存）

    def velocity(self, xy: np.ndarray, t: float) -> np.ndarray:
        xy = np.atleast_2d(xy)
        v = np.broadcast_to(np.asarray(self.mean_current), xy.shape).copy()

        # ガウス渦: v_theta(r) = Vmax * (r/R) * exp(0.5*(1 - (r/R)^2))  （r=R で最大）
        d = xy[:, None, :] - self.eddy_centers[None, :, :]          # (N,E,2)
        r = np.linalg.norm(d, axis=-1) + 1e-9                         # (N,E)
        rn = r / self.eddy_radius[None, :]
        vt = self.eddy_strength[None, :] * rn * np.exp(0.5 * (1.0 - rn**2))
        perp = np.stack([-d[..., 1], d[..., 0]], axis=-1) / r[..., None]
        v += np.sum(vt[..., None] * perp, axis=1)

        # 潮流（回転楕円）
        w = 2.0 * np.pi / M2_PERIOD_S
        ph = w * t + self.tidal_phase
        v += np.array([self.tidal_amp[0] * np.cos(ph), self.tidal_amp[1] * np.sin(ph)])

        # 風圧流
        v += self.leeway * np.asarray(self.wind)
        return v


@dataclass
class DriftSimulator:
    field: VelocityField
    diffusivity: float = 5.0      # 水平拡散係数 K [m^2/s]
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def step(self, xy: np.ndarray, t: float, dt: float) -> np.ndarray:
        """RK2（中点法）+ ランダムウォーク: dx = v dt + sqrt(2K dt) * N(0,1)."""
        k1 = self.field.velocity(xy, t)
        k2 = self.field.velocity(xy + 0.5 * dt * k1, t + 0.5 * dt)
        noise = np.sqrt(2.0 * self.diffusivity * dt) * self.rng.standard_normal(xy.shape)
        return xy + dt * k2 + noise

    def run(self, xy0: np.ndarray, t0: float, t1: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """(T,) 時刻と (T,N,2) 軌跡を返す."""
        ts = np.arange(t0, t1 + 1e-9, dt)
        out = np.empty((ts.size,) + np.shape(xy0))
        xy = np.array(xy0, dtype=float)
        for i, t in enumerate(ts):
            out[i] = xy
            xy = self.step(xy, t, dt)
        return ts, out
