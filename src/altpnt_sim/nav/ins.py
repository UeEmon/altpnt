"""INS 誤差モデル（水平 2D・速度誤差ベース）.

INS 出力速度 = 真速度 + バイアス b(t) + 白色雑音
  b(t) = b0 + ランダムウォーク
航法級 INS の典型ドリフト ~1 NM/h（≈0.5 m/s 相当）を既定値とする。
Schuler 振動・姿勢誤差結合は Phase 2 で拡張予定。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class INSModel:
    bias0_sigma: float = 0.35          # 初期速度バイアス σ [m/s/軸]
    bias_rw: float = 2e-3              # バイアスランダムウォーク [m/s/√s]
    vel_noise: float = 0.05            # 速度白色雑音 σ [m/s]
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(1))

    def __post_init__(self):
        self.bias = self.bias0_sigma * self.rng.standard_normal(2)

    def measure(self, v_true: np.ndarray, dt: float) -> np.ndarray:
        self.bias = self.bias + self.bias_rw * np.sqrt(dt) * self.rng.standard_normal(2)
        return v_true + self.bias + self.vel_noise * self.rng.standard_normal(2)
