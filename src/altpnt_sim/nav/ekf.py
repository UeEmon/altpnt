"""INS バウンディング用 EKF.

状態 x = [px, py, bx, by]（位置 + INS 速度バイアス）
  予測:  p_{k+1} = p_k + (v_ins - b) dt,   b_{k+1} = b_k
  観測:  任意の非線形 h(p)（HF Rx-TDOA、ブイ網 TDOA など）を update() に渡す
カイ二乗ゲートで外れ値（位相スリップ・マルチパス）を棄却する。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import chi2


@dataclass
class InsBoundingEKF:
    x: np.ndarray
    P: np.ndarray
    vel_noise: float = 0.05
    bias_rw: float = 2e-3
    gate_prob: float = 0.999

    @classmethod
    def init(cls, p0, pos_sigma: float, bias_sigma: float, **kw) -> "InsBoundingEKF":
        x = np.r_[np.asarray(p0, float), 0.0, 0.0]
        P = np.diag([pos_sigma**2] * 2 + [bias_sigma**2] * 2)
        return cls(x, P, **kw)

    @property
    def pos(self) -> np.ndarray:
        return self.x[:2]

    def predict(self, v_ins: np.ndarray, dt: float) -> None:
        F = np.eye(4)
        F[0, 2] = F[1, 3] = -dt
        self.x[:2] += (v_ins - self.x[2:]) * dt
        Q = np.diag([(self.vel_noise * dt) ** 2] * 2 + [self.bias_rw**2 * dt] * 2)
        self.P = F @ self.P @ F.T + Q

    def update(self, z: np.ndarray, h: Callable[[np.ndarray], np.ndarray],
               Hp: Callable[[np.ndarray], np.ndarray], R: np.ndarray) -> bool:
        """z: 観測, h(p)->予測観測, Hp(p)->∂h/∂p (K,2). 受理なら True."""
        p = self.pos
        y = z - h(p)
        H = np.zeros((len(z), 4))
        H[:, :2] = Hp(p)
        S = H @ self.P @ H.T + R
        d2 = float(y @ np.linalg.solve(S, y))
        if d2 > chi2.ppf(self.gate_prob, len(z)):
            return False
        K = np.linalg.solve(S, H @ self.P).T
        self.x = self.x + K @ y
        IKH = np.eye(4) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T      # Joseph 形式
        return True
