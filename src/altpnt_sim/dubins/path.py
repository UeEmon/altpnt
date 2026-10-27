"""Dubins パス（最小旋回半径制約下の最短経路）.

6 種の語 LSL, RSR, LSR, RSL, RLR, LRL を numpy で一括評価して最短を選ぶ。
（引継書の「LSC」は LSL の誤記と解釈。）
定式化は Shkel & Lumelsky (2001) / A. Walker の dubins.c に準拠。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WORDS = ("LSL", "RSR", "LSR", "RSL", "RLR", "LRL")
_TURN = {"L": 1.0, "S": 0.0, "R": -1.0}
TWO_PI = 2.0 * np.pi


def _mod2pi(a):
    return np.mod(a, TWO_PI)


@dataclass
class DubinsPath:
    q0: np.ndarray          # (x, y, heading)
    rho: float              # 旋回半径 [m]
    word: str
    seg: np.ndarray         # 正規化セグメント長 (3,)

    @property
    def length(self) -> float:
        return float(self.seg.sum() * self.rho)

    def sample(self, step: float) -> np.ndarray:
        """弧長 step [m] ごとの (M,3) 配列 [x, y, heading]. 終点を含む."""
        s = np.append(np.arange(0.0, self.length, step), self.length)
        return self.at(s)

    def at(self, s: np.ndarray) -> np.ndarray:
        """弧長 s [m]（配列）における姿勢. ベクトル化."""
        s = np.clip(np.atleast_1d(s) / self.rho, 0.0, self.seg.sum())
        # 各セグメント開始時の正規化姿勢
        starts = np.zeros((3, 3))
        q = np.array([0.0, 0.0, self.q0[2]])
        for i, (ch, L) in enumerate(zip(self.word, self.seg)):
            starts[i] = q
            q = _advance(q[None, :], np.array([L]), _TURN[ch])[0]
        bounds = np.cumsum(self.seg)
        idx = np.searchsorted(bounds, s, side="left").clip(0, 2)
        local = s - np.concatenate([[0.0], bounds[:-1]])[idx]
        out = np.empty((s.size, 3))
        for i, ch in enumerate(self.word):
            m = idx == i
            if m.any():
                out[m] = _advance(np.repeat(starts[i][None], m.sum(), 0), local[m], _TURN[ch])
        out[:, :2] = out[:, :2] * self.rho + self.q0[:2]
        out[:, 2] = _mod2pi(out[:, 2])
        return out


def _advance(q: np.ndarray, t: np.ndarray, k: float) -> np.ndarray:
    """正規化座標（半径1）で姿勢 q を長さ t だけ進める. k=+1:左, -1:右, 0:直進."""
    x, y, th = q[:, 0], q[:, 1], q[:, 2]
    if k == 0.0:
        return np.stack([x + t * np.cos(th), y + t * np.sin(th), th], axis=1)
    return np.stack([
        x + k * (np.sin(th + k * t) - np.sin(th)),
        y + k * (-np.cos(th + k * t) + np.cos(th)),
        th + k * t,
    ], axis=1)


def dubins_all(q0, q1, rho: float) -> np.ndarray:
    """全 6 語の正規化セグメント長 (6,3)。不可能な語は NaN."""
    q0 = np.asarray(q0, float)
    q1 = np.asarray(q1, float)
    dx, dy = q1[0] - q0[0], q1[1] - q0[1]
    d = np.hypot(dx, dy) / rho
    th = np.arctan2(dy, dx) if d > 1e-12 else 0.0
    a, b = _mod2pi(q0[2] - th), _mod2pi(q1[2] - th)
    sa, sb, ca, cb = np.sin(a), np.sin(b), np.cos(a), np.cos(b)
    cab = np.cos(a - b)
    res = np.full((6, 3), np.nan)

    with np.errstate(invalid="ignore"):
        # LSL
        p2 = 2 + d * d - 2 * cab + 2 * d * (sa - sb)
        if p2 >= 0:
            t1 = np.arctan2(cb - ca, d + sa - sb)
            res[0] = [_mod2pi(t1 - a), np.sqrt(p2), _mod2pi(b - t1)]
        # RSR
        p2 = 2 + d * d - 2 * cab + 2 * d * (sb - sa)
        if p2 >= 0:
            t1 = np.arctan2(ca - cb, d - sa + sb)
            res[1] = [_mod2pi(a - t1), np.sqrt(p2), _mod2pi(t1 - b)]
        # LSR
        p2 = -2 + d * d + 2 * cab + 2 * d * (sa + sb)
        if p2 >= 0:
            p = np.sqrt(p2)
            t0 = np.arctan2(-ca - cb, d + sa + sb) - np.arctan2(-2.0, p)
            res[2] = [_mod2pi(t0 - a), p, _mod2pi(t0 - _mod2pi(b))]
        # RSL
        p2 = -2 + d * d + 2 * cab - 2 * d * (sa + sb)
        if p2 >= 0:
            p = np.sqrt(p2)
            t0 = np.arctan2(ca + cb, d - sa - sb) - np.arctan2(2.0, p)
            res[3] = [_mod2pi(a - t0), p, _mod2pi(b - t0)]
        # RLR
        c = (6.0 - d * d + 2 * cab + 2 * d * (sa - sb)) / 8.0
        if abs(c) <= 1:
            phi = np.arctan2(ca - cb, d - sa + sb)
            p = _mod2pi(TWO_PI - np.arccos(c))
            t = _mod2pi(a - phi + _mod2pi(p / 2.0))
            res[4] = [t, p, _mod2pi(a - b - t + _mod2pi(p))]
        # LRL
        c = (6.0 - d * d + 2 * cab + 2 * d * (sb - sa)) / 8.0
        if abs(c) <= 1:
            phi = np.arctan2(ca - cb, d + sa - sb)
            p = _mod2pi(TWO_PI - np.arccos(c))
            t = _mod2pi(-a - phi + p / 2.0)
            res[5] = [t, p, _mod2pi(_mod2pi(b) - a - t + _mod2pi(p))]
    return res


def dubins_shortest(q0, q1, rho: float) -> DubinsPath:
    segs = dubins_all(q0, q1, rho)
    total = np.nansum(segs, axis=1)
    total[np.isnan(segs).any(axis=1)] = np.inf
    k = int(np.argmin(total))
    return DubinsPath(np.asarray(q0, float), rho, WORDS[k], segs[k])
