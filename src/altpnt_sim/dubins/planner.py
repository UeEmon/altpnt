"""PNT グリッド展開・キャリブレーション飛行計画.

対潜用ソノブイ（ASW）と位置補正用 G サイズ RF ブイ（PNT）は別ノードとして扱う。
本プランナは PNT ブイ（5〜9 本）の投下順序と各投下点での機首方位を候補列挙し、
Dubins 経路長の合計が最小になる計画を選ぶ。投下後はグリッド中心周りの
キャリブレーション周回（円軌道）を付与する。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .path import DubinsPath, dubins_shortest


def hex_grid(center, n: int, spacing: float) -> np.ndarray:
    """中心 1 本 + 周囲 (n-1) 本のリング配置. n は 5〜9."""
    if not 5 <= n <= 9:
        raise ValueError("PNT グリッドのブイ数は 5〜9 本")
    ang = np.linspace(0.0, 2 * np.pi, n - 1, endpoint=False)
    ring = np.stack([np.cos(ang), np.sin(ang)], axis=1) * spacing
    return np.vstack([[0.0, 0.0], ring]) + np.asarray(center, float)


def chain_paths(poses: np.ndarray, rho: float) -> list[DubinsPath]:
    return [dubins_shortest(poses[i], poses[i + 1], rho) for i in range(len(poses) - 1)]


@dataclass
class Plan:
    order: np.ndarray                 # 投下順（グリッド点インデックス）
    poses: np.ndarray                 # 経由姿勢 (K,3)  先頭は開始姿勢
    paths: list[DubinsPath]
    orbit_center: np.ndarray
    orbit_radius: float
    words: list[str] = field(default_factory=list)

    @property
    def length(self) -> float:
        return float(sum(p.length for p in self.paths))


@dataclass
class PNTGridPlanner:
    rho: float = 3000.0               # 最小旋回半径 [m]（例: 150 m/s, 30°バンク ≒ 4 km）
    orbit_radius: float | None = None

    def _poses_for(self, start, pts: np.ndarray, order: np.ndarray) -> np.ndarray:
        seq = pts[order]
        nxt = np.vstack([seq[1:], seq[-1] + (seq[-1] - seq[-2])])
        hdg = np.arctan2(nxt[:, 1] - seq[:, 1], nxt[:, 0] - seq[:, 0])
        return np.vstack([np.asarray(start, float), np.column_stack([seq, hdg])])

    def candidates(self, pts: np.ndarray) -> list[np.ndarray]:
        """リング巡回（時計/反時計 × 開始点）× 中心を最初/最後 の候補順序."""
        n_ring = len(pts) - 1
        ring = np.arange(1, n_ring + 1)
        out = []
        for s in range(n_ring):
            for direction in (1, -1):
                r = np.roll(ring[::direction], -s)
                out.append(np.concatenate([[0], r]))
                out.append(np.concatenate([r, [0]]))
        return out

    def plan(self, start, pts: np.ndarray) -> Plan:
        best = None
        for order in self.candidates(pts):
            poses = self._poses_for(start, pts, order)
            paths = chain_paths(poses, self.rho)
            L = sum(p.length for p in paths)
            if best is None or L < best[0]:
                best = (L, order, poses, paths)
        _, order, poses, paths = best
        center = pts.mean(axis=0)
        spread = np.max(np.linalg.norm(pts - center, axis=1))
        r_orb = self.orbit_radius or max(self.rho, 0.6 * spread)
        return Plan(order, poses, paths, center, r_orb, [p.word for p in paths])


class OrbitArc:
    """反時計回りの円周回セグメント（DubinsPath と同じ length / at(s) インタフェース）."""

    def __init__(self, center, radius: float, ang0: float, length: float):
        self.c, self.r, self.ang0, self._len = np.asarray(center, float), radius, ang0, length

    @property
    def length(self) -> float:
        return self._len

    def at(self, s: np.ndarray) -> np.ndarray:
        ang = self.ang0 + np.atleast_1d(s) / self.r
        return np.column_stack([self.c[0] + self.r * np.cos(ang), self.c[1] + self.r * np.sin(ang),
                                np.mod(ang + np.pi / 2, 2 * np.pi)])


def end_pose(seg) -> np.ndarray:
    return seg.at(np.array([seg.length]))[0]


def orbit_segments(last_pose, center, radius: float, duration_s: float, speed: float, rho: float) -> list:
    """現在姿勢 → 周回円の接線姿勢へ Dubins 遷移 + 周回円弧."""
    c = np.asarray(center, float)
    off = np.asarray(last_pose[:2]) - c
    ang0 = np.arctan2(off[1], off[0]) if np.hypot(*off) > 1.0 else last_pose[2] - np.pi / 2
    entry = np.array([c[0] + radius * np.cos(ang0), c[1] + radius * np.sin(ang0), ang0 + np.pi / 2])
    return [dubins_shortest(last_pose, entry, rho), OrbitArc(c, radius, ang0, speed * duration_s)]


def sample_segments(segs: list, speed: float, dt: float) -> np.ndarray:
    """セグメント列を通しの弧長で一定速度サンプリング（継ぎ目でも等間隔）. (T,5) [t,x,y,vx,vy]."""
    L = np.array([g.length for g in segs])
    bounds = np.concatenate([[0.0], np.cumsum(L)])
    s = np.arange(0.0, bounds[-1] + 1e-9, speed * dt)
    k = np.clip(np.searchsorted(bounds, s, side="right") - 1, 0, len(segs) - 1)
    xyh = np.empty((s.size, 3))
    for i, g in enumerate(segs):
        m = k == i
        if m.any():
            xyh[m] = g.at(s[m] - bounds[i])
    v = speed * np.column_stack([np.cos(xyh[:, 2]), np.sin(xyh[:, 2])])
    return np.column_stack([s / speed, xyh[:, :2], v])


def sample_trajectory(paths: list[DubinsPath], speed: float, dt: float,
                      orbit: tuple[np.ndarray, float, float] | None = None,
                      tail: list[DubinsPath] | None = None) -> np.ndarray:
    """経路列（+ 任意の周回 orbit=(center, radius, duration_s) + tail）を時間サンプリング."""
    segs = list(paths)
    if orbit is not None:
        segs += orbit_segments(end_pose(segs[-1]), orbit[0], orbit[1], orbit[2], speed, paths[0].rho)
    segs += list(tail or [])
    return sample_segments(segs, speed, dt)
