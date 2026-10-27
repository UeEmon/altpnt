"""球面の正距方位図法（azimuthal equidistant）.

長距離シナリオでは、航跡と EKF の状態を原点まわりの正距方位図法の平面 [m] で持ち、
観測（大圏距離）を計算するときだけ緯度経度に戻す。原点から 2,000 km 程度までなら
航空機の運動を平面で扱う誤差は観測誤差に比べて無視できる。
"""
from __future__ import annotations

import numpy as np

R_M = 6_371_000.0


def aeqd_forward(lat, lon, lat0: float, lon0: float):
    p, l = np.radians(lat), np.radians(lon)
    p0, l0 = np.radians(lat0), np.radians(lon0)
    cosc = np.clip(np.sin(p0) * np.sin(p) + np.cos(p0) * np.cos(p) * np.cos(l - l0), -1, 1)
    c = np.arccos(cosc)
    k = np.where(c < 1e-12, 1.0, c / np.sin(np.where(c < 1e-12, 1.0, c)))
    x = R_M * k * np.cos(p) * np.sin(l - l0)
    y = R_M * k * (np.cos(p0) * np.sin(p) - np.sin(p0) * np.cos(p) * np.cos(l - l0))
    return x, y


def aeqd_inverse(x, y, lat0: float, lon0: float):
    x, y = np.asarray(x, float), np.asarray(y, float)
    p0, l0 = np.radians(lat0), np.radians(lon0)
    rho = np.hypot(x, y)
    c = rho / R_M
    safe = np.where(rho < 1e-9, 1.0, rho)
    lat = np.arcsin(np.clip(np.cos(c) * np.sin(p0) + y * np.sin(c) * np.cos(p0) / safe, -1, 1))
    lon = l0 + np.arctan2(x * np.sin(c), safe * np.cos(p0) * np.cos(c) - y * np.sin(p0) * np.sin(c))
    lat = np.where(rho < 1e-9, p0, lat)
    lon = np.where(rho < 1e-9, l0, lon)
    return np.degrees(lat), (np.degrees(lon) + 540) % 360 - 180


def gc_distance_m(lat1, lon1, lat2, lon2):
    p1, l1, p2, l2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin((l2 - l1) / 2) ** 2
    return 2 * R_M * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
