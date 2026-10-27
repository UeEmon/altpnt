"""逆TDOA（Rx-TDOA）: 位置既知の複数送信局（アンカー）からの到達時間差で受信者位置を求める.

測定量は距離差 [m]（= 到達時間差 × 伝搬速度）で扱う:
    z_i = |p - A_i| - |p - A_ref|   (i ≠ ref)
受信機の時計バイアスは差分で消える。複数受信者 (N,2) を一括処理する。
"""
from __future__ import annotations

import numpy as np


def _ranges(p: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """p: (N,2) or (2,), anchors: (M,2) -> (N,M)."""
    p = np.atleast_2d(p)
    return np.linalg.norm(p[:, None, :] - anchors[None, :, :], axis=-1)


def rx_tdoa_ranges(p, anchors, ref: int = 0) -> np.ndarray:
    """距離差 (N, M-1)."""
    r = _ranges(p, anchors)
    return np.delete(r - r[:, [ref]], ref, axis=1)


def rx_tdoa_jacobian(p, anchors, ref: int = 0) -> np.ndarray:
    """∂z/∂p : (N, M-1, 2)."""
    p = np.atleast_2d(p)
    d = p[:, None, :] - anchors[None, :, :]
    u = d / np.linalg.norm(d, axis=-1, keepdims=True)
    return np.delete(u - u[:, [ref], :], ref, axis=1)


def tdoa_cov(n_anchor: int, sigma: np.ndarray | float, ref: int = 0) -> np.ndarray:
    """各アンカーの TOA 誤差 σ_i（独立）から距離差の共分散 (M-1,M-1) を作る（基準局誤差の相関を含む）."""
    s2 = np.broadcast_to(np.asarray(sigma, float) ** 2, (n_anchor,))
    D = np.delete(np.eye(n_anchor), ref, axis=0)
    D[:, ref] = -1.0
    return D @ np.diag(s2) @ D.T


def solve_rx_tdoa(z: np.ndarray, anchors: np.ndarray, p0: np.ndarray, ref: int = 0,
                  W: np.ndarray | None = None, iters: int = 10) -> np.ndarray:
    """重み付き Gauss-Newton. z: (N,M-1), p0: (N,2) 初期値 -> (N,2). 受信者方向にベクトル化."""
    p = np.array(np.atleast_2d(p0), float)
    z = np.atleast_2d(z)
    W = np.eye(z.shape[1]) if W is None else W
    for _ in range(iters):
        H = rx_tdoa_jacobian(p, anchors, ref)                     # (N,K,2)
        r = z - rx_tdoa_ranges(p, anchors, ref)                   # (N,K)
        HtW = np.einsum("nki,kl->nil", H, W)                      # (N,2,K)
        A = np.einsum("nik,nkj->nij", HtW, H)                     # (N,2,2)
        b = np.einsum("nik,nk->ni", HtW, r)
        dp = np.linalg.solve(A, b[..., None])[..., 0]
        p += dp
        if np.max(np.abs(dp)) < 1e-4:
            break
    return p


def gdop(p, anchors, ref: int = 0) -> np.ndarray:
    """水平 DOP（TOA 誤差 1 に対する位置誤差 RMS 倍率）. p:(N,2) -> (N,)."""
    H = rx_tdoa_jacobian(p, anchors, ref)
    C = tdoa_cov(len(anchors), 1.0, ref)
    Ci = np.linalg.inv(C)
    A = np.einsum("nki,kl,nlj->nij", H, Ci, H)
    return np.sqrt(np.trace(np.linalg.inv(A), axis1=1, axis2=2))
