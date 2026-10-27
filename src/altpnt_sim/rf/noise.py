"""HF ノイズ・フェージング生成器.

RTL-SDR V4 の収録データ（rtl_sdr の .cu8: uint8 IQ インタリーブ、または complex .npy）を
ノイズ源として読み込む。ファイルが無い場合は合成モデル（AWGN + Watterson 型ガウス
ドップラー拡散フェージング）にフォールバックする。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import lfilter


def load_rtlsdr_cu8(path: str | Path, max_samples: int | None = None) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint8, count=-1 if max_samples is None else 2 * max_samples)
    iq = (raw[0::2].astype(np.float32) - 127.5) + 1j * (raw[1::2].astype(np.float32) - 127.5)
    return (iq / 127.5).astype(np.complex64)


def load_iq(path: str | Path, max_samples: int | None = None) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        iq = np.load(path).astype(np.complex64)
        return iq if max_samples is None else iq[:max_samples]
    return load_rtlsdr_cu8(path, max_samples)


def gaussian_doppler_fading(n: int, fs: float, doppler_spread_hz: float, rng: np.random.Generator) -> np.ndarray:
    """平均電力 1 の複素ガウス過程（ガウス型ドップラースペクトル）. 1 次 IIR 近似でベクトル化."""
    w = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    a = np.exp(-2 * np.pi * doppler_spread_hz / fs)
    g = lfilter([np.sqrt(1 - a * a)], [1, -a], w)
    return g / np.sqrt(np.mean(np.abs(g) ** 2))


@dataclass
class NoiseInjector:
    iq_path: str | Path | None = None
    doppler_spread_hz: float = 0.5        # 中緯度 HF の「穏やか」条件
    rician_k_db: float = 6.0              # 地表波は直接成分が強いので Rician
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(3))

    def __post_init__(self):
        self._iq = None
        if self.iq_path and Path(self.iq_path).exists():
            iq = load_iq(self.iq_path)
            self._iq = (iq - iq.mean()) / (np.std(iq) + 1e-12)

    @property
    def source(self) -> str:
        return "rtlsdr" if self._iq is not None else "synthetic"

    def noise(self, n: int) -> np.ndarray:
        """単位電力の雑音サンプル."""
        if self._iq is not None:
            start = self.rng.integers(0, max(1, len(self._iq) - n))
            seg = self._iq[start:start + n]
            if len(seg) < n:
                seg = np.resize(seg, n)
            return seg
        return (self.rng.standard_normal(n) + 1j * self.rng.standard_normal(n)) / np.sqrt(2)

    def fading_db(self, n: int, fs: float = 1.0) -> np.ndarray:
        """Rician フェージングによる SNR 変動 [dB]（時系列 n 点, サンプルレート fs）."""
        k = 10 ** (self.rician_k_db / 10)
        g = gaussian_doppler_fading(n, fs, self.doppler_spread_hz, self.rng)
        h = np.sqrt(k / (k + 1)) + np.sqrt(1 / (k + 1)) * g
        return 20 * np.log10(np.abs(h) + 1e-12)

    def add_noise(self, sig: np.ndarray, snr_db: float) -> np.ndarray:
        p_sig = np.mean(np.abs(sig) ** 2)
        return sig + np.sqrt(p_sig / 10 ** (snr_db / 10)) * self.noise(len(sig))
