"""DDC（デジタルダウンコンバータ）と VHF 120 kbps 帯域収容の検証.

RTL-SDR（例 2.4 MS/s, HF ダイレクトサンプリング）の IQ から 3〜6 kHz の OFDM 帯域を
切り出して間引き、VHF リンクで送れるかを判定する。
送り方は 2 通り:
  - "iq"      : 間引き後 IQ をそのまま量子化して送る
  - "feature" : エッジで TOA/位相などの特徴量だけ抽出して送る（推奨）
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

from ..constants import VHF_LINK_RATE_BPS


def ddc(iq: np.ndarray, fs_in: float, f_offset: float, fs_out: float) -> np.ndarray:
    """f_offset [Hz] を 0 Hz に移し fs_out へ有理数リサンプル（アンチエイリアス FIR 込み）."""
    n = np.arange(len(iq))
    bb = iq * np.exp(-2j * np.pi * f_offset / fs_in * n)
    fr = Fraction(fs_out / fs_in).limit_denominator(10000)
    return resample_poly(bb, fr.numerator, fr.denominator).astype(np.complex64)


@dataclass
class LinkBudget:
    link_bps: float = VHF_LINK_RATE_BPS
    framing_overhead: float = 0.15       # ヘッダ・FEC・同期語

    def iq_bps(self, fs_out: float, bits_per_component: int) -> float:
        return fs_out * 2 * bits_per_component * (1 + self.framing_overhead)

    def feature_bps(self, n_anchors: int, update_hz: float, bits_per_anchor: int = 96,
                    header_bits: int = 128) -> float:
        """1 アンカーあたり TOA(48bit) + 位相(24bit) + SNR/品質(24bit) を想定."""
        return (header_bits + n_anchors * bits_per_anchor) * update_hz * (1 + self.framing_overhead)

    def fits(self, bps: float) -> bool:
        return bps <= self.link_bps

    def max_bits_for_iq(self, fs_out: float) -> int:
        return int(np.floor(self.link_bps / (fs_out * 2 * (1 + self.framing_overhead))))
