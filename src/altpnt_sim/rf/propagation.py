"""電波伝搬モデル.

HF 地表波（アンカー → ブイ/哨戒機）:
  - SNR の距離減衰（簡易: 自由空間 + 地表波付加減衰 [dB/km]）
  - TOA 誤差: OFDM パイロット包絡線相関の CRLB
  - 伝搬遅延バイアス: Loran の ASF に相当する「二次位相遅延」を空間的に滑らかな場で表現
    （海水塩分・海面粗度で変動。受信側は名目値しか知らない → 残差が系統誤差になる）
VHF/UHF（ブイ ⇔ 哨戒機）:
  - フリスの伝送公式、2 波（直接波 + 海面反射）モデル、波浪によるアンテナ傾斜・遮蔽損失、BER
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import erfc

from ..constants import C, K_BOLTZ, T0, R_EARTH, K_FACTOR

# ---------------------------------------------------------------- HF 地表波


@dataclass
class HFGroundWave:
    freq_hz: float = 8e6
    bandwidth_hz: float = 5e3             # パイロットトーン付き OFDM（3〜6 kHz）
    integ_time_s: float = 1.0             # 相関積分時間
    snr_ref_db: float = 30.0              # 基準距離での SNR（帯域内）
    d_ref_m: float = 100e3
    extra_atten_db_per_km: float = 0.02   # 海上地表波の付加減衰（周波数・海況依存）
    asf_ns_per_km: float = 0.3            # 二次位相遅延の距離比例分 [ns/km]
    asf_model_err: float = 0.3            # 受信側モデルの相対誤差（0.3 → 30% が残差として残る）
    ripple_m: float = 40.0                # 空間的リップル振幅 [m]
    ripple_wavelength_m: float = 150e3
    min_sigma_m: float = 5.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(2))

    def __post_init__(self):
        self._ripple_dir = None

    def snr_db(self, d_m: np.ndarray) -> np.ndarray:
        d = np.maximum(d_m, 1.0)
        return (self.snr_ref_db - 20 * np.log10(d / self.d_ref_m)
                - self.extra_atten_db_per_km * (d - self.d_ref_m) / 1e3)

    def toa_sigma_m(self, snr_db: np.ndarray) -> np.ndarray:
        """包絡線 TOA の CRLB: σ_τ = 1 / (2π β √(2 SNR·B·T)), β = B/√12."""
        beta = self.bandwidth_hz / np.sqrt(12.0)
        en0 = 10 ** (np.asarray(snr_db) / 10) * self.bandwidth_hz * self.integ_time_s
        return np.maximum(C / (2 * np.pi * beta * np.sqrt(2 * en0)), self.min_sigma_m)

    def carrier_phase_sigma_m(self, snr_db: np.ndarray) -> np.ndarray:
        """搬送波位相の測距 σ（整数不確定性解決後）. Phase 2 で位相スリップと合わせて使用."""
        lam = C / self.freq_hz
        en0 = 10 ** (np.asarray(snr_db) / 10) * self.bandwidth_hz * self.integ_time_s
        return lam / (2 * np.pi * np.sqrt(2 * en0))

    def excess_path_m(self, p: np.ndarray, anchors: np.ndarray, sea_state: float = 3.0) -> np.ndarray:
        """真の伝搬遅延バイアスのうち受信側モデルで補正しきれない残差 [m]. (N,M)."""
        p = np.atleast_2d(p)
        if self._ripple_dir is None:
            ang = self.rng.uniform(0, 2 * np.pi, len(anchors))
            self._ripple_dir = np.stack([np.cos(ang), np.sin(ang)], 1)
            self._ripple_phase = self.rng.uniform(0, 2 * np.pi, len(anchors))
            self._asf_sign = self.rng.choice([-1.0, 1.0], len(anchors))
        d = np.linalg.norm(p[:, None, :] - anchors[None], axis=-1)
        asf = self.asf_ns_per_km * 1e-9 * C * (d / 1e3) * (1 + 0.1 * (sea_state - 3))
        k = 2 * np.pi / self.ripple_wavelength_m
        ripple = self.ripple_m * np.sin(k * (p @ self._ripple_dir.T) + self._ripple_phase)
        return self.asf_model_err * self._asf_sign * asf + ripple

    def measure_toa(self, p: np.ndarray, anchors: np.ndarray, fading_db: np.ndarray | float = 0.0,
                    sea_state: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
        """擬似 TOA 距離 (N,M) と各 σ (N,M) を返す（受信機時計バイアスは差分で消えるので省略）."""
        p = np.atleast_2d(p)
        d = np.linalg.norm(p[:, None, :] - anchors[None], axis=-1)
        sig = self.toa_sigma_m(self.snr_db(d) + fading_db)
        z = d + self.excess_path_m(p, anchors, sea_state) + sig * self.rng.standard_normal(d.shape)
        return z, sig


# ---------------------------------------------------------------- VHF/UHF リンク


def radio_horizon_m(h1: float, h2: float) -> float:
    return np.sqrt(2 * K_FACTOR * R_EARTH * h1) + np.sqrt(2 * K_FACTOR * R_EARTH * h2)


def friis_path_loss_db(d_m, freq_hz) -> np.ndarray:
    lam = C / freq_hz
    return 20 * np.log10(4 * np.pi * np.maximum(d_m, 1.0) / lam)


def sea_reflection_coeff(grazing_rad, freq_hz, pol: str = "V", eps_r: float = 81.0, sigma: float = 4.0):
    """海面の Fresnel 反射係数（複素）."""
    eps_c = eps_r - 1j * sigma / (2 * np.pi * freq_hz * 8.854e-12)
    s, c = np.sin(grazing_rad), np.cos(grazing_rad)
    root = np.sqrt(eps_c - c**2)
    if pol.upper() == "H":
        return (s - root) / (s + root)
    return (eps_c * s - root) / (eps_c * s + root)


def two_ray_path_loss_db(d_m, ht_m, hr_m, freq_hz, wave_rms_m: float = 0.5, pol: str = "V") -> np.ndarray:
    """2 波モデル（直接波 + 海面反射波）の伝搬損失 [dB]. 海面粗度で反射を減衰させる."""
    d = np.maximum(np.asarray(d_m, float), 1.0)
    lam = C / freq_hz
    r_dir = np.hypot(d, ht_m - hr_m)
    r_ref = np.hypot(d, ht_m + hr_m)
    psi = np.arctan2(ht_m + hr_m, d)
    gamma = sea_reflection_coeff(psi, freq_hz, pol)
    rough = np.exp(-2 * (2 * np.pi * wave_rms_m * np.sin(psi) / lam) ** 2)
    k = 2 * np.pi / lam
    field_ratio = np.exp(-1j * k * r_dir) / r_dir + rough * gamma * np.exp(-1j * k * r_ref) / r_ref
    return -20 * np.log10(np.abs(field_ratio) * lam / (4 * np.pi))


def wave_loss_db(rng: np.random.Generator, n: int, hs_m: float = 2.0, ant_height_m: float = 0.8,
                 tilt_sigma_deg: float = 15.0, shadow_db: float = 12.0) -> np.ndarray:
    """波浪によるアンテナ傾斜損失 + 波の谷での遮蔽損失（時間サンプル n 個）."""
    tilt = np.deg2rad(np.abs(tilt_sigma_deg * rng.standard_normal(n)))
    tilt_loss = -20 * np.log10(np.maximum(np.cos(tilt), 0.1))
    eta = (hs_m / 4.0) * rng.standard_normal(n)              # 海面変位（Hs = 4σ）
    shadow = np.where(eta < -(ant_height_m - hs_m / 4.0), shadow_db, 0.0)
    return tilt_loss + shadow


def ber_bpsk(ebn0_db) -> np.ndarray:
    return 0.5 * erfc(np.sqrt(10 ** (np.asarray(ebn0_db) / 10)))


@dataclass
class VHFLink:
    freq_hz: float = 160e6
    tx_power_dbm: float = 30.0          # 1 W
    g_tx_dbi: float = 0.0
    g_rx_dbi: float = 2.0
    noise_figure_db: float = 6.0
    rx_bw_hz: float = 250e3
    bit_rate: float = 120e3
    misc_loss_db: float = 3.0

    def ebn0_db(self, path_loss_db) -> np.ndarray:
        pr = self.tx_power_dbm + self.g_tx_dbi + self.g_rx_dbi - path_loss_db - self.misc_loss_db
        n_dbm = 10 * np.log10(K_BOLTZ * T0 * self.rx_bw_hz * 1e3) + self.noise_figure_db
        return pr - n_dbm + 10 * np.log10(self.rx_bw_hz / self.bit_rate)

    def packet_success(self, path_loss_db, bits: int) -> np.ndarray:
        return (1.0 - ber_bpsk(self.ebn0_db(path_loss_db))) ** bits
