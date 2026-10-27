"""エッジノード: DDC + 帯域収容判定 → シナリオ実行（INS バウンディング EKF）.

RTL-SDR 収録ファイルが data/rtlsdr/ にあればノイズ源として使い、無ければ合成ノイズ。
"""
import json
import os
from pathlib import Path

import numpy as np

from altpnt_sim.rf.ddc import LinkBudget, ddc
from altpnt_sim.rf.noise import NoiseInjector
from altpnt_sim.scenario import main as run_scenario

link_bps = float(os.getenv("LINK_RATE_BPS", "120000"))
iq_files = sorted(Path("data/rtlsdr").glob("*.cu8")) + sorted(Path("data/rtlsdr").glob("*.npy"))
inj = NoiseInjector(iq_path=iq_files[0] if iq_files else None)

fs_in, fs_out = 2.4e6, 8e3
x = inj.noise(int(fs_in * 0.1))
y = ddc(x, fs_in, f_offset=100e3, fs_out=fs_out)
lb = LinkBudget(link_bps=link_bps)
report = {
    "noise_source": inj.source,
    "ddc_out_samples": int(y.size),
    "iq_8bit_bps": lb.iq_bps(fs_out, 8), "iq_8bit_fits": lb.fits(lb.iq_bps(fs_out, 8)),
    "max_iq_bits_per_component": lb.max_bits_for_iq(fs_out),
    "feature_bps_4anchors_1hz": lb.feature_bps(4, 1.0),
}
print(json.dumps(report, ensure_ascii=False, indent=2))
run_scenario(["--out", "out/edge"])
