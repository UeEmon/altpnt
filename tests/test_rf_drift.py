import numpy as np

from altpnt_sim.drift import DriftSimulator, OceanCurrentField
from altpnt_sim.rf.ddc import LinkBudget, ddc
from altpnt_sim.rf.noise import NoiseInjector
from altpnt_sim.rf.propagation import (HFGroundWave, friis_path_loss_db, radio_horizon_m,
                                       two_ray_path_loss_db)


def test_drift_mean_motion():
    f = OceanCurrentField(eddy_strength=np.array([0.0]), tidal_amp=(0.0, 0.0), wind=(0.0, 0.0))
    sim = DriftSimulator(f, diffusivity=0.0)
    _, tr = sim.run(np.zeros((5, 2)), 0, 3600, 60)
    assert np.allclose(tr[-1], np.array(f.mean_current) * 3600, atol=1e-6)


def test_friis_and_two_ray():
    assert abs(friis_path_loss_db(1000.0, 160e6) - 76.5) < 0.5
    pl = two_ray_path_loss_db(np.array([5e3, 50e3]), 1500.0, 0.8, 160e6)
    assert np.all(np.isfinite(pl)) and pl[1] > pl[0]
    assert 120e3 < radio_horizon_m(1500.0, 0.8) < 170e3


def test_hf_toa_sigma_decreases_with_snr():
    hf = HFGroundWave()
    s = hf.toa_sigma_m(np.array([0.0, 10.0, 20.0]))
    assert s[0] > s[1] > s[2]


def test_ddc_and_link_budget():
    inj = NoiseInjector()
    assert inj.source == "synthetic"
    y = ddc(inj.noise(240_000), 2.4e6, 100e3, 8e3)
    assert abs(y.size - 800) <= 2
    lb = LinkBudget()
    assert not lb.fits(lb.iq_bps(8e3, 8))       # 8bit IQ は 120kbps を超える
    assert lb.fits(lb.iq_bps(8e3, 6))
    assert lb.fits(lb.feature_bps(4, 1.0))
