"""電離層の監視網（送信所の近くの受信局で他局を受信して補正値を作る）のテスト."""
import numpy as np

from altpnt_sim.ionomon import MonitorCorrector, dPdh, gc_km, midpoint, monitor_paths

ST = {"青森": (40.8, 140.7), "千葉": (35.6, 140.1), "広島": (34.4, 132.5), "鹿児島": (31.6, 130.6),
      "沖縄": (26.2, 127.7), "硫黄島": (24.8, 141.3)}
NAMES, LL = list(ST), list(ST.values())
OP = (18.56, 143.51)


def test_midpoint_and_sensitivity():
    m = midpoint((0.0, 0.0), (0.0, 10.0))
    assert abs(m[0]) < 1e-9 and abs(m[1] - 5.0) < 1e-9
    assert dPdh(600) > dPdh(2000) > 0                 # 短い経路ほど反射高さに敏感


def test_paths_reciprocal_only_between_monitored_stations():
    ps = monitor_paths(LL, NAMES, ["硫黄島"])
    assert ps and all(5 in (p.a, p.b) for p in ps) and not any(p.reciprocal for p in ps)
    ps_all = monitor_paths(LL, NAMES, NAMES)
    assert all(p.reciprocal for p in ps_all) and all(p.dist_km >= 400 for p in ps_all)
    ps_rx = monitor_paths(LL, NAMES, [], rx_only={"沖ノ鳥島": (20.42, 136.08)})
    assert len(ps_rx) == len(LL) and all(p.b == -1 for p in ps_rx)


def test_correction_reduces_error_more_when_monitors_are_close():
    mids = [midpoint(s, OP) for s in LL]
    far = MonitorCorrector(mids, monitor_paths(LL, NAMES, NAMES), 500, 5, 1800, 10.0, 3000, seed=1)
    near = MonitorCorrector(mids, monitor_paths(LL, NAMES, NAMES, rx_only={"沖ノ鳥島": (20.42, 136.08)}),
                            500, 5, 1800, 10.0, 3000, seed=1)
    assert (near.resid_unit <= far.resid_unit + 1e-9).all() and near.resid_unit.mean() < far.resid_unit.mean()
    # 予測した残差の大きさと、実際に補正した後の残差が合う
    for mc in (far, near):
        e = np.array([mc.buoy_truth(i) - mc.correction(i) for i in range(20, mc.T)])
        ratio = e.std(axis=0) / mc.resid_unit
        assert (0.6 < ratio).all() and (ratio < 1.5).all()


def test_message_rate_is_tiny():
    mids = [midpoint(s, OP) for s in LL]
    mc = MonitorCorrector(mids, monitor_paths(LL, NAMES, NAMES), 500, 5, 1800, 1.0, 10)
    assert mc.message_bps() < 10                       # 局ごとの補正値を 60 秒ごと: 数 bps
    assert mc.message_bps(n_values=256) < 200          # 格子 16×16 でも 200 bps 未満
    assert gc_km((0, 0), (0, 1)) > 100
