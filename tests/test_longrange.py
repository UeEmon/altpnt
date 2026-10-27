"""長距離シナリオ（上空波 Rx-TDOA + 電離層バイアス拡張 EKF）のテスト. VOACAP は保存済み出力で代用."""
from pathlib import Path

import numpy as np
import yaml

from altpnt_sim import longrange as L
from altpnt_sim.geo import aeqd_forward, aeqd_inverse, gc_distance_m
from altpnt_sim.hfprop import parse_voacap

ROOT = Path(__file__).resolve().parents[1]
RAW = (Path(__file__).parent / "data" / "voacap_iwoto_guam_202610.out").read_text()


def test_aeqd_roundtrip_and_distance():
    lat0, lon0 = 27.0, 142.0
    lat = np.array([35.0, 18.6, 27.0]); lon = np.array([140.5, 143.5, 142.0])
    x, y = aeqd_forward(lat, lon, lat0, lon0)
    la, lo = aeqd_inverse(x, y, lat0, lon0)
    assert np.allclose(la, lat, atol=1e-9) and np.allclose(lo, lon, atol=1e-9)
    # 原点からの距離は正距方位図法で保存される
    assert np.allclose(np.hypot(x, y), gc_distance_m(lat0, lon0, lat, lon), rtol=1e-9)


class FakeBackend:
    def predict(self, req):
        return parse_voacap(RAW)


def _cfg():
    cfg = yaml.safe_load((ROOT / "config" / "longrange_default.yaml").read_text(encoding="utf-8"))
    sky = yaml.safe_load((ROOT / "config" / "skywave_japan.yaml").read_text(encoding="utf-8"))
    cfg["route"]["op_area"]["nm"] = 300
    cfg["route"]["orbit"]["duration_h"] = 0.5
    cfg["epoch"] = {"year": 2026, "month": 10, "ssn": 100}
    return cfg, sky


def test_route_phases_continuous():
    cfg, _ = _cfg()
    traj, origin, op_xy, op_ll, phases = L.build_route(cfg)
    step = np.hypot(*np.diff(traj[:, 1:3], axis=0).T)
    assert np.allclose(step, cfg["aircraft"]["speed"] * cfg["dt"], atol=0.5)
    assert list(phases) == ["進出", "作戦海域", "帰投"]
    assert np.hypot(traj[-1, 1] - traj[0, 1], traj[-1, 2] - traj[0, 2]) < cfg["aircraft"]["speed"] * cfg["dt"] + 1   # 出発点に戻る（最後の 1 刻み以内）


def test_ekf_bounds_ins_and_is_consistent():
    cfg, sky = _cfg()
    out = L.run(cfg, sky, FakeBackend(), depart_utc=22)
    s = L.summarize(out)["modes"]
    assert s["realtime"]["final_km"] < s["ins_only"]["final_km"]
    assert s["differential"]["作戦海域"]["mean_km"] <= s["realtime"]["作戦海域"]["mean_km"] + 0.05
    for mode in ("model", "realtime"):
        for ph in ("進出", "作戦海域", "帰投"):
            assert 0.3 < s[mode][ph]["consistency"] < 3.0       # 推定誤差の見積もりが大きく外れない


def test_aggregate_monte_carlo():
    a = {"depart_utc": 22, "ssn": 100, "ssn_source": "config", "stations": ["A"], "duration_h": 1,
         "modes": {"m": {"P": {"mean_km": 1.0, "p95_km": 2.0, "max_km": 3.0, "consistency": 1.0, "RNP-10": "100%"},
                         "final_km": 1.0}}}
    b = {**a, "modes": {"m": {"P": {"mean_km": 3.0, "p95_km": 4.0, "max_km": 5.0, "consistency": 2.0, "RNP-10": "50%"},
                              "final_km": 3.0}}}
    g = L.aggregate([a, b])
    assert g["runs"] == 2 and g["modes"]["m"]["P"]["mean_km"] == 2.0 and g["modes"]["m"]["P"]["max_km"] == 5.0
    assert g["modes"]["m"]["P"]["RNP-10"] == "75%" and g["modes"]["m"]["final_km"] == 2.0
