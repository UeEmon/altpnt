"""交代機シナリオ（航空機間のデータ引き継ぎなし）のテスト. VOACAP は保存済み出力で代用."""
from pathlib import Path

import numpy as np
import yaml

from altpnt_sim import buoyrelay as RL
from altpnt_sim.hfprop import parse_voacap

ROOT = Path(__file__).resolve().parents[1]
RAW = (Path(__file__).parent / "data" / "voacap_iwoto_guam_202610.out").read_text()


class FakeBackend:
    def predict(self, req):
        return parse_voacap(RAW)


def _cfg():
    cfg = yaml.safe_load((ROOT / "config" / "buoyrelay_default.yaml").read_text(encoding="utf-8"))
    sky = yaml.safe_load((ROOT / "config" / "skywave_japan.yaml").read_text(encoding="utf-8"))
    cfg["route"]["op_area"]["nm"] = 150
    cfg["route"]["orbit"]["duration_h"] = 0.75
    cfg["buoys"]["lifetime_h"] = 0.75
    cfg["buoys"]["redeploy"] = {"enabled": True, "overlap_h": 0.25, "min_remaining_h": 0.25}
    cfg["epoch"] = {"year": 2026, "month": 10, "ssn": 100}
    return cfg, sky


def test_relief_aircraft_restarts_network_estimate_without_handover():
    cfg, sky = _cfg()
    out = RL.run(cfg, sky, FakeBackend(), 22, "aircraft")
    assert len(out["acs"]) == 2 and out["buoys_used"] >= 2 * cfg["buoys"]["n"]
    a1 = out["acs"][1]
    a0, b0 = out["op_win"][1]
    assert a1["last_fix"] >= a0                                       # 交代機もブイだけから測位している
    assert np.nanmax(a1["err"][a0:b0]) < 20e3                         # 事前情報（計画上の中心 ± 20 km）から外れない
    assert out["acs"][1]["net"] is not out["acs"][0]["net"]           # 推定器は機体ごと（受け渡しなし）


def test_buoy_side_estimator_runs():
    cfg, sky = _cfg()
    out = RL.run(cfg, sky, FakeBackend(), 22, "buoy")
    s = RL.summarize(out)
    assert all(a["op_mean_km"] is not None for a in s["aircraft"])
    assert s["buoys_used"] >= 2 * cfg["buoys"]["n"]
