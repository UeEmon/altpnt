"""ブイ中心構成（HF はブイで受信、母機は UHF TOA＋VHF TDOA）のテスト. VOACAP は保存済み出力で代用."""
from pathlib import Path

import yaml

from altpnt_sim import buoynet as BN
from altpnt_sim.hfprop import parse_voacap

ROOT = Path(__file__).resolve().parents[1]
RAW = (Path(__file__).parent / "data" / "voacap_iwoto_guam_202610.out").read_text()


class FakeBackend:
    def predict(self, req):
        assert 3.0 <= min(req["frequencies"]) and max(req["frequencies"]) <= 30.0     # 3〜30 MHz
        return parse_voacap(RAW)


def _cfg():
    cfg = yaml.safe_load((ROOT / "config" / "buoynet_default.yaml").read_text(encoding="utf-8"))
    sky = yaml.safe_load((ROOT / "config" / "skywave_japan.yaml").read_text(encoding="utf-8"))
    cfg["route"]["op_area"]["nm"] = 200
    cfg["route"]["orbit"]["duration_h"] = 1.0
    cfg["epoch"] = {"year": 2026, "month": 10, "ssn": 100}
    return cfg, sky


def test_buoy_network_bounds_aircraft_error():
    cfg, sky = _cfg()
    out = BN.run(cfg, sky, FakeBackend(), depart_utc=22)
    s = BN.summarize(out)["modes"]
    op = "作戦海域"
    assert s["realtime"][op]["mean_km"] < s["ins_only"][op]["mean_km"]
    net = s["realtime"]["ブイ網"]
    assert net["rel_mean_km"] < net["abs_mean_km"]           # 相対位置は精密、絶対位置は共通の電離層誤差で決まる
    assert 0.3 < s["realtime"][op]["consistency"] < 3.0
    assert s["realtime"]["counts"]["uhf"] > 0 and s["realtime"]["counts"]["vhf"] > 0


def test_aircraft_does_not_use_hf_outside_op_area():
    cfg, sky = _cfg()
    out = BN.run(cfg, sky, FakeBackend(), depart_utc=22)
    r, ins = out["results"]["realtime"], out["results"]["ins_only"]
    t = out["t"]
    a, _ = out["phases"]["作戦海域"]
    pre = t < a
    assert (r["err"][pre] == ins["err"][pre]).all()          # 進出中は INS 単独と同一（母機は HF を受信しない）


def test_single_link_still_works():
    cfg, sky = _cfg()
    cfg["links"]["vhf"]["enabled"] = False
    cfg["modes"] = ["realtime"]
    s = BN.summarize(BN.run(cfg, sky, FakeBackend(), depart_utc=22))["modes"]["realtime"]
    assert s["counts"]["vhf"] == 0 and s["counts"]["uhf"] > 0
