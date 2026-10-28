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


def test_commanded_plan_holds_last_frequency():
    from altpnt_sim.hfprop.plan import LinkHour

    def lh(h, f):
        return LinkHour(h, True, "sky", f, 0.9, 60.0, "1F2", 280.0, 0.0, 20.0, 1500.0, 0.99, 0.7)

    links = [[lh(h, 7.5 + h) for h in range(24)]]                  # 1 局、時刻ごとに違う周波数
    p = BN.CommandedPlan(links, horizon_h=2, update_min=10, deploy_only=True)
    assert p.freqs(3) is None                                        # 指示前は受信しない
    p.maybe_command(0.0, 3, in_range=True)
    assert p.freqs(3) == [10.5] and p.freqs(4) == [11.5]
    assert p.freqs(9) == [11.5]                                       # 指示の範囲を過ぎたら最後の周波数のまま
    p.maybe_command(3600.0, 4, in_range=True)
    assert p.count == 1                                               # deploy_only: 指示し直さない


def test_link_for_freq_rejects_unreliable_frequency():
    from altpnt_sim.hfprop.plan import link_for_freq
    block = parse_voacap(RAW)[5]
    ok = link_for_freq(block, 1300.0, 19.0, rel_min=0.8)
    assert ok.usable and ok.freq_mhz == 19.0
    bad = link_for_freq(block, 1300.0, 30.0 if block["muf_mhz"] < 25 else 3.0, rel_min=0.99)
    assert not bad.usable


def test_emission_designator_sets_bandwidth():
    from altpnt_sim.hfprop.chuck import chuck_request, parse_emission
    assert parse_emission("3K00J2D") == ("3K00", "J2D", 3000.0)
    assert parse_emission("24K0J2D")[2] == 24000.0
    req = chuck_request((24.78, 141.32), (13.4, 144.8), 2026, 10, 100, [9.0], {"emission": "6K00G7W"})
    assert req["bandwidth"] == "6K00" and req["modulation"] == "G7W" and req["snr_input"] == "bandwidth"


def test_wider_band_and_simultaneous_reception_reduce_toa_noise():
    from altpnt_sim.hfprop.plan import LinkHour
    lh = LinkHour(0, True, "sky", 9.0, 0.95, 42.0, "1F2", 280.0, 0.0, 20.0, 1500.0, 0.99, 0.7)   # 弱い信号
    narrow = BN.buoy_toa_sigma_m(lh, 3000.0, 10.0)
    wide = BN.buoy_toa_sigma_m(lh, 24000.0, 10.0)
    tdm = BN.buoy_toa_sigma_m(lh, 3000.0, 10.0 / 6)            # 時分割（6 局）
    assert wide < narrow < tdm
    assert BN.vhf_report_bps(8, 10.0) < 120_000                   # 測定結果の送信は VHF 120 kbps に十分収まる


def test_buoys_stop_after_lifetime():
    cfg, sky = _cfg()
    cfg["buoys"]["lifetime_h"] = 0.5
    cfg["buoys"]["redeploy"] = {"enabled": False}
    cfg["modes"] = ["realtime"]
    out = BN.run(cfg, sky, FakeBackend(), depart_utc=22)
    t = out["t"]
    a, b = out["phases"]["作戦海域"]
    r = out["results"]["realtime"]
    after = (t > a + 0.5 * 3600 + 60) & (t < b)
    assert out["buoy_end_h"] is not None and (r["n_buoy"][after] == 0).all()


def test_redeploy_keeps_network_alive():
    cfg, sky = _cfg()
    cfg["route"]["orbit"]["duration_h"] = 1.5
    cfg["buoys"]["lifetime_h"] = 0.75
    cfg["buoys"]["redeploy"] = {"enabled": True, "overlap_h": 0.25, "min_remaining_h": 0.25}
    cfg["modes"] = ["realtime"]
    out = BN.run(cfg, sky, FakeBackend(), depart_utc=22)
    s = BN.summarize(out)["modes"]["realtime"]
    assert s["counts"]["deployments"] == 3 and s["counts"]["buoys_used"] == 3 * cfg["buoys"]["n"]   # 0, 0.5, 1.0 h に投下
    t, r = out["t"], out["results"]["realtime"]
    a, b = out["phases"]["作戦海域"]
    mid = (t > a + 0.8 * 3600) & (t < a + 1.2 * 3600)              # 1 組目が止まった後も 2 組目で測位が続く
    assert (r["n_buoy"][mid] >= 3).all()
