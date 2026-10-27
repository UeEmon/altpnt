"""CHUCK/VOACAP 連携と上空波の時間帯別評価のテスト（VOACAP 本体は不要: 実出力を保存したものを使う）."""
from pathlib import Path

import math
import yaml

from altpnt_sim.hfprop import parse_voacap, plan_link, position_error
from altpnt_sim.hfprop.chuck import chuck_request
from altpnt_sim.hfprop.plan import LinkHour, destination, gc_dist_az, group_path_km
from altpnt_sim.skywave import DEFAULT_CFG, run, summarize

RAW = (Path(__file__).parent / "data" / "voacap_iwoto_guam_202610.out").read_text()
CFG = yaml.safe_load(DEFAULT_CFG.read_text(encoding="utf-8"))


def test_parse_all_fields():
    hours = parse_voacap(RAW)
    assert [b["hour_utc"] for b in hours] == list(range(24))
    rows = hours[5]["rows"]
    assert [r["freq_mhz"] for r in rows] == CFG["frequencies"]
    r19 = next(r for r in rows if r["freq_mhz"] == 19.0)
    # CHUCK の parse_output と同じ値（この出力では 05UTC/19MHz: SNR 71, REL 1.00, TANGLE 19.4）
    assert r19["snr_dbhz"] == 71.0 and r19["rel"] == 1.0 and r19["takeoff_deg"] == 19.4
    assert r19["mode"] and r19["mode"][1:3] in ("F2", "F1", " E", "ES")
    assert 150 < r19["vhite_km"] < 500 and r19["delay_ms"] > 4


def test_plan_link_chooses_reliable_frequency():
    d, _ = gc_dist_az(24.78, 141.32, 13.4, 144.8)
    lk = plan_link(parse_voacap(RAW), d, rel_min=0.8)
    assert len(lk) == 24
    used = [lh for lh in lk if lh.usable]
    assert used and all(lh.kind == "sky" and lh.rel >= 0.8 for lh in used)
    assert all(lh.freq_mhz <= lh.muf_mhz * 1.1 for lh in used)          # MUF を大きく超えない
    assert all(0.3 < lh.dPdh < 1.2 and 0.9 < lh.dPdD <= 1.01 for lh in used)


def test_group_path_and_geometry():
    assert group_path_km(1852, 300) > 1852
    assert group_path_km(1852, 350) > group_path_km(1852, 300)
    lat, lon = destination(34, 135, 90, 1000 * 1.852)
    d, az = gc_dist_az(34, 135, lat, lon)
    assert abs(d - 1852) < 1 and abs(math.degrees(az) - 90) < 0.5


def _lh(kind="sky", h=280.0):
    return LinkHour(0, kind != "none", kind, 10.0, 0.95, 60.0, "1F2", h, 0.0, 20.0, 1500.0, 0.99, 0.7)


def test_position_error_tiers_and_minimum_stations():
    stations = [(40.82, 140.74), (35.61, 140.12), (34.39, 132.46), (31.6, 130.56), (26.21, 127.68), (24.78, 141.32)]
    rx = destination(34, 135, 180, 1852)
    links = [_lh() for _ in stations]
    e_model, n = position_error(rx, stations, links, "model")
    e_rt, _ = position_error(rx, stations, links, "realtime")
    e_diff, _ = position_error(rx, stations, links, "differential")
    assert n == 6 and e_model > e_rt > e_diff > 0
    assert 3.5 < e_model / e_rt < 4.5                         # σ_h 20 km / 5 km
    e_none, n2 = position_error(rx, stations, [_lh("none")] * 4 + [_lh(), _lh()], "realtime")
    assert math.isnan(e_none) and n2 == 2


class FakeBackend:
    """全リンクで同じ保存済み出力を返す（CLI の流れの確認用）."""

    def predict(self, req):
        assert req["frequencies"] == CFG["frequencies"] and req["snr_input"] == "bandwidth"
        return parse_voacap(RAW)


def test_run_and_summary_with_fake_backend():
    cfg = dict(CFG, epochs=CFG["epochs"][:1], points={"南1000NM": CFG["points"]["南1000NM"]})
    res = run(cfg, FakeBackend())
    assert len(res["errors"]) == 24 and len(res["plan"]) == 24 * len(cfg["stations"])
    s = summarize(res)["南1000NM"]["2026-10"]
    assert s["hours"] == 24 and s["realtime"]["median_drms_km"] > s["differential"]["median_drms_km"]


def test_chuck_request_matches_api_fields():
    req = chuck_request((24.78, 141.32), (13.4, 144.8), 2026, 10, 100, [7.5, 9.0], CFG["link"])
    for k in ("tx_lat", "rx_lon", "year", "month", "ssn", "frequencies", "power_w", "tx_antenna", "tx_length_m",
              "bandwidth", "inband_snr_db"):
        assert k in req


def test_resolve_ssn_effective_and_fallback():
    from altpnt_sim.skywave import resolve_ssn

    ok = lambda url, y, m: (87.4, {"used_count": 3})                      # noqa: E731
    none = lambda url, y, m: (None, {"error": "観測が2局そろいません"})     # noqa: E731
    assert resolve_ssn({"year": 2026, "month": 10, "ssn": 95}, None) == (95.0, "config")
    v, src = resolve_ssn({"year": 2026, "month": 10, "ssn": "effective"}, "http://chuck", ok)
    assert v == 87.4 and src.startswith("effective")
    v, src = resolve_ssn({"year": 2026, "month": 10, "ssn": "effective", "ssn_fallback": 100}, "http://chuck", none)
    assert v == 100.0 and "2局" in src
    import pytest
    with pytest.raises(ValueError):
        resolve_ssn({"year": 2026, "month": 10, "ssn": "effective"}, None)
