"""aqua-drift-simulator 連携のテスト（aqua_drift 未インストールならスキップ）."""
import numpy as np
import pytest
import yaml

aqua = pytest.importorskip("aqua_drift")
from aqua_drift import physics as aphys  # noqa: E402
from aqua_drift.models import CurrentFieldConfig, Position  # noqa: E402

from altpnt_sim.drift import AquaDriftAdapter, AquaDropModel  # noqa: E402
from altpnt_sim.scenario import DEFAULT_CFG, build_trajectory, deploy_buoys  # noqa: E402

FIELD = {"reference_position": {"latitude": 30.0, "longitude": 135.0, "depth_ft": 0.0},
         "base_velocity": {"east_kt": 0.6, "north_kt": 0.2, "vertical_fps": 0.0},
         "gradient_per_nm": [[0.01, 0.004, 0.0], [-0.004, -0.01, 0.0], [0.0, 0.0, 0.0]]}


def test_adapter_matches_current_at():
    """numpy 一括評価が aqua_drift.physics.current_at と一致する（原点が p_ref からずれていても）."""
    ad = AquaDriftAdapter(FIELD, origin_latlon=(30.2, 134.9))
    cfg = CurrentFieldConfig.model_validate(FIELD)
    xy = np.random.default_rng(0).uniform(-50e3, 50e3, (30, 2))
    ll = ad.to_latlon(xy)
    ref = np.array([[v.east_kt, v.north_kt] for v in
                    (aphys.current_at(cfg, Position(latitude=a, longitude=b, depth_ft=0.0)) for a, b in ll)])
    # 差は aqua の local_offset_m（平均緯度）と平面 ENU 近似の違いのみ: ±50 km で < 1 mm/s（1 時間で < 4 m）
    np.testing.assert_allclose(ad.velocity(xy), ref * aphys.KNOT_TO_MPS, atol=1e-3)


def test_fall_matches_aqua():
    dm = AquaDropModel()
    d, tf = dm.fall(1500.0, np.array([150.0, 0.0]))
    assert 40 < tf < 70 and 300 < np.hypot(*d) < 1500


def test_release_correction_reduces_miss():
    cfg = yaml.safe_load(DEFAULT_CFG.read_text(encoding="utf-8"))
    cfg["dt"] = 0.1
    tr, drops, td, *_ = build_trajectory(cfg)
    miss = {}
    for corr in (False, True):
        cfg["drift"]["drop"]["release_correction"] = corr
        (_, r), _, _ = deploy_buoys(cfg, tr, drops, td, np.random.default_rng(0))
        miss[corr] = r.miss_m[1:].mean()           # 初回は風推定なし
    assert miss[True] < 0.8 * miss[False]
