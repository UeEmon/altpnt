import yaml

from altpnt_sim.scenario import DEFAULT_CFG, run, summarize


def test_default_scenario_bounds_ins():
    cfg = yaml.safe_load(DEFAULT_CFG.read_text(encoding="utf-8"))
    cfg["pnt_grid"]["orbit_duration_s"] = 900            # テスト短縮
    s = summarize(run(cfg))
    a = s["aircraft"]
    assert a["hf"]["final_m"] < a["ins_only"]["final_m"] / 5
    assert a["hf_grid"]["orbit"]["p95_m"] < 1000
    assert a["hf"]["transit"] == a["hf_grid"]["transit"]   # 投下前は同一観測 → 同一結果
