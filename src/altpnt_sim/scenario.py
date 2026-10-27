"""統合シナリオ: GPS ロスト → PNT グリッド展開（Dubins）→ キャリブレーション周回 → 帰投.

同一の真値・同一の観測系列に対して 3 方式の EKF を走らせて比較する:
  ins_only : INS 単独（バウンディング無し）
  hf       : INS + 機上 HF Rx-TDOA（欠測あり）
  hf_grid  : INS + 機上 HF Rx-TDOA + PNT ブイ網の VHF TDOA

ブイ自身の絶対位置は、各ブイの HF Rx-TDOA 測位を漂流 KF（位置・速度）で時間平滑して求める。

実行: python -m altpnt_sim.scenario [--config config/scenario_default.yaml] [--out out] [--llm]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

import warnings

from .drift import HAVE_AQUA, AquaDriftAdapter, AquaDropModel, DriftSimulator, OceanCurrentField
from .dubins import PNTGridPlanner, hex_grid
from .dubins.path import dubins_shortest
from .dubins.planner import end_pose, orbit_segments, sample_segments
from .nav import INSModel, InsBoundingEKF, gdop, rx_tdoa_jacobian, rx_tdoa_ranges, solve_rx_tdoa
from .nav.tdoa import tdoa_cov
from .rf.noise import NoiseInjector
from .rf.propagation import HFGroundWave, VHFLink, radio_horizon_m, two_ray_path_loss_db, wave_loss_db

DEFAULT_CFG = Path(__file__).resolve().parents[2] / "config" / "scenario_default.yaml"


# ------------------------------------------------------------------ 経路
def build_trajectory(cfg):
    ac, g = cfg["aircraft"], cfg["pnt_grid"]
    dt, v, rho = cfg["dt"], ac["speed"], ac["turn_radius"]
    pts = hex_grid(g["center"], g["n_buoys"], g["spacing"])
    plan = PNTGridPlanner(rho=rho).plan(ac["start"], pts)
    segs = list(plan.paths) + orbit_segments(end_pose(plan.paths[-1]), plan.orbit_center, plan.orbit_radius,
                                             g["orbit_duration_s"], v, rho)
    t_orbit_end = sum(sg.length for sg in segs) / v
    home = np.r_[ac["start"][:2], ac["start"][2] + np.pi]
    ret = dubins_shortest(end_pose(segs[-1]), home, rho)
    traj = sample_segments(segs + [ret], v, dt)
    t_drop = np.cumsum([p.length for p in plan.paths]) / v
    phases = {"transit": (0.0, t_drop[0]), "deploy": (t_drop[0], t_drop[-1]),
              "orbit": (t_drop[-1], t_orbit_end), "return": (t_orbit_end, traj[-1, 0])}
    return traj, pts[plan.order], t_drop, plan, phases, ret


# ------------------------------------------------------------------ ブイ（真値・推定）
def drift_engine(cfg) -> str:
    eng = cfg["drift"].get("engine", "builtin")
    if eng == "aqua" and not HAVE_AQUA:
        warnings.warn("aqua_drift 未インストールのため builtin 漂流モデルで代替します（pip install -e '.[aqua]'）")
        return "builtin"
    return eng


def make_field(cfg, surface_wind=None):
    dr = cfg["drift"]
    if drift_engine(cfg) == "aqua":
        aq = dr.get("aqua", {})
        return AquaDriftAdapter(aq.get("current_field"), tuple(aq["origin_latlon"]) if aq.get("origin_latlon") else None,
                                leeway=aq.get("leeway", 0.0), wind_mps=surface_wind or tuple(dr["wind"]))
    return OceanCurrentField(mean_current=tuple(dr["mean_current"]), wind=tuple(dr["wind"]))


def deploy_buoys(cfg, traj, drops, t_drop, rng):
    """着水点・着水時刻. aqua 有効時は自由落下と投下補正、無効時は予定点に即時着水."""
    dp = cfg["drift"].get("drop", {})
    if drift_engine(cfg) != "aqua" or not dp.get("enabled", True):
        return None, drops.astype(float), t_drop
    model = AquaDropModel(dp.get("wind"), release_correction=dp.get("release_correction", True),
                          splash_obs_sigma_m=dp.get("splash_obs_sigma_m", 100.0), rng=rng)
    res = model.deploy(traj, drops, t_drop, cfg["aircraft"]["altitude"])
    return (model, res), res.splash_xy, res.splash_t


def simulate_buoys(cfg, drops, t_drop, T, rng, field=None):
    dtb = cfg["hf"]["update_s"]
    dr = cfg["drift"]
    sim = DriftSimulator(field or make_field(cfg), diffusivity=dr["diffusivity"], rng=rng)
    tb = np.arange(0.0, T + dtb, dtb)
    N = len(drops)
    pos = np.empty((tb.size, N, 2))
    xy = drops.astype(float).copy()
    for i, t in enumerate(tb):
        pos[i] = xy
        act = t >= t_drop
        if act.any():
            xy[act] = sim.step(xy[act], t, dtb)
    return tb, pos


def estimate_buoys(cfg, tb, truth, t_drop, anchors, hf, rng):
    """各ブイの HF Rx-TDOA 測位 → 等速 KF で平滑（ブイ方向にベクトル化）."""
    N, M = truth.shape[1], len(anchors)
    dtb = tb[1] - tb[0]
    inj = NoiseInjector(rng=rng)
    fade = inj.fading_db(tb.size * N, fs=1 / dtb).reshape(tb.size, N) if cfg["hf"]["buoy_diversity"] else 0.0
    F = np.eye(4); F[0, 2] = F[1, 3] = dtb
    qa = 1e-5                                     # 漂流速度の変動（加速度 PSD）[m^2/s^3]
    Q = qa * np.array([[dtb**3 / 3, 0, dtb**2 / 2, 0], [0, dtb**3 / 3, 0, dtb**2 / 2],
                       [dtb**2 / 2, 0, dtb, 0], [0, dtb**2 / 2, 0, dtb]])
    Hm = np.hstack([np.eye(2), np.zeros((2, 2))])
    x = np.zeros((N, 4)); P = np.zeros((N, 4, 4)); init = np.zeros(N, bool)
    est = np.full((tb.size, N, 2), np.nan); cov = np.full((tb.size, N, 2, 2), np.nan)
    for i, t in enumerate(tb):
        act = t >= t_drop
        if not act.any():
            continue
        fd = fade[i, act] if np.ndim(fade) else 0.0
        z, sig = hf.measure_toa(truth[i, act], anchors, fading_db=fd[:, None] if np.ndim(fd) else 0.0)
        zt = z[:, 1:] - z[:, [0]]
        p0 = np.where(init[act, None], x[act, :2], truth[i, act] + rng.normal(0, 2e3, (act.sum(), 2)))
        fix = solve_rx_tdoa(zt, anchors, p0)
        H = rx_tdoa_jacobian(fix, anchors)
        C = np.stack([tdoa_cov(M, s) for s in sig])
        Rfix = np.linalg.inv(np.einsum("nki,nkl,nlj->nij", H, np.linalg.inv(C), H))
        idx = np.flatnonzero(act)
        new = idx[~init[idx]]
        if new.size:
            k = np.searchsorted(idx, new)
            x[new] = np.c_[fix[k], np.zeros((new.size, 2))]
            P[new] = 0.0
            P[new, :2, :2] = Rfix[k]
            P[new, 2:, 2:] = np.eye(2) * 0.5**2
            init[new] = True
        old = idx[np.isin(idx, new, invert=True)]
        if old.size:
            k = np.searchsorted(idx, old)
            xp = x[old] @ F.T
            Pp = F @ P[old] @ F.T + Q
            S = Hm @ Pp @ Hm.T + Rfix[k]
            K = Pp @ Hm.T @ np.linalg.inv(S)
            x[old] = xp + np.einsum("nij,nj->ni", K, fix[k] - xp[:, :2])
            P[old] = (np.eye(4) - K @ Hm) @ Pp
        est[i, idx] = x[idx, :2]
        cov[i, idx] = P[idx, :2, :2]
    return est, cov


# ------------------------------------------------------------------ 実行
def run(cfg: dict) -> dict:
    rng = np.random.default_rng(cfg["seed"])
    anchors = np.asarray(cfg["anchors_m"], float)
    ac, hfc, vc = cfg["aircraft"], cfg["hf"], cfg["vhf"]
    dt, alt = cfg["dt"], ac["altitude"]

    traj, drops, t_drop, plan, phases, _ = build_trajectory(cfg)
    t, p_true, v_true = traj[:, 0], traj[:, 1:3], traj[:, 3:5]
    T = t[-1]

    hf = HFGroundWave(freq_hz=hfc["freq_hz"], bandwidth_hz=hfc["bandwidth_hz"],
                      snr_ref_db=hfc["snr_ref_db"], rng=np.random.default_rng(cfg["seed"] + 1))
    drop_info, splash, t_splash = deploy_buoys(cfg, traj, drops, t_drop, np.random.default_rng(cfg["seed"] + 6))
    field = make_field(cfg, drop_info[0].surface_wind_mps() if drop_info else None)
    tb, b_true = simulate_buoys(cfg, splash, t_splash, T, np.random.default_rng(cfg["seed"] + 2), field)
    b_est, b_cov = estimate_buoys(cfg, tb, b_true, t_splash, anchors, hf, np.random.default_rng(cfg["seed"] + 3))

    # --- 観測系列の事前生成（全方式で共通）
    ins = INSModel(cfg["ins"]["bias0_sigma"], cfg["ins"]["bias_rw"], cfg["ins"]["vel_noise"],
                   rng=np.random.default_rng(cfg["seed"] + 4))
    v_ins = np.array([ins.measure(v, dt) for v in v_true])

    ac_fade = NoiseInjector(rng=np.random.default_rng(cfg["seed"] + 5)).fading_db(len(t), fs=1 / dt)
    hf_meas = {}
    k_hf = int(round(hfc["update_s"] / dt))
    for i in range(0, len(t), k_hf):
        if rng.random() < hfc["aircraft_outage_prob"]:
            continue
        z, sig = hf.measure_toa(p_true[i], anchors, fading_db=ac_fade[i])
        hf_meas[i] = (z[0, 1:] - z[0, 0], sig[0])

    link = VHFLink(freq_hz=vc["freq_hz"])
    horizon = radio_horizon_m(alt, cfg["pnt_grid"]["buoy_ant_height"])
    vhf_meas = {}
    k_v = int(round(vc["update_s"] / dt))
    sig_r = np.hypot(vc["ranging_sigma_m"], vc["buoy_sync_sigma_m"])
    for i in range(0, len(t), k_v):
        j = min(int(t[i] // (tb[1] - tb[0])), len(tb) - 1)
        act = (t[i] >= t_splash) & ~np.isnan(b_est[j, :, 0])
        if act.sum() < 3:
            continue
        bt = b_true[j, act]
        dh = np.linalg.norm(bt - p_true[i], axis=1)
        pl = two_ray_path_loss_db(dh, alt, cfg["pnt_grid"]["buoy_ant_height"], vc["freq_hz"])
        pl = pl + wave_loss_db(rng, act.sum(), hs_m=vc["hs_m"])
        ok = (dh < horizon) & (rng.random(act.sum()) < link.packet_success(pl, bits=256))
        if ok.sum() < 3:
            continue
        rho = np.hypot(dh[ok], alt) + sig_r * rng.standard_normal(ok.sum())
        idx = np.flatnonzero(act)[ok]
        vhf_meas[i] = (rho[1:] - rho[0], b_est[j, idx], b_cov[j, idx])

    # --- EKF 3 方式
    def hf_update(ekf, z, sig):
        ekf.update(z, lambda p: rx_tdoa_ranges(p, anchors)[0], lambda p: rx_tdoa_jacobian(p, anchors)[0],
                   tdoa_cov(len(anchors), sig))

    def vhf_update(ekf, z, bpos, bcov):
        def slant(p):
            return np.hypot(np.linalg.norm(p - bpos, axis=1), alt)

        def h(p):
            s = slant(p)
            return s[1:] - s[0]

        def Hp(p):
            u = (p - bpos) / slant(p)[:, None]
            return u[1:] - u[0]

        Hg = Hp(ekf.pos)
        if np.sqrt(np.trace(np.linalg.pinv(Hg.T @ Hg))) > vc.get("max_dop", 6.0):
            return                          # ブイ網が遠方で一方向に固まり幾何条件が悪い → 使わない
        u = (ekf.pos - bpos) / slant(ekf.pos)[:, None]
        var = sig_r**2 + np.einsum("ni,nij,nj->n", u, bcov, u)
        D = np.hstack([-np.ones((len(bpos) - 1, 1)), np.eye(len(bpos) - 1)])
        ekf.update(z, h, Hp, D @ np.diag(var) @ D.T)

    results = {}
    p_init = p_true[0] + rng.normal(0, ac["init_pos_sigma"], 2)       # 全方式で共通の初期推定
    for mode in ("ins_only", "hf", "hf_grid"):
        ekf = InsBoundingEKF.init(p_init, ac["init_pos_sigma"],
                                  cfg["ins"]["bias0_sigma"], vel_noise=cfg["ins"]["vel_noise"],
                                  bias_rw=cfg["ins"]["bias_rw"])
        est = np.empty_like(p_true); sd = np.empty(len(t))
        for i in range(len(t)):
            if i:
                ekf.predict(v_ins[i - 1], dt)
            if mode != "ins_only" and i in hf_meas:
                hf_update(ekf, *hf_meas[i])
            if mode == "hf_grid" and i in vhf_meas:
                vhf_update(ekf, *vhf_meas[i])
            est[i] = ekf.pos
            sd[i] = np.sqrt(np.trace(ekf.P[:2, :2]))
        results[mode] = {"est": est, "err": np.linalg.norm(est - p_true, axis=1), "sd": sd}

    buoy_err = np.linalg.norm(b_est - b_true, axis=-1)
    return {"cfg": cfg, "t": t, "p_true": p_true, "drops": drops, "t_drop": t_drop, "splash": splash,
            "t_splash": t_splash, "drop": drop_info[1] if drop_info else None, "engine": drift_engine(cfg), "plan": plan,
            "phases": phases, "tb": tb, "b_true": b_true, "b_est": b_est, "buoy_err": buoy_err,
            "anchors": anchors, "results": results,
            "n_hf": len(hf_meas), "n_vhf": len(vhf_meas), "horizon_m": horizon,
            "gdop_center": float(gdop(np.asarray(cfg["pnt_grid"]["center"], float), anchors)[0])}


def summarize(out: dict) -> dict:
    t = out["t"]
    s = {"words": out["plan"].words, "path_length_km": round(out["plan"].length / 1e3, 1),
         "duration_min": round(t[-1] / 60, 1), "gdop_at_grid_center": round(out["gdop_center"], 2),
         "drift_engine": out["engine"],
         "hf_epochs": out["n_hf"], "vhf_epochs": out["n_vhf"],
         "buoy_pos_err_final_m": np.round(out["buoy_err"][-1], 1).tolist(), "aircraft": {}}
    if out["drop"] is not None:
        d = out["drop"]
        s["drop"] = {"splash_miss_m": np.round(d.miss_m, 1).tolist(),
                     "fall_time_s": np.round(d.splash_t - d.release_t, 1).tolist(),
                     "wind_est_mps": np.round(d.wind_est_mps, 2).tolist()}
    for mode, r in out["results"].items():
        s["aircraft"][mode] = {}
        for ph, (a, b) in out["phases"].items():
            m = (t >= a) & (t <= b)
            e = r["err"][m]
            s["aircraft"][mode][ph] = {"mean_m": round(float(e.mean()), 1),
                                       "p95_m": round(float(np.percentile(e, 95)), 1),
                                       "max_m": round(float(e.max()), 1)}
        s["aircraft"][mode]["final_m"] = round(float(r["err"][-1]), 1)
    return s


def plot(out: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))
    P = out["p_true"] / 1e3
    a1.plot(P[:, 0], P[:, 1], "k-", lw=0.8, label="真航跡")
    for mode, c in (("ins_only", "tab:red"), ("hf", "tab:orange"), ("hf_grid", "tab:blue")):
        E = out["results"][mode]["est"] / 1e3
        a1.plot(E[:, 0], E[:, 1], "-", c=c, lw=0.6, alpha=0.8, label=mode)
    B = out["b_true"] / 1e3
    for n in range(B.shape[1]):
        a1.plot(B[:, n, 0], B[:, n, 1], "g-", lw=1.5)
    a1.plot(*(out["drops"] / 1e3).T, "g^", ms=7, mfc="none", label="PNT ブイ予定投下点")
    a1.plot(*(out["splash"] / 1e3).T, "g.", ms=6, label="着水点")
    a1.set_aspect("equal"); a1.set_xlabel("East [km]"); a1.set_ylabel("North [km]")
    a1.set_title("航跡・ブイ漂流（アンカーは図外）"); a1.legend(fontsize=8); a1.grid(alpha=0.3)

    t = out["t"] / 60
    for mode, c in (("ins_only", "tab:red"), ("hf", "tab:orange"), ("hf_grid", "tab:blue")):
        a2.semilogy(t, np.maximum(out["results"][mode]["err"], 1), c=c, lw=0.8, label=mode)
    for k, (ph, (a, b)) in enumerate(out["phases"].items()):
        a2.axvspan(a / 60, b / 60, color=f"C{k}", alpha=0.07)
        a2.text((a + b) / 120, 1.3, ph, ha="center", fontsize=8)
    a2.set_xlabel("GPS ロスト後経過時間 [min]"); a2.set_ylabel("水平位置誤差 [m]")
    a2.set_title("自機位置誤差（INS バウンディング効果）"); a2.legend(); a2.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _set_jp_font():
    try:
        import matplotlib
        from matplotlib import font_manager
        for name in ("Noto Sans CJK JP", "Hiragino Sans", "IPAexGothic", "Yu Gothic"):
            if any(name in f.name for f in font_manager.fontManager.ttflist):
                matplotlib.rcParams["font.family"] = name
                return
    except Exception:
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--out", default="out")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--llm", action="store_true", help="結果要約を LiteLLM 経由で解析")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    out = run(cfg)
    s = summarize(out)
    od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    (od / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(s, ensure_ascii=False, indent=2))
    if not a.no_plot:
        _set_jp_font(); plot(out, od / "scenario.png"); print(f"plot -> {od / 'scenario.png'}")
    if a.llm:
        from .gateway import LiteLLMClient
        print(LiteLLMClient().analyze_summary(s))


if __name__ == "__main__":
    main()
