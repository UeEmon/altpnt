"""長距離シナリオ: 上空波 Rx-TDOA で INS の誤差増大を抑える EKF（周波数は CHUCK で時刻ごとに選ぶ）.

流れ
  1. 出発点 → 作戦海域（例: 1,000 NM 先）→ 周回 → 帰投の航跡を、航路中点を原点とする
     正距方位図法の平面 [m] で作る（Dubins 経路 + 周回円）。
  2. 航路に沿って一定間隔の地点ごとに、各送信所から CHUCK（VOACAP）で 24 時間の予測を取り、
     時刻ごとの周波数・伝搬モード・仮想反射高度を決める（hfprop.plan_link）。
  3. 真値の観測: 各送信所の群路長 P(D, h') + 電離層の残差 e_i(t) + TOA 雑音。
     e_i(t) は補正方式ごとの大きさ σ_i と相関時間（iono_tau_s）を持つ Gauss-Markov 過程。
       model        … σ = dP/dh'·20 km（気候学モデルのみ）
       realtime     … σ = dP/dh'·5 km（電離層をリアルタイムに推定）
       differential … 作戦海域の参照ブイから半径 diff.radius_km 以内では σ = diff.sigma_km、外では realtime
  4. EKF 状態 = [位置 2, INS 速度バイアス 2, 送信所ごとの電離層バイアス N]。TDOA（局間差）で
     受信機の時計誤差を消す。電離層バイアスは e = σ·u（u は連続な Gauss-Markov）とみなし、周波数・モード・
     補正方式で σ が変わったときは推定値と共分散を σ の比で伸縮する（初期化すると相関を捨てて自信過剰になる）。

実行: CHUCK_APP_DIR=... VOACAP_BIN=... VOACAP_MODEL=... altpnt-longrange --out out/longrange
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

from .dubins.path import dubins_shortest
from .dubins.planner import end_pose, orbit_segments, sample_segments
from .geo import aeqd_forward, aeqd_inverse, gc_distance_m
from .hfprop import ChuckEngineBackend, ChuckHttpBackend, chuck_request, plan_link
from .hfprop.plan import destination, group_path_km, hops_of
from .nav.ins import INSModel
from .nav.tdoa import tdoa_cov
from .constants import C
from .skywave import resolve_ssn
from .spaceweather import Timeline
from scipy.stats import chi2

DEFAULT_CFG = Path(__file__).resolve().parents[2] / "config" / "longrange_default.yaml"
RNP_M = {"RNP-10": 18520.0, "RNP-4": 7408.0, "RNP-2": 3704.0}
HAL_M = {"RNP-10": 2 * 18520.0, "RNP-4": 2 * 7408.0}      # 封じ込め（containment）限界 = 2×RNP


# ------------------------------------------------------------------ 航跡
def build_route(cfg):
    lr = cfg["route"]
    s_lat, s_lon = lr["start"]
    o_lat, o_lon = destination(s_lat, s_lon, lr["op_area"]["bearing"], lr["op_area"]["nm"] * 1.852)
    lat0, lon0 = (s_lat + o_lat) / 2, (s_lon + o_lon) / 2
    sx, sy = aeqd_forward(s_lat, s_lon, lat0, lon0)
    ox, oy = aeqd_forward(o_lat, o_lon, lat0, lon0)
    ac = cfg["aircraft"]
    v, rho = ac["speed"], ac["turn_radius"]
    hdg = math.atan2(oy - sy, ox - sx)
    start = np.array([float(sx), float(sy), hdg])
    orb = orbit_segments(start, (float(ox), float(oy)), lr["orbit"]["radius_km"] * 1e3,
                         lr["orbit"]["duration_h"] * 3600, v, rho)
    home = np.array([float(sx), float(sy), hdg + math.pi])
    ret = dubins_shortest(end_pose(orb[-1]), home, rho)
    traj = sample_segments(orb + [ret], v, cfg["dt"])
    t_arrive = orb[0].length / v
    t_leave = t_arrive + orb[1].length / v
    phases = {"進出": (0.0, t_arrive), "作戦海域": (t_arrive, t_leave), "帰投": (t_leave, traj[-1, 0])}
    return traj, (lat0, lon0), (float(ox), float(oy)), (o_lat, o_lon), phases


# ------------------------------------------------------------------ 時刻別リンク（CHUCK）
def link_table(cfg, sky_cfg, traj, origin, stations, backend, ssn, muf_factors=(1.0,)):
    """航路の地点ごと × 送信所ごとの 24 時間のリンク計画.

    戻り値: (各時刻 → 地点番号, {muf_factor: links[地点][局][時]})。muf_factor < 1 は電離層嵐で MUF が下がった状態。
    """
    step = max(1, int(cfg["route_sample_km"] * 1e3 / (cfg["aircraft"]["speed"] * cfg["dt"])))
    idx = np.arange(0, len(traj), step)
    lat, lon = aeqd_inverse(traj[idx, 1], traj[idx, 2], *origin)
    ep = cfg["epoch"]
    tables = {f: [] for f in muf_factors}
    for la, lo in zip(np.atleast_1d(lat), np.atleast_1d(lon)):
        per = {f: [] for f in muf_factors}
        for st in stations:
            d_km = float(gc_distance_m(st[0], st[1], la, lo)) / 1e3
            hours = backend.predict(chuck_request(tuple(st), (float(la), float(lo)), ep["year"], ep["month"], ssn,
                                                  sky_cfg["frequencies"], sky_cfg["link"]))
            for f in muf_factors:
                per[f].append(plan_link(hours, d_km, sky_cfg["rel_min"], sky_cfg["ground_wave_km"], muf_factor=f))
        for f in muf_factors:
            tables[f].append(per[f])
    sample_of = np.minimum(np.round(np.arange(len(traj)) / step).astype(int), len(idx) - 1)
    return sample_of, tables


# ------------------------------------------------------------------ 観測モデル
def group_path_m(lat, lon, st, lh) -> float:
    d = float(gc_distance_m(st[0], st[1], lat, lon))
    if lh.kind == "ground":
        return d
    return group_path_km(d / 1e3, lh.vhite_km, hops_of(lh.mode)) * 1e3


def toa_sigma_m(lh, bandwidth_hz=5000.0) -> float:
    if lh.kind == "ground":
        return 50.0
    inband = (lh.snr_dbhz or 0) - 10 * math.log10(bandwidth_hz)
    beta = bandwidth_hz / math.sqrt(12)
    return max(C / (2 * math.pi * beta * math.sqrt(2 * 10 ** (inband / 10) * bandwidth_hz)), 30.0)


def resid_sigma_m(lh, tier, in_diff, cfg) -> float:
    if lh.kind == "ground":
        return cfg["ground_wave_sigma_km"] * 1e3
    if tier == "differential" and in_diff:
        s = cfg["diff"]["sigma_km"] * 1e3
    else:
        s = lh.dPdh * cfg["sigma_h_km"]["realtime" if tier == "differential" else tier] * 1e3
    return s * (1 + 2 * (lh.mprob or 0))


# ------------------------------------------------------------------ 実行
def _measure_update(x, Pm, use, lk, z_all, stations, origin, R_sig):
    """use の局で TDOA 観測を組み立て、(イノベーション y, H, R, S, NIS) を返す."""
    n = len(x)
    la, lo = aeqd_inverse(x[0], x[1], *origin)
    h0 = np.array([group_path_m(la, lo, stations[s], lk[s]) for s in use])
    J = np.zeros((len(use), 2))
    for ax_ in range(2):
        dx = np.zeros(2); dx[ax_] = 10.0
        la2, lo2 = aeqd_inverse(x[0] + dx[0], x[1] + dx[1], *origin)
        J[:, ax_] = (np.array([group_path_m(la2, lo2, stations[s], lk[s]) for s in use]) - h0) / 10.0
    rho = np.array([z_all[s] for s in use])
    pred = h0 + x[[4 + s for s in use]]
    y = (rho[1:] - rho[0]) - (pred[1:] - pred[0])
    H = np.zeros((len(use) - 1, n))
    H[:, :2] = J[1:] - J[0]
    for r, s in enumerate(use[1:]):
        H[r, 4 + s] = 1.0
    H[:, 4 + use[0]] -= 1.0
    R = tdoa_cov(len(use), np.array([R_sig[s] for s in use]))
    S = H @ Pm @ H.T + R
    return y, H, R, S, float(y @ np.linalg.solve(S, y))


def run(cfg: dict, sky_cfg: dict, backend, depart_utc: float, with_extra=(), ssn_url=None, events=None) -> dict:
    stations_d = dict(sky_cfg["stations"])
    for n_ in with_extra:
        stations_d[n_] = sky_cfg["extra_stations"][n_]
    names, stations = list(stations_d), [tuple(v) for v in stations_d.values()]
    N = len(stations)
    ssn, ssn_src = resolve_ssn(cfg["epoch"], ssn_url)

    traj, origin, op_xy, op_ll, phases = build_route(cfg)
    t, P_true, V_true = traj[:, 0], traj[:, 1:3], traj[:, 3:5]
    dt = cfg["dt"]
    lat_t, lon_t = aeqd_inverse(P_true[:, 0], P_true[:, 1], *origin)
    tl = Timeline(events, t, names, cfg["epoch"]["year"], cfg["epoch"]["month"], depart_utc)
    f_storm = tl.storm_cfg.get("muf_factor", 1.0) if tl.storm_cfg else 1.0
    sample_of, tables = link_table(cfg, sky_cfg, traj, origin, stations, backend, ssn, sorted({1.0, f_storm}))
    hour_of = ((depart_utc + t / 3600.0) % 24).astype(int)
    t_arrive = phases["作戦海域"][0]
    in_diff = (np.hypot(P_true[:, 0] - op_xy[0], P_true[:, 1] - op_xy[1]) <= cfg["diff"]["radius_km"] * 1e3) & (t >= t_arrive)

    ins = INSModel(cfg["ins"]["bias0_sigma"], cfg["ins"]["bias_rw"], cfg["ins"]["vel_noise"],
                   rng=np.random.default_rng(cfg["seed"] + 1))
    v_ins = np.array([ins.measure(v, dt) for v in V_true])
    taus = cfg["iono_tau_s"] if isinstance(cfg["iono_tau_s"], dict) else {m: cfg["iono_tau_s"] for m in cfg["modes"]}
    # 電離層残差の単位 Gauss-Markov 過程（相関時間ごと。駆動乱数は全方式で共通）
    drive = np.random.default_rng(cfg["seed"] + 3).standard_normal((len(t), N))
    u0 = np.random.default_rng(cfg["seed"] + 2).standard_normal(N)
    U_by_tau = {}
    for tau in {float(v) for v in taus.values()}:
        a = math.exp(-dt / tau)
        U = np.empty((len(t), N)); u = u0.copy()
        for i in range(len(t)):
            U[i] = u
            u = a * u + math.sqrt(1 - a * a) * drive[i]
        U_by_tau[tau] = U
    noise = np.random.default_rng(cfg["seed"] + 4).standard_normal((len(t), N))
    k_meas = int(round(cfg["meas_interval_s"] / dt))
    p0 = P_true[0] + np.random.default_rng(cfg["seed"] + 5).normal(0, cfg["aircraft"]["init_pos_sigma"], 2)
    integ = cfg.get("integrity", {})
    imode, pfa, k_hpl = integ.get("mode", "fde"), integ.get("pfa", 1e-3), integ.get("k_hpl", 5.33)
    adapt = integ.get("adaptive") or {}            # 検知が続いたら嵐とみなして電離層残差の見積もりを広げる
    sw_alert = bool(integ.get("space_weather_alert"))   # 外部の宇宙天気警報（例: CHUCK の宇宙天気）で嵐を知る
    infl_k = float(adapt.get("inflate", integ.get("inflate", 3.0)))

    results = {}
    for mode in cfg["modes"]:
        tau = float(taus.get(mode, taus.get("realtime", 1800)))
        a1, U = math.exp(-dt / tau), U_by_tau[tau]
        n = 4 + N
        x = np.zeros(n); x[:2] = p0
        Pm = np.zeros((n, n))
        Pm[:2, :2] = np.eye(2) * cfg["aircraft"]["init_pos_sigma"] ** 2
        Pm[2:4, 2:4] = np.eye(2) * cfg["ins"]["bias0_sigma"] ** 2
        sig_e = np.full(N, 1e3)
        seen = [False] * N
        F = np.eye(n); F[0, 2] = F[1, 3] = -dt; F[4:, 4:] *= a1
        qv, qb = (cfg["ins"]["vel_noise"] * dt) ** 2, cfg["ins"]["bias_rw"] ** 2 * dt
        est = np.empty((len(t), 2)); n_used = np.zeros(len(t), int); sd = np.empty(len(t)); hpl = np.empty(len(t))
        stats = {"epochs": 0, "detected": 0, "excluded": 0, "coasted": 0, "inflated_epochs": 0}
        recent: list[int] = []
        infl = 1.0
        for i in range(len(t)):
            if i:
                x[:2] += (v_ins[i - 1] - x[2:4]) * dt
                x[4:] *= a1
                Q = np.zeros(n); Q[:2] = qv; Q[2:4] = qb; Q[4:] = sig_e**2 * (1 - a1 * a1)
                Pm = F @ Pm @ F.T + np.diag(Q)
            if mode != "ins_only" and i % k_meas == 0:
                table = tables[f_storm] if tl.storm[i] > 0.5 else tables[1.0]
                lk = [table[sample_of[i]][s][hour_of[i]] for s in range(N)]
                use = [s for s in range(N) if lk[s].usable
                       and not tl.blocked(i, s, lat_t[i], lon_t[i], stations[s], lk[s].kind)]
                storm_flag = (adapt and sum(recent[-int(adapt.get("window", 30)):]) >= int(adapt.get("trigger", 5))) \
                    or (sw_alert and tl.storm[i] > 0.05)
                new_infl = infl_k if storm_flag else 1.0
                if new_infl != infl:
                    # 見積もりを広げる／戻す: 推定値はそのまま、分散だけ足し引きする（モデルの誤りへの備え）
                    for s in range(N):
                        if seen[s]:
                            Pm[4 + s, 4 + s] = max(Pm[4 + s, 4 + s] + sig_e[s] ** 2 * ((new_infl / infl) ** 2 - 1), 1.0)
                            sig_e[s] *= new_infl / infl
                    infl = new_infl
                stats["inflated_epochs"] += infl > 1
                for s in use:
                    sg = resid_sigma_m(lk[s], mode, bool(in_diff[i]), cfg) * infl
                    if not seen[s]:                 # 初めて使う局: 事前分布 N(0, σ²)
                        x[4 + s] = 0.0
                        Pm[4 + s, :] = 0.0; Pm[:, 4 + s] = 0.0
                        Pm[4 + s, 4 + s] = sg**2
                        seen[s] = True
                    elif abs(sg / sig_e[s] - 1) > 1e-6:
                        # 周波数・モード・補正方式で残差の大きさが変わった: e = σ·u（u は連続）として比で伸縮し、
                        # それまでの推定と位置との相関は保つ
                        r = sg / sig_e[s]
                        x[4 + s] *= r
                        Pm[4 + s, :] *= r; Pm[:, 4 + s] *= r
                    sig_e[s] = sg
                n_used[i] = len(use)
                if len(use) >= 2:
                    stats["epochs"] += 1
                    tier = mode
                    z_all, R_sig = {}, {}
                    for s in use:
                        z_all[s] = (group_path_m(lat_t[i], lon_t[i], stations[s], lk[s])
                                    + resid_sigma_m(lk[s], mode, bool(in_diff[i]), cfg) * tl.sigma_factor(i) * U[i, s]
                                    + (tl.storm_bias_m(i, lk[s].dPdh, "differential" if (tier == "differential" and in_diff[i]) else
                                                       ("realtime" if tier == "differential" else tier)) if lk[s].kind == "sky" else 0.0)
                                    + toa_sigma_m(lk[s]) * noise[i, s])
                        R_sig[s] = toa_sigma_m(lk[s])
                    y, H, R, S, nis = _measure_update(x, Pm, use, lk, z_all, stations, origin, R_sig)
                    ok = True
                    fail = imode != "none" and nis > chi2.ppf(1 - pfa, len(y))
                    recent.append(int(fail))
                    if fail:
                        stats["detected"] += 1
                        ok = False
                        if imode == "fde" and len(use) >= 4:
                            best = None
                            for s_ex in use:            # 1 局ずつ除いて検定し直す
                                sub = [s for s in use if s != s_ex]
                                cand = _measure_update(x, Pm, sub, lk, z_all, stations, origin, R_sig)
                                ratio = cand[4] / chi2.ppf(1 - pfa, len(cand[0]))
                                if ratio <= 1 and (best is None or ratio < best[0]):
                                    best = (ratio, s_ex, cand)
                            if best is not None:
                                _, s_ex, (y, H, R, S, nis) = best
                                stats["excluded"] += 1
                                ok = True
                                # 除外した局のバイアスは信用を落として再収束させる
                                Pm[4 + s_ex, :] = 0.0; Pm[:, 4 + s_ex] = 0.0
                                Pm[4 + s_ex, 4 + s_ex] = (3 * sig_e[s_ex]) ** 2
                        if not ok:
                            stats["coasted"] += 1      # 使える組がない: INS で惰性航法
                    if ok:
                        K = np.linalg.solve(S, H @ Pm).T
                        x = x + K @ y
                        IKH = np.eye(n) - K @ H
                        Pm = IKH @ Pm @ IKH.T + K @ R @ K.T
            est[i] = x[:2]
            sd[i] = math.sqrt(max(Pm[0, 0] + Pm[1, 1], 0.0))
            hpl[i] = k_hpl * math.sqrt(max(np.linalg.eigvalsh(Pm[:2, :2])[-1], 0.0))
        err = np.hypot(*(est - P_true).T)
        results[mode] = {"est": est, "err": err, "n_used": n_used, "sd": sd, "hpl": hpl, "stats": stats}

    return {"t": t, "P_true": P_true, "lat": lat_t, "lon": lon_t, "origin": origin, "op_ll": op_ll,
            "phases": phases, "results": results, "stations": names, "station_ll": stations,
            "depart_utc": depart_utc, "ssn": ssn, "ssn_source": ssn_src, "events": events or [],
            "storm": tl.storm, "flare": tl.flare}


def summarize(out: dict) -> dict:
    t = out["t"]
    s = {"depart_utc": out["depart_utc"], "ssn": out["ssn"], "ssn_source": out["ssn_source"], "events": out["events"],
         "stations": out["stations"], "duration_h": round(t[-1] / 3600, 1), "modes": {}}
    for mode, r in out["results"].items():
        m = {}
        for ph, (a, b) in out["phases"].items():
            e = r["err"][(t >= a) & (t <= b)] / 1e3
            sdv = r["sd"][(t >= a) & (t <= b)] / 1e3
            m[ph] = {"mean_km": round(float(e.mean()), 2), "p95_km": round(float(np.percentile(e, 95)), 2),
                     "consistency": round(float(np.sqrt(np.mean(e**2)) / max(np.sqrt(np.mean(sdv**2)), 1e-9)), 2),
                     "max_km": round(float(e.max()), 2),
                     **{k: f"{round(100 * float(np.mean(e * 1e3 <= v)))}%" for k, v in RNP_M.items()},
                     # 保護レベルが誤差を下回った時間（完全性の破れ）と、保護レベルが警報限界内に収まった時間（可用性）
                     "integrity_fail": f"{round(100 * float(np.mean(r['err'][(t >= a) & (t <= b)] > r['hpl'][(t >= a) & (t <= b)])), 1)}%",
                     **{f"avail_{k}": f"{round(100 * float(np.mean(r['hpl'][(t >= a) & (t <= b)] <= v)))}%" for k, v in HAL_M.items()}}
        m["final_km"] = round(float(r["err"][-1] / 1e3), 2)
        m["fde"] = r["stats"]
        s["modes"][mode] = m
    return s


def aggregate(runs: list[dict]) -> dict:
    """モンテカルロの各回の summarize を平均する（RNP 適合率も平均、consistency は二乗平均）."""
    if len(runs) == 1:
        return dict(runs[0], runs=1)
    base = {k: runs[0].get(k) for k in ("depart_utc", "ssn", "ssn_source", "stations", "duration_h", "events",
                                         "emissions", "vhf_report_bps", "buoy_end_h", "deploy_h")}
    base["runs"] = len(runs)
    base["modes"] = {}
    for mode, m0 in runs[0]["modes"].items():
        mm = {}
        for ph, d0 in m0.items():
            if not isinstance(d0, dict):
                mm[ph] = round(float(np.mean([r["modes"][mode][ph] for r in runs])), 2)
                continue
            d = {}
            for k in d0:
                vals = [r["modes"][mode][ph][k] for r in runs]
                if isinstance(d0[k], str):
                    mean = float(np.mean([float(v[:-1]) for v in vals]))
                    d[k] = f"{round(mean, 1)}%" if k == "integrity_fail" else f"{round(mean)}%"
                elif k == "max_km":
                    d[k] = round(float(np.max(vals)), 2)
                elif k == "consistency":
                    d[k] = round(float(np.sqrt(np.mean(np.square(vals)))), 2)
                else:
                    d[k] = round(float(np.mean(vals)), 2)
            mm[ph] = d
        base["modes"][mode] = mm
    return base


LABEL = {"ins_only": "INS 単独", "model": "モデル補正", "realtime": "リアルタイム推定", "differential": "＋参照ブイ差分"}
COLOR = {"ins_only": "tab:red", "model": "tab:orange", "realtime": "tab:blue", "differential": "tab:green"}


def plot(out: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for nm in ("Noto Sans CJK JP", "Hiragino Sans", "IPAexGothic"):
        if any(nm in f.name for f in font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = nm
            break
    fig = plt.figure(figsize=(15, 6.5))
    a1 = fig.add_subplot(1, 2, 1)
    a1.plot(out["lon"], out["lat"], "k-", lw=1, label="真航跡")
    for mode in ("model", "realtime"):
        if mode in out["results"]:
            la, lo = aeqd_inverse(*out["results"][mode]["est"].T, *out["origin"])
            a1.plot(lo, la, "-", c=COLOR[mode], lw=0.6, alpha=0.8, label=LABEL[mode])
    for nm, (la, lo) in zip(out["stations"], out["station_ll"]):
        a1.plot(lo, la, "b^", ms=8)
        a1.text(lo + 0.4, la, nm, fontsize=9, color="navy")
    a1.plot(out["op_ll"][1], out["op_ll"][0], "g*", ms=12, label="作戦海域（参照ブイ）")
    a1.set_xlabel("経度 [°E]"); a1.set_ylabel("緯度 [°N]"); a1.grid(alpha=0.3); a1.legend(fontsize=8)
    a1.set_aspect(1 / math.cos(math.radians(np.mean(out["lat"]))))
    a1.set_title("航跡と送信所")
    a2 = fig.add_subplot(1, 2, 2)
    th = out["t"] / 3600
    for mode, r in out["results"].items():
        a2.semilogy(th, np.maximum(r["err"] / 1e3, 0.01), c=COLOR[mode], lw=0.9, label=LABEL[mode])
        if mode in ("model", "realtime") and out.get("events"):
            a2.semilogy(th, r["hpl"] / 1e3, "--", c=COLOR[mode], lw=0.7, alpha=0.8, label=LABEL[mode] + " 保護レベル")
    for key, col, lab in (("storm", "purple", "電離層嵐"), ("flare", "gold", "太陽フレア")):
        sig = np.asarray(out.get(key, []), float)
        if sig.size and sig.max() > 0:
            a2.fill_between(th, 0.01, 200, where=sig > 0.05, color=col, alpha=0.12, step="mid", label=lab)
    for k, v in (("RNP-10", 18.52), ("RNP-4", 7.408)):
        a2.axhline(v / 2, ls=":", c="gray", lw=0.8)
        a2.text(th[-1], v / 2, f" {k}\n(95%: 2drms)", fontsize=7, va="center", color="gray")
    for j, (ph, (a, b)) in enumerate(out["phases"].items()):
        a2.axvspan(a / 3600, b / 3600, color=f"C{j}", alpha=0.06)
        a2.text((a + b) / 7200, 0.012, ph, ha="center", fontsize=8)
    ref = next((r for m, r in out["results"].items() if m != "ins_only"), None)
    if ref is not None:
        ax3 = a2.twinx()
        meas = ref["n_used"] > 0
        ax3.step(th[meas], ref["n_used"][meas], where="post", c="gray", lw=0.6, alpha=0.6)
        ax3.set_ylim(0, 3 * len(out["stations"]))
        ax3.set_ylabel("使えた送信所数", color="gray")
    a2.set_xlabel(f"GPS ロスト後の経過時間 [h]（出発 {out['depart_utc']:02.0f} UTC）")
    a2.set_ylabel("水平位置誤差 [km]")
    a2.set_ylim(0.01, 200)
    a2.grid(alpha=0.3, which="both"); a2.legend(fontsize=8, loc="upper left")
    ev = "・".join(e["type"] + (f"({e['station']})" if e.get("station") else "") for e in out.get("events", [])) or "異常なし"
    a2.set_title(f"上空波 Rx-TDOA による INS バウンディング（SSN {out['ssn']:.0f}、{ev}）")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="長距離シナリオ（上空波 Rx-TDOA + EKF、周波数は CHUCK）")
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--out", default="out/longrange")
    ap.add_argument("--chuck-url")
    ap.add_argument("--ssn-from")
    ap.add_argument("--with", dest="extra", nargs="*", default=[])
    ap.add_argument("--depart-utc", type=float, nargs="*", help="出発時刻 UTC（複数可）。省略時は config")
    ap.add_argument("--runs", type=int, default=1, help="モンテカルロ回数（乱数の種を変えて平均）")
    ap.add_argument("--scenario", nargs="*", help="config の scenarios から選ぶ（省略時は nominal のみ）")
    ap.add_argument("--integrity", choices=["none", "detect", "fde"], help="完全性監視の方式（config を上書き）")
    ap.add_argument("--adaptive", action="store_true", help="検知が続いたら電離層残差の見積もりを広げる")
    ap.add_argument("--sw-alert", action="store_true", help="宇宙天気警報で嵐の期間を知り、見積もりを広げる")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    sky_path = Path(cfg["skywave_config"])
    if not sky_path.is_absolute():
        sky_path = Path(a.config).resolve().parent / sky_path
    sky_cfg = yaml.safe_load(sky_path.read_text(encoding="utf-8"))
    od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    cache = od.parent / "chuck_cache"
    backend = ChuckHttpBackend(a.chuck_url, cache_dir=cache) if a.chuck_url else ChuckEngineBackend(cache_dir=cache)
    ssn_url = a.ssn_from or a.chuck_url or sky_cfg.get("chuck_url")
    if a.integrity:
        cfg["integrity"] = dict(cfg.get("integrity", {}), mode=a.integrity)
    if a.adaptive:
        cfg["integrity"] = dict(cfg.get("integrity", {}), adaptive=cfg.get("integrity", {}).get("adaptive_cfg", {"window": 30, "trigger": 5, "inflate": 3.0}))
    if a.sw_alert:
        cfg["integrity"] = dict(cfg.get("integrity", {}), space_weather_alert=True)
    scen = cfg.get("scenarios", {"nominal": []})
    summaries = []
    for name in a.scenario or ["nominal"]:
        for dep in a.depart_utc or cfg["depart_utc"]:
            runs = []
            for k in range(a.runs):
                c = dict(cfg, seed=cfg["seed"] + 1000 * k)
                out = run(c, sky_cfg, backend, dep, a.extra, ssn_url, events=scen[name])
                runs.append(summarize(out))
                if k == 0:
                    plot(out, od / f"longrange_{name}_dep{int(dep):02d}utc.png")
            g = aggregate(runs)
            g["scenario"] = name
            summaries.append(g)
    (od / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
