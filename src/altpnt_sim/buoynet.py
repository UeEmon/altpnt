"""ブイ中心構成のシナリオ: RF ブイが HF を受信し、母機はブイ網を基準に UHF/VHF で自機位置を求める.

構成（2026-10-28 確定の前提）
  - HF 送信所（3〜30 MHz、周波数は CHUCK で時刻ごとに選ぶ）→ RF ブイの SDR が複数局を受信
  - ブイは測定結果を VHF（120 kbps）で母機へ送る。母機は UHF でブイにコマンドを送る
  - 母機は HF を受信しない。ブイは作戦海域にだけ置く（進出・帰投は INS 単独）
  - 母機の測位は 2 つを併用:
      UHF TOA … 母機の UHF 送信を各ブイが受け、HF で合わせた時計で到着時刻を記録 → 到着時刻差（ブイ間）で母機位置
      VHF TDOA … 各ブイの VHF 送信の到着時刻差を母機の受信機で測る

推定の 2 段構え
  1. ブイ網の推定器（KF）: 状態 = ブイごとの [x, y, vx, vy, 時計オフセット, 時計ドリフト]（距離換算 m）
     + 送信所ごとの電離層バイアス（全ブイ共通。ブイ間は数十 km で、遠方の送信所までの経路がほぼ同じため）
     観測 = 各ブイでの各送信所の TOA = 群路長 + 電離層バイアス + 時計オフセット + 雑音（送信所の時刻は既知）
     ※ 共通の電離層バイアスはブイ網全体の平行移動として残る（相対位置は精密、絶対位置は時間平均で改善）
  2. 母機の EKF: 状態 = [位置 2, INS 速度バイアス 2]。ブイ網の推定値と共分散を使って UHF/VHF の
     到着時刻差を処理する（ブイの位置・時計の不確かさは観測誤差として取り込む）

SDR は局ごとに周波数が違うので、時分割で順番に受信する（cycle_s ごとに全局を一巡）とみなす。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

from .drift import DriftSimulator, OceanCurrentField
from .dubins.planner import hex_grid
from .geo import aeqd_inverse, gc_distance_m
from .hfprop import ChuckEngineBackend, ChuckHttpBackend, chuck_request, plan_link
from .hfprop.plan import link_for_freq
from .constants import C
from .hfprop.chuck import link_bandwidth_hz, parse_emission
from .longrange import build_route, group_path_m, aggregate as _aggregate
from .nav.ins import INSModel
from .nav.tdoa import tdoa_cov
from .skywave import resolve_ssn
from .spaceweather import Timeline

DEFAULT_CFG = Path(__file__).resolve().parents[2] / "config" / "buoynet_default.yaml"
RNP_M = {"RNP-10": 18520.0, "RNP-4": 7408.0, "RNP-2": 3704.0, "RNP-1": 1852.0}


# ------------------------------------------------------------------ HF リンク（作戦海域の中心で時刻別に計画）
def op_links(sky_cfg, stations, op_ll, epoch, ssn, backend, emissions=None):
    """作戦海域の中心での周波数計画. 戻り値: (links[局][時] = その時刻の最適な周波数, raw[局] = 24 時間の全周波数, 距離 km).

    emissions[局] は送信所ごとの電波型式（帯域幅は電波型式で決まり、CHUCK の要求 SNR の帯域換算に使う）。
    """
    links, raw, dist = [], [], []
    for k, st in enumerate(stations):
        d_km = float(gc_distance_m(st[0], st[1], *op_ll)) / 1e3
        hours = backend.predict(chuck_request(tuple(st), tuple(op_ll), epoch["year"], epoch["month"], ssn,
                                              sky_cfg["frequencies"], sky_cfg["link"],
                                              emission=emissions[k] if emissions else None))
        links.append(plan_link(hours, d_km, sky_cfg["rel_min"], sky_cfg["ground_wave_km"]))
        raw.append({b["hour_utc"]: b for b in hours})
        dist.append(d_km)
    return links, raw, dist


class CommandedPlan:
    """UHF コマンドでブイに指示された受信周波数（送信所ごと）.

    母機は CHUCK の計画のうち、今の時刻から horizon_h 時間分をコマンドで送る。ブイは指示された時刻の周波数を
    受信し、指示の範囲を過ぎたら最後に指示された周波数を使い続ける。母機がブイ網の UHF 範囲にいる間は
    update_min ごとに指示し直す（deploy_only なら投下時の 1 回だけ）。
    """

    def __init__(self, links, horizon_h: int, update_min: float, deploy_only: bool):
        self.links, self.horizon, self.update_s, self.once = links, max(1, int(horizon_h)), update_min * 60, deploy_only
        self.table: dict[int, list[float | None]] = {}
        self.last_t = -1e18
        self.count = 0

    def maybe_command(self, t_s: float, hour: int, in_range: bool) -> None:
        if not in_range or (self.once and self.count) or t_s - self.last_t < self.update_s:
            return
        self.table = {(hour + k) % 24: [per[(hour + k) % 24].freq_mhz for per in self.links] for k in range(self.horizon)}
        self.last_hour = (hour + self.horizon - 1) % 24
        self.last_t, self.count = t_s, self.count + 1

    def freqs(self, hour: int) -> list[float | None] | None:
        if not self.table:
            return None
        return self.table.get(hour, self.table[self.last_hour])


def iono_sigma_m(lh, tier, cfg) -> float:
    if lh.kind == "ground":
        return cfg["ground_wave_sigma_km"] * 1e3
    return lh.dPdh * cfg["sigma_h_km"][tier] * 1e3 * (1 + 2 * (lh.mprob or 0))


def buoy_toa_sigma_m(lh, bandwidth_hz: float, integ_s: float) -> float:
    """ブイでの包絡線 TOA の CRLB. 帯域幅（電波型式）と積分時間（同時受信なら測定間隔そのもの）で決まる."""
    if lh.kind == "ground":
        return 50.0
    inband = (lh.snr_dbhz or 0) - 10 * math.log10(bandwidth_hz)
    beta = bandwidth_hz / math.sqrt(12)
    return max(C / (2 * math.pi * beta * math.sqrt(2 * 10 ** (inband / 10) * bandwidth_hz * integ_s)), 10.0)


def vhf_report_bps(n_stations: int, cycle_s: float, bits_per_station: int = 96, header_bits: int = 192,
                   overhead: float = 0.15) -> float:
    """ブイ 1 本が VHF で送る測定結果のビットレート（局ごとの TOA・品質・周波数 ＋ 位置・時計のヘッダ）."""
    return (header_bits + n_stations * bits_per_station) / cycle_s * (1 + overhead)


def slant(p, q, alt):
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + alt * alt)


# ------------------------------------------------------------------ 実行
def run(cfg: dict, sky_cfg: dict, backend, depart_utc: float, with_extra=(), ssn_url=None, events=None) -> dict:
    stations_d = dict(sky_cfg["stations"])
    for n_ in with_extra:
        stations_d[n_] = sky_cfg["extra_stations"][n_]
    names, stations = list(stations_d), [tuple(v) for v in stations_d.values()]
    N = len(stations)
    ssn, ssn_src = resolve_ssn(cfg["epoch"], ssn_url)

    traj, origin, op_xy, op_ll, phases = build_route(cfg)
    t, P_true, V_true = traj[:, 0], traj[:, 1:3], traj[:, 3:5]
    dt, alt = cfg["dt"], cfg["aircraft"]["altitude"]
    T = len(t)
    t_arr = phases["作戦海域"][0]
    i_arr = int(np.searchsorted(t, t_arr))
    hb0 = cfg["hf_buoy"]
    by_st = hb0.get("emission_by_station", {}) or {}
    emissions = [by_st.get(nm, sky_cfg["link"].get("emission")) for nm in names]
    bw_hz = [parse_emission(e)[2] if e else link_bandwidth_hz(sky_cfg["link"]) for e in emissions]
    links, raw, dist_op = op_links(sky_cfg, stations, op_ll, cfg["epoch"], ssn, backend, emissions)
    cmdc = cfg.get("command", {})
    hour_of = ((depart_utc + t / 3600.0) % 24).astype(int)
    tl = Timeline(events, t, names, cfg["epoch"]["year"], cfg["epoch"]["month"], depart_utc)

    # ---- 真値: INS、ブイの投下と漂流、ブイの時計
    rng_ins = np.random.default_rng(cfg["seed"] + 1)
    ins = INSModel(cfg["ins"]["bias0_sigma"], cfg["ins"]["bias_rw"], cfg["ins"]["vel_noise"], rng=rng_ins)
    v_ins = np.array([ins.measure(v, dt) for v in V_true])
    bc = cfg["buoys"]
    B = bc["n"]
    drops = hex_grid(op_xy, B, bc["spacing_km"] * 1e3)
    dr = cfg["drift"]
    sim = DriftSimulator(OceanCurrentField(mean_current=tuple(dr["mean_current"]), wind=tuple(dr["wind"]),
                                           eddy_strength=np.array([0.0])),
                         diffusivity=dr["diffusivity"], rng=np.random.default_rng(cfg["seed"] + 6))
    bpos = np.full((T, B, 2), np.nan)
    xy = drops.copy()
    for i in range(i_arr, T):
        bpos[i] = xy
        xy = sim.step(xy, t[i], dt)
    rng_clk = np.random.default_rng(cfg["seed"] + 7)
    ck = cfg["buoys"]["clock"]
    clk = np.zeros((T, B))
    c, cd = rng_clk.normal(0, ck["offset0_m"], B), rng_clk.normal(0, ck["drift0_mps"], B)
    for i in range(i_arr, T):
        clk[i] = c
        cd = cd + ck["drift_rw"] * math.sqrt(dt) * rng_clk.standard_normal(B)
        c = c + cd * dt

    # 電離層残差の単位 Gauss-Markov 過程（送信所ごと。全ブイ共通）
    taus = cfg["iono_tau_s"]
    drive = np.random.default_rng(cfg["seed"] + 3).standard_normal((T, N))
    u0 = np.random.default_rng(cfg["seed"] + 2).standard_normal(N)
    U_by_tau = {}
    for tau in {float(v) for v in taus.values()}:
        a = math.exp(-dt / tau)
        U = np.empty((T, N)); u = u0.copy()
        for i in range(T):
            U[i] = u
            u = a * u + math.sqrt(1 - a * a) * drive[i]
        U_by_tau[tau] = U
    noise_hf = np.random.default_rng(cfg["seed"] + 4).standard_normal((T, B, N))
    noise_lk = np.random.default_rng(cfg["seed"] + 8).standard_normal((T, 2, B))
    p0 = P_true[0] + np.random.default_rng(cfg["seed"] + 5).normal(0, cfg["aircraft"]["init_pos_sigma"], 2)

    hb, lkc = cfg["hf_buoy"], cfg["links"]
    k_cyc = int(round(hb["cycle_s"] / dt))
    # 同時受信なら各局を測定間隔いっぱい積分できる。時分割なら局数で割った時間だけ
    integ_s = hb["cycle_s"] if hb.get("simultaneous", True) else hb["cycle_s"] / N
    i_end = i_arr + int(round(cfg["buoys"].get("lifetime_h", 8.0) * 3600 / dt))    # ブイの稼働時間（電池）
    vhf_bps = vhf_report_bps(N, hb["cycle_s"])
    results = {}
    for mode in cfg["modes"]:
        tau = float(taus.get(mode, 1800.0))
        a1, U = math.exp(-dt / tau), U_by_tau[tau]
        # ---- 母機 EKF
        x = np.zeros(4); x[:2] = p0
        P = np.diag([cfg["aircraft"]["init_pos_sigma"] ** 2] * 2 + [cfg["ins"]["bias0_sigma"] ** 2] * 2)
        F = np.eye(4); F[0, 2] = F[1, 3] = -dt
        Qa = np.diag([(cfg["ins"]["vel_noise"] * dt) ** 2] * 2 + [cfg["ins"]["bias_rw"] ** 2 * dt] * 2)
        # ---- ブイ網 KF（投下時に初期化）
        nb = 6 * B + N
        xn = Pn = None
        sig_e = np.zeros(N)
        seen = np.zeros(N, bool)
        est = np.empty((T, 2)); sd = np.empty(T)
        net_abs = np.full(T, np.nan); net_rel = np.full(T, np.nan); n_buoy = np.zeros(T, int)
        stats = {"hf_cycles": 0, "uhf": 0, "vhf": 0, "commands": 0, "stale_freq_lost": 0}
        plan = CommandedPlan(links, cmdc.get("horizon_h", 1), cmdc.get("update_min", 10), cmdc.get("deploy_only", False))
        Sc = np.zeros((2, 2))          # ブイ網全体の平行移動（重心の推定誤差）の共分散
        last_fix, handed_over = -1, True
        for i in range(T):
            if i:
                x[:2] += (v_ins[i - 1] - x[2:]) * dt
                P = F @ P @ F.T + Qa
            if mode == "ins_only":
                est[i] = x[:2]; sd[i] = math.sqrt(P[0, 0] + P[1, 1])
                continue
            if i == i_arr:
                # 投下: 母機は自分の推定位置を基準に投下点を決めるので、ブイの事前位置は母機の推定誤差を含む
                xn = np.zeros(nb); Pn = np.zeros((nb, nb))
                off = x[:2] - P_true[i]
                for b in range(B):
                    j = 6 * b
                    xn[j:j + 2] = drops[b] + off
                    Pn[j:j + 2, j:j + 2] = P[:2, :2] + np.eye(2) * 500.0**2
                    Pn[j + 2:j + 4, j + 2:j + 4] = np.eye(2) * 0.5**2
                    Pn[j + 4, j + 4] = ck["offset0_m"] ** 2
                    Pn[j + 5, j + 5] = ck["drift0_mps"] ** 2
                for b1 in range(B):             # 投下位置の誤差は全ブイ共通（母機の位置誤差）
                    for b2 in range(B):
                        if b1 != b2:
                            Pn[6 * b1:6 * b1 + 2, 6 * b2:6 * b2 + 2] = P[:2, :2]
            if xn is not None and i > i_arr:
                # ブイ網の予測（漂流は等速＋ランダムウォーク、時計はオフセット＋ドリフト、電離層は Gauss-Markov）
                Fn = np.eye(nb)
                Qn = np.zeros(nb)
                for b in range(B):
                    j = 6 * b
                    Fn[j, j + 2] = Fn[j + 1, j + 3] = dt
                    Fn[j + 4, j + 5] = dt
                    Qn[j + 2:j + 4] = bc["accel_psd"] * dt
                    Qn[j + 5] = ck["drift_rw"] ** 2 * dt
                    Qn[j + 4] = ck.get("offset_rw", 0.01) ** 2 * dt
                Fn[6 * B:, 6 * B:] *= a1
                Qn[6 * B:] = sig_e**2 * (1 - a1 * a1)
                xn = Fn @ xn
                Pn = Fn @ Pn @ Fn.T + np.diag(Qn)
            alive = i_arr <= i < i_end
            if xn is not None and alive and (i - i_arr) % k_cyc == 0:
                # ---- ブイの HF 受信（時分割で全局を一巡したとみなす）
                # ブイは UHF で指示された周波数を受信する（指示が古いと MUF の変化で受からなくなる）
                near = bool(np.any(np.hypot(*(bpos[i] - P_true[i]).T) <= lkc["range_km"] * 1e3))
                plan.maybe_command(t[i], int(hour_of[i]), near or i == i_arr)
                fq = plan.freqs(int(hour_of[i]))
                lk = [link_for_freq(raw[s][int(hour_of[i])], dist_op[s], fq[s] if fq else None,
                                    sky_cfg["rel_min"], sky_cfg["ground_wave_km"]) for s in range(N)]
                stats["stale_freq_lost"] += sum(1 for s in range(N) if links[s][hour_of[i]].usable and not lk[s].usable)
                stats["commands"] = plan.count
                cen = np.nanmean(bpos[i], axis=0)
                cla, clo = aeqd_inverse(cen[0], cen[1], *origin)
                use = [s for s in range(N) if lk[s].usable
                       and not tl.blocked(i, s, float(cla), float(clo), stations[s], lk[s].kind)]
                for s in use:
                    sg = iono_sigma_m(lk[s], mode, cfg)
                    if not seen[s]:
                        xn[6 * B + s] = 0.0
                        Pn[6 * B + s, :] = 0.0; Pn[:, 6 * B + s] = 0.0
                        Pn[6 * B + s, 6 * B + s] = sg**2
                        seen[s] = True
                    elif abs(sg / sig_e[s] - 1) > 1e-6:
                        r = sg / sig_e[s]
                        xn[6 * B + s] *= r
                        Pn[6 * B + s, :] *= r; Pn[:, 6 * B + s] *= r
                    sig_e[s] = sg
                if len(use) >= 3:
                    stats["hf_cycles"] += 1
                    zs, hs, Hs, Rs = [], [], [], []
                    for b in range(B):
                        la_t, lo_t = aeqd_inverse(bpos[i, b, 0], bpos[i, b, 1], *origin)
                        j = 6 * b
                        la_e, lo_e = aeqd_inverse(xn[j], xn[j + 1], *origin)
                        for s in use:
                            sg_true = iono_sigma_m(lk[s], mode, cfg) * tl.sigma_factor(i)
                            e_true = sg_true * U[i, s] + (tl.storm_bias_m(i, lk[s].dPdh, mode) if lk[s].kind == "sky" else 0.0)
                            sig_toa = math.hypot(buoy_toa_sigma_m(lk[s], bw_hz[s], integ_s), hb["per_buoy_decorr"] * sg_true)
                            zs.append(group_path_m(float(la_t), float(lo_t), stations[s], lk[s]) + e_true
                                      + clk[i, b] + sig_toa * noise_hf[i, b, s])
                            g0 = group_path_m(float(la_e), float(lo_e), stations[s], lk[s])
                            hrow = np.zeros(nb)
                            for ax_ in range(2):
                                d = np.zeros(2); d[ax_] = 10.0
                                la2, lo2 = aeqd_inverse(xn[j] + d[0], xn[j + 1] + d[1], *origin)
                                hrow[j + ax_] = (group_path_m(float(la2), float(lo2), stations[s], lk[s]) - g0) / 10.0
                            hrow[j + 4] = 1.0
                            hrow[6 * B + s] = 1.0
                            hs.append(g0 + xn[j + 4] + xn[6 * B + s])
                            Hs.append(hrow); Rs.append(sig_toa**2)
                    H = np.array(Hs); R = np.diag(Rs)
                    y = np.array(zs) - np.array(hs)
                    S = H @ Pn @ H.T + R
                    K = np.linalg.solve(S, H @ Pn).T
                    xn = xn + K @ y
                    IKH = np.eye(nb) - K @ H
                    Pn = IKH @ Pn @ IKH.T + K @ R @ K.T
            if xn is not None and alive:
                A = np.zeros((2, nb))
                for b in range(B):
                    A[:, 6 * b:6 * b + 2] += np.eye(2) / B
                Sc = A @ Pn @ A.T
                be = np.array([xn[6 * b:6 * b + 2] for b in range(B)])
                errs = be - bpos[i]
                net_abs[i] = float(np.hypot(*errs.mean(axis=0)))
                net_rel[i] = float(np.sqrt(np.mean(np.sum((errs - errs.mean(axis=0)) ** 2, axis=1))))
                # ---- 母機: UHF TOA（ブイで計測）と VHF TDOA（母機で計測）
                inr = [b for b in range(B) if np.hypot(*(bpos[i, b] - P_true[i])) <= lkc["range_km"] * 1e3]
                n_buoy[i] = len(inr)
                for kind, k_int, sig in (("uhf", lkc["uhf"]["interval_s"], lkc["uhf"]["sigma_m"]),
                                         ("vhf", lkc["vhf"]["interval_s"], lkc["vhf"]["sigma_m"])):
                    if not lkc[kind]["enabled"] or (i - i_arr) % int(round(k_int / dt)) or len(inr) < 3:
                        continue
                    stats[kind] += 1
                    ch = 0 if kind == "uhf" else 1
                    rho = np.array([slant(P_true[i], bpos[i, b], alt) + clk[i, b] + sig * noise_lk[i, ch, b] for b in inr])
                    pred = np.array([slant(x[:2], xn[6 * b:6 * b + 2], alt) + xn[6 * b + 4] for b in inr])
                    u = np.array([(x[:2] - xn[6 * b:6 * b + 2]) / slant(x[:2], xn[6 * b:6 * b + 2], alt) for b in inr])
                    y = (rho[1:] - rho[0]) - (pred[1:] - pred[0])
                    H = np.zeros((len(inr) - 1, 4)); H[:, :2] = u[1:] - u[0]
                    # ブイ網の不確かさ（位置・時計）を観測誤差に入れる
                    G = np.zeros((len(inr) - 1, nb))
                    for r, b in enumerate(inr):
                        j = 6 * b
                        row = -u[r]
                        if r == 0:
                            G[:, j:j + 2] -= row; G[:, j + 4] -= 1.0
                        else:
                            G[r - 1, j:j + 2] += row; G[r - 1, j + 4] += 1.0
                    R = tdoa_cov(len(inr), sig) + G @ Pn @ G.T
                    S = H @ P @ H.T + R
                    K = np.linalg.solve(S, H @ P).T
                    x = x + K @ y
                    P = (np.eye(4) - K @ H) @ P @ (np.eye(4) - K @ H).T + K @ R @ K.T
                    last_fix, handed_over = i, False
            # 母機の位置はブイ網に対して決まる: 絶対位置の不確かさ = 相対の共分散 + ブイ網全体の平行移動
            fixing = last_fix >= 0 and i - last_fix <= 60
            if not fixing and not handed_over:
                P[:2, :2] += Sc            # ブイ網を離れた: 網の絶対誤差を母機の共分散へ引き継ぐ
                handed_over = True
            est[i] = x[:2]
            sd[i] = math.sqrt(max(P[0, 0] + P[1, 1] + (np.trace(Sc) if fixing else 0.0), 0.0))
        err = np.hypot(*(est - P_true).T)
        results[mode] = {"est": est, "err": err, "sd": sd, "net_abs": net_abs, "net_rel": net_rel,
                         "n_buoy": n_buoy, "stats": stats}

    return {"t": t, "P_true": P_true, "lat": aeqd_inverse(P_true[:, 0], P_true[:, 1], *origin)[0],
            "lon": aeqd_inverse(P_true[:, 0], P_true[:, 1], *origin)[1], "origin": origin, "op_ll": op_ll,
            "phases": phases, "results": results, "stations": names, "station_ll": stations, "buoys": bpos,
            "depart_utc": depart_utc, "ssn": ssn, "ssn_source": ssn_src, "events": events or [],
            "storm": tl.storm, "flare": tl.flare,
            "links_at_op": [[(lh.freq_mhz, lh.usable) for lh in per] for per in links],
            "buoy_end_h": float(t[min(i_end, T - 1)] / 3600) if i_end < T else None,
            "emissions": dict(zip(names, emissions)), "vhf_report_bps": round(vhf_bps, 1)}


def summarize(out: dict) -> dict:
    t = out["t"]
    s = {"depart_utc": out["depart_utc"], "ssn": out["ssn"], "ssn_source": out["ssn_source"], "events": out["events"],
         "stations": out["stations"], "duration_h": round(t[-1] / 3600, 1), "emissions": out["emissions"],
         "vhf_report_bps": out["vhf_report_bps"], "buoy_end_h": out["buoy_end_h"], "modes": {}}
    for mode, r in out["results"].items():
        m = {}
        for ph, (a, b) in out["phases"].items():
            sel = (t >= a) & (t <= b)
            e = r["err"][sel] / 1e3
            sdv = r["sd"][sel] / 1e3
            m[ph] = {"mean_km": round(float(e.mean()), 3), "p95_km": round(float(np.percentile(e, 95)), 3),
                     "max_km": round(float(e.max()), 3),
                     "consistency": round(float(np.sqrt(np.mean(e**2)) / max(np.sqrt(np.mean(sdv**2)), 1e-9)), 2),
                     **{k: f"{round(100 * float(np.mean(e * 1e3 <= v)))}%" for k, v in RNP_M.items()}}
        a, b = out["phases"]["作戦海域"]
        sel = (t >= a) & (t <= b) & np.isfinite(r["net_abs"])
        if sel.any():
            na, nr = r["net_abs"][sel] / 1e3, r["net_rel"][sel] / 1e3
            last = sel & (t >= b - 1800)
            m["ブイ網"] = {"abs_mean_km": round(float(na.mean()), 3), "abs_last30min_km": round(float(np.mean(r["net_abs"][last]) / 1e3), 3),
                         "rel_mean_km": round(float(nr.mean()), 4)}
        m["final_km"] = round(float(r["err"][-1] / 1e3), 3)
        m["counts"] = r["stats"]
        s["modes"][mode] = m
    return s


def aggregate(runs: list[dict]) -> dict:
    g = _aggregate([{k: v for k, v in r.items()} for r in runs]) if len(runs) > 1 else dict(runs[0], runs=1)
    return g


LABEL = {"ins_only": "INS 単独", "model": "モデル補正", "realtime": "リアルタイム推定"}
COLOR = {"ins_only": "tab:red", "model": "tab:orange", "realtime": "tab:blue"}


def plot(out: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for nm in ("Noto Sans CJK JP", "Hiragino Sans", "IPAexGothic"):
        if any(nm in f.name for f in font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = nm
            break
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.2))
    a1.plot(out["lon"], out["lat"], "k-", lw=1, label="真航跡")
    for nm, (la, lo) in zip(out["stations"], out["station_ll"]):
        a1.plot(lo, la, "b^", ms=8); a1.text(lo + 0.4, la, nm, fontsize=9, color="navy")
    a1.plot(out["op_ll"][1], out["op_ll"][0], "g*", ms=12, label="作戦海域（RF ブイ網）")
    a1.set_xlabel("経度 [°E]"); a1.set_ylabel("緯度 [°N]"); a1.grid(alpha=0.3); a1.legend(fontsize=8)
    a1.set_aspect(1 / math.cos(math.radians(float(np.mean(out["lat"])))))
    a1.set_title("航跡・送信所・ブイ網")
    th = out["t"] / 3600
    for mode, r in out["results"].items():
        a2.semilogy(th, np.maximum(r["err"] / 1e3, 1e-3), c=COLOR[mode], lw=0.9, label=f"母機 {LABEL[mode]}")
        if mode != "ins_only":
            a2.semilogy(th, np.maximum(r["net_abs"] / 1e3, 1e-3), ":", c=COLOR[mode], lw=1.0,
                        label=f"ブイ網の絶対誤差（{LABEL[mode]}）")
    for k, v in (("RNP-4", 7.408), ("RNP-1", 1.852)):
        a2.axhline(v / 2, ls=":", c="gray", lw=0.8)
        a2.text(th[-1], v / 2, f" {k}", fontsize=7, va="center", color="gray")
    for j, (ph, (a, b)) in enumerate(out["phases"].items()):
        a2.axvspan(a / 3600, b / 3600, color=f"C{j}", alpha=0.06)
        a2.text((a + b) / 7200, 1.3e-3, ph, ha="center", fontsize=8)
    a2.set_ylim(1e-3, 100)
    a2.set_xlabel(f"GPS ロスト後の経過時間 [h]（出発 {out['depart_utc']:02.0f} UTC）")
    a2.set_ylabel("水平位置誤差 [km]")
    a2.grid(alpha=0.3, which="both"); a2.legend(fontsize=7, loc="upper left")
    a2.set_title(f"ブイ中心構成（HF はブイで受信、母機は UHF TOA＋VHF TDOA）SSN {out['ssn']:.0f}")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="ブイ中心構成のシナリオ（HF はブイで受信、母機は UHF/VHF でブイ網を基準に測位）")
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--out", default="out/buoynet")
    ap.add_argument("--chuck-url")
    ap.add_argument("--ssn-from")
    ap.add_argument("--with", dest="extra", nargs="*", default=[])
    ap.add_argument("--depart-utc", type=float, nargs="*")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--scenario", nargs="*")
    ap.add_argument("--links", choices=["both", "uhf", "vhf"], help="母機の測位に使うリンク（config を上書き）")
    ap.add_argument("--command-once", action="store_true", help="受信周波数の指示を投下時の 1 回だけにする")
    ap.add_argument("--emission", help="全送信所の電波型式（例 3K00J2D, 24K0J2D）。config を上書き")
    ap.add_argument("--lifetime-h", type=float, help="ブイの稼働時間 [h]（既定 8）")
    ap.add_argument("--orbit-h", type=float, help="作戦海域での周回時間 [h]")
    ap.add_argument("--command-horizon", type=int, help="1 回の指示で送る周波数計画の時間幅 [h]")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    sky_path = Path(cfg["skywave_config"])
    if not sky_path.is_absolute():
        sky_path = Path(a.config).resolve().parent / sky_path
    sky_cfg = yaml.safe_load(sky_path.read_text(encoding="utf-8"))
    if a.emission:
        sky_cfg["link"]["emission"] = a.emission
        cfg["hf_buoy"]["emission_by_station"] = {}
    cfg.setdefault("command", {})
    if a.lifetime_h:
        cfg["buoys"]["lifetime_h"] = a.lifetime_h
    if a.orbit_h:
        cfg["route"]["orbit"]["duration_h"] = a.orbit_h
    if a.command_once:
        cfg["command"]["deploy_only"] = True
    if a.command_horizon:
        cfg["command"]["horizon_h"] = a.command_horizon
    if a.links:
        cfg["links"]["uhf"]["enabled"] = a.links in ("both", "uhf")
        cfg["links"]["vhf"]["enabled"] = a.links in ("both", "vhf")
    od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    cache = od.parent / "chuck_cache"
    backend = ChuckHttpBackend(a.chuck_url, cache_dir=cache) if a.chuck_url else ChuckEngineBackend(cache_dir=cache)
    ssn_url = a.ssn_from or a.chuck_url or sky_cfg.get("chuck_url")
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
                    plot(out, od / f"buoynet_{name}_dep{int(dep):02d}utc.png")
            g = aggregate(runs)
            g["scenario"] = name
            summaries.append(g)
    (od / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
