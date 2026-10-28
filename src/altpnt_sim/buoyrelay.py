"""交代機シナリオ: 航空機どうしはデータを受け渡さず、ブイと航空機の間だけでやり取りする.

前提（2026-10-28 確定）
  - 作戦海域での 1 機あたりの在空時間は 8 時間。継続するときは交代機が到着する（前の機体の離脱と同時）
  - 航空機間のデータ引き継ぎはしない。交代機が使えるのはブイから届く情報だけ
  - ブイの稼働は 8 時間、作戦海域では在場機が再投下できる

ブイ網の位置・時計をどこで推定するかで 2 方式を比べる
  aircraft … 機上で推定（ブイは HF の到着時刻をそのまま送る）。交代機は到着時に推定をやり直す
             （ブイの位置の事前情報は作戦海域の計画上の中心 ± prior_km しかない）
  buoy     … ブイ自身が推定（各ブイが自分の HF 受信から自位置・時計を推定し続け、結果を VHF で送る）。
             ブイどうしは通信しないので推定はブイごとに独立。推定がブイに残るので交代機もすぐ使える
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

from .buoynet import (CommandedPlan, buoy_toa_sigma_m, iono_sigma_m, op_links, slant)
from .drift import DriftSimulator, OceanCurrentField
from .dubins.planner import hex_grid
from .geo import aeqd_inverse
from .hfprop import ChuckEngineBackend, ChuckHttpBackend
from .hfprop.chuck import link_bandwidth_hz, parse_emission
from .hfprop.plan import link_for_freq
from .longrange import build_route, group_path_m
from .nav.ins import INSModel
from .nav.tdoa import tdoa_cov
from .skywave import resolve_ssn

DEFAULT_CFG = Path(__file__).resolve().parents[2] / "config" / "buoyrelay_default.yaml"


# ------------------------------------------------------------------ ブイ網の推定器（1 つの KF が持つブイの組）
class NetEstimator:
    """slots のブイの [x, y, vx, vy, 時計, 時計ドリフト] と、送信所ごとの電離層バイアス（この推定器内で共通）."""

    def __init__(self, slots, N, bc, i0, dt):
        self.slots = list(slots)
        self.ix = {m: 6 * k for k, m in enumerate(self.slots)}
        self.N, self.bc, self.dt = N, bc, dt
        self.n = 6 * len(self.slots) + N
        self.io = 6 * len(self.slots)
        self.x = np.zeros(self.n)
        self.P = np.zeros((self.n, self.n))
        self.t_i = i0
        self.sig_e = np.zeros(N)
        self.seen = np.zeros(N, bool)
        self.ready = set()

    def init_slot(self, m, pos, pos_cov, clk_sigma, drift_sigma, shared=None):
        j = self.ix[m]
        self.x[j:j + 6] = 0.0
        self.P[j:j + 6, :] = 0.0; self.P[:, j:j + 6] = 0.0
        self.x[j:j + 2] = pos
        self.P[j:j + 2, j:j + 2] = pos_cov
        self.P[j + 2:j + 4, j + 2:j + 4] = np.eye(2) * 0.5**2
        self.P[j + 4, j + 4] = clk_sigma**2
        self.P[j + 5, j + 5] = drift_sigma**2
        for m2 in shared or []:           # 同時に投下した組の位置誤差は共通（投下した機体の位置誤差）
            if m2 != m and m2 in self.ready:
                j2 = self.ix[m2]
                self.P[j:j + 2, j2:j2 + 2] = shared[m2]
                self.P[j2:j2 + 2, j:j + 2] = shared[m2]
        self.ready.add(m)

    def propagate(self, i, tau):
        d = (i - self.t_i) * self.dt
        if d <= 0:
            return
        ck = self.bc["clock"]
        F = np.eye(self.n); Q = np.zeros(self.n)
        for m, j in self.ix.items():
            F[j, j + 2] = F[j + 1, j + 3] = d
            F[j + 4, j + 5] = d
            Q[j + 2:j + 4] = self.bc["accel_psd"] * d
            Q[j + 5] = ck["drift_rw"] ** 2 * d
            Q[j + 4] = ck.get("offset_rw", 0.01) ** 2 * d
        a = math.exp(-d / tau)
        F[self.io:, self.io:] *= a
        Q[self.io:] = self.sig_e**2 * (1 - a * a)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + np.diag(Q)
        self.t_i = i

    def set_iono(self, use, sig):
        for s in use:
            k = self.io + s
            if not self.seen[s]:
                self.x[k] = 0.0
                self.P[k, :] = 0.0; self.P[:, k] = 0.0
                self.P[k, k] = sig[s] ** 2
                self.seen[s] = True
            elif abs(sig[s] / self.sig_e[s] - 1) > 1e-6:
                r = sig[s] / self.sig_e[s]
                self.x[k] *= r
                self.P[k, :] *= r; self.P[:, k] *= r
            self.sig_e[s] = sig[s]

    def hf_update(self, live, use, z, lk, stations, origin):
        """z[(m, s)] = ブイ m での送信所 s の TOA 観測（距離換算）, R[(m, s)] = 分散."""
        zs, hs, Hs, Rs = [], [], [], []
        for m in live:
            if m not in self.ready:
                continue
            j = self.ix[m]
            la_e, lo_e = aeqd_inverse(self.x[j], self.x[j + 1], *origin)
            for s in use:
                g0 = group_path_m(float(la_e), float(lo_e), stations[s], lk[s])
                h = np.zeros(self.n)
                for ax in range(2):
                    dd = np.zeros(2); dd[ax] = 10.0
                    la2, lo2 = aeqd_inverse(self.x[j] + dd[0], self.x[j + 1] + dd[1], *origin)
                    h[j + ax] = (group_path_m(float(la2), float(lo2), stations[s], lk[s]) - g0) / 10.0
                h[j + 4] = 1.0
                h[self.io + s] = 1.0
                zs.append(z[(m, s)][0]); Rs.append(z[(m, s)][1])
                hs.append(g0 + self.x[j + 4] + self.x[self.io + s]); Hs.append(h)
        if not zs:
            return
        H = np.array(Hs); R = np.diag(Rs); y = np.array(zs) - np.array(hs)
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T
        self.x = self.x + K @ y
        IKH = np.eye(self.n) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

    def pos(self, m):
        j = self.ix[m]
        return self.x[j:j + 2], self.x[j + 4]

    def cov3(self, ms):
        """ms のブイの [x, y, 時計] の共分散（3×len × 3×len）."""
        idx = [k for m in ms for k in (self.ix[m], self.ix[m] + 1, self.ix[m] + 4)]
        return self.P[np.ix_(idx, idx)]


# ------------------------------------------------------------------ 実行
def run(cfg, sky_cfg, backend, depart_utc, estimator, ssn_url=None, with_extra=()):
    stations_d = dict(sky_cfg["stations"])
    for n_ in with_extra:
        stations_d[n_] = sky_cfg["extra_stations"][n_]
    names, stations = list(stations_d), [tuple(v) for v in stations_d.values()]
    N = len(stations)
    ssn, ssn_src = resolve_ssn(cfg["epoch"], ssn_url)
    dt, alt = cfg["dt"], cfg["aircraft"]["altitude"]
    K = int(cfg["relief"]["aircraft"])
    onst = cfg["route"]["orbit"]["duration_h"] * 3600
    traj, origin, op_xy, op_ll, phases = build_route(cfg)
    Ta = len(traj)
    off = [int(round(k * onst / dt)) for k in range(K)]             # 交代機は前の機体の離脱と同時に到着
    T = off[-1] + Ta
    tg = np.arange(T) * dt
    i_arr0 = int(np.searchsorted(traj[:, 0], phases["作戦海域"][0]))
    i_leave0 = int(np.searchsorted(traj[:, 0], phases["作戦海域"][1]))
    op_win = [(off[k] + i_arr0, off[k] + i_leave0) for k in range(K)]
    hour_of = ((depart_utc + tg / 3600.0) % 24).astype(int)

    hb, lkc, bc = cfg["hf_buoy"], cfg["links"], cfg["buoys"]
    by_st = hb.get("emission_by_station", {}) or {}
    emissions = [by_st.get(nm, sky_cfg["link"].get("emission")) for nm in names]
    bw_hz = [parse_emission(e)[2] if e else link_bandwidth_hz(sky_cfg["link"]) for e in emissions]
    links, raw, dist_op = op_links(sky_cfg, stations, op_ll, cfg["epoch"], ssn, backend, emissions)

    # ---- 投下計画（在場機が投下）
    B = bc["n"]
    life_i = int(round(bc["lifetime_h"] * 3600 / dt))
    rd = bc.get("redeploy", {}) or {}
    end_op = op_win[-1][1]
    dep = [op_win[0][0]]
    if rd.get("enabled", True):
        step = life_i - int(round(rd.get("overlap_h", 0.5) * 3600 / dt))
        if step <= 0:
            raise ValueError("再投下の間隔（稼働時間 − 重なり時間）が 0 以下です")
        while dep[-1] + step < end_op - int(round(rd.get("min_remaining_h", 0.5) * 3600 / dt)):
            dep.append(dep[-1] + step)
    G = len(dep); M = B * G
    gen = np.repeat(np.arange(G), B)
    t_dep = np.array([dep[g] for g in gen]); t_die = t_dep + life_i
    grid = hex_grid(op_xy, B, bc["spacing_km"] * 1e3)
    drops = np.vstack([grid] * G)
    dr = cfg["drift"]
    sim = DriftSimulator(OceanCurrentField(mean_current=tuple(dr["mean_current"]), wind=tuple(dr["wind"]),
                                           eddy_strength=np.array([0.0])),
                         diffusivity=dr["diffusivity"], rng=np.random.default_rng(cfg["seed"] + 6))
    bpos = np.full((T, M, 2), np.nan); xy = drops.copy()
    for i in range(dep[0], T):
        d_ = t_dep <= i
        bpos[i, d_] = xy[d_]
        if d_.any():
            xy[d_] = sim.step(xy[d_], tg[i], dt)
    ck = bc["clock"]
    rc = np.random.default_rng(cfg["seed"] + 7)
    clk = np.zeros((T, M)); c = rc.normal(0, ck["offset0_m"], M); cd = rc.normal(0, ck["drift0_mps"], M)
    for i in range(dep[0], T):
        clk[i] = c
        cd = cd + ck["drift_rw"] * math.sqrt(dt) * rc.standard_normal(M)
        c = c + cd * dt
    tau = float(cfg["iono_tau_s"]["realtime"])
    a = math.exp(-dt / tau)
    drive = np.random.default_rng(cfg["seed"] + 3).standard_normal((T, N))
    U = np.empty((T, N)); u = np.random.default_rng(cfg["seed"] + 2).standard_normal(N)
    for i in range(T):
        U[i] = u; u = a * u + math.sqrt(1 - a * a) * drive[i]
    nz_hf = np.random.default_rng(cfg["seed"] + 4).standard_normal((T, M, N)).astype(np.float32)
    nz_lk = np.random.default_rng(cfg["seed"] + 8).standard_normal((T, 2, M)).astype(np.float32)

    # ---- 航空機（各機独立の INS）
    acs = []
    for k in range(K):
        ins = INSModel(cfg["ins"]["bias0_sigma"], cfg["ins"]["bias_rw"], cfg["ins"]["vel_noise"],
                       rng=np.random.default_rng(cfg["seed"] + 100 + k))
        v_ins = np.array([ins.measure(v, dt) for v in traj[:, 3:5]])
        p0 = traj[0, 1:3] + np.random.default_rng(cfg["seed"] + 200 + k).normal(0, cfg["aircraft"]["init_pos_sigma"], 2)
        P = np.diag([cfg["aircraft"]["init_pos_sigma"] ** 2] * 2 + [cfg["ins"]["bias0_sigma"] ** 2] * 2)
        acs.append({"x": np.r_[p0, 0, 0], "P": P, "v_ins": v_ins, "err": np.full(T, np.nan),
                    "net": None, "last_fix": -1, "handed": True, "Sc": np.zeros((2, 2))})
    F = np.eye(4); F[0, 2] = F[1, 3] = -dt
    Qa = np.diag([(cfg["ins"]["vel_noise"] * dt) ** 2] * 2 + [cfg["ins"]["bias_rw"] ** 2 * dt] * 2)

    # ---- ブイ自身の推定（estimator == "buoy"）: ブイごとに独立の KF
    own = {}
    plan = CommandedPlan(links, cfg["command"]["horizon_h"], cfg["command"]["update_min"], False)
    k_cyc = int(round(hb["cycle_s"] / dt)); integ_s = hb["cycle_s"]
    k_link = {k: int(round(lkc[k]["interval_s"] / dt)) for k in ("uhf", "vhf")}
    on_station = np.full(T, -1, int)
    for k, (a0, b0) in enumerate(op_win):
        on_station[a0:b0] = k
    net_abs = np.full(T, np.nan)

    for i in range(T):
        # 航空機の INS 予測
        for k, ac in enumerate(acs):
            li = i - off[k]
            if 0 < li < Ta:
                ac["x"][:2] += (ac["v_ins"][li - 1] - ac["x"][2:]) * dt
                ac["P"] = F @ ac["P"] @ F.T + Qa
        kk = on_station[i]
        live = [m for m in range(M) if t_dep[m] <= i < t_die[m]]
        # 投下（在場機）
        if i in dep and kk >= 0:
            ac = acs[kk]; li = i - off[kk]
            g = dep.index(i)
            slots = [m for m in range(M) if gen[m] == g]
            drop_err = ac["x"][:2] - traj[li, 1:3]
            Pd = ac["P"][:2, :2] + ac["Sc"] + np.eye(2) * 500.0**2
            if estimator == "buoy":
                for m in slots:
                    own[m] = NetEstimator([m], N, bc, i, dt)
                    own[m].init_slot(m, drops[m] + drop_err, Pd, ck["offset0_m"], ck["drift0_mps"])
            elif ac["net"] is not None:
                for m in slots:
                    ac["net"].init_slot(m, drops[m] + drop_err, Pd, ck["offset0_m"], ck["drift0_mps"],
                                        shared={m2: ac["P"][:2, :2] + ac["Sc"] for m2 in slots})
        # 交代機の到着: 機上推定ならブイ網の推定をやり直す（事前情報は計画上の中心のみ）
        if estimator == "aircraft" and kk >= 0 and acs[kk]["net"] is None:
            ac = acs[kk]
            ac["net"] = NetEstimator(range(M), N, bc, i, dt)
            pk = cfg["relief"]["prior_km"] * 1e3
            for m in live:
                age = (i - t_dep[m]) * dt
                if i in dep and t_dep[m] == i:
                    pos, Pp = drops[m] + (ac["x"][:2] - traj[i - off[kk], 1:3]), ac["P"][:2, :2] + np.eye(2) * 500.0**2
                else:
                    pos, Pp = np.asarray(op_xy, float), np.eye(2) * pk**2
                ac["net"].init_slot(m, pos, Pp, ck["offset0_m"] + ck["drift0_mps"] * age, ck["drift0_mps"])
            for m in range(M):              # まだ投下されていない組も、投下時に初期化できるよう枠だけ持つ
                pass
        # ---- ブイの HF 受信と、ブイ網の推定
        if live and (i - dep[0]) % k_cyc == 0:
            if kk >= 0:
                plan.maybe_command(tg[i], int(hour_of[i]), True)
            fq = plan.freqs(int(hour_of[i]))
            lk = [link_for_freq(raw[s][int(hour_of[i])], dist_op[s], fq[s] if fq else None,
                                sky_cfg["rel_min"], sky_cfg["ground_wave_km"]) for s in range(N)]
            use = [s for s in range(N) if lk[s].usable]
            sig = {s: iono_sigma_m(lk[s], "realtime", cfg) for s in use}
            z = {}
            for m in live:
                la, lo = aeqd_inverse(bpos[i, m, 0], bpos[i, m, 1], *origin)
                for s in use:
                    st = math.hypot(buoy_toa_sigma_m(lk[s], bw_hz[s], integ_s), hb["per_buoy_decorr"] * sig[s])
                    z[(m, s)] = (group_path_m(float(la), float(lo), stations[s], lk[s]) + sig[s] * U[i, s]
                                 + clk[i, m] + st * float(nz_hf[i, m, s]), st**2)
            if len(use) >= 3:
                if estimator == "buoy":
                    for m in live:
                        if m in own:
                            own[m].propagate(i, tau); own[m].set_iono(use, sig)
                            own[m].hf_update([m], use, z, lk, stations, origin)
                elif kk >= 0 and acs[kk]["net"] is not None:   # 機上推定はブイのデータを受けている機体だけ
                    net = acs[kk]["net"]
                    net.propagate(i, tau); net.set_iono(use, sig)
                    net.hf_update(live, use, z, lk, stations, origin)
        # ---- 在場機の測位（UHF TOA ＋ VHF TDOA）
        if kk >= 0 and live:
            ac = acs[kk]; li = i - off[kk]
            ptrue = traj[li, 1:3]
            inr = [m for m in live if np.hypot(*(bpos[i, m] - ptrue)) <= lkc["range_km"] * 1e3]
            if estimator == "buoy":
                inr = [m for m in inr if m in own]
                for m in inr:
                    own[m].propagate(i, tau)
                pos_of = {m: own[m].pos(m) for m in inr}
                C3 = np.zeros((3 * len(inr), 3 * len(inr)))
                for r, m in enumerate(inr):           # ブイ同士は通信しない: 推定は独立（ブロック対角）
                    C3[3 * r:3 * r + 3, 3 * r:3 * r + 3] = own[m].cov3([m])
                if inr:   # ブイ網全体のずれ（共通の電離層誤差）: 各ブイの位置誤差の平均を完全相関とみなす
                    ac["Sc"] = np.mean([own[m].cov3([m])[:2, :2] for m in inr], axis=0)
            else:
                net = ac["net"]
                inr = [m for m in inr if net is not None and m in net.ready]
                if inr:
                    net.propagate(i, tau)
                    pos_of = {m: net.pos(m) for m in inr}
                    C3 = net.cov3(inr)
                    A = np.zeros((2, net.n))
                    for m in inr:
                        A[:, net.ix[m]:net.ix[m] + 2] += np.eye(2) / len(inr)
                    ac["Sc"] = A @ net.P @ A.T
            if inr:
                net_abs[i] = float(np.hypot(*np.mean([pos_of[m][0] - bpos[i, m] for m in inr], axis=0)))
            for kind in ("uhf", "vhf"):
                if not lkc[kind]["enabled"] or (i - dep[0]) % k_link[kind] or len(inr) < 3:
                    continue
                sg = lkc[kind]["sigma_m"]; ch = 0 if kind == "uhf" else 1
                x, P = ac["x"], ac["P"]
                rho = np.array([slant(ptrue, bpos[i, m], alt) + clk[i, m] + sg * float(nz_lk[i, ch, m]) for m in inr])
                pred = np.array([slant(x[:2], pos_of[m][0], alt) + pos_of[m][1] for m in inr])
                u_ = np.array([(x[:2] - pos_of[m][0]) / slant(x[:2], pos_of[m][0], alt) for m in inr])
                y = (rho[1:] - rho[0]) - (pred[1:] - pred[0])
                H = np.zeros((len(inr) - 1, 4)); H[:, :2] = u_[1:] - u_[0]
                Gm = np.zeros((len(inr) - 1, 3 * len(inr)))
                for r in range(len(inr)):
                    row = np.r_[-u_[r], 1.0]
                    if r == 0:
                        Gm[:, 0:3] -= row
                    else:
                        Gm[r - 1, 3 * r:3 * r + 3] += row
                R = tdoa_cov(len(inr), sg) + Gm @ C3 @ Gm.T
                S = H @ P @ H.T + R
                Kg = np.linalg.solve(S, H @ P).T
                ac["x"] = x + Kg @ y
                ac["P"] = (np.eye(4) - Kg @ H) @ P @ (np.eye(4) - Kg @ H).T + Kg @ R @ Kg.T
                ac["last_fix"], ac["handed"] = i, False
        # 離脱時: ブイ網全体のずれを機体の共分散へ引き継ぐ
        for k, ac in enumerate(acs):
            li = i - off[k]
            if 0 <= li < Ta:
                if not (ac["last_fix"] >= 0 and i - ac["last_fix"] <= 60) and not ac["handed"]:
                    ac["P"][:2, :2] += ac["Sc"]; ac["handed"] = True
                ac["err"][i] = float(np.hypot(*(ac["x"][:2] - traj[li, 1:3])))

    return {"tg": tg, "off": off, "Ta": Ta, "op_win": op_win, "acs": acs, "dep_h": [round(d * dt / 3600, 2) for d in dep],
            "buoys_used": M, "net_abs": net_abs, "estimator": estimator, "ssn": ssn, "ssn_source": ssn_src,
            "phases": phases, "dt": dt}


def summarize(out):
    tg, dt = out["tg"], out["dt"]
    s = {"estimator": out["estimator"], "deploy_h": out["dep_h"], "buoys_used": out["buoys_used"], "aircraft": []}
    for k, ac in enumerate(out["acs"]):
        a0, b0 = out["op_win"][k]
        e = ac["err"][a0:b0] / 1e3
        first = ac["err"][a0:a0 + int(3600 / dt)] / 1e3
        ret = ac["err"][b0:out["off"][k] + out["Ta"]] / 1e3
        s["aircraft"].append({"op_mean_km": round(float(np.nanmean(e)), 3), "op_p95_km": round(float(np.nanpercentile(e, 95)), 3),
                              "op_first_hour_mean_km": round(float(np.nanmean(first)), 3),
                              "op_rnp1": f"{round(100 * float(np.nanmean(e <= 0.926)))}%",
                              "return_mean_km": round(float(np.nanmean(ret)), 3) if ret.size else None,
                              "final_km": round(float(ret[-1]), 3) if ret.size else None})
    na = out["net_abs"] / 1e3
    s["net_abs_mean_km"] = round(float(np.nanmean(na)), 3)
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description="交代機シナリオ（航空機間のデータ引き継ぎなし）")
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--out", default="out/buoyrelay")
    ap.add_argument("--estimator", nargs="*", default=["aircraft", "buoy"], choices=["aircraft", "buoy"])
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--chuck-url"); ap.add_argument("--ssn-from")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    sky_path = Path(cfg["skywave_config"])
    if not sky_path.is_absolute():
        sky_path = Path(a.config).resolve().parent / sky_path
    sky = yaml.safe_load(sky_path.read_text(encoding="utf-8"))
    od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    cache = od.parent / "chuck_cache"
    be = ChuckHttpBackend(a.chuck_url, cache_dir=cache) if a.chuck_url else ChuckEngineBackend(cache_dir=cache)
    ssn_url = a.ssn_from or a.chuck_url or sky.get("chuck_url")
    res = {}
    for est in a.estimator:
        runs = []
        for r in range(a.runs):
            c = dict(cfg, seed=cfg["seed"] + 1000 * r)
            out = run(c, sky, be, cfg["depart_utc"], est, ssn_url)
            runs.append(summarize(out))
            if r == 0:
                plot(out, od / f"buoyrelay_{est}.png")
        agg = {"estimator": est, "runs": len(runs), "deploy_h": runs[0]["deploy_h"], "buoys_used": runs[0]["buoys_used"],
               "net_abs_mean_km": round(float(np.mean([x["net_abs_mean_km"] for x in runs])), 3), "aircraft": []}
        for k in range(len(runs[0]["aircraft"])):
            d = {}
            for key in runs[0]["aircraft"][k]:
                vals = [x["aircraft"][k][key] for x in runs]
                if isinstance(vals[0], str):
                    d[key] = f"{round(float(np.mean([float(v[:-1]) for v in vals])))}%"
                elif vals[0] is not None:
                    d[key] = round(float(np.mean(vals)), 3)
            agg["aircraft"].append(d)
        res[est] = agg
    (od / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


def plot(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for nm in ("Noto Sans CJK JP", "Hiragino Sans", "IPAexGothic"):
        if any(nm in f.name for f in font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = nm
            break
    fig, ax = plt.subplots(figsize=(12, 5.5))
    th = out["tg"] / 3600
    for k, ac in enumerate(out["acs"]):
        ax.semilogy(th, np.maximum(ac["err"] / 1e3, 1e-3), lw=0.9, label=f"{k + 1} 機目")
        a0, b0 = out["op_win"][k]
        ax.axvspan(a0 * out["dt"] / 3600, b0 * out["dt"] / 3600, color=f"C{k}", alpha=0.07)
    for d in out["dep_h"]:
        ax.axvline(d, ls="--", c="gray", lw=0.7)
    ax.semilogy(th, np.maximum(out["net_abs"] / 1e3, 1e-3), "k:", lw=0.8, label="ブイ網全体のずれ")
    ax.set_ylim(1e-2, 100); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    ax.set_xlabel("1 機目の GPS ロストからの経過時間 [h]（破線: ブイの投下）"); ax.set_ylabel("水平位置誤差 [km]")
    lab = {"aircraft": "機上で推定（交代機は推定をやり直す）", "buoy": "ブイ自身が推定（推定がブイに残る）"}[out["estimator"]]
    ax.set_title(f"交代機シナリオ: {lab}")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
