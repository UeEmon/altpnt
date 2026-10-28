"""主たる作戦海域（第一列島線と第二列島線の間、南限 北緯 15 度、東は東経 150 度付近まで）の面的な評価.

格子点ごとに、ブイ網の位置の「瞬時の推定誤差」の指標を時刻別に求める。
  各送信所の観測 = 群路長（または地表波の距離）+ ブイの時計 + 電離層誤差
  未知数 = ブイ網の位置（東・北）とブイの時計 → 重み付き最小二乗の共分散から誤差（drms）を出す
  電離層誤差は送信所ごとに独立（σ = iono_sigma_m）。監視網の補正があれば補正後に残る分（resid_unit 倍）
これは 1 時点の値で、実際の推定器は電離層誤差の時間変化を平均するのでこれより小さくなる。
海域内の比較（どこが良くどこが悪いか、補正でどれだけ下がるか）に使い、絶対値はシミュレーションで確かめる。

実行:
  VOACAP_BIN=... VOACAP_MODEL=... CHUCK_APP_DIR=... altpnt-oparea --out out/oparea
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import yaml

from .buoynet import iono_sigma_m, op_links
from .hfprop import ChuckEngineBackend, ChuckHttpBackend
from .hfprop.plan import gc_dist_az
from .ionomon import MonitorCorrector, midpoint, monitor_paths
from .skywave import resolve_ssn

DEFAULT_CFG = Path(__file__).resolve().parents[2] / "config" / "oparea_default.yaml"

# 作戦海域の境界（lat, lon）。西: 第一列島線（九州〜南西諸島〜先島諸島〜台湾〜ルソン）、南: 北緯 15 度、
# 東: 東経 150 度、北: 本州南岸（伊豆半島〜四国〜九州）
AREA = [(31.0, 130.6), (28.3, 129.5), (26.2, 127.7), (24.8, 125.3), (24.45, 123.0), (22.0, 121.5),
        (18.5, 122.3), (15.0, 121.6), (15.0, 150.0), (32.0, 150.0), (34.6, 139.9), (33.4, 135.5), (32.0, 131.8)]
CHAIN1 = [(33.6, 130.4), (31.0, 130.6), (28.3, 129.5), (26.2, 127.7), (24.8, 125.3), (24.3, 124.2), (24.45, 123.0),
          (23.5, 121.0), (18.5, 121.5), (14.5, 121.0)]
CHAIN2 = [(34.7, 139.4), (33.1, 139.8), (30.5, 140.3), (27.1, 142.2), (24.8, 141.3), (20.5, 144.9), (15.2, 145.75),
          (13.4, 144.8)]


def in_area(lat, lon) -> bool:
    from matplotlib.path import Path as MPath
    return bool(MPath([(lo, la) for la, lo in AREA]).contains_point((lon, lat)))


def grid_points(step_deg=1.5):
    pts = []
    for la in np.arange(15.0 + step_deg / 2, 34.6, step_deg):
        for lo in np.arange(121.5 + step_deg / 2, 150.0, step_deg):
            if in_area(la, lo):
                pts.append((round(float(la), 2), round(float(lo), 2)))
    return pts


def snapshot_drms(point, stations, lks, sig_m):
    """ブイ網の位置 + 時計を各送信所の観測から最小二乗で求めたときの位置誤差 drms [m]（使える局が 3 未満なら nan）."""
    rows, w = [], []
    for s, (st, lk) in enumerate(zip(stations, lks)):
        if not lk.usable or not math.isfinite(sig_m[s]):
            continue
        _, az = gc_dist_az(point[0], point[1], st[0], st[1])
        away = az + math.pi                                   # 送信所から遠ざかる向き
        g = 1.0 if lk.kind == "ground" else float(lk.dPdD or 1.0)
        rows.append([g * math.sin(away), g * math.cos(away), 1.0])
        w.append(1.0 / sig_m[s] ** 2)
    if len(rows) < 3:
        return float("nan")
    H = np.array(rows); W = np.diag(w)
    try:
        C = np.linalg.inv(H.T @ W @ H)
    except np.linalg.LinAlgError:
        return float("nan")
    return float(math.sqrt(max(C[0, 0] + C[1, 1], 0.0)))


def evaluate(cfg, sky, backend, stations_d, tier="realtime", monitor=None, points=None):
    names, stations = list(stations_d), [tuple(v) for v in stations_d.values()]
    ssn = cfg["epoch"]["ssn"] if isinstance(cfg["epoch"]["ssn"], (int, float)) else cfg["epoch"].get("ssn_fallback", 100)
    emissions = [sky["link"].get("emission")] * len(stations)
    paths = monitor_paths(stations, names, monitor["sites"], monitor.get("min_km", 400), monitor.get("max_km", 4000),
                          {k: tuple(v) for k, v in (monitor.get("rx_only") or {}).items()}) if monitor else []
    out = []
    for p in points or grid_points(cfg.get("grid_deg", 1.5)):
        links, raw, dist = op_links(sky, stations, p, cfg["epoch"], ssn, backend, emissions)
        resid = np.ones(len(stations))
        if paths:
            mc = MonitorCorrector([midpoint(st, p) for st in stations], paths, float(monitor["corr_km"][tier]),
                                  float(cfg["sigma_h_km"][tier]), float(cfg["iono_tau_s"][tier]), 1.0, 4,
                                  sigma_obs_m=monitor.get("sigma_obs_m", 100.0), update_s=monitor.get("update_s", 60.0),
                                  latency_s=monitor.get("latency_s", 60.0))
            resid = mc.resid_unit
        per_h, n_use, n_ground = [], [], []
        for h in range(24):
            lks = [links[s][h] for s in range(len(stations))]
            sig = [iono_sigma_m(lk, tier, cfg) * (resid[s] if lk.kind == "sky" else 1.0) if lk.usable else float("nan")
                   for s, lk in enumerate(lks)]
            per_h.append(snapshot_drms(p, stations, lks, sig))
            n_use.append(sum(lk.usable for lk in lks)); n_ground.append(sum(lk.usable and lk.kind == "ground" for lk in lks))
        a = np.array(per_h)
        out.append({"lat": p[0], "lon": p[1], "drms_med_km": round(float(np.nanmedian(a)) / 1e3, 3) if np.isfinite(a).any() else None,
                    "drms_p90_km": round(float(np.nanpercentile(a, 90)) / 1e3, 3) if np.isfinite(a).any() else None,
                    "hours_lt3": int(np.sum(~np.isfinite(a))), "min_stations": int(min(n_use)), "ground": int(max(n_ground)),
                    "dist_km": [round(d) for d in dist], "resid": [round(float(r), 2) for r in resid]})
    return out


def plot(results: dict, path: Path, key="drms_med_km", vmax=None, extra_marks=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Polygon
    from mpl_toolkits.basemap import Basemap
    for nm in ("Noto Sans CJK JP", "IPAexGothic"):
        if any(nm in f.name for f in font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = nm
            break
    n = len(results)
    cols = 2 if n > 1 else 1
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7.2 * cols, 6.2 * rows), squeeze=False)
    vals = [r[key] for res in results.values() for r in res["points"] if r[key] is not None]
    vmax = vmax or float(np.nanpercentile(vals, 95))
    for ax, (title, res) in zip(axes.flat, results.items()):
        m = Basemap(projection="merc", llcrnrlat=12, urcrnrlat=42, llcrnrlon=119, urcrnrlon=157, resolution="l", ax=ax)
        m.fillcontinents(color="#e6e2d8"); m.drawmapboundary(fill_color="#eef4f9"); m.drawcoastlines(linewidth=0.3, color="#777")
        m.drawparallels(range(15, 45, 5), labels=[1, 0, 0, 0], fontsize=7, color="#ccc", linewidth=0.3)
        m.drawmeridians(range(120, 160, 10), labels=[0, 0, 0, 1], fontsize=7, color="#ccc", linewidth=0.3)
        ax.add_patch(Polygon([m(lo, la) for la, lo in AREA], closed=True, fc="none", ec="#d68910", lw=1.2))
        for ch, c in ((CHAIN1, "#c0392b"), (CHAIN2, "#1f5fa8")):
            x, y = m([q[1] for q in ch], [q[0] for q in ch]); m.plot(x, y, ls="--", color=c, lw=1.2)
        pts = [r for r in res["points"] if r[key] is not None]
        x, y = m([r["lon"] for r in pts], [r["lat"] for r in pts])
        sc = m.scatter(x, y, c=[r[key] for r in pts], s=140, marker="s", cmap="viridis_r", vmin=0, vmax=vmax, edgecolors="none")
        for r, xx, yy in zip(pts, x, y):
            ax.text(xx, yy, f"{r[key]:.1f}", fontsize=5.5, ha="center", va="center", color="w")
        for nm_, (la, lo) in res["stations"].items():
            xx, yy = m(lo, la); m.plot(xx, yy, "^", ms=7, color="#7d3c98", mec="k", mew=0.5)
        for nm_, (la, lo) in (res.get("rx_only") or {}).items():
            xx, yy = m(lo, la); m.plot(xx, yy, "s", ms=7, color="#48c9b0", mec="#117a65")
        for nm_, (la, lo) in (extra_marks or {}).items():
            xx, yy = m(lo, la); m.plot(xx, yy, "o", ms=9, mfc="none", mec="#e74c3c", mew=1.6)
            ax.annotate(nm_, (xx, yy), xytext=(5, 5), textcoords="offset points", fontsize=8, color="#c0392b")
        ax.set_title(title, fontsize=10)
        fig.colorbar(sc, ax=ax, shrink=0.75, label="ブイ網の位置誤差の指標 [km]（1 時点、24 時間の中央値）")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="作戦海域の面的な評価（格子点ごとのブイ網の位置誤差の指標）")
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--out", default="out/oparea")
    ap.add_argument("--chuck-url")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    sky = yaml.safe_load((Path(a.config).resolve().parent / cfg["skywave_config"]).read_text(encoding="utf-8"))
    od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    be = ChuckHttpBackend(a.chuck_url, cache_dir=od / "chuck_cache") if a.chuck_url else ChuckEngineBackend(cache_dir=od / "chuck_cache")
    results = {}
    for case in cfg["cases"]:
        st = {k: v for k, v in sky["stations"].items() if k in case["tx"]}
        mon = copy.deepcopy(cfg["monitor"]) if case.get("monitor") else None
        if mon:
            mon["sites"] = list(st)
        pts = evaluate(cfg, sky, be, st, case.get("tier", "realtime"), mon)
        results[case["name"]] = {"points": pts, "stations": st, "rx_only": (mon or {}).get("rx_only")}
    (od / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    plot(results, od / "oparea_drms.png", extra_marks=cfg.get("representative"))
    for k, v in results.items():
        d = [p["drms_med_km"] for p in v["points"] if p["drms_med_km"] is not None]
        print(k, "points", len(v["points"]), "median", round(float(np.median(d)), 2), "p90", round(float(np.percentile(d, 90)), 2))


if __name__ == "__main__":
    main()
