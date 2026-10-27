"""上空波 Rx-TDOA の時間帯別評価（CHUCK で周波数計画 → 使える送信所 → 位置誤差）.

実行:
  # CHUCK 本家のエンジンを直接使う（推奨: 計算回数の上限なし）
  CHUCK_APP_DIR=../VOCAP-COMPATIBILITY-/app VOACAP_BIN=/usr/local/bin/voacapl \\
  VOACAP_MODEL=/usr/local/share/voacapl/itshfbc altpnt-skywave --out out/skywave
  # 稼働中の CHUCK の API を使う（結果はキャッシュ）
  altpnt-skywave --chuck-url http://localhost:18082 --out out/skywave
  # 当月を実効 SSN（GIRO の foF2 観測で校正）で計算: config の epochs に {ssn: effective, ssn_fallback: 100}
  altpnt-skywave --ssn-from http://localhost:18082 --out out/skywave_now
  # 候補局を足して比較
  altpnt-skywave --with 南鳥島 --out out/skywave_mnt

出力: frequency_plan.csv（送信所×評価点×時刻の周波数）, position_error.csv, summary.json, skywave_*.png
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import yaml

from .hfprop import ChuckEngineBackend, ChuckHttpBackend, chuck_request, fetch_effective_ssn, plan_link, position_error
from .hfprop.plan import destination, gc_dist_az

DEFAULT_CFG = Path(__file__).resolve().parents[2] / "config" / "skywave_japan.yaml"
RNP = {"RNP-10": 18.52, "RNP-4": 7.408, "RNP-2": 3.704}     # 95% 位置誤差 [km]


def resolve_points(cfg) -> dict[str, tuple[float, float]]:
    out = {}
    for name, p in cfg["points"].items():
        out[name] = tuple(p["latlon"]) if "latlon" in p else destination(*cfg["center"], p["bearing"], p["nm"] * 1.852)
    return out


def resolve_ssn(ep: dict, ssn_url: str | None, fetch=fetch_effective_ssn) -> tuple[float, str]:
    """季節ごとの SSN. ssn: effective なら CHUCK の実効 SSN（当月のみ）を使い、取れなければ ssn_fallback."""
    if ep["ssn"] != "effective":
        return float(ep["ssn"]), "config"
    fallback = ep.get("ssn_fallback")
    if ssn_url:
        try:
            value, snap = fetch(ssn_url, ep["year"], ep["month"])
            if value is not None:
                return float(value), f"effective({snap.get('used_count')}局)"
            reason = snap.get("error") or "値なし"
        except Exception as e:  # noqa: BLE001  通信失敗も想定値に戻す
            reason = str(e)
    else:
        reason = "--ssn-from / --chuck-url が未指定"
    if fallback is None:
        raise ValueError(f"{ep['year']}-{ep['month']:02d}: 実効 SSN を取得できず ssn_fallback もありません（{reason}）")
    return float(fallback), f"fallback（{reason}）"


def run(cfg: dict, backend, with_extra: list[str] | None = None, ssn_url: str | None = None) -> dict:
    stations = dict(cfg["stations"])
    for name in with_extra or []:
        stations[name] = cfg["extra_stations"][name]
    st_names, st_pos = list(stations), [tuple(v) for v in stations.values()]
    points = resolve_points(cfg)
    kw = dict(ground_wave_sigma_km=cfg["ground_wave_sigma_km"], diff_sigma_km=cfg["diff_sigma_km"],
              bandwidth_hz=float(cfg["link"].get("bandwidth", "5K00").upper().replace("K", ".")) * 1e3
              if "bandwidth" in cfg["link"] else 5000.0)
    plan_rows, err_rows, ssn_used = [], [], {}
    for ep in cfg["epochs"]:
        ssn, src = resolve_ssn(ep, ssn_url)
        ssn_used[f'{ep["year"]}-{ep["month"]:02d}'] = {"ssn": ssn, "source": src}
        for pname, rx in points.items():
            links = []
            for sname, tx in zip(st_names, st_pos):
                d, _ = gc_dist_az(*tx, *rx)
                hours = backend.predict(chuck_request(tx, rx, ep["year"], ep["month"], ssn, cfg["frequencies"], cfg["link"]))
                lk = plan_link(hours, d, cfg["rel_min"], cfg["ground_wave_km"])
                links.append(lk)
                for lh in lk:
                    plan_rows.append(dict(year=ep["year"], month=ep["month"], point=pname, station=sname, hour_utc=lh.hour_utc,
                                          dist_km=round(d), kind=lh.kind, freq_mhz=lh.freq_mhz, rel=lh.rel, snr_dbhz=lh.snr_dbhz,
                                          mode=lh.mode, vhite_km=lh.vhite_km, mprob=lh.mprob, muf_mhz=lh.muf_mhz))
            for h in range(24):
                per = [lk[h] for lk in links]
                row = dict(year=ep["year"], month=ep["month"], point=pname, hour_utc=h,
                           n_usable=sum(lh.usable for lh in per),
                           stations="/".join(n for n, lh in zip(st_names, per) if lh.usable))
                for tier in cfg["tiers"]:
                    e, _ = position_error(rx, st_pos, per, tier, **kw)
                    row[f"drms_km_{tier}"] = None if not math.isfinite(e) else round(e, 2)
                err_rows.append(row)
    return {"stations": st_names, "points": list(points), "plan": plan_rows, "errors": err_rows, "tiers": cfg["tiers"],
            "ssn": ssn_used}


def summarize(res: dict) -> dict:
    """評価点×季節×補正方式ごとの 24 時間中の可用率（3 局以上）と RNP 適合率（2·drms を 95% 値とみなす）."""
    out: dict = {}
    for r in res["errors"]:
        key = f'{r["year"]}-{r["month"]:02d}'
        d = out.setdefault(r["point"], {}).setdefault(key, {"hours": 0, "available": 0, **{t: {} for t in res["tiers"]}})
        d["hours"] += 1
        d["available"] += r["n_usable"] >= 3
        for t in res["tiers"]:
            e = r[f"drms_km_{t}"]
            for name, lim in RNP.items():
                d[t][name] = d[t].get(name, 0) + (e is not None and 2 * e <= lim)
            d[t].setdefault("_errs", []).append(e)
    for p in out.values():
        for d in p.values():
            for t in res["tiers"]:
                errs = [e for e in d[t].pop("_errs") if e is not None]
                d[t]["median_drms_km"] = round(float(np.median(errs)), 1) if errs else None
                for name in RNP:
                    d[t][name] = f'{round(100 * d[t][name] / d["hours"])}%'
            d["available"] = f'{round(100 * d["available"] / d["hours"])}%'
    return out


def plot(res: dict, path: Path, tier: str = "realtime") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors, font_manager
    for n in ("Noto Sans CJK JP", "Hiragino Sans", "IPAexGothic"):
        if any(n in f.name for f in font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = n
            break
    epochs = sorted({(r["year"], r["month"]) for r in res["errors"]})
    pts = res["points"]
    fig, axes = plt.subplots(1, len(epochs), figsize=(4.2 * len(epochs) + 1.5, 0.42 * len(pts) + 2.2), sharey=True)
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad("#bbbbbb")
    norm = colors.LogNorm(vmin=0.5, vmax=200)
    for ax, (y, m) in zip(axes, epochs):
        Z = np.full((len(pts), 24), np.nan)
        for r in res["errors"]:
            if (r["year"], r["month"]) == (y, m) and r[f"drms_km_{tier}"] is not None:
                Z[pts.index(r["point"]), r["hour_utc"]] = r[f"drms_km_{tier}"]
        im = ax.imshow(np.ma.masked_invalid(Z), aspect="auto", cmap=cmap, norm=norm)
        ax.set_title(f"{y}年{m}月")
        ax.set_xticks(range(0, 24, 3))
        ax.set_xticklabels([f"{h}" for h in range(0, 24, 3)])
        ax.set_xlabel("UTC [時]（JST = UTC+9）")
        ax.set_yticks(range(len(pts)))
        ax.set_yticklabels(pts)
        lim = RNP["RNP-10"] / 2
        for i in range(len(pts)):
            for h in range(24):
                if np.isfinite(Z[i, h]) and Z[i, h] <= lim:
                    ax.plot(h, i, "k.", ms=2.5)
    cb = fig.colorbar(im, ax=axes, shrink=0.85)
    cb.set_label("位置誤差 drms [km]（灰色: 3局未満で測位不可）")
    fig.suptitle(f"上空波 Rx-TDOA の時間帯別位置誤差（補正方式: {tier}、送信所: {'・'.join(res['stations'])}）\n"
                 f"点は RNP-10（95% 18.5 km）を満たす時刻。周波数は CHUCK/VOACAP で時刻ごとに選択", fontsize=10)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="上空波 Rx-TDOA の時間帯別評価（CHUCK で周波数計画）")
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--out", default="out/skywave")
    ap.add_argument("--chuck-url", help="稼働中の CHUCK の URL（省略時は CHUCK_APP_DIR のエンジンを直接使う）")
    ap.add_argument("--chuck-user")
    ap.add_argument("--chuck-password")
    ap.add_argument("--ssn-from", help="実効 SSN を取る CHUCK の URL（省略時は --chuck-url）")
    ap.add_argument("--with", dest="extra", nargs="*", default=[], help="extra_stations から追加する送信所")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    cache = od.parent / "chuck_cache"
    if a.chuck_url:
        auth = (a.chuck_user, a.chuck_password) if a.chuck_user else None
        backend = ChuckHttpBackend(a.chuck_url, cache_dir=cache, auth=auth)
    else:
        backend = ChuckEngineBackend(cache_dir=cache)
    res = run(cfg, backend, a.extra, ssn_url=a.ssn_from or a.chuck_url)
    for name, rows in (("frequency_plan.csv", res["plan"]), ("position_error.csv", res["errors"])):
        with open(od / name, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    s = summarize(res)
    (od / "summary.json").write_text(json.dumps({"stations": res["stations"], "ssn": res["ssn"], "summary": s}, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    for t in res["tiers"]:
        plot(res, od / f"skywave_{t}.png", t)
    print(json.dumps({"stations": res["stations"], "ssn": res["ssn"], "out": str(od)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
