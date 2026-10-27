"""CHUCK（HF 伝搬予算プログラム）の呼び出し.

2 通りの呼び方を用意する。どちらも CHUCK の /api/predict と同じ入力（辞書）を受け、
VOACAP の生出力を parse_voacap で全項目に分解して返す。

- ChuckEngineBackend: CHUCK の app/engine.py を直接 import して VOACAP を回す（大量計算向け）。
    CHUCK_APP_DIR      … CHUCK リポジトリの app ディレクトリ
    VOACAP_BIN / VOACAP_MODEL … voacapl 本体と itshfbc データ（CHUCK の Dockerfile と同じもの）
- ChuckHttpBackend: 稼働中の CHUCK に POST /api/predict する。
    公開版は 1 時間あたりの計算回数に上限があるので、結果をディスクにキャッシュし、同じ条件は再計算しない。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from .voacap import parse_voacap


def chuck_request(tx: tuple[float, float], rx: tuple[float, float], year: int, month: int, ssn: float,
                  freqs: list[float], link: dict) -> dict:
    """CHUCK /api/predict の入力辞書を作る. link は config の link 節（電力・アンテナ・雑音・要求 SNR など）."""
    req = {
        "tx_lat": tx[0], "tx_lon": tx[1], "rx_lat": rx[0], "rx_lon": rx[1],
        "year": year, "month": month, "ssn": ssn, "frequencies": list(freqs),
        "power_w": link.get("power_w", 1000), "noise": link.get("noise", 155),
        "min_angle": link.get("min_angle", 3), "reliability": link.get("reliability", 90),
        "tx_antenna": link.get("tx_antenna", "monopole"), "rx_antenna": link.get("rx_antenna", "whip"),
        "modulation": link.get("modulation", "manual"),
    }
    for k, v in link.get("tx_params", {"length_m": 10}).items():
        req["tx_" + k] = v
    for k, v in link.get("rx_params", {}).items():
        req["rx_" + k] = v
    if "bandwidth" in link:
        req.update(snr_input="bandwidth", bandwidth=link["bandwidth"], inband_snr_db=link.get("inband_snr_db", 6))
    else:
        req["required_snr"] = link.get("required_snr", 43)
    return req


def _key(req: dict) -> str:
    return hashlib.sha256(json.dumps(req, sort_keys=True).encode()).hexdigest()[:24]


class _Cached:
    def __init__(self, cache_dir: str | Path | None):
        self.cache = Path(cache_dir) if cache_dir else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)

    def predict(self, req: dict) -> list[dict]:
        if self.cache:
            f = self.cache / (_key(req) + ".json")
            if f.exists():
                return json.loads(f.read_text())
        hours = parse_voacap(self._raw(req))
        if self.cache:
            f.write_text(json.dumps(hours))
        return hours

    def _raw(self, req: dict) -> str:  # pragma: no cover
        raise NotImplementedError


class ChuckEngineBackend(_Cached):
    def __init__(self, app_dir: str | Path | None = None, cache_dir: str | Path | None = None):
        super().__init__(cache_dir)
        app_dir = Path(app_dir or os.environ.get("CHUCK_APP_DIR", ""))
        if not (app_dir / "engine.py").exists():
            raise FileNotFoundError("CHUCK の app ディレクトリ（engine.py）が見つかりません: CHUCK_APP_DIR を設定してください")
        if str(app_dir) not in sys.path:
            sys.path.insert(0, str(app_dir))
        import engine  # CHUCK app/engine.py

        self._mod = engine
        self._engine = engine.Engine()
        if not self._engine.ready():
            raise RuntimeError("voacapl が見つかりません: VOACAP_BIN / VOACAP_MODEL を設定してください")

    def _raw(self, req: dict) -> str:
        return self._engine.run(self._mod.validate(req))["raw_output"]


class ChuckHttpBackend(_Cached):
    def __init__(self, base_url: str | None = None, cache_dir: str | Path | None = "out/chuck_cache",
                 auth: tuple[str, str] | None = None, timeout: float = 90.0):
        super().__init__(cache_dir)
        self.base = (base_url or os.environ.get("CHUCK_URL", "http://localhost:18082")).rstrip("/")
        self.auth = auth
        self.timeout = timeout

    def _raw(self, req: dict) -> str:
        body = json.dumps(req).encode()
        headers = {"Content-Type": "application/json"}
        if self.auth:
            import base64
            headers["Authorization"] = "Basic " + base64.b64encode(f"{self.auth[0]}:{self.auth[1]}".encode()).decode()
        for attempt in range(3):
            r = urllib.request.Request(self.base + "/api/predict", data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(r, timeout=self.timeout) as resp:
                    return json.loads(resp.read())["raw_output"]
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 2:      # 公開版の計算回数上限
                    time.sleep(60)
                    continue
                raise RuntimeError(f"CHUCK /api/predict が {e.code}: {e.read()[:200]!r}") from e
        raise RuntimeError("CHUCK の計算回数上限に達しました")
