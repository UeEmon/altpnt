"""LiteLLM Proxy 用の最小クライアント（OpenAI 互換 /v1/chat/completions）.

アプリ側は論理モデル名（edge-fast / cloud-analyst）だけを指定し、実体の切替は
config/litellm_config.yaml に閉じ込める。標準ライブラリのみで動く（エッジ低リソース前提）。
"""
from __future__ import annotations

import json
import os
import urllib.request


class LiteLLMClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None, timeout: float = 30.0):
        self.base_url = (base_url or os.getenv("LITELLM_BASE_URL", "http://localhost:4000")).rstrip("/")
        self.api_key = api_key or os.getenv("LITELLM_MASTER_KEY", "sk-local-dev")
        self.model = model or os.getenv("LLM_MODEL", "cloud-analyst")
        self.timeout = timeout

    def chat(self, messages: list[dict], max_tokens: int = 512, **params) -> str:
        body = json.dumps({"model": self.model, "messages": messages,
                           "max_tokens": max_tokens, **params}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions", data=body, method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]

    def analyze_summary(self, summary: dict) -> str:
        sys = ("あなたは航法システム評価の補助者です。代替PNTシミュレーション結果の要約を読み、"
               "誤差の傾向・劣化要因・次に試すべきパラメータを簡潔に日本語で述べてください。")
        return self.chat([{"role": "system", "content": sys},
                          {"role": "user", "content": json.dumps(summary, ensure_ascii=False)}])
