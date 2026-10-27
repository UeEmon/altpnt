"""クラウド/親局 API: PNT グリッド展開計画・シナリオ実行・LLM 状況解析."""
from pathlib import Path

import numpy as np
import yaml
from fastapi import FastAPI
from pydantic import BaseModel

from altpnt_sim.dubins import PNTGridPlanner, hex_grid
from altpnt_sim.gateway import LiteLLMClient
from altpnt_sim.scenario import DEFAULT_CFG, run, summarize

app = FastAPI(title="altpnt-sim cloud")


class PlanReq(BaseModel):
    start: tuple[float, float, float]
    center: tuple[float, float]
    n_buoys: int = 7
    spacing: float = 15000.0
    turn_radius: float = 4000.0


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/plan")
def plan(req: PlanReq):
    pts = hex_grid(req.center, req.n_buoys, req.spacing)
    p = PNTGridPlanner(rho=req.turn_radius).plan(req.start, pts)
    return {"order": p.order.tolist(), "words": p.words, "length_m": p.length,
            "drop_points": pts[p.order].tolist(), "orbit_center": p.orbit_center.tolist(),
            "orbit_radius": p.orbit_radius}


@app.post("/scenario")
def scenario(config_path: str | None = None, analyze: bool = False):
    cfg = yaml.safe_load(Path(config_path or DEFAULT_CFG).read_text(encoding="utf-8"))
    s = summarize(run(cfg))
    if analyze:
        s["llm_analysis"] = LiteLLMClient().analyze_summary(s)
    return s
