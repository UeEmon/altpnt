"""HF 伝搬予測（CHUCK / VOACAP）を使った上空波 Rx-TDOA の時間帯別評価.

CHUCK（UeEmon/CHUCK-public-, VOACAP 互換の HF 伝搬予算プログラム）を呼び出し、
送信所ごと・時刻ごとに使う周波数を決め、その時刻に使える送信所の組から
HDOP と位置誤差を求める。
"""
from .voacap import parse_voacap
from .chuck import ChuckEngineBackend, ChuckHttpBackend, chuck_request
from .plan import LinkHour, plan_link, position_error

__all__ = ["parse_voacap", "ChuckEngineBackend", "ChuckHttpBackend", "chuck_request",
           "LinkHour", "plan_link", "position_error"]
