#!/usr/bin/env bash
# RTL-SDR V4 で HF 帯ノイズを収録（HF は V4 内蔵アップコンバータで直接受信）
# 使い方: scripts/capture_rtlsdr.sh 8000000 60   # 8 MHz を 60 秒
set -euo pipefail
FREQ=${1:-8000000}
SECS=${2:-60}
RATE=2400000
OUT="data/rtlsdr/hf_${FREQ}_$(date +%Y%m%d_%H%M%S).cu8"
mkdir -p data/rtlsdr
rtl_sdr -f "$FREQ" -s "$RATE" -n $((RATE * SECS)) "$OUT"
echo "saved: $OUT  （edge コンテナ／NoiseInjector が自動で使用）"
