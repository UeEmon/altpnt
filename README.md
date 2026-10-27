# altpnt-sim — GPS拒否環境下 代替PNT・ソノブイネットワーク位置補正シミュレータ

開発引継書（`docs/開発引継書_代替PNTシミュレータ.md`）と提案書（`docs/提案書_代替PNTシステム.md`）に基づく新規プロジェクト。
**Phase 1（P0）**のコア基盤と、Phase 2 の数理・RFモデルの最初の版を実装済み。

## クイックスタート

```bash
pip install -e ".[dev,aqua]"                # aqua = aqua-drift-simulator（git から取得）
pytest -q                                   # 16 テスト（aqua 未導入時は連携テスト 3 件をスキップ）
altpnt-run --out out/default                # 既定シナリオ → out/default/summary.json, scenario.png
altpnt-run --config config/scenario_degraded_hf.yaml --out out/degraded
```

Docker（エッジ／クラウド／LiteLLM の分離構成）:

```bash
cp .env.example .env        # API キーと OLLAMA_API_BASE を設定
docker compose up --build   # cloud API: http://localhost:8080/docs, LiteLLM: :4000
```

**本番環境:** 既存の Oracle Cloud VM（[UeEmon/oracle-cloud-deploy](https://github.com/UeEmon/oracle-cloud-deploy)）に相乗りしています。
`https://altpnt.<IP>.sslip.io/docs`（CHUCK 管理サイトと同じパスワード）。`main` へのマージは 5 分以内に自動で反映されます。
VM では API（`cloud`）だけが動き、LiteLLM とエッジのシナリオ実行は起動しません。

単独の VM に立てる場合は [`deploy/oci/README.md`](deploy/oci/README.md)（Resource Manager 用 Terraform スタック）。

## 構成

```
src/altpnt_sim/
  drift/ocean.py        漂流積分（RK2+拡散, ブイ方向ベクトル化）と内蔵流速場（一様流+渦+M2潮流+風圧流）
  drift/aqua.py         aqua-drift-simulator 連携
                        AquaDriftAdapter: 潮流場 v = base + G(p - p_ref) を局所 ENU で一括評価
                        AquaDropModel  : 高度別風・自由落下で着水点を計算、直前の着水点から平均風を推定して投下補正
  dubins/path.py        Dubins 6語（LSL,RSR,LSR,RSL,RLR,LRL）の一括評価・弧長サンプリング
  dubins/planner.py     PNT ブイ 5〜9 本のリング配置、投下順序の候補評価、周回円への遷移、等速連続サンプリング
  nav/tdoa.py           逆TDOA（Rx-TDOA）: 測定モデル・ヤコビアン・重み付きGauss-Newton・GDOP（受信者方向ベクトル化）
  nav/ins.py            INS 速度バイアス誤差モデル
  nav/ekf.py            INS バウンディング EKF（位置+速度バイアス, χ²ゲート, Joseph形式）
  rf/propagation.py     HF地表波（SNR減衰, 包絡線TOAのCRLB, ASF相当の伝搬遅延残差）
                        VHF（フリス, 2波海面反射+粗度, 波浪傾斜・遮蔽損失, BER, リンク収支）
  rf/noise.py           RTL-SDR .cu8/.npy 読込, 合成 Rician/ガウスドップラー・フェージング
  rf/ddc.py             DDC（ミキシング+有理数リサンプル）, 120kbps 収容判定
  gateway/llm_client.py LiteLLM（OpenAI互換）最小クライアント（標準ライブラリのみ）
  scenario.py           統合シナリオ（ins_only / hf / hf_grid の3方式を同一観測で比較）
services/edge, services/cloud   コンテナ（edge: DDC+シナリオ, cloud: FastAPI /plan /scenario）
config/litellm_config.yaml      論理モデル名 edge-fast / cloud-analyst → Ollama / Claude / GPT
```

## シナリオのモデル

1. GPS ロスト時点から、沿岸の友軍 HF 局 4 局（150〜550 km）を基準に東方へ進出
2. グリッド中心まで Dubins 経路で移動しながら PNT ブイ 7 本を投下（ASW ブイとは別ノード）。
   高度 1500 m から約 52 秒落下し、風で流されて着水（aqua_drift.wind）。着水後は aqua の潮流場で漂流
3. グリッド周回（キャリブレーション）60 分 → 開始点へ帰投
4. 各ブイ: HF Rx-TDOA 測位を漂流 KF（位置・速度）で平滑して自位置を確定
5. 航空機: INS + 機上 HF Rx-TDOA（10 s 毎, 欠測あり）+ ブイ網の VHF タイムスタンプ TDOA（2 s 毎, 見通し内・パケット成功時・幾何 DOP ≤ 6）

## 現時点の結果（seed=42, 漂流・投下 = aqua）

| 方式 | 周回中 平均 / p95 | 帰投中 平均 / p95 | 帰投完了時 |
|:-|:-:|:-:|:-:|
| INS 単独 | 2,027 / 2,705 m | 3,359 / 3,868 m | 3,921 m |
| 既定（機上HF欠測30%）: hf | 87 / 148 m | 68 / 113 m | 49 m |
| 既定: hf_grid | 88 / 156 m | 77 / 136 m | 41 m |
| 劣化（機上HF欠測97%）: hf | 243 / 364 m | 160 / 256 m | 127 m |
| 劣化: hf_grid | **89 / 162 m** | 262 / 450 m | 228 m |

投下（aqua_drift.wind, 高度 1500 m, 150 m/s）: 落下 52 s、着水ずれは補正なし平均 ~550 m → 補正あり ~280 m
（残りは放出タイミングでは消せない横風成分。経路側で補正するには aqua の layer/route 方式の取り込みが必要）。

読み取り:
- 機上で HF が安定して受かる限り、ブイ網の追加効果は小さい。ブイの絶対位置も同じ HF 地表波で決めているため、
  空間相関のある伝搬遅延誤差（ASF 相当）はブイと航空機で共通に残り、差分で消えない。
- ブイ網が効くのは**機上 HF が使えないとき**（劣化ケースで周回中 p95 が 364→162 m）。
  グリッドを離れるとブイが一方向に固まり幾何が悪化するため、DOP ゲートで観測を打ち切る（ゲート無しでは帰投時 1.3 km まで悪化した）。
- 誤差の数値は HF モデルの仮定（基準 SNR 30 dB @100 km、CRLB 到達、ASF 残差 30%）に強く依存する。
  提案書の CEP 1〜2.5 km より楽観的なので、Phase 2 で実測ノイズに置き換えて校正すること。

## 次の作業（引継書 Phase 対応）

- Phase 1 残: Apple Silicon 上での compose 動作確認、aqua の layer/route（投下点への経路・横風補正）と Dubins 計画の統合検討
- Phase 2: 搬送波位相（整数不確定性・位相スリップ）、RTL-SDR 実測 IQ の投入（`data/rtlsdr/*.cu8`）、
  地球曲率/測地線距離、ブイ位置誤差の時間相関を EKF に反映（現在は白色扱いで楽観側）
- Phase 3: LiteLLM 経由の状況解析をシナリオ出力に接続（`altpnt-run --llm` は実装済み）、
  シリアル 115200bps 制限での実データフロー

未解決の論点は `docs/OPEN_ISSUES.md` を参照。
