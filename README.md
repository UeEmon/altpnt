# altpnt-sim — GPS拒否環境下 代替PNT・ソノブイネットワーク位置補正シミュレータ

開発引継書（`docs/開発引継書_代替PNTシミュレータ.md`）と提案書（`docs/提案書_代替PNTシステム.md`）に基づく新規プロジェクト。
**Phase 1（P0）**のコア基盤と、Phase 2 の数理・RFモデルの最初の版を実装済み。


> **確定した前提（2026-10-28）**: 周波数は 3〜30 MHz。**母機は HF を受信しない**。RF ブイの SDR が複数局の HF を受信して
> VHF で母機に送り、母機は UHF でブイにコマンドを送る。母機の測位は UHF TOA と VHF TDOA の併用。ブイは作戦海域のみ。
> この前提のシナリオは **`altpnt-buoynet`**（1 機）と **`altpnt-relay`**（交代機、航空機間のデータ引き継ぎなし）。構成・結果・仮定は [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。
> `altpnt-run` と `altpnt-longrange` は機上で HF を受信する旧前提のシナリオで、比較用に残している。

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

## 上空波の時間帯別評価（CHUCK で周波数計画）

既存の HF 伝搬予算プログラム CHUCK（本家 UeEmon/VOCAP-COMPATIBILITY-、VOACAP 互換。公開版は [UeEmon/CHUCK-public-](https://github.com/UeEmon/CHUCK-public-)）を呼び、
送信所×評価点×時刻ごとに周波数を選びます（信頼度 `rel_min` 以上の候補から選択）。そのうえで、その時刻に使える送信所の組から位置誤差を出します。
VOACAP の生出力から、伝搬モード・仮想反射高度・マルチパス確率も読み、上空波の距離誤差モデルに使います。

```bash
# CHUCK のエンジンを直接使う（計算回数の上限なし。voacapl は CHUCK の Dockerfile と同じ手順でビルド）
CHUCK_APP_DIR=../VOCAP-COMPATIBILITY-/app VOACAP_BIN=/usr/local/bin/voacapl \
VOACAP_MODEL=/usr/local/share/voacapl/itshfbc altpnt-skywave --out out/skywave
# 稼働中の CHUCK の API を使う（公開版は 1 時間 60 回まで。結果はキャッシュして再計算しない）
altpnt-skywave --chuck-url http://localhost:18082 --out out/skywave
altpnt-skywave --with 南鳥島 与那国 --out out/skywave_ext     # 候補局を足して比較
altpnt-skywave --ssn-from http://localhost:18082 --out out/skywave_now   # 実効 SSN の取得先を変える（既定は config の chuck_url）
```

当月（`ssn: effective`）は、`chuck_url` の CHUCK 公開版（https://168-110-7-225.sslip.io）から実効 SSN を取得します。取れなければ `ssn_fallback` を使い、どちらを使ったかを `summary.json` に記録します。
設定は `config/skywave_japan.yaml`（送信所、評価点、季節と SSN、周波数の候補、回線条件、補正方式）。
出力は `frequency_plan.csv`（周波数運用表）、`position_error.csv`、`summary.json`、補正方式ごとの時間帯別誤差の図。
計算エンジンは本家と公開版で同一（同じ入力で結果が完全一致することを確認済み）。
補正方式: `model`（反射高度 σ20 km）/ `realtime`（σ5 km）/ `differential`（近くの参照点で差分補正、距離誤差 σ0.2 km は仮定）。
VOACAP は地表波を計算しないため、`ground_wave_km` 以内は地表波として固定の誤差を置いています。

## 長距離シナリオ（上空波 Rx-TDOA + EKF）

房総沖で GPS を失い、1,000 NM 先の作戦海域へ進出して 4 時間周回し、帰投するまで（約 11 時間）を評価します。
航路に沿って 150 km ごとに CHUCK で送信所ごと・時刻ごとの周波数を決め、使える局が入れ替わる中で
EKF（位置・INS 速度バイアス・送信所ごとの電離層バイアス）が INS の誤差をどこまで抑えられるかを、補正方式ごとに比べます。

```bash
CHUCK_APP_DIR=../VOCAP-COMPATIBILITY-/app VOACAP_BIN=... VOACAP_MODEL=... \
altpnt-longrange --depart-utc 22 10 --runs 8 --out out/longrange           # 6 局
altpnt-longrange --with 南鳥島 与那国 --runs 8 --out out/longrange_8        # 8 局
```

結果（6 局、出発 22 UTC、モンテカルロ 8 回の平均。作戦海域での平均誤差 / 帰投完了時）:
INS 単独 11.9 / 26.9 km、モデル補正 5.5 / 1.4 km、リアルタイム推定 3.6 / 1.2 km、作戦海域の参照ブイで差分補正 0.55 / 0.8 km。
出発時刻（22 UTC と 10 UTC）による差は小さい。`consistency`（実際の RMS 誤差 / EKF が見積もる誤差）は 0.6〜1.4。

前提と注意: 電離層の残差は送信所ごとに独立な Gauss-Markov 過程（相関時間はモデル補正 3 時間、ほかは 30 分）で、
EKF はその統計を正しく知っている。磁気嵐・太陽フレアによる突発的な変化や、局間で相関した誤差は入れていない。
出発直後に地表波で INS の速度バイアスを較正できることが効いている（地表波なしではモデル補正の作戦海域誤差が 5.5 → 9.2 km）。

## ブイ中心構成（確定した前提）: altpnt-buoynet

```bash
CHUCK_APP_DIR=../VOCAP-COMPATIBILITY-/app VOACAP_BIN=... VOACAP_MODEL=... \
altpnt-buoynet --runs 6 --out out/buoynet                 # UHF TOA ＋ VHF TDOA
altpnt-buoynet --links uhf --runs 4 --out out/buoynet_uhf  # UHF だけ（vhf も可）
altpnt-buoynet --scenario storm --runs 4 --out out/buoynet_storm
```

ブイ網の推定器（各ブイの位置・漂流速度・時計 ＋ 全ブイ共通の送信所ごとの電離層バイアス）で HF の TOA を処理し、
母機の EKF はその推定値と共分散を使って UHF/VHF の到着時刻差を処理します。結果と仮定は [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 次の作業（引継書 Phase 対応）

- Phase 1 残: Apple Silicon 上での compose 動作確認、aqua の layer/route（投下点への経路・横風補正）と Dubins 計画の統合検討
- Phase 2: 搬送波位相（整数不確定性・位相スリップ）、RTL-SDR 実測 IQ の投入（`data/rtlsdr/*.cu8`）、
  地球曲率/測地線距離、ブイ位置誤差の時間相関を EKF に反映（現在は白色扱いで楽観側）
- Phase 3: LiteLLM 経由の状況解析をシナリオ出力に接続（`altpnt-run --llm` は実装済み）、
  シリアル 115200bps 制限での実データフロー

未解決の論点は `docs/OPEN_ISSUES.md` を参照。
