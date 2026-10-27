# Oracle Cloud (OCI) へのデプロイ

Ubuntu 22.04 の Compute インスタンス 1 台に Docker Compose で `cloud`（FastAPI）・`litellm` を常駐させ、
`edge` で既定シナリオを 1 回実行します。既定では **SSH(22) だけを開け**、API は SSH トンネル経由で使います。

## Resource Manager（コンソール）で立ち上げる

1. `altpnt-oci-stack.zip`（このディレクトリの `*.tf`・`cloud-init.yaml`・`schema.yaml` をまとめた zip）を用意
   ```bash
   cd deploy/oci && zip altpnt-oci-stack.zip *.tf cloud-init.yaml schema.yaml
   ```
2. OCI コンソール → 開発者サービス → Resource Manager → スタック → **スタックの作成** → zip をアップロード
3. 変数を入力: コンパートメント、**SSH 公開鍵**、接続元 CIDR（自分の IP/32 推奨）。API キーは空のままで可
4. **適用(Apply)** → 完了後「出力」の `public_ip` / `api_tunnel` を確認
5. セットアップ（Docker 導入・ビルド）に 5〜10 分。進捗は `ssh ubuntu@<IP> 'tail -f /var/log/altpnt-setup.log'`
   最後に `=== altpnt ready` が出れば完了

## 使い方

```bash
ssh -N -L 8080:127.0.0.1:8080 -L 4000:127.0.0.1:4000 ubuntu@<IP>   # 手元で実行
# → http://localhost:8080/docs   （/plan, /scenario）
```

LLM 解析を使う場合は SSH で `/opt/altpnt/.env` に `ANTHROPIC_API_KEY` を書き、
`cd /opt/altpnt && docker compose up -d litellm cloud`。

更新: `cd /opt/altpnt && git pull && docker compose up -d --build cloud`

## 注意

- **A1（Arm）の容量不足**で作成に失敗することがある。`ad_index` を 1, 2 に変えるか時間を置いて再実行。
- `expose_api = true` にすると 8080 を `allowed_cidr` に公開する（認証なしの API なので CIDR は必ず絞る）。
- Terraform 変数で渡した API キーはインスタンスメタデータ（user_data）に残る。SSH で後から設定する方が安全。
- 削除: スタックの **破棄(Destroy)** で VCN ごと消える。
