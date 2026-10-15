# 実装計画（フェーズ別チェックリスト）

フェーズは番号順に進める。順序は入れ替えない。
各フェーズは「完了条件」をすべて満たし、コミットしてから次へ進む。
途中で止まった場合は、そのフェーズの末尾の「メモ」に理由を書き残す。

凡例: `[ ]` 未着手 / `[x]` 完了

---

## Phase 0: 事前準備

目的: 開発・検証・デプロイに必要な環境と認証情報を揃える。

- [x] Node 22、Docker（Compose v2）、AWS CLI が入っている
- [x] `aws sts get-caller-identity` が通る（リージョン ap-northeast-1）
- [x] Lambda 実行ロール（AWSLambdaBasicExecutionRole のみ）を作り、ARN を `.env` に記入
- [x] `.env.example` を `.env` にコピーし、ANTHROPIC_API_KEY・VOYAGE_API_KEY を記入
- [x] `make setup` が通る（pre-commit 登録、イメージ取得）
- [x] `make db-up` → `make psql` で `\dx` に vector が出る

完了条件: 上のすべてにチェックが付く。

---

## Phase 1: TypeScript 基盤と Claude ストリーミング

目的: TS のビルド環境を整え、Claude の SSE をサーバ側で受け取れることを確認する。

- [x] `npm init`、TypeScript、esbuild、`@anthropic-ai/sdk` を導入（`npx`、ホストにグローバル導入しない）
- [x] `tsconfig.json`（strict、ESM、target ES2022）
- [x] `src/llm.ts`: `messages.stream()` を `for await` で読み、テキスト差分を標準出力へ逐次書き出す
- [x] TS SDK で temperature を渡せるか確認し、結果を `docs/notes.md` に 1 行記録
- [x] モデル ID（claude-sonnet-5-5 / claude-haiku-4-5）と料金を公式ドキュメントで確認し、`src/pricing.ts` に定数化
- [x] `usage` から input_tokens / output_tokens / cache_read_input_tokens を取り出し、円換算して表示

完了条件: `npx tsx src/llm.ts "質問"` で回答が逐次表示され、末尾にトークン数と費用が出る。

メモ: TS のビルド環境

```
git switch -c phase1-ts-setup
node -v
npm init -y
npm pkg set type=module
npm install @anthropic-ai/sdk
npm install -D typescript esbuild tsx @types/node
```

---

## Phase 2: ナレッジベース構築（pgvector）

目的: ページ本文をチャンク化し、Voyage で埋め込み、Postgres + pgvector で類似検索できる状態にする。

- [x] `db/init/001_schema.sql` の `chunks` テーブルを確認（`embedding vector(256)`、`model` 列でモデル別に保持）
- [x] `scripts/build_kb.py`: `reference/okubo-shamisen/build_kb.py` の `chunk_page` を移植してチャンク化
- [x] `scripts/embedders.py`: Voyage 呼び出し（input_type=document、output_dimension=256）とディスクキャッシュ
- [x] `scripts/load_kb.py`: チャンクと埋め込みを `chunks` に投入（モデル名付き）
- [x] `scripts/export_kb.py`: `chunks` を `kb.json` に書き出す（Phase 5 の Lambda 同梱用）
- [x] `src/db.ts`: `pg` で接続し、`<=>`（コサイン距離）で top_k 検索
- [x] `src/search.ts`: 質問を Voyage（input_type=query）で埋め込み、検索結果を返す

完了条件: `npx tsx src/search.ts "紋紅木とは"` で `column-equipment` のチャンクが上位に出る。

メモ: 処理の詳細は docs/kb-pipeline.md、判断とつまずきは docs/notes.md。
「紋紅木とは」→ column-equipment が 1 位（score 0.xx）。

---

## Phase 3: 検索精度の評価と Embedding 選定

目的: 2 つの Embedding モデルを同じ評価セットで比較し、採用するモデルを数値で決める。

- [ ] `scripts/eval_embeddings.py`: FAQ 質問文を埋め込み、`corpus/eval/positive.md` の 33 問で Recall@1 / Recall@3 / MRR を算出
- [ ] `corpus/eval/negative.md` の 28 問について、上位候補の類似度分布（最大・中央値）を算出
- [ ] voyage-3.5-lite と voyage-3.5 を同じ 256 次元で比較し、表にする
- [ ] 採用モデルで 1024 次元（Voyage の既定）も 1 回測り、256 との差（Recall@k / MRR / kb.json サイズ）を表に加える。DB には入れず評価スクリプト内で計算する
- [ ] pgvector の全件比較と HNSW インデックス（`db/init/002_hnsw.sql`）で top_k の一致率と所要時間を比較
- [ ] 採用モデルと閾値の方針を `docs/eval-report.md` に記録

完了条件: `docs/eval-report.md` に比較表（Recall@1 / Recall@3 / MRR / 拒否セットの分布 / 次元 256 と 1024 の差 / HNSW 一致率と速度）がある。

メモ:

---

## Phase 4: 検索 API と SSE 配信、コストログ

目的: ブラウザから質問を送ると、検索→生成→ストリーミング配信まで一気通貫で動く。

- [ ] `src/server.ts`: `/ask` エンドポイント（Node http か Hono）
- [ ] プロンプトは `reference/okubo-shamisen/engine.php` の規則を踏襲（NO_ANSWER、`[SOURCE: n]` 必須、`<context>` と `<question>` の分離）
- [ ] `[SOURCE: n]` の引用検証（存在しない番号を参照したら拒否に落とす）
- [ ] Claude のストリームをブラウザへ SSE（`text/event-stream`）で転送
- [ ] `src/usage.ts`: 1 問ごとに `{ts, tier, model, input_tokens, output_tokens, cache_read_tokens, cost_jpy, ttft_ms, total_ms}` を `logs/usage.jsonl` に追記
- [ ] system プロンプトに `cache_control` を付け、2 回目以降で cache_read_tokens が増えることを確認
- [ ] `web/index.html`: 質問欄と逐次表示の最小 UI
- [ ] `scripts/usage_summary.py`: `logs/usage.jsonl` から 1 問あたり費用・TTFT の平均を集計

完了条件: ブラウザで回答が逐次表示され、`logs/usage.jsonl` に 1 行増え、集計表が出る。

メモ:

---

## Phase 5: AWS Lambda へのデプロイ（レスポンスストリーミング）

目的: 同じ TS コードを Lambda Function URL から配信し、ブラウザでストリーミングされることを確認する。

- [ ] `src/lambda.ts`: `awslambda.streamifyResponse` でハンドラを書く
- [ ] 検索は DB ではなく同梱の `kb.json` を全件比較（Lambda からローカル DB に届かないため。README に理由を書く）
- [ ] esbuild で 1 ファイルに束ね、`kb.json` と一緒に zip（`infra/build.sh`）
- [ ] `infra/deploy.sh`: 関数作成・更新、環境変数（鍵）の設定、Function URL（InvokeMode: RESPONSE_STREAM）
- [ ] CloudWatch Logs に usage の JSON が出ることを確認
- [ ] `web/index.html` のエンドポイントを Function URL に切り替えて動作確認

完了条件: Function URL に対してブラウザから質問し、逐次表示される。CloudWatch にログが残る。

メモ:

---

## Phase 6: ドキュメント

目的: 構成・計測結果・設計判断をまとめ、第三者が読んで再現できる README にする。

- [ ] README: 何を作ったか、元になった本番ボットの紹介
- [ ] 構成図（ローカル / Lambda）
- [ ] 計測表: 段ごとの 1 問あたり費用・TTFT、Embedding 比較と次元比較（256 / 1024）（Phase 3 の表を転載）
- [ ] 設計判断: 埋め込みの非対称、閾値 1 本の限界と判定段、引用検証、Lambda で DB を使わない理由
- [ ] 使い方（ローカル / Lambda）
- [ ] 今後の拡張: Bedrock、RDS、CDK、TTS/STT（邦楽の固有名詞の CER 比較）

完了条件: README を読んだだけで `make db-up` から Lambda デプロイまで再現できる。

メモ:
