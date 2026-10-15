# CLAUDE.md

津軽三味線教室の公式サイトで稼働している FAQ ボット（PHP、4 段構成）を題材に、
RAG アプリケーションの中核を TypeScript で再実装し、精度評価とコスト計測まで含めて公開するリポジトリ。

- 進行は `docs/PLAN.md` のフェーズ順に行う。フェーズの順序は入れ替えず、完了条件を満たしてチェックを付けてから次へ進む。
- 開発・検証はローカルの Docker（Postgres + pgvector、Python ツール用コンテナ）で行い、配信は AWS Lambda（レスポンスストリーミング）へデプロイする。

## 作業の約束
- TypeScript のコードには設計意図をコメントで添える（型の選び方、async/await、`for await` による SSE の読み方など）。
- Python（KB 構築・評価スクリプト）は説明コメントも適切な量で添える。
- ホスト環境を汚さない。Python は Docker コンテナ（`make py`）、Node は `npx`。
- 鍵はチャットに貼らない。`.env`（gitignore 済み）か `read -s` で渡す。Lambda では環境変数に置く。
- コミット前に鍵の混入を確認する（`scripts/check-secrets.sh`。`make setup` で pre-commit に登録される）。
- 作業はブランチを切って PR にする。main に直接 push しない。
- フェーズを 1 つ終えるごとにコミットし、`docs/PLAN.md` のチェックを更新する。

## スコープ
| 要件 | 到達する状態 |
|---|---|
| TypeScript での開発 | RAG の本体（検索・生成・配信）を TS で書く |
| LLM のストリーミング応答 | Claude の SSE をサーバで受け、ブラウザへ SSE で流す |
| トークン・コスト管理 | リクエストごとに入力/出力トークン数を記録し、1 問あたりの費用を集計 |
| AWS での開発 | Lambda のレスポンスストリーミング + Function URL にデプロイし、ブラウザで流れることを確認 |
| RAG 精度検証・Embedding 選定・pgvector | Postgres + pgvector（Docker）で検索。Embedding 2 モデルを Recall@k / MRR で比較 |
| 検証コード・評価レポート | 計測結果を README に表で載せる |

スコープ外（今後の拡張として README に記す）: TTS/STT、Bedrock、CDK/Terraform、AWS 上の DB（Aurora/RDS）。
採用しないもの（運用コストの理由）: Aurora Serverless v2、RDS、NAT Gateway、EC2。

## リポジトリ構成
```
corpus/      ナレッジの原本（FAQ・ページ本文・評価セット）
reference/   本番ボット（PHP）からの参考コピー。そのままでは動かない
docs/        PLAN.md（フェーズ別チェックリスト）、設計メモ
db/init/     Postgres 初期化 SQL（pgvector 拡張、chunks テーブル）
docker/      Python ツール用コンテナの Dockerfile
scripts/     Python（チャンク化・埋め込み・投入・評価）とシェル補助
src/         TypeScript（検索・生成・SSE 配信・Lambda ハンドラ）
web/         動作確認用の HTML
infra/       Lambda デプロイ用スクリプト
logs/        usage.jsonl（gitignore）
```

### corpus の形式
- `corpus/faq/*.md`（8 本・63 エントリ）: `## id | 見出し` → `Q:` 行が複数 → `A:` → `LINK:`
- `corpus/pages/*.md`（17 本）: 先頭に `<!-- page_key: -->` `<!-- title: -->`。サイトの公開本文
- `corpus/eval/positive.md`: 言い換え 33 問 `Q: 質問 => 期待する FAQ の id`
- `corpus/eval/negative.md`: 回答してはいけない 28 問 `Q: 質問 [分類]`

### reference/okubo-shamisen の中身
- `build_kb.py`（チャンク化 `chunk_page`）、`embedders.py`（Voyage 呼び出しとキャッシュ）、`textmatch.py`（Tier0）
- `eval.py`（評価・閾値の掃引・判定段）、`eval-report.md`（実測結果）
- `engine.php`（4 段構成の本体。生成プロンプトと引用検証）、`router-prompt.txt`（FAQ 判定プロンプト）

## 本番ボットから引き継ぐ前提知識
- 4 段構成: Tier0 文字バイグラム一致（τ0=0.85）→ Tier1 埋め込み類似で定型回答（τ1=0.90）→
  Tier1r 上位 3 候補（類似度 0.60 以上）を Haiku 4.5 に渡し FAQ の id か NONE を返させる →
  Tier2 チャンク検索（top_k=12、τ2=0.40）+ 生成。根拠外は NO_ANSWER で拒否、`[SOURCE: n]` の引用を検証する。
- Voyage の input_type: 質問どうしを比べるときは両方 `query`。本文チャンクは `document`、質問は `query`。
- 閾値 1 本では「オンラインレッスンはありますか？」（記載なし）が「体験レッスンはありますか？」に吸われる。
  誤答ゼロには τ1=0.90 が必要だが、言い換えのカバー率が 18% に落ちる。これが LLM 判定段を足した理由（eval-report.md）。
- Voyage: voyage-3.5-lite、output_dimension=256。
- anthropic Python SDK 1.x は temperature 引数が外れており `extra_body={"temperature": 0}` で渡した。TS SDK に同じ制約がないか最初に確認する。
- Haiku 4.5 は temperature を受け付けるが、Haiku 5.5 は 400 を返す。

## 設計の決め事
- 生成は claude-sonnet-5-5（コスト抑制）。モデル ID と料金は公式ドキュメントで確認してから固定する。
- 判定段を入れる場合は claude-haiku-4-5、temperature 0、max_tokens 16。
- プロンプトは本番の規則を踏襲: 文脈外は NO_ANSWER、`[SOURCE: n]` 必須、ユーザー入力は system に混ぜず `<context>` と `<question>` に分ける（engine.php 参照）。
- 費用ログは 1 行 JSON: `{ts, tier, model, input_tokens, output_tokens, cache_read_tokens, cost_jpy, ttft_ms, total_ms}`。ttft_ms（最初の 1 文字までの時間）は必ず取る。
- system に cache_control を付け、cache_read_input_tokens が増えることを確認する。
- Lambda からはローカル DB に届かないため、ベクトル同梱の `kb.json` を全件比較にする。理由と「本番は RDS に置き換える」方針を README に書く。
- Embedding は voyage-3.5-lite と voyage-3.5 を同じ次元（256）で比較し、結果を見て採用を決める。
  256 は本番からの引き継ぎで根拠の記録がないため、採用モデルで 1024 も 1 回測り、256 を選ぶ根拠を README に残す。

## 事前準備
- Node 22、Docker（Compose v2）、AWS CLI（`aws sts get-caller-identity` が通る、ap-northeast-1）
- Lambda 実行ロール（AWSLambdaBasicExecutionRole のみ）の ARN
- `.env.example` を `.env` にコピーし、ANTHROPIC_API_KEY・VOYAGE_API_KEY・LAMBDA_ROLE_ARN を記入
- `make setup`（pre-commit 登録、イメージ取得）→ `make db-up`

## ローカル環境（Docker）
| コマンド | 内容 |
|---|---|
| `make setup` | pre-commit 登録、`pgvector/pgvector:pg16` と Python ツールイメージの準備 |
| `make db-up` / `make db-down` | Postgres + pgvector の起動 / 停止 |
| `make db-reset` | ボリュームを消して `db/init/*.sql` から再作成 |
| `make psql` | DB に接続 |
| `make py ARGS="scripts/xxx.py"` | Python ツール用コンテナでスクリプトを実行（`.env` を読み込む） |

## README の骨組み
1. 何を作ったか。元になった本番ボット（教室サイトの 4 段構成）の紹介
2. 構成図
3. 計測結果: 段ごとの 1 問あたり費用・TTFT・Embedding 比較表
4. 設計判断: 埋め込みの非対称、閾値 1 本の限界と判定段、引用検証
5. 使い方（ローカル / Lambda）
6. 今後の拡張: Bedrock、RDS、CDK、TTS/STT（邦楽の固有名詞の CER 比較）
