# CLAUDE.md

教室サイト（津軽三味線・大久保教室）の FAQ ボットを題材に、LLM アプリの必須スキルを
1 日（5 時間）で「経験した」と言える成果物を作り、ポートフォリオにするリポジトリ。
作業は Mac のターミナルの Claude Code で行う（AWS 認証情報はローカルのプロファイル）。

## 作業者について（進め方の約束）
- TypeScript は勉強中。TS のコードには「なぜそう書くか」を短く添える（型、async/await、`for await` による SSE の読み方など）。
- Python はそれなりに読める。評価スクリプトの説明は不要。
- AWS は EC2・Amplify・Lambda・Aurora・S3・Route 53 の簡単な部分は経験あり。Terraform は未経験。
- グローバル環境を汚さない。Python は venv か Docker、Node は npx。
- 鍵はチャットに貼らない。`.env`（gitignore 済み）か `read -s` で渡す。
- コミット前に鍵が混ざっていないか必ず確認する。
- 作業はブランチを切って PR にする。main に直接 push しない。

## 要件と到達状態
| 要件 | 到達する状態 |
|---|---|
| TypeScript での開発 | RAG の本体（検索・生成・配信）を TS で書く |
| LLM のストリーミング応答 | Claude の SSE をサーバで受け、ブラウザへ SSE で流す |
| トークン・コスト管理 | リクエストごとに入力/出力トークン数を記録し、1 問あたりの費用を集計 |
| AWS での開発 | Lambda のレスポンスストリーミング + Function URL にデプロイし、ブラウザで流れることを確認 |
| RAG 精度検証・Embedding 選定・pgvector | Postgres + pgvector（Docker）で検索。Embedding 2 モデルを Recall@k / MRR で比較 |
| 検証コード・評価レポート | 計測結果を README に表で載せる |

やらないこと（2 日目以降）: TTS/STT、Bedrock、CDK/Terraform、AWS 上の DB（Aurora/RDS）。
作らないもの（費用）: Aurora Serverless v2、RDS、NAT Gateway、EC2。

## リポジトリの中身
- `corpus/faq/*.md`（8 本・63 エントリ）: `## id | 見出し` → `Q:` 行が複数 → `A:` → `LINK:`
- `corpus/pages/*.md`（17 本）: 先頭に `<!-- page_key: -->` `<!-- title: -->`。サイトの公開本文
- `corpus/eval/positive.md`: 言い換え 33 問 `Q: 質問 => 期待する FAQ の id`
- `corpus/eval/negative.md`: 回答してはいけない 28 問 `Q: 質問 [分類]`
- `reference/okubo-shamisen/`: 本番ボット（WordPress テーマ内、PHP）からの参考コピー。そのままでは動かない
  - `build_kb.py`（チャンク化 `chunk_page`）、`embedders.py`（Voyage 呼び出しとキャッシュ）、`textmatch.py`（Tier0）
  - `eval.py`（評価・閾値の掃引・判定段）、`eval-report.md`（実測結果）
  - `engine.php`（4 段構成の本体。生成プロンプトと引用検証）、`router-prompt.txt`（FAQ 判定プロンプト）

## 本番ボットで学んだこと（前提知識）
- 4 段構成: Tier0 文字バイグラム一致（τ0=0.85）→ Tier1 埋め込み類似で定型回答（τ1=0.90）→
  Tier1r 上位 3 候補（類似度 0.60 以上）を Haiku 4.5 に渡し FAQ の id か NONE を返させる →
  Tier2 チャンク検索（top_k=12、τ2=0.40）+ 生成。根拠外は NO_ANSWER で拒否、`[SOURCE: n]` の引用を検証する。
- Voyage の input_type: 質問どうしを比べるときは両方 `query`。本文チャンクは `document`、質問は `query`。
- 閾値 1 本では「オンラインレッスンはありますか？」（記載なし）が「体験レッスンはありますか？」に吸われる。
  誤答ゼロには τ1=0.90 が必要だが、言い換えのカバー率が 18% に落ちる。これが LLM 判定段を足した理由（eval-report.md）。
- Voyage: voyage-3.5-lite、output_dimension=256。無料枠内。
- anthropic Python SDK 1.x は temperature 引数が外れており `extra_body={"temperature": 0}` で渡した。TS SDK に同じ落とし穴がないか最初に確認する。
- Haiku 4.5 は temperature を受け付けるが、Haiku 5.5 は 400 を返す。

## 設計の決め事
- 生成は claude-sonnet-5-5（コスト抑制）。モデル ID と料金は当日に公式で確認。
- 判定段を入れるなら claude-haiku-4-5、temperature 0、max_tokens 16。
- プロンプトは本番の規則を踏襲: 文脈外は NO_ANSWER、`[SOURCE: n]` 必須、ユーザー入力は system に混ぜず `<context>` と `<question>` に分ける（engine.php 参照）。
- 費用ログは 1 行 JSON: `{ts, tier, model, input_tokens, output_tokens, cache_read_tokens, cost_jpy, ttft_ms, total_ms}`。ttft_ms（最初の 1 文字までの時間）は必ず取る。
- 任意: system に cache_control を付け、cache_read_input_tokens が増えることを確認。
- Lambda からは DB に届かないので、ベクトル同梱の kb.json を全件比較にし、理由を README に書く。

## 前日準備
- Node 22、Docker、Python 3.11 以上、AWS CLI（`aws sts get-caller-identity` が通る、ap-northeast-1）
- Lambda 実行ロール（AWSLambdaBasicExecutionRole のみ）の ARN
- ANTHROPIC_API_KEY・VOYAGE_API_KEY を `.env` に
- `docker pull pgvector/pgvector:pg16`

## 5 時間の計画
| 時間 | 作業 | 完了の確認 |
|---|---|---|
| 0:00–0:45 | npm init、TypeScript、esbuild。@anthropic-ai/sdk の messages.stream() でターミナルに流す | 逐次表示される |
| 0:45–1:45 | Docker で Postgres + pgvector。`chunks(id, page_key, title, heading, text, embedding vector(256))`。Python でチャンク化→Voyage（document）→投入。TS から `<=>` で top_k 検索 | 「紋紅木とは」で道具コラムが上位 |
| 1:45–2:45 | /ask（Node http か Hono）。質問→Voyage（query）→pgvector→Claude ストリーミング→ブラウザへ SSE。usage からトークンを取り円換算して logs/usage.jsonl へ。簡単な HTML | ブラウザに流れ、JSONL に 1 行増える |
| 2:45–3:45 | Lambda へ。awslambda.streamifyResponse + Function URL（InvokeMode: RESPONSE_STREAM）。esbuild で 1 ファイルに束ねて zip。鍵は Lambda 環境変数 | Function URL で流れる。CloudWatch にログ |
| 3:45–4:30 | voyage-3.5-lite と voyage-3.5 を評価セットで比較（Recall@1/@3、MRR、拒否 28 問の上位スコア分布）。pgvector の全件比較と HNSW の一致と速度も 1 行 | 表が出る |
| 4:30–5:00 | README: 構成図、計測表、設計判断、「本番は RDS に置き換え」 | push 済み |

遅れたら削る順: Embedding 比較 → HTML 画面（curl で可）→ pgvector（JSON 全件比較へ）。AWS デプロイは削らない。

## README の骨組み
1. 何を作ったか。元になった本番ボット（教室サイトの 4 段構成）の紹介
2. 構成図
3. 計測結果: 段ごとの 1 問あたり費用・TTFT・Embedding 比較表
4. 設計判断: 埋め込みの非対称、閾値 1 本の限界と判定段、引用検証
5. 使い方（ローカル / Lambda）
6. 次にやること: Bedrock、RDS、CDK、TTS/STT（邦楽の固有名詞の CER 比較）
