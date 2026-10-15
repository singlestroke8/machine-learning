# 作業メモ

検証で分かったこと、確認した値、判断の理由を 1 行ずつ残す。
README の「設計判断」と「計測結果」の材料にする。

## API・SDK の挙動

- 2026-10-15 TS SDK（@anthropic-ai/sdk@0.134.0）は temperature を型として受け付ける
- 2026-10-15 claude-sonnet-5-5 + temperature: 0 → 400 invalid_request_error "`temperature` is deprecated for this model."
- 2026-10-15 claude-haiku-4-5 + temperature: 0, max_tokens: 16 → OK（in=19 out=5）。判定段はこの設定で使える

## 料金・モデル ID の確認
<!-- 単位: USD / 100 万トークン。src/pricing.ts を変えたら必ずここにも書く -->
| 確認日 | モデル | 入力 | 出力 | キャッシュ読み | キャッシュ書き(5分) | 出典 |
|---|---|---|---|---|---|---|
| 2026-10-15 | claude-sonnet-5-5 | $2 | $10 | $0.10（入力の 0.05 倍） | $2.50 | https://platform.claude.com/docs/en/about-claude/pricing.md |
| 2026-10-15 | claude-haiku-4-5 | $1 | $5 | $0.10（入力の 0.1 倍） | $1.25 | 同上 |

為替: 1 USD = 157.9000 円（2026-10-15 時点）

## 計測（試し打ち）
| 日付 | モデル | 質問 | in | out | cache_read | 円 | TTFT ms | total ms | 備考 |
|---|---|---|---|---|---|---|---|---|---|

## 判断と理由
- 2026-10-16 チャンク化・埋め込み・投入・書き出しを 4 本に分けた。チャンクを見直しても Voyage の再課金が起きない
- 2026-10-16 埋め込みのキャッシュの鍵に input_type を含めた。同じ文でも query と document でベクトルが変わる
- 2026-10-16 load_kb はモデル単位で全削除→全投入（1 トランザクション）。チャンク id が通し番号でずれるため差分更新しない
- 2026-10-16 kb.json は DB から書き出す（DB を正とする）。ローカル検索と Lambda で同じ中身を使うため 

## 未解決・あとで確認
- ページ冒頭の目次チャンク（見出しなし）が検索上位に混ざる。Phase 3 の評価で除外するか判断
- HNSW + WHERE model の組み合わせで topK 件に届かない可能性。Phase 3 の一致率測定で確認
- 256 次元の根拠がない → Phase 3 で 1024 と比較