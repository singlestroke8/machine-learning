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
- 

## 未解決・あとで確認
- 