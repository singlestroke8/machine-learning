# ナレッジベース構築パイプライン（Phase 2）

サイトのページ本文を検索できる形（チャンク + ベクトル）にして、Postgres と kb.json に置くまでの流れ。

```
corpus/pages/*.md → .cache/chunks.json → .cache/embeddings/ → chunks テーブル → dist/kb.json
     build_kb.py         embedders.py          load_kb.py          export_kb.py
```

## 実行順

| 順 | コマンド | 入力 → 出力 | 課金 |
|---|---|---|---|
| 1 | `make py ARGS="scripts/build_kb.py"` | `corpus/pages/*.md` → `.cache/chunks.json` | なし |
| 2 | `make py ARGS="scripts/embedders.py"` | `chunks.json` → `.cache/embeddings/` | Voyage（未キャッシュ分のみ） |
| 3 | `make py ARGS="scripts/load_kb.py"` | チャンク + ベクトル → `chunks` テーブル | なし（キャッシュを使う） |
| 4 | `make py ARGS="scripts/export_kb.py"` | `chunks` テーブル → `dist/kb.json` | なし |
| 検索 | `npx tsx src/search.ts "質問"` | 質問 → 上位チャンク | Voyage（質問 1 件分） |

コンテンツを追加・修正したら 1〜4 を順に回す。比較用のモデルは `--model voyage-3.5` を付ける。

---

## scripts/build_kb.py

ページ本文を、見出しごとの短い文章（チャンク）に切り分ける。API も DB も使わない。

1. ページ先頭の `<!-- page_key: -->` と `<!-- title: -->` から、ページ名とタイトルを取り出す。
2. 本文を上から読み、`## 見出し` が出るたびに区切る。
3. 区切った塊が 450 字を超えたら段落の境目で分け、60 字未満なら捨てる。
4. 各チャンクに `c0000` から順に番号を振り、`.cache/chunks.json` に書き出す。

## scripts/embedders.py

`.cache/chunks.json` の 246 チャンクを Voyage に送り、1 つずつ 256 個の数値（ベクトル）に変換する。結果はファイルに保存し、同じ文章を二度課金しない。

1. チャンクを読む（`build_kb.py` の出力）。
2. キャッシュを探す。モデル・次元・`input_type`・本文から鍵を作り、保存済みならファイルから読む。
3. キャッシュに無いものだけを 128 件ずつ Voyage に送る。課金されるのはここだけ。
4. 返ってきたベクトルが 256 次元かを確かめ、1 件ずつ `.cache/embeddings/` に保存する。
5. 件数・キャッシュから読んだ数・課金トークン数を表示する。

## scripts/load_kb.py

チャンクとそのベクトルを、Postgres の `chunks` テーブルに 1 行ずつ入れる。

1. `.cache/chunks.json`（`build_kb.py` の出力）を読む。
2. `Embedder` でベクトルを用意する。`embedders.py` を実行済みなら全件キャッシュから読むので課金は 0。
3. DB に接続し、`vector` 型を Python から扱えるようにする。
4. 同じモデルの古い行を消し、246 行を入れる。この 2 つは 1 つのトランザクションにまとめ、途中で失敗したら元に戻る。
5. モデル別の行数を表示する。

## scripts/export_kb.py

`chunks` テーブルの 1 モデル分を、Lambda に同梱する `dist/kb.json` に書き出す。

1. DB から、指定したモデルの行を id 順に読む。
2. 0 件なら止める（`load_kb.py` の実行忘れ）。
3. ベクトルを小数 4 桁に丸めて、JSON にできる形にする。
4. モデル名・次元・作成日時を付けて `dist/kb.json` に書き出す（`dist/` は `.gitignore` 済み）。
5. 件数とファイルサイズを表示する。

## src/db.ts

質問ベクトルに近いチャンクを、pgvector で検索する。

1. DB への接続を 1 つの入れ物（Pool）にまとめて使い回す。最初に使うときに作る。
2. 質問のベクトルを、pgvector が読める文字列 `"[0.1,0.2,...]"` に変換する。
3. 同じモデルの行の中から、コサイン距離（`<=>`）が小さい順に topK 件を取り出す。
4. 類似度（1 − 距離）を付けて、型付きの配列で返す。

## src/search.ts

質問文を埋め込み、`db.ts` で検索して結果を表示する。

1. `.env` を読む（CLI として実行したときだけ）。
2. 質問文を Voyage に送り、256 次元のベクトルにする。本文は `document`、質問は `query` で埋め込む。
3. そのベクトルを `searchChunks`（`db.ts`）に渡し、近いチャンクを取り出す。
4. 順位・類似度・ページ・見出し・本文の冒頭を表示する。埋め込みと DB 検索の所要時間も分けて出す。